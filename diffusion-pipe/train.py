import argparse
import os
import wandb
# Disable comfy_kitchen during training to avoid autograd errors
import sys
sys.modules["comfy_kitchen"] = None
from datetime import datetime, timezone, timedelta
import shutil
import glob
import time
import random
import json
import inspect
import math
import subprocess
import re
from pathlib import Path
from collections import defaultdict

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule

# needed for broadcasting Queue in dataset.py
mp.current_process().authkey = b'afsaskgfdjh4'

wandb_enable = False

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DIFFUSION_PIPE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DIFFUSION_PIPE_ROOT.parent
_UNEXPANDED_ENV_VAR = re.compile(r'\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*')

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--reset_dataloader', action='store_true', help='Start dataloader from scratch when resuming from checkpoint, i.e. only load the optimizer states.')
parser.add_argument('--reset_optimizer', action='store_true')
parser.add_argument('--reset_optimizer_params', action='store_true')
parser.add_argument('--regenerate_cache', action='store_true', help='Force regenerate cache.')
parser.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
parser.add_argument('--trust_cache', action='store_true', help='Load from metadata cache files if they exist, without checking if any fingerprints have changed. Can make loading much faster for large datasets.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


def _dataset_config_uses_precomputed_wan22_embeddings(dataset_config):
    directories = dataset_config.get('directory', [])
    if len(directories) == 0:
        return False
    for directory in directories:
        if directory.get('precomputed_embeddings', dataset_config.get('precomputed_embeddings')) != 'wan22_ti2v_embeddings':
            return False
    return True


def _resolve_config_path(value, *, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty path string')
    expanded = value.replace('${REPO_ROOT}', str(REPO_ROOT))
    expanded = os.path.expanduser(os.path.expandvars(expanded))
    unresolved = _UNEXPANDED_ENV_VAR.search(expanded)
    if unresolved:
        raise ValueError(
            f'{field_name} references an unset environment variable: {unresolved.group(0)}'
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _resolve_main_config_paths(config):
    for key in ('output_dir', 'dataset'):
        config[key] = _resolve_config_path(config[key], field_name=key)

    for index, eval_dataset in enumerate(config.get('eval_datasets', [])):
        if isinstance(eval_dataset, str):
            config['eval_datasets'][index] = _resolve_config_path(
                eval_dataset,
                field_name=f'eval_datasets[{index}]',
            )
        else:
            eval_dataset['config'] = _resolve_config_path(
                eval_dataset['config'],
                field_name=f'eval_datasets[{index}].config',
            )

    model_config = config['model']
    for key in ('ckpt_path', 'prompt_prefix_embedding_path'):
        if model_config.get(key):
            model_config[key] = _resolve_config_path(
                model_config[key],
                field_name=f'model.{key}',
            )

    adapter_config = config.get('adapter', {})
    if adapter_config.get('init_from_existing'):
        adapter_config['init_from_existing'] = _resolve_config_path(
            adapter_config['init_from_existing'],
            field_name='adapter.init_from_existing',
        )

    visualization_config = config.get('epoch_visualization', {})
    for key in ('prompt_file', 'sample_script'):
        if visualization_config.get(key):
            visualization_config[key] = _resolve_config_path(
                visualization_config[key],
                field_name=f'epoch_visualization.{key}',
            )


def _resolve_dataset_config_paths(dataset_config, *, field_prefix='dataset'):
    for index, directory in enumerate(dataset_config.get('directory', [])):
        directory['path'] = _resolve_config_path(
            directory['path'],
            field_name=f'{field_prefix}.directory[{index}].path',
        )


def _progress_log_path():
    configured_path = os.environ.get('TRAIN_PROGRESS_LOG')
    if configured_path:
        return Path(os.path.expanduser(os.path.expandvars(configured_path)))
    return REPO_ROOT / 'outputs' / 'train_progress.log'


def log_progress(message):
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    line = f'[{timestamp}] rank={os.getenv("RANK", "?")} local_rank={os.getenv("LOCAL_RANK", "?")} {message}\n'
    print(line, end='', flush=True)
    progress_log_path = _progress_log_path()
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_log_path, 'a', encoding='utf-8') as f:
        f.write(line)


def _log_epoch_visualization_to_wandb(video_path, prompt_path, meta_path, finished_epoch, current_step):
    if getattr(wandb, 'run', None) is None:
        raise RuntimeError('wandb.run is not initialized')
    run_step = getattr(wandb.run, 'step', None)
    if run_step is None:
        log_step = current_step
    else:
        # Keep epoch visualization logs strictly monotonic relative to the active run.
        log_step = max(int(run_step), int(current_step)) + 1
    payload = {
        'eval/sample_video': wandb.Video(str(video_path), fps=16, format='mp4'),
        'eval/sample_epoch': finished_epoch,
        'epoch': finished_epoch,
    }
    if prompt_path.exists():
        payload['eval/sample_prompt'] = prompt_path.read_text(encoding='utf-8').strip()
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            payload['eval/sample_meta'] = json.load(f)
    wandb.log(payload, step=log_step)


def maybe_run_epoch_visualization(config, run_dir, finished_epoch, current_step):
    vis_config = config.get('epoch_visualization', {})
    if not vis_config.get('enabled', False):
        return
    epoch_dir = Path(run_dir) / f'epoch{finished_epoch}'
    adapter_path = epoch_dir / 'adapter_model.safetensors'
    model_path = epoch_dir / 'model.safetensors'
    if not adapter_path.exists() and not model_path.exists():
        return

    output_dir = epoch_dir / 'visualization'
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = vis_config.get('prompt_file', str(REPO_ROOT / 'env' / 'visualization_prompt.txt'))
    sample_script = vis_config.get('sample_script', str(REPO_ROOT / 'env' / 'sample_wan22_ti2v_lora.py'))
    sample_size = vis_config.get('size', '2560*1536')
    frame_num = int(vis_config.get('frame_num', 101))
    sample_steps = int(vis_config.get('sample_steps', 30))
    seed = int(vis_config.get('seed', 1234))
    visual_slot_count = int(config['model'].get('visual_slot_count', 0))
    visual_slot_rows = int(config['model'].get('visual_slot_rows', 4))
    visual_slot_cols = int(config['model'].get('visual_slot_cols', 4))
    prompt_prefix_text = config['model'].get('prompt_prefix_text', '')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    env['PYTHONNOUSERSITE'] = '1'
    env['NVCC_PREPEND_FLAGS'] = env.get('NVCC_PREPEND_FLAGS', '')
    # The sampling subprocess must not inherit the active training process-group
    # environment, otherwise it can interfere with the live TCPStore/NCCL state.
    for key in (
        'RANK',
        'WORLD_SIZE',
        'LOCAL_RANK',
        'LOCAL_WORLD_SIZE',
        'GROUP_RANK',
        'ROLE_RANK',
        'ROLE_NAME',
        'NODE_RANK',
        'MASTER_ADDR',
        'MASTER_PORT',
        'TORCHELASTIC_RUN_ID',
        'TORCHELASTIC_RESTART_COUNT',
        'TORCHELASTIC_MAX_RESTARTS',
        'TORCHELASTIC_ERROR_FILE',
        'TORCH_DISTRIBUTED_DEFAULT_PORT',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING',
        'TORCH_NCCL_ENABLE_MONITORING',
        'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC',
        'NCCL_ASYNC_ERROR_HANDLING',
        'NCCL_SOCKET_IFNAME',
        'NCCL_IB_HCA',
        'NCCL_P2P_DISABLE',
        'NCCL_SHM_DISABLE',
        'DEEPSPEED_LAUNCHER',
        'DEEPSPEED_HOSTFILE',
    ):
        env.pop(key, None)

    sample_cmd = [
        sys.executable,
        sample_script,
        '--adapter_dir', str(epoch_dir),
        '--prompt_file', str(prompt_file),
        '--output_dir', str(output_dir),
        '--size', str(sample_size),
        '--frame_num', str(frame_num),
        '--sample_steps', str(sample_steps),
        '--seed', str(seed),
        '--visual_slot_count', str(visual_slot_count),
        '--visual_slot_rows', str(visual_slot_rows),
        '--visual_slot_cols', str(visual_slot_cols),
    ]
    if prompt_prefix_text:
        sample_cmd.extend(['--prompt_prefix_text', str(prompt_prefix_text)])
    sample_log = output_dir / 'sample_stdout.log'
    log_progress(f'starting epoch visualization for epoch={finished_epoch}')
    video_path = output_dir / 'sample_preview.mp4'
    prompt_path = output_dir / 'sample_prompt.txt'
    meta_path = output_dir / 'sample_meta.json'
    with open(sample_log, 'a', encoding='utf-8') as sample_stdout:
        result = subprocess.run(
            sample_cmd,
            check=False,
            env=env,
            stdout=sample_stdout,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'epoch visualization sampling failed for epoch={finished_epoch} '
                f'with returncode={result.returncode}'
            )
        if wandb_enable and video_path.exists():
            log_progress(
                f'epoch visualization video ready for epoch={finished_epoch}; '
                'auto-upload disabled, upload manually if needed'
            )
    log_progress(f'epoch visualization complete for epoch={finished_epoch}')


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.state = defaultdict(dict)
        self.param_groups = []

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config or 'save_every_n_examples' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP[transformer_dtype]
    if diffusion_model_dtype := model_config.get('diffusion_model_dtype', None):
        model_config['diffusion_model_dtype'] = DTYPE_MAP[diffusion_model_dtype]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_every_n_examples', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)
    config.setdefault('x_axis_examples', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss, 'step': step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
            if wandb_enable:
                wandb.log({f'{name}/loss': avg_loss, 'step': step})

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        if wandb_enable:
            wandb.log({'eval/eval_time_sec': duration, 'step': step})
        pbar.close()


def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    empty_cuda_cache()
    model.prepare_block_swap_training()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', str(args.master_port))

    return world_size, rank, local_rank


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()


if __name__ == '__main__':
    # With multiple GPUs / large batch sizes, the dataloader can trigger "too many open files" errors unless we do this.
    torch.multiprocessing.set_sharing_strategy('file_system')
    deepspeed.utils.set_log_level_from_string('info')
    apply_patches()

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    _resolve_main_config_paths(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Cache-only mode never builds the training engine, so using NCCL here only adds
    # unnecessary GPU-side collectives during startup. On B200 this can hang in NCCL
    # transport setup before dataset caching even begins.
    dist_backend = 'gloo' if args.cache_only else None
    deepspeed.init_distributed(dist_backend=dist_backend, timeout=timedelta(hours=12))

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(dist.get_rank())

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)
    _resolve_dataset_config_paths(dataset_config)

    eval_dataset_configs = []
    for eval_dataset in config['eval_datasets']:
        config_path = eval_dataset if type(eval_dataset) == str else eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        _resolve_dataset_config_paths(eval_dataset_config, field_prefix=f'eval_datasets[{len(eval_dataset_configs)}]')
        eval_dataset_configs.append(eval_dataset_config)

    model_type = config['model']['type']
    if (
        model_type == 'wan'
        and _dataset_config_uses_precomputed_wan22_embeddings(dataset_config)
        and all(_dataset_config_uses_precomputed_wan22_embeddings(cfg) for cfg in eval_dataset_configs)
    ):
        config['model'].setdefault('skip_text_encoder_model_load', True)
        config['model'].setdefault('skip_vae_load', True)

    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl
        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos
        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2
        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models.wan import wan
        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma
        model = chroma.ChromaPipeline(config)
    elif model_type == 'hidream':
        from models import hidream
        model = hidream.HiDreamPipeline(config)
    elif model_type == 'sd3':
        from models import sd3
        model = sd3.SD3Pipeline(config)
    elif model_type == 'cosmos_predict2' or model_type == 'anima':
        from models import cosmos_predict2
        model = cosmos_predict2.CosmosPredict2Pipeline(config)
    elif model_type == 'omnigen2':
        from models import omnigen2
        model = omnigen2.OmniGen2Pipeline(config)
    elif model_type == 'qwen_image':
        from models import qwen_image
        model = qwen_image.QwenImagePipeline(config)
    elif model_type == 'hunyuan_image':
        from models import hunyuan_image
        model = hunyuan_image.HunyuanImagePipeline(config)
    elif model_type == 'auraflow':
        from models import auraflow
        model = auraflow.AuraFlowPipeline(config)
    elif model_type == 'z_image':
        from models import z_image
        model = z_image.ZImagePipeline(config)
    elif model_type == 'hunyuan_video_15':
        from models import hunyuan_video_15
        model = hunyuan_video_15.HunyuanVideo15Pipeline(config)
    elif model_type == 'flux2':
        from models import flux2
        model = flux2.Flux2Pipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')
    log_progress(f'model constructed type={model_type}')

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    micro_batch_size_per_gpu = config.get('micro_batch_size_per_gpu', 1)
    if isinstance(micro_batch_size_per_gpu, int):
        micro_batch_size_per_gpu = {None: micro_batch_size_per_gpu}
    elif isinstance(micro_batch_size_per_gpu, list):
        micro_batch_size_per_gpu = {x[0]: x[1] for x in micro_batch_size_per_gpu}

    eval_micro_batch_size_per_gpu = config.get('eval_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(eval_micro_batch_size_per_gpu, int):
        eval_micro_batch_size_per_gpu = {None: eval_micro_batch_size_per_gpu}
    elif isinstance(eval_micro_batch_size_per_gpu, list):
        eval_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_micro_batch_size_per_gpu}

    image_micro_batch_size_per_gpu = config.get('image_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(image_micro_batch_size_per_gpu, int):
        image_micro_batch_size_per_gpu = {None: image_micro_batch_size_per_gpu}
    elif isinstance(image_micro_batch_size_per_gpu, list):
        image_micro_batch_size_per_gpu = {x[0]: x[1] for x in image_micro_batch_size_per_gpu}

    eval_image_micro_batch_size_per_gpu = config.get('eval_image_micro_batch_size_per_gpu', eval_micro_batch_size_per_gpu)
    if isinstance(eval_image_micro_batch_size_per_gpu, int):
        eval_image_micro_batch_size_per_gpu = {None: eval_image_micro_batch_size_per_gpu}
    elif isinstance(eval_image_micro_batch_size_per_gpu, list):
        eval_image_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_image_micro_batch_size_per_gpu}

    default_micro_batch_size_per_gpu = list(micro_batch_size_per_gpu.values())[0]

    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_config = {
        'train_micro_batch_size_per_gpu': default_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        # Can't do gradient clipping with gradient release, since there are no grads at the end of the step anymore.
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }
    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, trust_cache=args.trust_cache, caching_batch_size=caching_batch_size)
    log_progress('dataset_manager constructed')

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(train_data)
    log_progress('train dataset registered')

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        eval_dataset_config = eval_dataset_configs[i]
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])
    log_progress(f'eval datasets registered count={len(eval_data_map)}')

    if args.dump_dataset:
        # only works for flux
        import torchvision
        dataset_manager.cache(unload_models=False)
        if is_main_process():
            with torch.no_grad():
                os.makedirs(args.dump_dataset, exist_ok=True)
                vae = model.vae.to('cuda')
                train_data.post_init(
                    0,
                    1,
                    1,
                    1,
                    1,
                )
                for i, item in enumerate(train_data):
                    latents = item['latents']
                    latents = latents / vae.config.scaling_factor
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        latents = latents + vae.config.shift_factor
                    img = vae.decode(latents.to(vae.device, vae.dtype)).sample.to(torch.float32)
                    img = img.squeeze(0)
                    img = ((img + 1) / 2).clamp(0, 1)
                    pil_img = torchvision.transforms.functional.to_pil_image(img)
                    pil_img.save(args.dump_dataset / f'{i}.png')
                    if i >= 100:
                        break
        dist.barrier()
        quit()

    log_progress('starting dataset_manager.cache')
    dataset_manager.cache()
    log_progress('dataset_manager.cache complete')
    if args.cache_only:
        quit()

    log_progress('starting load_diffusion_model')
    model.load_diffusion_model()
    log_progress('load_diffusion_model complete')

    if adapter_config := config.get('adapter', None):
        log_progress('starting configure_adapter')
        model.configure_adapter(adapter_config)
        log_progress('configure_adapter complete')
        is_adapter = True
        if init_from_existing := adapter_config.get('init_from_existing', None):
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    # if this is a new run, create a new dir for it
    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(config['dataset'], run_dir)
        for eval_dataset in config['eval_datasets']:
            shutil.copy(eval_dataset['config'], run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    dist.barrier()
    if resume_from_checkpoint is True:  # No specific folder provided, use most recent
        run_dir = get_most_recent_run_dir(config['output_dir'])
    elif isinstance(resume_from_checkpoint, str):  # Specific folder provided
        run_dir = os.path.join(config['output_dir'], resume_from_checkpoint)
        if not os.path.exists(run_dir):
            raise ValueError(f"Checkpoint directory {run_dir} does not exist")
    else:  # Not resuming, use most recent (newly created) dir
        run_dir = get_most_recent_run_dir(config['output_dir'])

    # WandB logging
    wandb_enable = config.get('monitoring', {}).get('enable_wandb', False)
    if wandb_enable and is_main_process():
        monitoring_config = config.get('monitoring', {})
        wandb_api_key = monitoring_config.get('wandb_api_key') or os.getenv('WANDB_API_KEY', '')
        wandb_tracker = monitoring_config.get('wandb_tracker_name') or os.getenv('WANDB_PROJECT', 'wan22_5b_b200_lora')
        wandb_run_name = monitoring_config.get('wandb_run_name') or os.getenv('WANDB_RUN_NAME', os.path.basename(run_dir))
        wandb_entity = monitoring_config.get('wandb_entity') or os.getenv('WANDB_ENTITY', '')
        logging_dir = run_dir
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        existing_wandb_info_path = os.path.join(run_dir, 'wandb_run.json')
        existing_wandb_info = None
        if os.path.exists(existing_wandb_info_path):
            with open(existing_wandb_info_path, 'r', encoding='utf-8') as f:
                existing_wandb_info = json.load(f)
        init_kwargs = {
            'project': wandb_tracker,
            'name': wandb_run_name,
            'config': config,
            'dir': logging_dir,
        }
        if wandb_entity:
            init_kwargs['entity'] = wandb_entity
        if resume_from_checkpoint and existing_wandb_info and existing_wandb_info.get('run_id'):
            init_kwargs['project'] = existing_wandb_info.get('project', wandb_tracker)
            init_kwargs['name'] = existing_wandb_info.get('run_name', wandb_run_name)
            init_kwargs['id'] = existing_wandb_info['run_id']
            init_kwargs['resume'] = 'allow'
            if existing_wandb_info.get('entity'):
                init_kwargs['entity'] = existing_wandb_info['entity']
        wandb.init(**init_kwargs)
        with open(os.path.join(run_dir, 'wandb_run.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'entity': wandb.run.entity,
                'project': wandb.run.project,
                'run_id': wandb.run.id,
                'run_name': wandb.run.name,
                'run_dir': run_dir,
            }, f, indent=2)

    # Block swapping
    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        # Don't automatically move to GPU, we'll do that ourselves.
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            # TODO: block swapping doesn't work with Deepspeed non-reentrant checkpoint, but PyTorch native one is fine. Some
            # weights end up on CPU where they shouldn't. Why? Are we giving anything up by not using the Deepspeed implementation?
            #checkpoint_func = deepspeed.checkpointing.non_reentrant_checkpoint
            from functools import partial
            checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=config['reentrant_activation_checkpointing'])
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1)
    partition_method=config.get('partition_method', 'parameters')
    partition_split = config.get('partition_split',[len(layers) / num_stages])
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split,
        loss_fn=model.get_loss_fn(),
        **additional_pipeline_module_kwargs
    )
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    if config['compile']:
        pipeline_model.compile()

    log_progress('starting deepspeed.initialize')
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        config=ds_config,
    )
    log_progress('deepspeed.initialize complete')
    # Newer Deepspeed versions fail when pipeline_stages>1 because of a check on this field which defaults to False. But, pipeline
    # parallelism has always relied on "Torch-style" backward(), so I think this is an oversight by Deepspeed devs and it's safe
    # to force this to True to get it to work.
    model_engine._support_torch_style_backward = True
    global_batch_size = model_engine.train_micro_batch_size_per_gpu() * model_engine.gradient_accumulation_steps() * model_engine.grid.get_data_parallel_world_size()
    print(f'Global batch size = {global_batch_size}')

    if save_every_n_examples := config.pop('save_every_n_examples', None):
        config['save_every_n_steps'] = save_every_n_examples // global_batch_size
        print(f"Computed save_every_n_steps = {config['save_every_n_steps']}")
    if eval_every_n_examples := config.pop('eval_every_n_examples', None):
        config['eval_every_n_steps'] = eval_every_n_examples // global_batch_size
        print(f"Computed eval_every_n_steps = {config['eval_every_n_steps']}")

    def get_optimizer(model_parameters):
        if len(model_parameters) == 0:
            return DummyOptimizer()

        optim_config = config['optimizer']
        optim_type = optim_config['type']
        optim_type_lower = optim_type.lower()

        if beta2_half_life := optim_config.pop('beta2_half_life', None):
            betas = optim_config['betas']
            assert len(betas) == 2
            betas[1] = 0.5 ** (global_batch_size / beta2_half_life)
            print(f'Computed beta2 = {betas[1]}')
            optim_config['betas'] = betas

        args = []
        kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}

        if optim_type_lower == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            # klass = deepspeed.ops.adam.FusedAdam
            klass = torch.optim.AdamW
        elif optim_type_lower == 'adamw8bit':
            import bitsandbytes
            klass = bitsandbytes.optim.AdamW8bit
        elif optim_type_lower == 'adamw_optimi':
            import optimi
            klass = optimi.AdamW
        elif optim_type_lower == 'stableadamw':
            import optimi
            klass = optimi.StableAdamW
        elif optim_type_lower == 'sgd':
            klass = torch.optim.SGD
        elif optim_type_lower == 'adamw8bitkahan':
            from optimizers import adamw_8bit
            klass = adamw_8bit.AdamW8bitKahan
        elif optim_type_lower == 'offload':
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            klass = CPUOffloadOptimizer
            args.append(torch.optim.AdamW)
            kwargs['fused'] = True
        elif optim_type_lower == 'automagic':
            from optimizers import automagic
            klass = automagic.Automagic
        elif optim_type_lower == 'genericoptim':
            from optimizers import generic_optim
            klass = generic_optim.GenericOptim
        else:
            import pytorch_optimizer
            klass = getattr(pytorch_optimizer, optim_type)

        if optim_config.get('gradient_release', False):
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}", ranks=[0])
            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                assert self.mpu.get_data_parallel_world_size() == 1, 'When using gradient release, data parallel world size must be 1. Make sure pipeline_stages = num_gpus.'
                return
            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)
            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1/gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1/gas)

            optimizer_dict = {}
            for pg in model.get_param_groups(model_parameters):
                param_kwargs = kwargs.copy()
                if isinstance(pg, dict):
                    # param group
                    for p in pg['params']:
                        param_kwargs['lr'] = pg['lr']
                        optimizer_dict[p] = klass([p], **param_kwargs)
                else:
                    # param
                    optimizer_dict[pg] = klass([pg], **param_kwargs)

            def optimizer_hook(p):
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))
        elif optim_type_lower == 'genericoptim':
            kwargs['compile'] = config['compile']
            kwargs['mpu'] = pipeline_model.mpu()
            new_param_groups = []
            param_groups = model.get_param_groups(model_parameters)
            for pg in param_groups:
                params = pg.pop('params')
                params_2d = []
                params_other = []
                for p in params:
                    if p.ndim == 2:
                        params_2d.append(p)
                    else:
                        params_other.append(p)
                pg_2d = pg.copy()
                pg_2d['params'] = params_2d
                if kwargs.get('second_moment_type', None) == 'sn':
                    pg_2d['subset_size'] = 'heuristics'
                for key in ('rank', 'proj_type', 'update_proj_gap'):
                    if key in kwargs:
                        pg_2d[key] = kwargs.pop(key)
                new_param_groups.append(pg_2d)
                pg_other = pg
                pg_other['params'] = params_other
                new_param_groups.append(pg_other)
            return klass(new_param_groups, *args, **kwargs)
        else:
            param_groups = model.get_param_groups(model_parameters)
            return klass(param_groups, *args, **kwargs)

    model_engine._configure_optimizer(get_optimizer, parameters_to_train)
    optimizer = model_engine.optimizer

    model.model_engine = model_engine
    if model_engine.is_pipe_parallel:
         grid = model_engine.grid
         model_engine.first_last_stage_group = dist.new_group(ranks=[grid.pp_group[0], grid.pp_group[-1]])



    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        micro_batch_size_per_gpu,
        model_engine.gradient_accumulation_steps(),
        image_micro_batch_size_per_gpu,
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            eval_micro_batch_size_per_gpu,
            config['eval_gradient_accumulation_steps'],
            eval_image_micro_batch_size_per_gpu,
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_num_dataloader_workers = int(config.get('num_dataloader_workers', 1))
    train_dataloader = dataset_util.PipelineDataLoader(
        train_data,
        model_engine,
        model_engine.gradient_accumulation_steps(),
        model,
        num_dataloader_workers=train_num_dataloader_workers,
    )
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    log_progress(
        f'train_dataloader ready steps_per_epoch={steps_per_epoch} '
        f'num_workers={train_num_dataloader_workers}'
    )

    total_training_steps = config['epochs'] * steps_per_epoch
    if 'max_steps' in config:
        total_training_steps = min(total_training_steps, int(config['max_steps']))
    warmup_steps = int(config['warmup_steps'])
    decay_steps = max(1, total_training_steps - warmup_steps)

    scheduler_type = config.get('lr_scheduler', 'constant')
    if scheduler_type == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=decay_steps,
        )
    elif scheduler_type == 'cosine':
        min_lr_ratio = float(config.get('cosine_min_lr_ratio', 0.0))
        if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
            raise ValueError(f'cosine_min_lr_ratio must be in [0, 1], got {min_lr_ratio}')

        def cosine_decay_lambda(current_step):
            progress = min(float(current_step) / float(decay_steps), 1.0)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_decay_lambda)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler: {scheduler_type}')
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    step = 1
    examples = global_batch_size
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        param_groups = optimizer.param_groups.copy()
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config and not args.reset_optimizer and not args.reset_optimizer_params,
            load_optimizer_states=not args.reset_optimizer,
        )
        if args.reset_optimizer_params:
            optimizer.param_groups = param_groups
        dist.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        saved_world_size = client_state.get('world_size')
        current_world_size = dist.get_world_size()
        if saved_world_size is not None and saved_world_size != current_world_size:
            raise RuntimeError(
                f'Full checkpoint resume requires the same world size. '
                f'Checkpoint saved with world_size={saved_world_size}, '
                f'current run has world_size={current_world_size}. '
                f'Use the original world size for a full resume, or restart from model weights instead.'
            )
        if args.reset_dataloader:
            train_dataloader.epoch = client_state['custom_loader']['epoch']
        else:
            train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        if 'examples' in client_state:
            examples = client_state['examples'] + global_batch_size
        else:
            examples = step * global_batch_size
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)
    log_progress('tb_writer and saver ready')

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model, model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

    # TODO: this is state we need to save and resume when resuming from checkpoint. It only affects logging.
    epoch_loss = 0
    num_steps = 0
    empty_cuda_cache()
    while True:
        if step == 1:
            log_progress('entering training loop')
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        loss = model_engine.train_batch(iterator).item()
        if step == 1:
            log_progress(f'first train_batch complete loss={loss}')
        epoch_loss += loss
        num_steps += 1
        train_dataloader.sync_epoch()

        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step, examples)
        finished_epoch = True if new_epoch != epoch else False

        x_axis = examples if config['x_axis_examples'] else step

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss, x_axis)
            if hasattr(optimizer, '_grad_norm'):
                tb_writer.add_scalar(f'train/grad_norm', optimizer._grad_norm, x_axis)
            if wandb_enable:
                wandb.log({'train/loss': loss, 'step': x_axis})
                if hasattr(optimizer, '_grad_norm'):
                    wandb.log({'train/grad_norm': optimizer._grad_norm, 'step': x_axis})
            if optimizer.__class__.__name__ == 'Prodigy':
                prodigy_d = get_prodigy_d(optimizer)
                tb_writer.add_scalar(f'train/prodigy_d', prodigy_d, x_axis)
            if optimizer.__class__.__name__ in ('Automagic', 'GenericOptim'):
                lrs, avg_lr = _get_automagic_lrs(optimizer)
                if avg_lr > 0:
                    tb_writer.add_histogram(f'train/automagic_lrs', lrs, x_axis)
                    tb_writer.add_scalar(f'train/automagic_avg_lr', avg_lr, x_axis)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model, model_engine, eval_dataloaders, tb_writer, x_axis, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

        if finished_epoch:
            if is_main_process():
                tb_writer.add_scalar(f'train/epoch_loss', epoch_loss/num_steps, epoch)
                if wandb_enable:
                    wandb.log({'train/epoch_loss': epoch_loss/num_steps, 'epoch': epoch})
            if is_main_process():
                try:
                    maybe_run_epoch_visualization(config, run_dir, epoch, step)
                except Exception as exc:
                    log_progress(f'epoch visualization failed for epoch={epoch}: {exc!r}')
            epoch_loss = 0
            num_steps = 0
            if new_epoch is None:
                final_model_name = f'epoch{epoch}'
                break
            epoch = new_epoch

        checkpointed, saved = saver.process_step(step, examples)
        if 'max_steps' in config and step >= config['max_steps']:
            final_model_name = f'step{step}'
            break
        step += 1
        examples += global_batch_size

    # Save final training state checkpoint and model, unless we just saved them.
    if not checkpointed:
        saver.save_checkpoint(step, examples)
    if not saved:
        saver.save_model(final_model_name)

    if is_main_process():
        print('TRAINING COMPLETE!')
