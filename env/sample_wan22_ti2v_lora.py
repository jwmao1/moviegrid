#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DP_REPO = REPO_ROOT / 'diffusion-pipe'

_bootstrap_parser = argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument('--wan_repo', default='')
_bootstrap_args, _ = _bootstrap_parser.parse_known_args()
WAN_REPO = Path(
    _bootstrap_args.wan_repo
    or os.environ.get('WAN_REPO', '')
    or REPO_ROOT / 'Wan2.2'
).expanduser().resolve()

sys.path.insert(0, str(WAN_REPO))
sys.path.insert(0, str(DP_REPO))

import wan  # noqa: E402
import wan.textimage2video as wan_ti2v_module  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.vae2_2 import unpatchify  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402
from models.wan.model import WanModel as DiffusionPipeWanModel  # noqa: E402


def load_lora_weights(model, adapter_dir: Path) -> None:
    import peft
    from peft import LoraConfig

    lora_config = LoraConfig.from_pretrained(adapter_dir)
    peft.get_peft_model(model, lora_config)
    adapter_state_dict = load_file(adapter_dir / 'adapter_model.safetensors')
    normalized_state_dict = {}
    model_parameters = {name for name, _ in model.named_parameters()}
    for key, value in adapter_state_dict.items():
        stripped = key.removeprefix('transformer.').removeprefix('diffusion_model.')
        if stripped in ('visual_slot_embeddings',):
            normalized_state_dict[stripped] = value
            continue
        candidates = [
            stripped,
            stripped.replace('.lora_A.weight', '.lora_A.default.weight'),
            stripped.replace('.lora_B.weight', '.lora_B.default.weight'),
            f'base_model.model.{stripped}',
            f'base_model.model.{stripped.replace(".lora_A.weight", ".lora_A.default.weight")}',
            f'base_model.model.{stripped.replace(".lora_B.weight", ".lora_B.default.weight")}',
        ]
        for candidate in candidates:
            if candidate in model_parameters:
                normalized_state_dict[candidate] = value
                break
    missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected LoRA keys: {unexpected}')
    if not normalized_state_dict:
        sample_adapter_keys = list(adapter_state_dict.keys())[:5]
        sample_model_keys = [name for name, _ in model.named_parameters() if 'lora' in name][:10]
        raise RuntimeError(
            'No LoRA weights matched the inference model parameters. '
            f'Adapter sample keys: {sample_adapter_keys}. '
            f'Model sample LoRA keys: {sample_model_keys}'
        )
    print(f'[sample] matched {len(normalized_state_dict)} lora tensors', flush=True)
    model.eval()
    return model


def load_full_model_weights(model, model_dir: Path) -> None:
    model_state_dict = load_file(model_dir / 'model.safetensors')
    normalized_state_dict = {}
    model_parameters = {name for name, _ in model.named_parameters()}
    for key, value in model_state_dict.items():
        stripped = key.removeprefix('transformer.').removeprefix('diffusion_model.')
        if stripped in model_parameters:
            normalized_state_dict[stripped] = value
    if not normalized_state_dict:
        sample_keys = list(model_state_dict.keys())[:10]
        raise RuntimeError(
            'No full-model weights matched the inference model parameters. '
            f'Model file sample keys: {sample_keys}'
        )
    missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected full-model keys: {unexpected}')
    print(f'[sample] matched {len(normalized_state_dict)} full-model tensors', flush=True)
    model.eval()
    return model


def init_visual_structure(pipeline, slot_count: int) -> None:
    pipeline.model.visual_slot_count = slot_count
    ref_param = next((p for p in pipeline.model.parameters() if not getattr(p, 'is_meta', False)), None)
    if ref_param is None:
        raise RuntimeError('Could not find a materialized model parameter for visual slot init')
    tokenizer = pipeline.text_encoder.tokenizer.tokenizer
    with torch.no_grad():
        for slot_idx in range(min(slot_count, pipeline.model.visual_slot_embeddings.size(0))):
            slot_token = f'<extra_id_{slot_idx}>'
            slot_token_id = tokenizer.convert_tokens_to_ids(slot_token)
            if slot_token_id == tokenizer.unk_token_id:
                break
            slot_embed = pipeline.text_encoder.model.token_embedding.weight[slot_token_id].to(
                device=ref_param.device,
                dtype=ref_param.dtype,
            )
            slot_projected = pipeline.model.text_embedding(slot_embed.unsqueeze(0)).squeeze(0)
            pipeline.model.visual_slot_embeddings[slot_idx].copy_(slot_projected)
    print(f'[sample] initialized {min(slot_count, pipeline.model.visual_slot_embeddings.size(0))} visual slot embeddings', flush=True)


def materialize_meta_parameters(model) -> None:
    ref_param = next((p for p in model.parameters() if not getattr(p, 'is_meta', False)), None)
    if ref_param is None:
        raise RuntimeError('Could not find a materialized model parameter for meta materialization')
    meta_params = [(name, param) for name, param in model.named_parameters() if getattr(param, 'is_meta', False)]
    for name, param in meta_params:
        set_module_tensor_to_device(
            model,
            name,
            device='cpu',
            dtype=param.dtype if param.dtype != torch.float32 or ref_param.dtype == torch.float32 else ref_param.dtype,
            value=torch.zeros(param.shape, dtype=(param.dtype if param.dtype != torch.float32 or ref_param.dtype == torch.float32 else ref_param.dtype)),
        )
    remaining = [name for name, param in model.named_parameters() if getattr(param, 'is_meta', False)]
    if remaining:
        raise RuntimeError(f'Unhandled meta parameters remain: {remaining}')
    if meta_params:
        print(f'[sample] materialized meta params: {[name for name, _ in meta_params]}', flush=True)


def decode_vae_in_chunks(vae, zs, latent_chunk_frames: int):
    if not isinstance(zs, list):
        raise TypeError("zs should be a list")
    if latent_chunk_frames <= 0:
        raise ValueError(f"latent_chunk_frames must be > 0, got {latent_chunk_frames}")

    results = []
    model = vae.model
    scale = vae.scale
    patch_size = 2

    with torch.cuda.amp.autocast(dtype=vae.dtype):
        for item_idx, latent in enumerate(zs):
            z = latent.unsqueeze(0).to(vae.device, non_blocking=True)
            model.clear_cache()

            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, model.z_dim, 1, 1, 1) + scale[0].view(1, model.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]

            latent_steps = z.shape[2]
            print(
                f'[sample] chunked vae decode item={item_idx} latent_steps={latent_steps} '
                f'chunk={latent_chunk_frames}',
                flush=True,
            )

            decoded_chunks = []
            for start in range(0, latent_steps, latent_chunk_frames):
                end = min(start + latent_chunk_frames, latent_steps)
                chunk_outputs = []
                for t_idx in range(start, end):
                    model._conv_idx = [0]
                    x = model.conv2(z[:, :, t_idx:t_idx + 1, :, :])
                    out = model.decoder(
                        x,
                        feat_cache=model._feat_map,
                        feat_idx=model._conv_idx,
                        first_chunk=(t_idx == 0),
                    )
                    out = unpatchify(out, patch_size=patch_size).float().clamp_(-1, 1)
                    chunk_outputs.append(out.squeeze(0).cpu())
                    del x, out

                decoded_chunks.append(torch.cat(chunk_outputs, dim=1))
                del chunk_outputs
                torch.cuda.empty_cache()
                print(
                    f'[sample] decoded latent chunk {start}:{end} '
                    f'-> frames={decoded_chunks[-1].shape[1]}',
                    flush=True,
                )

            model.clear_cache()
            results.append(torch.cat(decoded_chunks, dim=1))
            del decoded_chunks, z
            torch.cuda.empty_cache()

    return results


def build_prompt(prompt_file: Path, prompt_prefix_text: str) -> str:
    prompt = prompt_file.read_text().strip()
    if prompt_prefix_text:
        prompt = f'{prompt_prefix_text} {prompt}'.strip()
    return prompt


def resolve_cases(args: argparse.Namespace):
    if args.prompt_manifest:
        manifest_path = Path(args.prompt_manifest)
        cases = []
        for line_no, raw_line in enumerate(manifest_path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line_no == 1 and line.startswith('case_name\t'):
                continue
            parts = raw_line.split('\t')
            if len(parts) != 3:
                raise ValueError(
                    f'Expected 3 tab-separated columns in prompt manifest, got {len(parts)} at line {line_no}: {raw_line!r}'
                )
            case_name, prompt_file, output_dir = parts
            cases.append({
                'case_name': case_name,
                'prompt_file': Path(prompt_file),
                'output_dir': Path(output_dir),
            })
        if not cases:
            raise ValueError(f'No cases found in prompt manifest: {manifest_path}')
        return cases

    if not args.prompt_file or not args.output_dir:
        raise ValueError('--prompt_file and --output_dir are required unless --prompt_manifest is provided')

    prompt_path = Path(args.prompt_file)
    return [{
        'case_name': prompt_path.stem,
        'prompt_file': prompt_path,
        'output_dir': Path(args.output_dir),
    }]


def write_summary(summary_tsv: Path, rows) -> None:
    lines = ['prompt\tstatus\toutput_dir\toutput_path']
    for row in rows:
        lines.append('\t'.join(row))
    summary_tsv.write_text('\n'.join(lines) + '\n')


def run_case(
    pipeline,
    cfg,
    args: argparse.Namespace,
    case_name: str,
    prompt_file: Path,
    output_dir: Path,
    adapter_dir: Path | None,
    width: int,
    height: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(prompt_file, args.prompt_prefix_text)

    print(f'[sample] case={case_name}', flush=True)
    print(f'[sample] prompt_file={prompt_file}', flush=True)
    print(f'[sample] output_dir={output_dir}', flush=True)
    print('[sample] starting video generation', flush=True)

    video = pipeline.generate(
        input_prompt=prompt,
        img=None,
        size=(width, height),
        frame_num=args.frame_num,
        shift=cfg.sample_shift,
        sampling_steps=cfg.sample_steps,
        guide_scale=cfg.sample_guide_scale,
        seed=args.seed,
        offload_model=True,
    )

    output_path = output_dir / 'sample_preview.mp4'
    print(f'[sample] saving video to {output_path}', flush=True)
    save_video(video.unsqueeze(0), save_file=str(output_path), fps=24)
    del video
    torch.cuda.empty_cache()

    (output_dir / 'sample_prompt.txt').write_text(prompt + '\n')
    (output_dir / 'sample_meta.json').write_text(json.dumps({
        'case_name': case_name,
        'prompt_file': str(prompt_file),
        'adapter_dir': str(adapter_dir),
        'base_only': args.base_only,
        'output_path': str(output_path),
        'seed': args.seed,
        'frame_num': args.frame_num,
        'sample_steps': cfg.sample_steps,
        'guide_scale': cfg.sample_guide_scale,
        'sample_shift': cfg.sample_shift,
        'size': args.size,
        'visual_slot_count': args.visual_slot_count,
        'visual_slot_rows': args.visual_slot_rows,
        'visual_slot_cols': args.visual_slot_cols,
        'prompt_prefix_text': args.prompt_prefix_text,
        'decode_chunk_latent_frames': args.decode_chunk_latent_frames,
    }, indent=2) + '\n')
    print('[sample] complete', flush=True)
    print(output_path, flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wan_repo',
        default=str(WAN_REPO),
        help='Path to the official Wan2.2 source checkout (or set WAN_REPO).',
    )
    parser.add_argument('--adapter_dir', default='')
    parser.add_argument('--base_only', action='store_true')
    parser.add_argument('--prompt_file', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--prompt_manifest', default='')
    parser.add_argument('--summary_tsv', default='')
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--size', default='2560*1536')
    parser.add_argument('--frame_num', type=int, default=101)
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--sample_shift', type=float, default=None)
    parser.add_argument('--guide_scale', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--visual_slot_count', type=int, default=16)
    parser.add_argument('--visual_slot_rows', type=int, default=4)
    parser.add_argument('--visual_slot_cols', type=int, default=4)
    parser.add_argument('--prompt_prefix_text', default='<grid 16>')
    parser.add_argument('--decode_chunk_latent_frames', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    cases = resolve_cases(args)
    print(f'[sample] adapter_dir={adapter_dir}', flush=True)
    if args.prompt_manifest:
        print(f'[sample] prompt_manifest={args.prompt_manifest}', flush=True)
    print(f'[sample] case_count={len(cases)}', flush=True)

    wan_ti2v_module.WanModel = DiffusionPipeWanModel
    cfg = WAN_CONFIGS['ti2v-5B']
    if args.sample_steps is not None:
        cfg.sample_steps = args.sample_steps
    if args.sample_shift is not None:
        cfg.sample_shift = args.sample_shift
    if args.guide_scale is not None:
        cfg.sample_guide_scale = args.guide_scale

    width, height = [int(x) for x in args.size.split('*')]
    print('[sample] building pipeline', flush=True)
    pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=True,
        init_on_cpu=True,
        convert_model_dtype=True,
    )
    materialize_meta_parameters(pipeline.model)
    pipeline.model.visual_slot_count = args.visual_slot_count
    pipeline.model.visual_slot_rows = args.visual_slot_rows
    pipeline.model.visual_slot_cols = args.visual_slot_cols

    if args.base_only:
        print('[sample] using base model only (no finetuned weights)', flush=True)
        init_visual_structure(pipeline, args.visual_slot_count)
    elif adapter_dir is not None and (adapter_dir / 'model.safetensors').exists():
        print('[sample] loading full-model weights', flush=True)
        pipeline.model = load_full_model_weights(pipeline.model, adapter_dir)
    else:
        print('[sample] initializing visual structure', flush=True)
        init_visual_structure(pipeline, args.visual_slot_count)
        print('[sample] loading lora weights', flush=True)
        if adapter_dir is None:
            raise RuntimeError('adapter_dir is required unless --base_only is specified')
        pipeline.model = load_lora_weights(pipeline.model, adapter_dir)

    original_decode = None
    if args.decode_chunk_latent_frames > 0:
        original_decode = pipeline.vae.decode

        def patched_decode(zs):
            return decode_vae_in_chunks(
                pipeline.vae,
                zs,
                latent_chunk_frames=args.decode_chunk_latent_frames,
            )

        pipeline.vae.decode = patched_decode
        print(
            f'[sample] enabled chunked vae decode: latent_chunk_frames='
            f'{args.decode_chunk_latent_frames}',
            flush=True,
        )

    summary_rows = []
    failures = []
    output_paths = []
    try:
        for case in cases:
            try:
                output_path = run_case(
                    pipeline,
                    cfg,
                    args,
                    case_name=case['case_name'],
                    prompt_file=case['prompt_file'],
                    output_dir=case['output_dir'],
                    adapter_dir=adapter_dir,
                    width=width,
                    height=height,
                )
                summary_rows.append((
                    case['case_name'],
                    'ok',
                    str(case['output_dir']),
                    str(output_path),
                ))
                output_paths.append(output_path)
            except Exception:
                traceback.print_exc()
                torch.cuda.empty_cache()
                summary_rows.append((
                    case['case_name'],
                    'fail',
                    str(case['output_dir']),
                    '',
                ))
                failures.append(case['case_name'])
    finally:
        if original_decode is not None:
            pipeline.vae.decode = original_decode

    if args.summary_tsv:
        write_summary(Path(args.summary_tsv), summary_rows)
        print(f'[sample] summary_tsv={args.summary_tsv}', flush=True)

    if failures:
        raise RuntimeError(f'Generation failed for cases: {failures}')

    if len(output_paths) == 1:
        print(output_paths[0])


if __name__ == '__main__':
    main()
