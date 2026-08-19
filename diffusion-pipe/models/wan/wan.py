import json
import re
import os.path
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
try:
    from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, get_t_distribution, slice_t_distribution, sample_t, load_state_dict
    from utils.offloading import ModelOffloader
except ModuleNotFoundError:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.abspath(_os.path.dirname(__file__)), '../../utils'))
    from common import AUTOCAST_DTYPE, get_lin_function, time_shift, get_t_distribution, slice_t_distribution, sample_t, load_state_dict
    from offloading import ModelOffloader
from .t5 import T5EncoderModel
from .vae2_1 import Wan2_1_VAE
from .vae2_2 import Wan2_2_VAE
from .model import (
    WanModel, sinusoidal_embedding_1d, add_visual_slot_structure
)
from .clip import CLIPModel
from .tokenizers import HuggingfaceTokenizer
from . import configs as wan_configs

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation']


class WanModelFromSafetensors(WanModel):
    @classmethod
    def from_pretrained(
        cls,
        weights_file,
        config,
        torch_dtype=torch.bfloat16,
        transformer_dtype=torch.bfloat16,
    ):
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = cls(**config)

        dtype_map = {
            name: (torch_dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype)
            for name, _ in model.named_parameters()
        }
        for shard in sorted(Path(weights_file).parent.glob(Path(weights_file).name)):
            with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    name = re.sub(r'^model\.diffusion_model\.', '', key)
                    set_module_tensor_to_device(
                        model,
                        name,
                        device='cpu',
                        dtype=dtype_map[name],
                        value=f.get_tensor(key),
                    )

        return model


def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)


# Wrapper to hold both VAE and CLIP, so we can move both to/from GPU together.
class VaeAndClip(nn.Module):
    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.clip = clip


class WanPipeline(BasePipeline):
    name = 'wan'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['WanAttentionBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)
        self.t_dist = get_t_distribution(self.model_config).to('cuda')
        self.dtype = self.model_config['dtype']

        # The official Wan top-level checkpoint folder. Must exist.
        ckpt_dir = Path(self.model_config['ckpt_path'])
        dtype = self.dtype
        self.ckpt_dir = ckpt_dir

        # transformer_path will either be a directory containing safetensors files, or directly point to a safetensors file
        self.transformer_path = Path(self.model_config.get('transformer_path', ckpt_dir))
        if self.transformer_path.is_dir():
            # If it's a directory, we assume the config JSON exists inside it, along with the safetensors files.
            self.original_model_config_path = self.transformer_path / 'config.json'
            safetensors_files = list(self.transformer_path.glob('*.safetensors'))
        else:
            # If it's a single file, the config JSON is assumed to be in the top-level checkpoint folder.
            self.original_model_config_path = ckpt_dir / 'config.json'
            if not self.original_model_config_path.exists():
                # Wan2.2 has subdirectories for the model. Automatically handle that.
                self.original_model_config_path = ckpt_dir / 'low_noise_model' / 'config.json'
            safetensors_files = [self.transformer_path]

        # get all weight keys
        weight_keys = set()
        for shard in safetensors_files:
            with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_keys.add(re.sub(r'^model\.diffusion_model\.', '', k))

        # SkyReels V2 uses 24 FPS. There seems to be no better way to autodetect this.
        if 'skyreels' in ckpt_dir.name.lower():
            skyreels = True
            self.framerate = 24
        else:
            skyreels = False

        with open(self.original_model_config_path) as f:
            self.json_config = json.load(f)
        # wogridattn_lora keeps visual slot conditioning but removes the added local
        # grid/slot attention modules from the Wan backbone.
        self.json_config['local_slot_attention_num_layers'] = 0

        model_type = self.json_config['model_type']
        model_dim = self.json_config['dim']

        def autodetect_error():
            raise RuntimeError(f'Could not autodetect model variant. model_type={model_type}, model_dim={model_dim}')

        if model_type == 't2v':
            if skyreels:
                # FPS is different so make sure to use a new cache dir
                self.name = 'skyreels_v2'
            if model_dim == 1536:
                wan_config = wan_configs.t2v_1_3B
            elif model_dim == 5120:
                # This config also works with Wan2.2 T2V.
                wan_config = wan_configs.t2v_14B
            else:
                autodetect_error()
        elif model_type == 'i2v':
            if 'blocks.0.cross_attn.k_img.weight' not in weight_keys:
                # Wan2.2 I2V
                model_type = 'i2v_v2'
                self.name = 'wan2.2_i2v'
                if model_dim == 5120:
                    wan_config = wan_configs.i2v_A14B
                else:
                    autodetect_error()
            else:
                if skyreels:
                    self.name = 'skyreels_v2_i2v'
                else:
                    self.name = 'wan_i2v'
                if model_dim == 1536: # There is no official i2v 1.3b model, but there is https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP
                    # This is a hack,
                    wan_config = wan_configs.t2v_1_3B
                    # The following lines are taken from https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_i2v_14B.py
                    wan_config.clip_model = 'clip_xlm_roberta_vit_h_14'
                    wan_config.clip_dtype = torch.float16
                    wan_config.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
                elif model_dim == 5120:
                    wan_config = wan_configs.i2v_14B
                else:
                    autodetect_error()
        elif model_type == 'flf2v':
            assert not skyreels
            self.name = 'wan_flf2v'
            if model_dim == 5120:
                wan_config = wan_configs.i2v_14B  # flf2v is same config as i2v
            else:
                autodetect_error()
        elif model_type == 'ti2v':
            self.name = 'wan2.2_5b'
            self.framerate = 24
            self.pixels_round_to_multiple = 32
            if model_dim == 3072:
                wan_config = wan_configs.ti2v_5B
            else:
                autodetect_error()
        else:
            raise RuntimeError(f'Unknown model_type: {model_type}')

        self.model_type = model_type
        self.json_config['model_type'] = model_type  # to handle the special i2v_v2 type we introduced
        self.wan_config = wan_config
        tokenizer_max_length = self.model_config.get('tokenizer_max_length', wan_config.text_len)
        if tokenizer_max_length is not None:
            tokenizer_max_length = int(tokenizer_max_length)
            if tokenizer_max_length <= 0:
                tokenizer_max_length = None
        self.tokenizer_max_length = tokenizer_max_length
        self.skip_text_encoder_model_load = bool(self.model_config.get('skip_text_encoder_model_load', False))
        self.skip_vae_load = bool(self.model_config.get('skip_vae_load', False))

        # This is the outermost class, which isn't a nn.Module
        t5_model_path = self.model_config['llm_path'] if self.model_config.get('llm_path', None) else os.path.join(ckpt_dir, wan_config.t5_checkpoint)
        self.t5_model_path = t5_model_path
        self.t5_tokenizer_path = ckpt_dir / wan_config.t5_tokenizer
        if self.skip_text_encoder_model_load:
            self.text_encoder = SimpleNamespace(
                model=None,
                tokenizer=HuggingfaceTokenizer(
                    name=self.t5_tokenizer_path,
                    seq_len=self.tokenizer_max_length,
                    clean='whitespace',
                ),
            )
        else:
            self.text_encoder = T5EncoderModel(
                text_len=wan_config.text_len,
                tokenizer_seq_len=self.tokenizer_max_length,
                dtype=dtype,
                device='cpu',
                checkpoint_path=t5_model_path,
                tokenizer_path=self.t5_tokenizer_path,
                shard_fn=None,
            )
        if self.text_encoder.model is not None and self.model_config.get('text_encoder_fp8', False):
            for name, p in self.text_encoder.model.named_parameters():
                if p.ndim == 2 and not ('token_embedding' in name or 'pos_embedding' in name):
                    p.data = p.data.to(torch.float8_e4m3fn)
        if self.text_encoder.model is not None:
            self.text_encoder.model.requires_grad_(False)
        self.visual_slot_count = int(self.model_config.get('visual_slot_count', 16))
        self.visual_slot_rows = int(self.model_config.get('visual_slot_rows', 4))
        self.visual_slot_cols = int(self.model_config.get('visual_slot_cols', 4))
        self.same_slot_attention_bias = float(self.model_config.get('same_slot_attention_bias', 0.0))
        self.local_slot_attention_enabled = False
        self.local_slot_attention_gate_init = 0.0
        self.local_slot_attention_num_layers = 0
        self.train_new_params_only = bool(self.model_config.get('train_new_params_only', False))
        self.partial_slot_condition_prob = float(self.model_config.get('partial_slot_condition_prob', 0.1))
        self.partial_slot_condition_min_slots = int(self.model_config.get('partial_slot_condition_min_slots', 1))
        self.partial_slot_condition_max_slots = int(self.model_config.get('partial_slot_condition_max_slots', 8))
        self.partial_slot_condition_rows = self.visual_slot_rows
        self.partial_slot_condition_cols = self.visual_slot_cols
        self.layout_loss_weight = float(self.config.get('layout_loss_weight', 0.0))
        self.layout_loss_boundary_width = int(self.config.get('layout_loss_boundary_width', 1))
        if self.layout_loss_boundary_width <= 0:
            raise ValueError('layout_loss_boundary_width must be positive')
        self._layout_loss_mask_cache = {}
        self.prompt_prefix_text = self.model_config.get('prompt_prefix_text', '').strip()
        prompt_prefix_embedding_path = self.model_config.get('prompt_prefix_embedding_path', None)
        self.prompt_prefix_embedding_path = Path(prompt_prefix_embedding_path) if prompt_prefix_embedding_path else None
        self._prompt_prefix_embeddings = None
        self._prompt_prefix_seq_len = 0

        vae_class = Wan2_2_VAE if model_type == 'ti2v' else Wan2_1_VAE
        if self.skip_vae_load:
            self.vae = None
        else:
            # Same here, this isn't a nn.Module.
            self.vae = vae_class(
                vae_pth=ckpt_dir / wan_config.vae_checkpoint,
                device='cpu',
                dtype=dtype,
            )
            self.vae.model.to(dtype)
            # These tensors need to be on the device the VAE will be moved to during caching.
            self.vae.scale = [entry.to('cuda') for entry in self.vae.scale]

        if model_type in ('i2v', 'flf2v'):
            self.clip = CLIPModel(
                dtype=dtype,
                device='cpu',
                checkpoint_path=ckpt_dir / wan_config.clip_checkpoint,
                tokenizer_path=ckpt_dir / wan_config.clip_tokenizer,
            )

    def _materialize_text_encoder(self):
        if self.text_encoder.model is not None:
            return
        self.text_encoder = T5EncoderModel(
            text_len=self.wan_config.text_len,
            tokenizer_seq_len=self.tokenizer_max_length,
            dtype=self.dtype,
            device='cpu',
            checkpoint_path=self.t5_model_path,
            tokenizer_path=self.t5_tokenizer_path,
            shard_fn=None,
        )
        if self.model_config.get('text_encoder_fp8', False):
            for name, p in self.text_encoder.model.named_parameters():
                if p.ndim == 2 and not ('token_embedding' in name or 'pos_embedding' in name):
                    p.data = p.data.to(torch.float8_e4m3fn)
        self.text_encoder.model.requires_grad_(False)

    def _get_layout_loss_mask(self, height, width, device, dtype):
        if self.layout_loss_weight <= 0:
            return None
        slot_rows = self.visual_slot_rows
        slot_cols = self.visual_slot_cols
        boundary_width = self.layout_loss_boundary_width
        if height % slot_rows != 0 or width % slot_cols != 0:
            raise RuntimeError(
                f'Latent grid {(height, width)} is not divisible by '
                f'layout {(slot_rows, slot_cols)}'
            )

        cell_h = height // slot_rows
        cell_w = width // slot_cols
        if boundary_width * 2 > min(cell_h, cell_w):
            raise RuntimeError(
                f'layout_loss_boundary_width={boundary_width} is too large for '
                f'latent cell size {(cell_h, cell_w)}'
            )

        cache_key = (height, width, slot_rows, slot_cols, boundary_width)
        cached = self._layout_loss_mask_cache.get(cache_key)
        if cached is None:
            # Build B directly in latent coordinates. A boundary lies between the
            # last latent location of one cell and the first location of the next,
            # so width=1 supervises both adjacent locations. Outer canvas edges are
            # intentionally excluded because they are not inter-cell boundaries.
            mask = torch.zeros((height, width), dtype=torch.float32)
            for row in range(1, slot_rows):
                y = row * cell_h
                mask[y - boundary_width:y + boundary_width, :] = 1.0
            for col in range(1, slot_cols):
                x = col * cell_w
                mask[:, x - boundary_width:x + boundary_width] = 1.0
            cached = mask.view(1, 1, 1, height, width).contiguous()
            self._layout_loss_mask_cache[cache_key] = cached
        return cached.to(device=device, dtype=dtype)

    def _build_partial_slot_condition_mask(self, batch_size, frames, height, width, device, dtype):
        if self.partial_slot_condition_prob <= 0:
            return None
        slot_rows = self.partial_slot_condition_rows
        slot_cols = self.partial_slot_condition_cols
        if height % slot_rows != 0 or width % slot_cols != 0:
            return None
        slot_h = height // slot_rows
        slot_w = width // slot_cols
        total_slots = slot_rows * slot_cols
        cond_mask = torch.zeros((batch_size, 1, frames, height, width), device=device, dtype=dtype)
        for batch_idx in range(batch_size):
            if torch.rand((), device=device).item() >= self.partial_slot_condition_prob:
                continue
            max_slots = min(
                total_slots,
                max(self.partial_slot_condition_min_slots, self.partial_slot_condition_max_slots),
            )
            num_slots = int(torch.randint(
                low=self.partial_slot_condition_min_slots,
                high=max_slots + 1,
                size=(1,),
                device=device,
            ).item())
            slot_perm = torch.randperm(total_slots, device=device)[:num_slots]
            for slot_idx in slot_perm.tolist():
                row = slot_idx // slot_cols
                col = slot_idx % slot_cols
                cond_mask[
                    batch_idx,
                    :,
                    :,
                    row * slot_h:(row + 1) * slot_h,
                    col * slot_w:(col + 1) * slot_w,
                ] = 1.0
        return cond_mask

    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)
        init_from_scratch = bool(self.model_config.get('init_from_scratch', False))

        if init_from_scratch:
            scratch_device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
            with torch.device(scratch_device):
                self.transformer = WanModel.from_config(self.json_config)
            for name, p in self.transformer.named_parameters():
                target_dtype = dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                if p.dtype != target_dtype:
                    p.data = p.data.to(target_dtype)
        elif self.transformer_path.is_file():
            self.transformer = WanModelFromSafetensors.from_pretrained(
                self.transformer_path,
                self.json_config,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            with init_empty_weights():
                self.transformer = WanModel.from_config(self.json_config)
            dtype_map = {
                name: (dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype)
                for name, _ in self.transformer.named_parameters()
            }
            for shard in sorted(self.transformer_path.glob('*.safetensors')):
                with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        set_module_tensor_to_device(
                            self.transformer,
                            key,
                            device='cpu',
                            dtype=dtype_map[key],
                            value=f.get_tensor(key),
                        )

        self.transformer.train()
        if getattr(self.transformer.visual_slot_embeddings, 'is_meta', False):
            set_module_tensor_to_device(
                self.transformer,
                'visual_slot_embeddings',
                device='cpu',
                dtype=dtype,
                value=torch.zeros((self.transformer.visual_slot_embeddings.shape[0], self.transformer.dim), dtype=dtype),
            )
            nn.init.normal_(self.transformer.visual_slot_embeddings, std=.02)
        for proj_name, in_features in (('region_rel_pos_proj', 2), ('region_geom_proj', 4)):
            proj = getattr(self.transformer, proj_name, None)
            if proj is None:
                continue
            for layer_idx, layer in enumerate(proj):
                if not isinstance(layer, nn.Linear):
                    continue
                weight_name = f'{proj_name}.{layer_idx}.weight'
                if getattr(layer.weight, 'is_meta', False):
                    set_module_tensor_to_device(
                        self.transformer,
                        weight_name,
                        device='cpu',
                        dtype=transformer_dtype,
                        value=torch.empty_like(layer.weight, device='cpu', dtype=transformer_dtype),
                    )
                    nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    bias_name = f'{proj_name}.{layer_idx}.bias'
                    if getattr(layer.bias, 'is_meta', False):
                        set_module_tensor_to_device(
                            self.transformer,
                            bias_name,
                            device='cpu',
                            dtype=transformer_dtype,
                            value=torch.zeros_like(layer.bias, device='cpu', dtype=transformer_dtype),
                        )
                    else:
                        layer.bias.data.zero_()
        for block in self.transformer.blocks:
            block.local_attn_enabled = False
            if getattr(block, 'local_attn_gate', None) is not None:
                block.local_attn_gate.data.fill_(0.0)
        self.transformer.visual_slot_count = self.visual_slot_count
        self.transformer.visual_slot_rows = self.visual_slot_rows
        self.transformer.visual_slot_cols = self.visual_slot_cols
        self.transformer.same_slot_attention_bias = 0.0
        self.transformer.local_slot_attention_enabled = self.local_slot_attention_enabled
        self.transformer.local_slot_attention_gate_init = self.local_slot_attention_gate_init
        self.transformer.local_slot_attention_num_layers = self.local_slot_attention_num_layers
        for block in self.transformer.blocks:
            block.self_attn.same_slot_attention_bias = 0.0
            block.local_attn_enabled = False
            if getattr(block, 'local_attn_gate', None) is not None:
                block.local_attn_gate.data.fill_(0.0)
        if self.visual_slot_count > 0 and not init_from_scratch and self.text_encoder.model is not None:
            with torch.no_grad():
                for slot_idx in range(min(self.visual_slot_count, self.transformer.visual_slot_embeddings.size(0))):
                    slot_token = f'<extra_id_{slot_idx}>'
                    slot_token_id = self.text_encoder.tokenizer.tokenizer.convert_tokens_to_ids(slot_token)
                    if slot_token_id == self.text_encoder.tokenizer.tokenizer.unk_token_id:
                        break
                    slot_embed = self.text_encoder.model.token_embedding.weight[slot_token_id].to(
                        self.transformer.visual_slot_embeddings.device,
                        self.transformer.visual_slot_embeddings.dtype,
                    )
                    slot_projected = self.transformer.text_embedding(slot_embed.unsqueeze(0)).squeeze(0)
                    self.transformer.visual_slot_embeddings[slot_idx].copy_(slot_projected)
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

        if self.train_new_params_only:
            for _, p in self.transformer.named_parameters():
                p.requires_grad_(False)
            self.transformer.visual_slot_embeddings.requires_grad_(True)
            for _, p in self.transformer.region_rel_pos_proj.named_parameters():
                p.requires_grad_(True)
            for _, p in self.transformer.region_geom_proj.named_parameters():
                p.requires_grad_(True)
            for block_idx, block in enumerate(self.transformer.blocks):
                if block_idx >= self.local_slot_attention_num_layers:
                    continue
                if getattr(block, 'local_attn', None) is not None:
                    for _, p in block.local_attn.named_parameters():
                        p.requires_grad_(True)
                if getattr(block, 'norm_local', None) is not None:
                    for _, p in block.norm_local.named_parameters():
                        p.requires_grad_(True)
                if getattr(block, 'local_attn_gate', None) is not None:
                    block.local_attn_gate.requires_grad_(True)

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        for name in ('visual_slot_embeddings',):
            param = getattr(self.transformer, name, None)
            if param is not None:
                param.requires_grad_(True)
                param.data = param.data.to(adapter_config['dtype'])
        for module_name in ('region_rel_pos_proj', 'region_geom_proj'):
            module = getattr(self.transformer, module_name, None)
            if module is not None:
                for _, p in module.named_parameters():
                    p.requires_grad_(True)
                    p.data = p.data.to(adapter_config['dtype'])

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        if self.vae is None:
            raise RuntimeError('VAE was skipped during initialization and is not available for this run.')
        vae = self.vae.model
        clip = self.clip.model if self.model_type in ('i2v', 'flf2v') else None
        return VaeAndClip(vae, clip)

    def get_text_encoders(self):
        # Return the inner nn.Module
        if self.cache_text_embeddings:
            if self.text_encoder.model is None:
                self._materialize_text_encoder()
            return [self.text_encoder.model]
        else:
            return []

    def get_num_text_encoders(self):
        return 1 if self.cache_text_embeddings else 0

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        peft_state_dict['diffusion_model.visual_slot_embeddings'] = self.transformer.visual_slot_embeddings.detach().cpu()
        for module_name in ('region_rel_pos_proj', 'region_geom_proj'):
            module = getattr(self.transformer, module_name, None)
            if module is not None:
                for name, param in module.named_parameters():
                    peft_state_dict[f'diffusion_model.{module_name}.{name}'] = param.detach().cpu()
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        if self.model_type == 'ti2v':
            round_side = 32
        else:
            round_side = 16
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=round_side,
            round_width=round_side,
        )

    def get_call_vae_fn(self, vae_and_clip):
        is_i2v = self.model_type in ('i2v', 'flf2v', 'i2v_v2')
        def fn(tensor):
            vae = vae_and_clip.vae
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            ret = {'latents': latents}

            if is_i2v:
                assert tensor.ndim == 5, f'i2v/flf2v must train on videos, got tensor with shape {tensor.shape}'
                assert tensor.shape[2] > 1, 'i2v/flf2v must train on videos, but got an image'
                first_frame = tensor[:, :, 0:1, ...].clone()

                if self.model_type == 'flf2v':
                    tensor[:, :, 1:-1, ...] = 0
                else:
                    tensor[:, :, 1:, ...] = 0

                # Image conditioning. Same shame as latents, first frame is unchanged, rest is 0.
                # NOTE: encoding 0s with the VAE doesn't give you 0s in the latents, I tested this. So we need to
                # encode the whole thing here, we can't just extract the first frame from the latents later and make
                # the rest 0. But what happens if you do that? Probably things get fried, but might be worth testing.
                y = vae_encode(tensor, self.vae)
                ret['y'] = y

            clip = vae_and_clip.clip
            if clip is not None:
                clip_context = self.clip.visual(first_frame.to(p.device, p.dtype))
                if self.model_type == 'flf2v':
                    last_frame = tensor[:, :, -1:, ...].clone()
                    # NOTE: dim=1 is a hack to pass clip_context without microbatching breaking the zeroth dim
                    clip_context = torch.cat([clip_context, self.clip.visual(last_frame.to(p.device, p.dtype))], dim=1)
                ret['clip_context'] = clip_context

            return ret
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # Args are lists
            p = next(text_encoder.parameters())
            if self.prompt_prefix_text:
                caption = [f'{self.prompt_prefix_text} {item}'.strip() for item in caption]
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(p.device)
            mask = mask.to(p.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.autocast(device_type=p.device.type, dtype=p.dtype):
                text_embeddings = text_encoder(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    def get_loss_fn(self):
        global_loss_weight = float(self.config.get('global_loss_weight', 0.2))
        layout_loss_weight = self.layout_loss_weight

        def loss_fn(output, label):
            if len(label) == 2:
                target, mask = label
                layout_mask_override = mask
                x_t = x_1 = t = None
            else:
                target, mask, x_t, x_1, t, layout_mask_override = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if 'pseudo_huber_c' in self.config:
                    c = self.config['pseudo_huber_c']
                    elementwise_loss = torch.sqrt((output-target)**2 + c**2) - c
                else:
                    elementwise_loss = F.mse_loss(output, target, reduction='none')

                loss_mask = None
                if mask.numel() > 0:
                    loss_mask = mask.to(output.device, torch.float32)
                    elementwise_loss = elementwise_loss * loss_mask

                global_loss = elementwise_loss.mean()
                total_loss = output.new_tensor(0.0)

                if global_loss_weight > 0:
                    total_loss = total_loss + global_loss_weight * global_loss

                if layout_loss_weight > 0:
                    if x_t is None or x_1 is None or t is None:
                        raise RuntimeError('layout_loss requires x_t, x_1, and t in the training labels')
                    # `output` predicts the flow term (x_0 - x_1). Reconstruct the
                    # clean video latent x_1 before applying layout supervision so the
                    # latent-grid boundary mask is enforced in the same space as the data latent.
                    t_expanded = t.view(-1, 1, 1, 1, 1).to(output.device, torch.float32)
                    x_t_fp32 = x_t.to(output.device, torch.float32)
                    x1_target = x_1.to(output.device, torch.float32)
                    x1_pred = x_t_fp32 - t_expanded * output
                    layout_mask = self._get_layout_loss_mask(
                        x1_pred.size(-2),
                        x1_pred.size(-1),
                        x1_pred.device,
                        x1_pred.dtype,
                    )
                    if layout_mask_override is not None and layout_mask_override.numel() > 0:
                        layout_mask = layout_mask * layout_mask_override.to(output.device, torch.float32)
                    layout_elementwise = F.mse_loss(x1_pred, x1_target, reduction='none')
                    # Match the paper's Grid Boundary Loss exactly:
                    #   ||B * (x1_pred - x1_target)||_2^2 / ||B||_1.
                    # The spatial boundary mask broadcasts over batch, channels, and
                    # frames, so expand it before computing ||B||_1. Otherwise the
                    # loss is unintentionally scaled by batch_size * channels * frames.
                    layout_mask = layout_mask.expand_as(layout_elementwise)
                    denom = layout_mask.sum().clamp_min(1.0)
                    layout_loss = (layout_elementwise * layout_mask).sum() / denom
                    total_loss = total_loss + layout_loss_weight * layout_loss

            return total_loss

        return loss_fn

    def _get_prompt_prefix_embeddings(self):
        if not self.prompt_prefix_text:
            return None, 0
        if self._prompt_prefix_embeddings is not None:
            return self._prompt_prefix_embeddings, self._prompt_prefix_seq_len
        if self.prompt_prefix_embedding_path is None or not self.prompt_prefix_embedding_path.exists():
            raise RuntimeError(
                f'prompt_prefix_text is set but prompt_prefix_embedding_path is missing: {self.prompt_prefix_embedding_path}'
            )
        payload = torch.load(self.prompt_prefix_embedding_path, map_location='cpu')
        self._prompt_prefix_embeddings = payload['text_embeddings'].to(torch.float32)
        self._prompt_prefix_seq_len = int(payload['seq_len'])
        return self._prompt_prefix_embeddings, self._prompt_prefix_seq_len

    def _prepend_prompt_prefix(self, text_embeddings_or_ids, seq_lens_or_text_mask):
        prefix_embeddings, prefix_len = self._get_prompt_prefix_embeddings()
        if prefix_embeddings is None or prefix_len <= 0:
            return text_embeddings_or_ids, seq_lens_or_text_mask

        if self.cache_text_embeddings:
            if not torch.is_tensor(text_embeddings_or_ids):
                text_embeddings = [torch.as_tensor(emb) for emb in text_embeddings_or_ids]
                if len(text_embeddings) == 0:
                    raise RuntimeError('empty text embedding batch')
                embedding_shape = text_embeddings[0].shape[1:]
                if any(emb.shape[1:] != embedding_shape for emb in text_embeddings):
                    raise RuntimeError('text embedding hidden dimensions differ within the batch')
                max_seq_len = max(emb.size(0) for emb in text_embeddings)
                text_embeddings_or_ids = text_embeddings[0].new_zeros(
                    (len(text_embeddings), max_seq_len, *embedding_shape)
                )
                for batch_idx, emb in enumerate(text_embeddings):
                    text_embeddings_or_ids[batch_idx, :emb.size(0)] = emb
            if not torch.is_tensor(seq_lens_or_text_mask):
                seq_lens_or_text_mask = torch.as_tensor(seq_lens_or_text_mask, dtype=torch.long)
            prefix = prefix_embeddings[:prefix_len].unsqueeze(0).expand(text_embeddings_or_ids.size(0), -1, -1)
            prefix = prefix.to(text_embeddings_or_ids.device, text_embeddings_or_ids.dtype)
            text_embeddings_or_ids = torch.cat([prefix, text_embeddings_or_ids], dim=1)
            seq_lens_or_text_mask = seq_lens_or_text_mask.to(dtype=torch.long) + prefix_len
            return text_embeddings_or_ids, seq_lens_or_text_mask

        captions = [f'{self.prompt_prefix_text} {caption}'.strip() for caption in text_embeddings_or_ids]
        return self.text_encoder.tokenizer(captions, return_mask=True, add_special_tokens=True)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']
        y = inputs['y'] if self.model_type in ('i2v', 'flf2v', 'i2v_v2') else None
        # No CLIP for i2v_v2 (Wan2.2)
        clip_context = inputs['clip_context'] if self.model_type in ('i2v', 'flf2v') else None

        if self.cache_text_embeddings:
            text_embeddings_or_ids = inputs['text_embeddings']
            seq_lens_or_text_mask = inputs['seq_lens']
        else:
            text_embeddings_or_ids, seq_lens_or_text_mask = self.text_encoder.tokenizer(inputs['caption'], return_mask=True, add_special_tokens=True)
        text_embeddings_or_ids, seq_lens_or_text_mask = self._prepend_prompt_prefix(
            text_embeddings_or_ids,
            seq_lens_or_text_mask,
        )

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target
        layout_loss_mask = None if mask is None else mask.clone()

        t = self.t_dist

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        t = slice_t_distribution(t, min_t=self.model_config.get('min_t', 0.0), max_t=self.model_config.get('max_t', 1.0))
        t = sample_t(t, bs, quantile=timestep_quantile).to(latents.device)
        t_raw = t

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        cond_mask = self._build_partial_slot_condition_mask(
            bs,
            num_frames,
            h,
            w,
            latents.device,
            latents.dtype,
        )
        if cond_mask is not None and cond_mask.any():
            x_t = torch.where(cond_mask.bool(), x_1, x_t)
            if mask is None:
                mask = (1.0 - cond_mask).to(x_t.dtype)
            else:
                mask = mask.to(x_t.dtype) * (1.0 - cond_mask)
            if layout_loss_mask is None:
                layout_loss_mask = torch.ones_like(cond_mask, dtype=x_t.dtype)
            else:
                layout_loss_mask = layout_loss_mask.to(x_t.dtype)

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000

        return (
            (x_t, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_context),
            (target, mask, x_t, x_1, t_raw, layout_loss_mask),
        )

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder.model
        layers = [InitialLayer(transformer, text_encoder)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model, text_encoder):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.i2v = (model.model_type == 'i2v')
        self.i2v_v2 = (model.model_type == 'i2v_v2')
        self.flf2v = (model.model_type == 'flf2v')
        if self.i2v or self.flf2v:
            self.img_emb = model.img_emb
        self.text_encoder = text_encoder
        self.freqs = model.freqs
        self.freq_dim = model.freq_dim
        self.dim = model.dim
        self.text_len = model.text_len
        self.visual_slot_count = getattr(model, 'visual_slot_count', 0)
        self.visual_slot_rows = getattr(model, 'visual_slot_rows', 4)
        self.visual_slot_cols = getattr(model, 'visual_slot_cols', 4)
        self.same_slot_attention_bias = getattr(model, 'same_slot_attention_bias', 0.0)
        self.visual_slot_embeddings = model.visual_slot_embeddings
        self.region_rel_pos_proj = model.region_rel_pos_proj
        self.region_geom_proj = model.region_geom_proj

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, y, t, text_embeddings_or_ids, seq_lens_or_text_mask, clip_fea = inputs
        bs, channels, f, h, w = x.shape
        if clip_fea.numel() == 0:
            clip_fea = None

        if self.text_encoder is not None:
            assert not torch.is_floating_point(text_embeddings_or_ids)
            with torch.no_grad():
                context = self.text_encoder(text_embeddings_or_ids, seq_lens_or_text_mask)
            context.requires_grad_(True)
            text_seq_lens = seq_lens_or_text_mask.gt(0).sum(dim=1).long()
        else:
            context = text_embeddings_or_ids
            text_seq_lens = seq_lens_or_text_mask

        # Keep the full tokenizer output and pad dynamically per batch instead of
        # clamping to the original 512-token context window here.
        text_seq_lens = text_seq_lens.to(dtype=torch.long)
        context = [emb[:int(length)] for emb, length in zip(context, text_seq_lens)]

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.i2v or self.flf2v or self.i2v_v2:
            mask = torch.zeros((bs, 4, f, h, w), device=x.device, dtype=x.dtype)
            mask[:, :, 0, ...] = 1
            if self.flf2v:
                mask[:, :, -1, ...] = 1
            y = torch.cat([mask, y], dim=1)
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        use_visual_structure = self.visual_slot_count > 0
        if use_visual_structure:
            slot_embeddings = None
            if self.visual_slot_count > 0:
                slot_embeddings = self.visual_slot_embeddings[:self.visual_slot_count].to(x[0].device, x[0].dtype)
            x, vision_token_masks, token_slot_ids, token_rel_pos, token_cell_geom = add_visual_slot_structure(
                x,
                grid_sizes,
                visual_slot_embeddings=slot_embeddings,
                slot_grid_shape=(self.visual_slot_rows, self.visual_slot_cols),
            )
        else:
            vision_token_masks = [torch.ones(u.size(1), dtype=torch.bool, device=u.device) for u in x]
            token_slot_ids = [torch.full((u.size(1),), -1, dtype=torch.long, device=u.device) for u in x]
            token_rel_pos = [torch.zeros((u.size(1), 2), dtype=u.dtype, device=u.device) for u in x]
            token_cell_geom = [torch.zeros((u.size(1), 4), dtype=u.dtype, device=u.device) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = [
            u + (
                self.region_rel_pos_proj(rel_pos.to(device=u.device, dtype=u.dtype))
                + self.region_geom_proj(cell_geom.to(device=u.device, dtype=u.dtype))
            ).unsqueeze(0)
            for u, rel_pos, cell_geom in zip(x, token_rel_pos, token_cell_geom)
        ]
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        vision_token_mask = torch.stack([
            torch.cat([mask, mask.new_zeros(seq_len - mask.size(0))]) for mask in vision_token_masks
        ])
        token_slot_ids = torch.stack([
            torch.cat([slot_ids, slot_ids.new_full((seq_len - slot_ids.size(0),), -1)]) for slot_ids in token_slot_ids
        ])

        # time embeddings
        time_embed_seq_len = seq_len
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            time_embed_seq_len = 1  # will broadcast
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, time_embed_seq_len)).to(x.device, torch.float32)
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # context
        max_text_len = max(int(u.size(0)) for u in context)
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(max_text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        context_lens = text_seq_lens.to(x.device)

        if self.i2v or self.flf2v:
            assert clip_fea is not None
            if self.flf2v:
                self.img_emb.emb_pos.data = self.img_emb.emb_pos.data.to(clip_fea.device, torch.float32)
                clip_fea = clip_fea.view(-1, 257, 1280)
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)
            context_lens = context_lens + context_clip.size(1)

        # pipeline parallelism needs everything on the GPU
        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)
        vision_token_mask = vision_token_mask.to(x.device)
        token_slot_ids = token_slot_ids.to(x.device)
        context_lens = context_lens.to(x.device)

        return make_contiguous(
            x,
            e,
            e0,
            seq_lens,
            grid_sizes,
            self.freqs,
            context,
            context_lens,
            vision_token_mask,
            token_slot_ids,
        )


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, context_lens, vision_token_mask, token_slot_ids = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.block(
            x,
            e0,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            vision_token_mask,
            token_slot_ids,
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(
            x,
            e,
            e0,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            vision_token_mask,
            token_slot_ids,
        )


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, context_lens, vision_token_mask, token_slot_ids = inputs
        if vision_token_mask is not None:
            filtered_x = []
            filtered_e = [] if e.size(1) != 1 else None
            original_seq_lens = []
            for i, grid_size in enumerate(grid_sizes.tolist()):
                original_seq_len = grid_size[0] * grid_size[1] * grid_size[2]
                original_seq_lens.append(original_seq_len)
                token_idx = torch.nonzero(vision_token_mask[i], as_tuple=False).squeeze(-1)[:original_seq_len]
                filtered_x.append(x[i, token_idx])
                if filtered_e is not None:
                    filtered_e.append(e[i, token_idx])
            max_original_seq_len = max(original_seq_lens)
            x = torch.stack([
                torch.cat([u, u.new_zeros(max_original_seq_len - u.size(0), u.size(1))], dim=0)
                for u in filtered_x
            ])
            if filtered_e is not None:
                e = torch.stack([
                    torch.cat([u, u.new_zeros(max_original_seq_len - u.size(0), u.size(1))], dim=0)
                    for u in filtered_e
                ])
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)
