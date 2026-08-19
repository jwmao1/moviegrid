# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def add_visual_slot_structure(
    token_sequences,
    grid_sizes,
    visual_slot_embeddings=None,
    slot_grid_shape=(4, 4),
):
    structured_sequences = []
    vision_token_masks = []
    token_slot_ids = []
    token_rel_pos = []
    token_cell_geom = []
    slot_count = 0 if visual_slot_embeddings is None else visual_slot_embeddings.size(0)

    for u, grid_size in zip(token_sequences, grid_sizes.tolist()):
        if slot_count > 0:
            f, h, w = [int(v) for v in grid_size]
            slot_rows, slot_cols = slot_grid_shape
            if h % slot_rows != 0 or w % slot_cols != 0:
                raise RuntimeError(
                    f'Cannot split patch grid {(f, h, w)} into slot grid {slot_grid_shape}'
                )
            slot_h = h // slot_rows
            slot_w = w // slot_cols
            token_grid = u.view(u.size(0), f, h, w, u.size(-1))
            structured_grid = token_grid.clone()
            slot_id_grid = torch.full((f, h, w), -1, dtype=torch.long, device=u.device)
            rel_pos_grid = torch.zeros((f, h, w, 2), dtype=token_grid.dtype, device=u.device)
            cell_geom_grid = torch.zeros((f, h, w, 4), dtype=token_grid.dtype, device=u.device)
            slot_idx = 0
            for row in range(slot_rows):
                for col in range(slot_cols):
                    slot_embedding = visual_slot_embeddings[min(slot_idx, slot_count - 1)].to(
                        token_grid.device, token_grid.dtype
                    ).view(1, 1, 1, 1, -1)
                    structured_grid[
                        :,
                        :,
                        row * slot_h:(row + 1) * slot_h,
                        col * slot_w:(col + 1) * slot_w,
                        :,
                    ] += slot_embedding
                    slot_id_grid[
                        :,
                        row * slot_h:(row + 1) * slot_h,
                        col * slot_w:(col + 1) * slot_w,
                    ] = slot_idx
                    rel_y = torch.linspace(
                        0.0, 1.0, steps=slot_h, device=u.device, dtype=token_grid.dtype
                    ).view(1, slot_h, 1, 1).expand(f, slot_h, slot_w, 1)
                    rel_x = torch.linspace(
                        0.0, 1.0, steps=slot_w, device=u.device, dtype=token_grid.dtype
                    ).view(1, 1, slot_w, 1).expand(f, slot_h, slot_w, 1)
                    rel_pos_grid[
                        :,
                        row * slot_h:(row + 1) * slot_h,
                        col * slot_w:(col + 1) * slot_w,
                        :,
                    ] = torch.cat([rel_x, rel_y], dim=-1)
                    cell_geom_grid[
                        :,
                        row * slot_h:(row + 1) * slot_h,
                        col * slot_w:(col + 1) * slot_w,
                        :,
                    ] = torch.tensor(
                        [
                            (col + 0.5) / slot_cols,
                            (row + 0.5) / slot_rows,
                            1.0 / slot_cols,
                            1.0 / slot_rows,
                        ],
                        device=u.device,
                        dtype=token_grid.dtype,
                    ).view(1, 1, 1, 4)
                    slot_idx += 1
            structured_sequences.append(structured_grid.reshape(u.size(0), -1, u.size(-1)))
            vision_token_masks.append(torch.ones(u.size(1), dtype=torch.bool, device=u.device))
            token_slot_ids.append(slot_id_grid.reshape(-1))
            token_rel_pos.append(rel_pos_grid.reshape(-1, 2))
            token_cell_geom.append(cell_geom_grid.reshape(-1, 4))
            continue
        structured_sequences.append(u)
        vision_token_masks.append(torch.ones(u.size(1), dtype=torch.bool, device=u.device))
        token_slot_ids.append(torch.full((u.size(1),), -1, dtype=torch.long, device=u.device))
        token_rel_pos.append(torch.zeros((u.size(1), 2), dtype=u.dtype, device=u.device))
        token_cell_geom.append(torch.zeros((u.size(1), 4), dtype=u.dtype, device=u.device))

    return structured_sequences, vision_token_masks, token_slot_ids, token_rel_pos, token_cell_geom


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, vision_token_mask=None):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        if vision_token_mask is None:
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
                seq_len, n, -1, 2))
            tail = x[i, seq_len:]
            x_out = None
        else:
            token_positions = torch.nonzero(vision_token_mask[i], as_tuple=False).squeeze(-1)
            token_positions = token_positions[:seq_len]
            x_i = torch.view_as_complex(x[i, token_positions].to(torch.float32).reshape(
                seq_len, n, -1, 2))
            x_out = x[i].to(torch.float32).clone()
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        if vision_token_mask is None:
            x_i = torch.cat([x_i, tail])
        else:
            x_out[token_positions] = x_i
            x_i = x_out

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.same_slot_attention_bias = 0.0

    def forward(self, x, seq_lens, grid_sizes, freqs, vision_token_mask=None, token_slot_ids=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs, vision_token_mask=vision_token_mask)
        k = rope_apply(k, grid_sizes, freqs, vision_token_mask=vision_token_mask)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x[..., :d]

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    # T2V and all Wan2.2 cross attn
    'default': WanCrossAttention,
    # Wan2.1 I2V only
    'wan2_1_i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 local_attn_enabled=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        # This variant intentionally removes the extra grid/local slot attention branch.
        # Slot embeddings and region geometry conditioning are still kept upstream.
        self.local_attn_enabled = False
        self.norm_local = None
        self.local_attn = None
        self.register_parameter('local_attn_gate', None)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        vision_token_mask=None,
        token_slot_ids=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, vision_token_mask=vision_token_mask, token_slot_ids=token_slot_ids)
        x = x + y * e[2].squeeze(2)
        del y

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 local_slot_attention_num_layers=0):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace', 'i2v_v2', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_slot_attention_num_layers = 0
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.visual_slot_count = 0
        self.visual_slot_rows = 4
        self.visual_slot_cols = 4
        self.same_slot_attention_bias = 0.0

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.visual_slot_embeddings = nn.Parameter(torch.zeros(64, dim))
        self.region_rel_pos_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.region_geom_proj = nn.Sequential(
            nn.Linear(4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        if model_type in ('i2v', 'flf2v'):
            cross_attn_type = 'wan2_1_i2v_cross_attn'
        else:
            cross_attn_type = 'default'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                window_size,
                qk_norm,
                cross_attn_norm,
                eps,
                local_attn_enabled=(idx < local_slot_attention_num_layers),
            )
            for idx in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    def forward(self, x, t, context, seq_len, y=None):
        if self.model_type == 'i2v':
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x]).to(device)
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
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        max_seq_len = int(seq_lens.max().item())
        seq_len = max(seq_len, max_seq_len)
        x = [
            u + (
                self.region_rel_pos_proj(rel_pos.to(device=u.device, dtype=u.dtype))
                + self.region_geom_proj(cell_geom.to(device=u.device, dtype=u.dtype))
            ).unsqueeze(0)
            for u, rel_pos, cell_geom in zip(x, token_rel_pos, token_cell_geom)
        ]
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])
        vision_token_mask = torch.stack([
            torch.cat([mask, mask.new_zeros(seq_len - mask.size(0))]) for mask in vision_token_masks
        ]).to(device)
        token_slot_ids = torch.stack([
            torch.cat([slot_ids, slot_ids.new_full((seq_len - slot_ids.size(0),), -1)]) for slot_ids in token_slot_ids
        ]).to(device)

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        elif t.size(1) != seq_len:
            expanded_t = []
            for t_row, mask in zip(t, vision_token_masks):
                num_real_tokens = int(mask.sum().item())
                if t_row.numel() < num_real_tokens:
                    raise RuntimeError(
                        f'Timestep row has {t_row.numel()} entries but needs at least {num_real_tokens}'
                    )
                real_t = t_row[:num_real_tokens]
                fill_value = real_t[0]
                expanded_row = real_t.new_full((seq_len,), fill_value)
                expanded_row[mask] = real_t
                expanded_t.append(expanded_row)
            t = torch.stack(expanded_t, dim=0)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        max_text_len = max(int(u.size(0)) for u in context)
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(max_text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        context_lens = torch.tensor(
            [u.size(0) for u in context],
            dtype=torch.long,
            device=device,
        )
        for block in self.blocks:
            x = block(
                x,
                e0,
                seq_lens,
                grid_sizes,
                self.freqs,
                context,
                context_lens,
                vision_token_mask,
                token_slot_ids,
            )

        x = self.head(x, e)
        if use_visual_structure:
            x = [u[mask] for u, mask in zip(x, vision_token_masks)]
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.normal_(self.visual_slot_embeddings, std=.02)
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
