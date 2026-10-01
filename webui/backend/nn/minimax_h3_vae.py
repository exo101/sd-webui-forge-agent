# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/vae.py
# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/audio_vae.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from backend import memory_management
from backend.attention import attention_function
from backend.operations import ForgeOperations as ops
from backend.operations import scaled_dot_product_attention

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]

# 音频 VAE latent 归一化统计（来自 DiffSynth，权重文件中不存储）
_AUDIO_LATENTS_MEAN = [
    -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373,
    0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
    -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614,
    -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797,
]

_AUDIO_LATENTS_STD = [
    1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
]


def _rms_norm(x, weight, eps):
    if weight is None:
        return F.rms_norm(x, (x.shape[-1],), eps=eps)
    return F.rms_norm(x, weight.shape, weight=weight.to(dtype=x.dtype, device=x.device), eps=eps)


def _apply_rope_split_half(x, freqs_cis):
    half = x.shape[-1] // 2
    diagonal = freqs_cis.diagonal(dim1=-2, dim2=-1).movedim(-1, -2)
    out = x.unflatten(-1, (2, -1)) * diagonal
    out[..., 0, :].addcmul_(x[..., half:], freqs_cis[..., 0, 1])
    out[..., 1, :].addcmul_(x[..., :half], freqs_cis[..., 1, 0])
    return out.reshape(x.shape)


def _rms_rope_split_half(x, freqs_cis, scale, eps, rot_dim):
    x_norm = F.rms_norm(x.float(), (x.shape[-1],), weight=scale.to(x.device), eps=eps)
    if rot_dim and rot_dim != x.shape[-1]:
        rotated = _apply_rope_split_half(x_norm[..., :rot_dim], freqs_cis)
        return torch.cat((rotated, x_norm[..., rot_dim:]), dim=-1).type_as(x)
    return _apply_rope_split_half(x_norm, freqs_cis).type_as(x)


# 3D causal CNN encoder

class CausalConv3d(ops.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x, pre_norm=None, spatial_pad=None, residual=None):
        pad_t, pad_h, pad_w = self.causal_padding
        if spatial_pad is None:
            spatial_pad = (pad_w, pad_w, pad_h, pad_h)
        front = 0 if x.shape[2] == 1 else pad_t * 2

        if pre_norm is not None or front or any(spatial_pad):
            if pre_norm is not None:
                x = F.silu(pre_norm(x), inplace=True)
            if any(spatial_pad):
                x = F.pad(x, (*spatial_pad, 0, 0), mode="reflect")
            if front:
                x = F.pad(x, (0, 0, 0, 0, front, 0), mode="constant")

        if x.shape[2] == 1 and pad_t:
            out = super().forward(x, autopad="causal_zero")
        else:
            out = super().forward(x)
        if residual is not None:
            out += residual
        return out


class TemporalIsolatedGroupNorm(ops.GroupNorm):

    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
            x = super().forward(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return super().forward(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
        )

    def forward(self, x):
        if self.space_stride == 2:
            return self.conv(x, spatial_pad=(0, 1, 0, 1))
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x, pre_norm=self.norm1)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return self.conv2(h, pre_norm=self.norm2, residual=x)


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(
                    ResnetBlock3D(
                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                        out_channels=block_mid[i_level],
                    )
                )
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(
                    block_mid[i_level],
                    block_out[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                )
            self.down.append(down)

        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(
            block_out[-1],
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        return self.conv_out(h, pre_norm=self.norm_out)


# ViT3D decoder

def create_token_ids(patch_dims, device, dtype):
    coords_list = []
    for dim_size in patch_dims:
        coords = torch.arange(0.5, dim_size, dtype=dtype, device=device)
        coords = coords / dim_size
        coords = 2.0 * coords - 1.0
        coords_list.append(coords)
    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        angles = (
            self.angle_scale
            * img_ids[:, :, :, None].float()
            * self.inv_freq.to(img_ids.device)[None, None, None, :]
        )
        angles = angles.flatten(2, 3)
        c, s = torch.cos(angles), torch.sin(angles)
        return torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        inner_dim = dim * mult
        self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, x, pre_norm, residual, residual_scale):
        h = self.w1(_rms_norm(x, pre_norm.weight, pre_norm.eps))
        gate, up = h.chunk(2, dim=-1)
        out = self.w2(F.silu(gate).mul_(up))
        if residual is None:
            return out
        return torch.addcmul(residual, out, residual_scale)


class Attention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.register_buffer("qk_norm_scale", torch.ones(dim_head), persistent=False)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, inner_dim, bias=bias)

    def forward(self, x, rotary_pos_emb, pre_norm, residual, residual_scale):
        batch_size, seq_len, _ = x.shape

        qkv = self.to_qkv(_rms_norm(x, pre_norm.weight, pre_norm.eps))
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        if rotary_pos_emb is not None:
            scale = self.qk_norm_scale.to(query.device)
            rot_dim = rotary_pos_emb.shape[-3] * 2
            query = _rms_rope_split_half(query, rotary_pos_emb, scale, self.norm_q.eps, rot_dim)
            key = _rms_rope_split_half(key, rotary_pos_emb, scale, self.norm_q.eps, rot_dim)
        else:
            query = _rms_norm(query, self.norm_q.weight, self.norm_q.eps)
            key = _rms_norm(key, self.norm_k.weight, self.norm_k.eps)

        query, key, value = (t.transpose(1, 2) for t in (query, key, value))
        out = attention_function(query, key, value, self.heads, skip_reshape=True)
        out = self.to_out(torch.nan_to_num(out))
        if residual is None:
            return out
        return torch.addcmul(residual, out, residual_scale)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.attn = Attention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.ff = FeedForward(dim=dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def _residual_scale(self, scale, x):
        if scale.dtype == x.dtype and scale.device == x.device:
            return scale
        return scale.to(dtype=x.dtype, device=x.device)

    def forward(self, x, rotary_pos_emb=None):
        x = self.attn(x, rotary_pos_emb, pre_norm=self.norm1,
                      residual=x, residual_scale=self._residual_scale(self.scale1, x))
        return self.ff(x, pre_norm=self.norm2,
                       residual=x, residual_scale=self._residual_scale(self.scale2, x))


class ViT3DDecoder(nn.Module):
    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=36, heads=32, dim_head=64, rope_theta=100.0,
                 rope_dim_ratio=0.75, bias=True, eps=1e-5, num_register_tokens=4):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        # 推理不使用；保留以匹配 checkpoint 中的键
        self.register_buffer("mask_token", torch.empty(1, 1, dim))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
             for _ in range(num_layers)]
        )

        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, x):
        B, C, latent_T, latent_H, latent_W = x.shape

        h = self.x_embedder(x.flatten(2).transpose(1, 2))  # [B, T*H*W, C]

        num_patches = h.shape[1]
        num_suffix = 1 + self.num_register_tokens

        h = torch.cat([h, self.register_tokens.to(dtype=h.dtype, device=h.device).expand(B, -1, -1), torch.zeros_like(h[:, 0:1, :])], dim=1)

        img_ids = create_token_ids((latent_T, latent_H, latent_W), x.device, x.dtype).expand(B, -1, -1)
        suffix_ids = torch.zeros((B, num_suffix, 3), device=x.device, dtype=img_ids.dtype)
        img_ids = torch.cat([img_ids, suffix_ids], dim=1)

        rotary_pos_emb = self.pos_embed(img_ids)

        for block in self.transformer_blocks:
            h = block(h, rotary_pos_emb)

        output = self.proj_out(self.norm_out(h))

        output = output[:, :num_patches, :]

        output = output.view(
            B, latent_T, latent_H, latent_W,
            self.out_channels, self.patch_size_t, self.patch_size, self.patch_size,
        )
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        output = output.reshape(
            B, self.out_channels,
            latent_T * self.patch_size_t,
            latent_H * self.patch_size,
            latent_W * self.patch_size,
        )
        return output


# Full VAE

class MiniMaxH3VideoVAE(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=24,
        z_channels=24,
        ch_mult=(1, 2, 2, 4, 4, 8),
        num_res_blocks=2,
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        clip_length=17,
        token_drop=3,
        tile_size=256,
        tile_overlap_min=64,
        tiling=True,
    ):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))
        self.vae_ratio_t = int(math.prod(time_down))

        self.clip_length = clip_length
        self.token_drop = token_drop
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)

        self.tiling = tiling
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = EncoderFCN3D(
            ch=ch,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            time_down=list(time_down),
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,
        )
        self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=self.vae_ratio_t,
            in_channels=z_channels,
            out_channels=out_ch,
        )

        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    # single-shot forward

    def _encode_moments(self, x):
        return self.quant_conv(self.encoder(x))

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z))

    def _normalize_pixels(self, x):
        return x.add(1.0).mul_(0.5).sub_(self.pixel_mean.to(x)).div_(self.pixel_std.to(x))

    def _finalize_pixels(self, part):
        part = part * self.pixel_std.to(device=part.device, dtype=torch.float32)
        return part.add_(self.pixel_mean.to(device=part.device, dtype=torch.float32)).clamp_(0.0, 1.0)

    def decode_output_shape(self, input_shape):
        b, c, t, h, w = input_shape
        if t == 1:
            frames = 1
        else:
            pad_tokens, num_chunks = self._decode_temporal_chunks(t)
            frames = self._decode_temporal_frame_plan(t + pad_tokens, num_chunks, pad_tokens)
        return (b, self.decoder.out_channels, frames, h * self.vae_ratio, w * self.vae_ratio)

    def _adaptive_encode(self, x):
        if self.tiling:
            return self.tiled_encode(x)
        return self._encode_moments(x)

    def _adaptive_decode(self, z):
        if self.tiling:
            return self.tiled_decode(z)
        return self._decode_pixels(z)

    # spatial tiling

    def split_tiles(self, input_len):
        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []

        N = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])

        return tile_start_idx, [tile_size] * N, overlaps

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)

        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)

        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def tiled_encode(self, x):
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        rows = []
        for i_pos, i_len in zip(y_idx, y_len):
            row = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]
                row.append(self._encode_moments(tile))
            rows.append(row)

        latent_y_overlap = [o // self.vae_ratio for o in y_overlap]
        latent_x_overlap = [o // self.vae_ratio for o in x_overlap]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _decode_tile_row(self, z_row, x_idx, x_len):
        free = memory_management.get_free_memory(z_row.device)
        # 128MB 为每 tile 激活的经验预算
        batch = int(max(1, min(4, free // (128 * 2**20 * z_row.shape[0]))))
        slices = [z_row[..., j_pos // self.vae_ratio:(j_pos + j_len) // self.vae_ratio] for j_pos, j_len in zip(x_idx, x_len)]
        for k in range(0, len(slices), batch):
            group = slices[k:k + batch]
            yield from self._decode_pixels(torch.cat(group)).chunk(len(group))

    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        canvas = None
        strip = None
        out_y = 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            tiles = self._decode_tile_row(z[..., zi:zi + zl, :], x_idx, x_len)
            new_strip = None
            left_tail = None
            out_x = 0
            for j in range(len(x_idx)):
                tile = next(tiles)
                if i > 0:
                    tile = self.blend(strip[..., :, x_idx[j]:x_idx[j] + x_len[j]], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                if i < len(y_idx) - 1:
                    if new_strip is None:
                        new_strip = torch.empty(*tile.shape[:-2], y_overlap[i], width, dtype=tile.dtype, device=tile.device)
                    new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -y_overlap[i]:, :]
                    tile = tile[..., :-y_overlap[i], :]
                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                tile_height = tile.shape[-2]
                out_x += tile.shape[-1]
                del tile
            strip = new_strip
            out_y += tile_height
        return canvas

    # temporal chunking

    def encode_temporal(self, x, device):
        z_list = []
        for i in range(math.ceil(x.shape[2] / self.clip_length)):
            clip_x = x[:, :, i * self.clip_length:(i + 1) * self.clip_length, :, :].to(device)
            if clip_x.shape[2] < self.clip_length:
                pad_frames = clip_x[:, :, -1:].repeat(1, 1, self.clip_length - clip_x.shape[2], 1, 1)
                clip_x = torch.cat([clip_x, pad_frames], dim=2)
            z_list.append(self._adaptive_encode(self._normalize_pixels(clip_x)))

        z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, :-self.token_drop]
        return z

    def _decode_temporal_pad_frames(self, z_len, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t

        z_len_before_pad = z_len - pad_tokens
        return sum(
            (intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0
             else self.vae_ratio_t)
            for k in range(pad_tokens)
        )

    def _decode_temporal_frame_plan(self, z_len, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames

        total_frames += final_overlap_frames
        return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)

    def _decode_temporal_chunks(self, z_len):
        pseudo_total_tokens = z_len + self.token_drop
        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
        pseudo_total_tokens += pad_tokens

        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1
        return pad_tokens, num_chunks

    def decode_temporal(self, z, output_buffer=None):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1

        if output_buffer is None:
            output_buffer = torch.empty(self.decode_output_shape(z.shape), dtype=torch.float32,
                                        device=memory_management.intermediate_device())

        pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])
        if pad_tokens > 0:
            pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
            z = torch.cat([z, pad_z], dim=2)

        dec = output_buffer
        dec_overlap = None
        write_pos = 0

        def write_part(part):
            nonlocal write_pos
            part_frames = part.shape[2]
            if part_frames <= 0:
                return
            part = self._finalize_pixels(part)
            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
            if copy_frames > 0:
                dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(
                    part[:, :, :copy_frames, :, :]
                )
                write_pos += copy_frames

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

            clip_dec = self._adaptive_decode(clip_z)

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                        dec_overlap = None
                    write_part(clip_dec_chunk)
                else:
                    dec_overlap = clip_dec_chunk.contiguous()

            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None

            del clip_dec, clip_z, clip_dec_chunk

        return dec

    def encode(self, x, device=None):
        # x: [B, 3, T, H, W] in [-1, 1] -> normalized latents [B, 24, T_lat, H/16, W/16]
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if device is None:
            device = x.device

        if x.shape[2] == 1:
            moments = self._adaptive_encode(self._normalize_pixels(x.to(device)))
            moments = moments[:, :, -1:, :, :]
        else:
            moments = self.encode_temporal(x, device)

        mean = torch.chunk(moments.float(), 2, dim=1)[0]

        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - latents_mean) / latents_std

    def encode_tiled(self, x, **kwargs):
        return self.encode(x)

    def decode_tiled(self, z, **kwargs):
        return self.decode(z)

    def decode(self, z, output_buffer=None):
        # z: [B, 24, T_lat, H_lat, W_lat] normalized latents -> float32 pixels [B, 3, T, H, W] in [0, 1]
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * latents_std + latents_mean

        if z.shape[2] == 1:
            dec = self._finalize_pixels(self._adaptive_decode(z)[:, :, -1:, :, :])
            if output_buffer is None:
                return dec
            output_buffer.copy_(dec)
            return output_buffer
        return self.decode_temporal(z, output_buffer)


# MiniMax H3 audio VAE: DAC-lineage waveform encoder + BigVGAN decoder

def snake(x, alpha, beta):
    t = torch.sin(alpha * x)
    return t.mul_(t).mul_((beta + 1e-9).reciprocal()).add_(x)


class Snake1d(nn.Module):
    """Snake activation with per-channel alpha (encoder side)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x):
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        return snake(x, alpha, alpha)


class SnakeBeta(nn.Module):
    """SnakeBeta := x + 1/beta * sin^2(alpha * x); alpha/beta stored in log scale."""

    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(in_features))
        self.beta = nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        alpha = torch.exp(self.alpha.to(dtype=x.dtype, device=x.device)).view(1, -1, 1)
        beta = torch.exp(self.beta.to(dtype=x.dtype, device=x.device)).view(1, -1, 1)
        return snake(x, alpha, beta)


# Alias-free (anti-aliased) activation: kaiser-windowed sinc resampling

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = F.conv_transpose1d(x, self.filter.to(dtype=x.dtype, device=x.device).expand(C, -1, -1), stride=self.stride, groups=C).mul_(self.ratio)
        x = x[..., self.pad_left:-self.pad_right]
        return x


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12):
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.to(dtype=x.dtype, device=x.device).expand(C, -1, -1), stride=self.stride, groups=C)


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    """upsample x2 -> pointwise activation -> downsample x2 (anti-aliased)."""

    def __init__(self, activation, up_ratio=2, down_ratio=2, up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# DAC encoder

class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)),
            Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y.add_(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim=16, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            weight_norm(nn.Conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048):
        super().__init__()
        block = [weight_norm(nn.Conv1d(1, d_model, kernel_size=7, padding=3))]
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]
        block += [
            Snake1d(d_model),
            weight_norm(nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


# Attention projection (encoder posterior head)

class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)).mul_(self.w1(x)))


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        B, N, C = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)).to(dtype=x.dtype, device=x.device)
        qkv = self.qkv(x) + bias
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(torch.mean(x, dim=1), self.out_dim)
        return self.proj(x)


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)

        self.norm2 = nn.LayerNorm(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=hidden_dim)

    def forward(self, x):
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return x.add_(self.mlp(self.norm2(x)))


# BigVGAN decoder

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d)))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1)))
                for _ in range(len(dilation))
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [Activation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt.add_(x)
        return x


class BigVGAN(nn.Module):
    """BigVGAN vocoder (MiniMax H3 32 kHz configuration).

    use_bias_at_final=False, use_tanh_at_final=False (output clamped to [-1, 1]).
    """

    def __init__(
        self,
        num_mels=2048,
        upsample_initial_channel=1024,
        upsample_rates=(5, 5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(nn.ConvTranspose1d(
                            upsample_initial_channel // (2 ** i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        ))
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, k, d))

        self.activation_post = Activation1d(activation=SnakeBeta(ch))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False))

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs.div_(self.num_kernels)

        x = self.activation_post(x)
        return self.conv_post(x).clamp_(-1.0, 1.0)


# Top-level VAE

class MiniMaxH3AudioVAE(nn.Module):
    """MiniMax H3 stereo audio VAE at 32 kHz.

    Latents are [B, 32, 2, T]: 32 channels, 2 stereo channels, T frames at
    40 latent frames per second (800 audio samples per latent frame). The
    stereo channels are processed independently by the mono encoder/decoder.
    Latents are normalized with the stored per-channel latents_mean/std.
    """

    def __init__(
        self,
        encoder_dim=64,
        encoder_rates=(2, 4, 4, 5, 5),
        latent_dim=2048,
        decoder_dim=1024,
        vae_latent_channels=32,
    ):
        super().__init__()
        self.sample_rate = 32000

        self.hop_length = 1
        for r in encoder_rates:
            self.hop_length *= r
        self.samples_per_latent = self.hop_length  # 800
        self.latents_per_second = self.sample_rate // self.hop_length  # 40
        self.output_sample_rate = self.sample_rate

        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.pre_block = AttnProjection(latent_dim, vae_latent_channels, num_heads=8)

        self.mean_proj = nn.Conv1d(vae_latent_channels, vae_latent_channels, 1)
        # checkpoint 中存在但推理不使用；保留以匹配键
        self.logs_proj = nn.Conv1d(vae_latent_channels, vae_latent_channels, 1)

        self.dec_in_proj = nn.Conv1d(vae_latent_channels, latent_dim, 1)
        self.decoder = BigVGAN(num_mels=latent_dim, upsample_initial_channel=decoder_dim)

        self.register_buffer("latents_mean", torch.tensor(_AUDIO_LATENTS_MEAN, dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(_AUDIO_LATENTS_STD, dtype=torch.float32))

    def decode(self, z):
        """Decode normalized latents [B, 32, 2, T] to stereo waveforms [B, 2, L] at 32 kHz."""
        b, c, s, t = z.shape
        z = z.permute(0, 2, 1, 3).reshape(b * s, c, t)
        mean = self.latents_mean.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        std = self.latents_std.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        z = z * std + mean
        x = self.dec_in_proj(z)
        x = self.decoder(x)  # [b * s, 1, L], already clamped to [-1, 1]
        return x.reshape(b, s, -1)

    def encode(self, waveform):
        """Encode stereo waveforms [B, 2, L] at 32 kHz (in [-1, 1]) to normalized latents [B, 32, 2, T].

        L is right-padded with zeros to a multiple of 800 samples; the returned
        posterior mean is used directly (no sampling).
        """
        b, s, length = waveform.shape
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        waveform = F.pad(waveform, (0, right_pad))
        x = waveform.reshape(b * s, 1, -1)
        x = self.encoder(x)  # [b * s, latent_dim, T]
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)  # [b * s, 32, T]
        z = self.mean_proj(x)
        mean = self.latents_mean.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        std = self.latents_std.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        z = (z - mean) / std
        return z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3)
