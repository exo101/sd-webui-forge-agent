# 移植自 ComfyUI 的 MiniMax H3（comfy/ldm/minimax），参考 https://github.com/comfyanonymous/ComfyUI
# MiniMax H3 音视频联合 DiT 的 Forge 原生实现：
# 单流 packed-token transformer，序列 [text | cond/ref 行 | audio | video]（audio/video 恒为末两段）
# 视频 latent 24ch（空间 /16、时间 /4、patch 1x2x2），音频 latent 32ch（立体声、40 latent fps）
# 条件为 Qwen3-VL 截断层 hidden states（5120 维、无 final norm）
#
# 与 ComfyUI 版本的差异：
# - 纯 torch.nn 模块；在 using_forge_operations 上下文中 op 类自动变量化（量化加载走 mixed_precision_ops）
# - denoise_mask / transformer_options / WrapperExecutor / prefetch 全部删除（扩展用不到）
# - 双 schedule 机制保留：采样器以常数 audio_scale = shift_v/shift_a 携带音频 latent，forward() 内部撤销
# - stream_blocks=True 时逐 block 流式换入 GPU（21GB int8 模型跑 16GB 卡）

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import memory_management
from backend.attention import attention_function
from backend.operations import using_forge_operations
from backend.quant_ops import QuantizedTensor
from backend.state_dict import detect_quantization, load_state_dict
from backend.utils import no_init_weights, calculate_parameters, weight_dtype

try:
    import comfy_kitchen as _ck
except Exception:
    _ck = None

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0

# split-half rope 融合 kernel 失败后永久回退纯 torch
_ROPE_KERNEL_OK = True


def time_shift_sigma(sigma, from_shift, to_shift):
    # invert sigma = s*b/(1+(s-1)*b) to the base grid, re-apply the other shift
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def patchify_video(latent, patch_size=(1, 2, 2)):
    # [B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    # [B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def pad_to_patch_size(img, patch_size=(1, 2, 2)):
    # circular pad to a multiple of the patch size（同 ComfyUI common_dit）
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    return F.pad(img, pad, mode="circular")


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    # linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    # area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    # origin + exclusive cumsum
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _ref_t_span(blk):
    # time-axis span a reference block occupies ahead of the target streams
    kind = blk["kind"]
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(blk["ref_audio_t"])
    if kind in ("video", "video_audio"):
        return max(float(blk["ref_audio_t"]), sum(_video_t_spans(blk["latent_t"])))
    return 0.0


def _audio_grid(cursor, t, w_low, w_high):
    # channel-major stereo rows: t advances per latent frame, w pinned to the grid extremes per stereo channel, h stays 0
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def rope_rotation_table(angles, dtype):
    """[S, rot_dim] pair angles -> [1, S, 1, rot_dim/2, 2, 2] rotation matrices.
    角度尾部的 0 是恒等 pair（c=1, s=0），用于把 96 维角度补齐到 head_dim。"""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


def _rms_rope_split_half_torch(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
    # 纯 torch 回退：逐字复刻 comfy_kitchen _rms_rope1 的 split-half 分支
    if k_scale is None:
        k_scale = q_scale

    def one(x, scale):
        x_float = x.float()
        rrms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + epsilon)
        x_norm = (x_float * rrms * scale.float()).to(x.dtype).float()
        freqs = freqs_cis.float()
        pairs = x_norm.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        return out.movedim(-1, -2).reshape_as(x).to(x.dtype)

    return one(q, q_scale), one(k, k_scale)


class TimeEmbedder(nn.Module):
    # 非 curve checkpoint 使用；fp32 频率嵌入，cos 在 sin 前
    def __init__(self, freq_dim, hidden, out):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
        self.proj_out = nn.Linear(hidden, out, bias=True)

    def forward(self, t):
        # t: [M] in [0, 1]; fp32 throughout
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(F.silu(self.proj_in(emb)))


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, gate_compress=False):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)
        self.to_gate_compress = None
        if gate_compress:
            # VSA gate，dense forward 不消费
            self.to_gate_compress = nn.Linear(hidden, inner, bias=False)

    def forward(self, x, rope_freqs=None):
        global _ROPE_KERNEL_OK
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # 逐头 RMSNorm + split-half rope；本地 kernel 要求覆盖整个 head_dim，
            # rope_freqs 的角度尾部已零填充为恒等 pair
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = self.q_norm.weight
            kw = self.k_norm.weight
            # 非 forge 补丁模块的 weight 可能留在 CPU（低显存/流式加载）
            if qw.device != x.device:
                qw = qw.to(x.device)
            if kw.device != x.device:
                kw = kw.to(x.device)
            if _ck is not None and _ROPE_KERNEL_OK and x.is_cuda:
                try:
                    q, k = _ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps)
                except Exception:
                    _ROPE_KERNEL_OK = False
                    q, k = _rms_rope_split_half_torch(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps)
            else:
                q, k = _rms_rope_split_half_torch(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps)
            q = q[0]
            k = k[0]
        else:
            # token_refiner 无 rope：纯逐头归一化
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))

        out = attention_function(
            q.reshape(1, s, -1), k.reshape(1, s, -1), v.reshape(1, s, -1), self.heads)
        return self.out_proj(out.squeeze(0))


class MLP(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.ffn = ffn
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        g = self.fc1(x)
        return self.fc2(F.silu(g[:, : self.ffn]) * g[:, self.ffn :])


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def forward(self, t_emb):
        # [M, t_dim] -> expand tensors of [M*modalities, hidden]
        x = self.linear(F.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_row(vecs, row, dtype):
    # row is a mod-row index, or a per-token LongTensor of mod-row indices
    return vecs[row].to(dtype)


def _mod_scale_shift(h, shift, scale, segments):
    # segments: [(start, stop, mod_row)] covering h contiguously
    for a, b, row in segments:
        h[a:b].mul_(1.0 + _mod_row(scale, row, h.dtype)).add_(_mod_row(shift, row, h.dtype))
    return h


def _mod_gate(x, gate, other, segments):
    # other is the fresh attn/mlp output: accumulate the gated residual into the stream in place
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], _mod_row(gate, row, x.dtype))
    return x


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps)
        self.mlp = MLP(hidden, ffn)

    def forward(self, x):
        # attn/mlp outputs are fresh: accumulate residuals in place
        x = self.attn(self.norm1(x)).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps):
        super().__init__()
        self.blocks = nn.ModuleList([
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps)
            for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(hidden, eps=final_eps)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 apply_silu=True, gate_compress=False):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps)
        self.norm2 = nn.RMSNorm(hidden, eps=eps)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, gate_compress=gate_compress)
        self.mlp = MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu)

    def forward(self, x, t_emb, mod_segments, rope_freqs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


def _pdd_head(head, h, n, start, stop, flow_shift):
    # PDD head bank：一步消费其跨越的 heads 的 dt 加权均值（row block 0 是完整 head，后面是偏移）
    grid = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    dt = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
    w = (dt / dt.sum()).to(h.dtype)
    weight, bias = head.weight, head.bias
    # 量化权重重解包（PDD 路径不走融合 kernel）
    if hasattr(weight, "dequantize"):
        weight = weight.dequantize()
    if bias is not None and hasattr(bias, "dequantize"):
        bias = bias.dequantize()
    weight = weight.to(device=h.device, dtype=h.dtype)
    if bias is not None:
        bias = bias.to(device=h.device, dtype=h.dtype)
    rows = weight.reshape(n, -1, weight.shape[1])
    brows = bias.reshape(n, -1)
    first = max(start, 1)
    return F.linear(h, rows[0] + torch.einsum("n,noi->oi", w[first - start:], rows[first:stop]),
                    brows[0] + torch.einsum("n,no->o", w[first - start:], brows[first:stop]))


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu)
        self.video_out = nn.Linear(hidden, video_dim, bias=True)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True)

    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
        # video_seg / audio_seg: (start, stop, row) of the target streams
        shift, scale = self.adaln_proj(t_emb)

        def mod(seg):
            a, b, row = seg
            return (self.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype))
                    + _mod_row(shift, row, shift.dtype)).to(torch.float32)

        weight = self.video_out.weight
        if hasattr(weight, "shape"):
            n = weight.shape[0] // self.video_out.out_features
        if n == 1:
            return self.video_out(mod(video_seg)), self.audio_out(mod(audio_seg))

        if sample_sigmas is None:
            raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        start, stop = (round(float(1.0 - time_shift_sigma(s, shifts[0], 1.0)) * n) for s in (sigma, sigma_next))
        start = min(start, n - 1)
        stop = max(stop, start + 1)
        return (_pdd_head(self.video_out, mod(video_seg), n, start, stop, shifts[0]),
                _pdd_head(self.audio_out, mod(audio_seg), n, start, stop, shifts[1]))


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]  # (kind, n_rows)
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]  # per segment: [n, 3] float64 (t, h, w)

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        row = text_len

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        # refs pack between text and the targets, so the target timeline starts after their spans
        cursor = float(text_len)
        for blk in refs or ():
            cursor += _ref_t_span(blk)

        if keyframes:
            # fl2va: keyframe cond rows right after text, sharing the target spatial grid;
            # anchors count from the target timeline origin, FRAME_RESCALE per pixel frame, 1.0 per audio latent frame
            for kf in keyframes:
                cond_t = cursor + FRAME_RESCALE * kf["resolved_frame_index"]
                video_latent = kf.get("latent")
                if video_latent is not None:
                    vt = video_latent.shape[2]
                    n = vt * frame_rows
                    segments.append(("cond", n))
                    pos.append(_video_grid(vt, frame, cond_t))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                audio_latent = kf.get("audio_latent")
                if audio_latent is not None:
                    rt = audio_latent.shape[-1]
                    segments.append(("cond_audio", rt * 2))
                    pos.append(_audio_grid(cond_t, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2

        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video
                    # rows, both sharing the cursor origin
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        # target audio then target video, always the last two segments
        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)  # [S, 3] float64
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        # contiguous segment table (start, stop, kind)
        # kinds: text / cond / cond_audio / ref_img / ref_audio / audio / video
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


class MiniMaxH3Model(nn.Module):
    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 adaln_curve_grid=None, gate_compress=False, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_dim = attention_head_dim
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.use_adaln_curves = adaln_curve_grid is not None
        # curve-form checkpoints replace the time embedder and full-width adaln weights
        # with a small shared basis of the time-embedding curve
        apply_silu = not self.use_adaln_curves
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                     time_embed_dim, norm_eps, qk_norm_eps, apply_silu=apply_silu,
                     gate_compress=gate_compress)
            for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_patch_dim, audio_latents_dim,
                                      final_norm_eps, apply_silu=apply_silu)

        # 大模型流式：扩展置 True 后，顶层模块与 blocks 在 forward 时按需换入 device
        self.stream_blocks = False

    # ------------------------------------------------------------------
    # 流式加载支持（模型驻留 CPU，逐 block 换入 GPU）
    # ------------------------------------------------------------------

    def _top_modules(self):
        mods = [self.video_patch_proj, self.audio_patch_proj, self.condition_proj,
                self.token_refiner, self.rope, self.final_layer]
        if self.use_adaln_curves:
            mods.append(None)  # placeholder to keep tuple shape stable
        else:
            mods.append(self.time_embedder)
        return mods

    def _on_device(self, module, device):
        for p in list(module.parameters()) + list(module.buffers()):
            if p.device != device:
                return False
        return True

    def _ensure_top_on_device(self, device):
        if not self.stream_blocks:
            return
        device = torch.device(device)
        for m in self._top_modules():
            if m is not None and not self._on_device(m, device):
                m.to(device, non_blocking=True)
        if self.use_adaln_curves and self.adaln_t_table.device != device:
            self.adaln_t_table.data = self.adaln_t_table.to(device, non_blocking=True)

    # ------------------------------------------------------------------

    def preprocess_text_embeds(self, text_states):
        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds."""
        self._ensure_top_on_device(text_states.device)
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)

    def rope_freqs(self, position_ids, device):
        # [S, 3] float64 -> [S, head_dim] fp32（尾部零填充为恒等 pair）
        pos = position_ids.to(torch.float32).to(device)
        inv = self.rope.inv_freq
        if inv.device != device:
            inv = inv.to(device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)      # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)              # [S, 48]
        angles = torch.cat((half, half), dim=-1)               # [S, 96]
        pad = self.attention_head_dim - angles.shape[-1]
        if pad > 0:
            angles = torch.cat((angles, torch.zeros(angles.shape[0], pad, dtype=angles.dtype, device=device)), dim=-1)
        return angles

    def _cond_video_rows(self, payload, device):
        """Concatenated visual condition rows (normalized latents -> patchified), with condition noise augmentation."""
        rows = []
        aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)
        seed = int(payload.get("seed", 0))
        # every condition intentionally restarts the same RNG stream
        for z in payload.get("cond_video_latents", []):
            r = patchify_video(z.to(torch.float32), self.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def _cond_audio_rows(self, payload, device):
        rows = []
        aug = payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)
        seed = int(payload.get("seed", 0)) + 1
        for z in payload.get("cond_audio_latents", []):
            r = pack_audio(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def forward(self, x, sigma_v, context, payload=None, sample_sigmas=None):
        """x = [video [1,24,T,H,W], audio [1,32,2,T]]；sigma_v 为 video 流的 sigma（float）。
        返回 [-video_out, -audio_out]（网络原始输出取负，与 ComfyUI 一致）。"""
        payload = payload or {}
        scale = float(payload.get("audio_scale", 1.0))
        audio_src = x[1]
        if scale != 1.0:
            # 采样器以 sigma_v/sigma_a 常数倍携带音频 latent，这里先撤销
            sigma_a = time_shift_sigma(sigma_v, self.sigma_shift_video, self.sigma_shift_audio)
            carry = sigma_a / max(sigma_v, 1e-6)
            x = [x[0], audio_src * carry]

        out = self._forward(x, sigma_v, context, payload, sample_sigmas)

        if scale != 1.0:
            # d/d(sigma_v) of the carried variable：把速度换算回携带变量域
            sigma_a = time_shift_sigma(sigma_v, self.sigma_shift_video, self.sigma_shift_audio)
            out[1] = ((1.0 - scale) * (audio_src * carry)
                      + (1.0 + (scale - 1.0) * sigma_a) * out[1])
        return out

    def _forward(self, x, sigma_v, context, payload, sample_sigmas):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = payload or {}
        device = video_x.device
        dtype = context.dtype  # compute dtype

        self._ensure_top_on_device(device)

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        # 调用方预建 layout 后放 payload 里复用
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                  keyframes=payload.get("keyframes"),
                                  refs=payload.get("refs"))
            payload["layout"] = layout

        sigma_v = float(sigma_v)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, self.sigma_shift_video, self.sigma_shift_audio))

        # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}

        unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments})
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        # embed
        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # cond_audio / ref_audio / audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            # adaln projections consume interpolated coordinates of the time-embedding curve
            table = self.adaln_t_table
            if table.device != device:
                table = table.to(device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)  # t in [0,1] -> fractional grid index
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)  # t=1.0 停在最后一段
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        # rotation table computed once per forward, consumed by the split-half rope
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

        # blocks（流式模式：2-block 窗口换入 GPU，窗口外的退回 offload device）
        offload = getattr(self, "offload_device", None)
        if self.stream_blocks:
            n_blocks = len(self.blocks)
            for i, block in enumerate(self.blocks):
                if not self._on_device(block, device):
                    block.to(device, non_blocking=True)
                if i + 1 < n_blocks and not self._on_device(self.blocks[i + 1], device):
                    self.blocks[i + 1].to(device, non_blocking=True)
                if i > 0 and offload is not None and not self._on_device(self.blocks[i - 1], offload):
                    self.blocks[i - 1].to(offload, non_blocking=True)
                h = block(h, t_emb, mod_segments, rope_freqs)
        else:
            for block in self.blocks:
                h = block(h, t_emb, mod_segments, rope_freqs)

        # target streams are single contiguous segments (audio then video, last two)
        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
        video_seg = (va, vb, t_row[seg_t["video"]])
        audio_seg = (aa, ab, t_row[seg_t["audio"]])
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v, sample_sigmas,
                                (self.sigma_shift_video, self.sigma_shift_audio))

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a)

        return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


# ----------------------------------------------------------------------
# 构建入口
# ----------------------------------------------------------------------


def _count_blocks(keys, prefix_fmt):
    count = 0
    while any(k.startswith(prefix_fmt.format(count)) for k in keys):
        count += 1
    return count


def detect_minimax_h3_config(state_dict):
    """从 state_dict 推导 MiniMax H3 超参（同 ComfyUI model_detection），非 H3 返回 None。"""
    if "video_patch_proj.weight" not in state_dict or "audio_patch_proj.weight" not in state_dict:
        return None
    keys = list(state_dict.keys())
    config = {}
    config["num_layers"] = _count_blocks(keys, "blocks.{}.")
    config["token_refiner_num_layers"] = _count_blocks(keys, "token_refiner.blocks.{}.")
    config["hidden_size"] = state_dict["video_patch_proj.weight"].shape[0]
    config["latents_dim"] = state_dict["final_layer.video_out.weight"].shape[0] // 4  # patch 1x2x2
    config["audio_latents_dim"] = state_dict["final_layer.audio_out.weight"].shape[0]
    config["attention_head_dim"] = state_dict["blocks.0.attn.q_norm.weight"].shape[0]
    qkv = state_dict["blocks.0.attn.qkv_proj.weight"]
    config["num_attention_heads"] = qkv.shape[0] // (3 * config["attention_head_dim"])
    config["ffn_hidden_size"] = state_dict["blocks.0.mlp.fc1.weight"].shape[0] // 2
    config["text_dim"] = state_dict["condition_proj.weight"].shape[1]
    if "adaln_t_table" in state_dict:
        # adaln shipped over a precomputed curve basis（无 time embedder）
        table = state_dict["adaln_t_table"].shape  # [grid, k]
        config["adaln_curve_grid"] = table[0]
        config["time_embed_dim"] = table[1]
    else:
        te = state_dict["time_embedder.proj_in.weight"]
        config["timestep_input_dim"] = te.shape[1]
        config["time_embed_hidden_size"] = te.shape[0]
        config["time_embed_dim"] = state_dict["time_embedder.proj_out.weight"].shape[0]
    config["rope_inv_freq_len"] = state_dict["rope.inv_freq"].shape[0]
    config["gate_compress"] = "blocks.0.attn.to_gate_compress.weight" in keys  # VSA-trained
    return config


def build_minimax_h3_model(state_dict, log_name=None):
    """按 Forge 标准 recipe 构建 MiniMax H3（量化/非量化双分支，同 loader.py 的 UNet 尾部）。"""
    config = detect_minimax_h3_config(state_dict)
    if config is None:
        raise ValueError("state_dict is not a MiniMax H3 DiT")

    load_device = memory_management.get_torch_device()
    offload_device = memory_management.unet_offload_device()
    state_dict_parameters = calculate_parameters(state_dict)
    state_dict_dtype = weight_dtype(state_dict)
    quant_config = detect_quantization(state_dict, is_unet=True)

    if quant_config is not None:
        # 量化模型：权重留在 initial_device（通常 CPU），运行时由 weights_manual_cast / 流式换入
        storage_dtype = state_dict_dtype
        memory_management.logger.info("MiniMax H3: Using MixedPrecision for Model")
        if memory_management.should_use_bf16(load_device):
            computation_dtype = torch.bfloat16
        elif memory_management.should_use_fp16(load_device, prioritize_performance=True):
            computation_dtype = torch.float16
        else:
            computation_dtype = torch.float32
        initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=storage_dtype)

        with no_init_weights():
            # 量化模型权重常驻 CPU/int8，forward 必须走 manual cast（设备 + 计算 dtype）
            with using_forge_operations(manual_cast_enabled=True, bnb_dtype=quant_config):
                model = MiniMaxH3Model(**config)
    else:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        storage_dtype = memory_management.unet_dtype(device=load_device, model_params=state_dict_parameters,
                                                     supported_dtypes=supported_dtypes, weight_dtype=state_dict_dtype)
        computation_dtype = memory_management.inference_cast(weight_dtype=storage_dtype, inference_device=load_device,
                                                             supported_dtypes=supported_dtypes)
        initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
        need_manual_cast = storage_dtype != computation_dtype
        to_args = dict(device=initial_device, dtype=storage_dtype)

        with no_init_weights():
            with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast):
                model = MiniMaxH3Model(**config).to(**to_args)

    load_state_dict(model, state_dict, log_name=log_name or "MiniMaxH3Model")

    # 标准 load 的 copy_ 不改变空权重的 dtype/grad 标志：统一收尾
    for p in model.parameters():
        p.requires_grad = False
        if not isinstance(p, QuantizedTensor) and p.dtype != computation_dtype:
            p.data = p.data.to(computation_dtype)

    model.storage_dtype = storage_dtype
    model.computation_dtype = computation_dtype
    model.load_device = load_device
    model.initial_device = initial_device
    model.offload_device = offload_device
    return model
