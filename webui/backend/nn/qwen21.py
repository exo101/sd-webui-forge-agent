# https://github.com/comfyanonymous/ComfyUI Qwen-Image-2.1
# Reference: https://github.com/QwenLM/Qwen-Image
# 三大性能件移植自 ComfyUI 的 qwen_image21：block-causal 分段注意力、前缀 KV cache、comfy_kitchen 融合 kernel

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import memory_management
from backend.attention import attention_function, attention_pytorch
from backend.args import dynamic_args
from backend.nn.flux import EmbedND, apply_rope1, timestep_embedding

try:
    import comfy_kitchen as _ck
except Exception:
    _ck = None


class ZeroCenteredRMSNorm(nn.Module):
    # stored weight is scale - 1, applied in fp32
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x):
        # not a forge-patched module: weight may stay on CPU under partial lowvram load
        w = self.weight
        if w.device != x.device:
            w = w.to(x.device)
        return F.rms_norm(x.float(), (x.shape[-1],), w.float() + 1.0, self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, bias=False):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=bias)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim, bias=False)

    def forward(self, timestep, dtype):
        return self.timestep_embedder(timestep_embedding(timestep.float(), 256).to(dtype))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate, proj = self.gate_up(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * proj)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False)])
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)

    def forward(self, x, pe, attn_fn, prefix_len):
        # (B, N, H, D) throughout: no transposes, the rope table is laid out to match
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, -1)
        k = self.to_k(x).view(B, N, self.heads, -1)
        v = self.to_v(x).view(B, N, self.heads, -1)

        if _ck is not None and x.is_cuda:
            # fused RMSNorm + RoPE in one kernel
            # the fused path reads the norm weights directly, bypassing the forge manual-cast forward:
            # under CPU-storage (quantized) loads they may still live on CPU, so pin them to q/k's device
            q, k = _ck.rms_rope(q, k, pe, self.norm_q.weight.to(q.device, q.dtype), self.norm_k.weight.to(k.device, k.dtype), self.norm_q.eps)
        else:
            q = self.norm_q(q)
            k = self.norm_k(k)
            q = apply_rope1(q, pe)
            k = apply_rope1(k, pe)

        q = q.flatten(start_dim=2)
        k = k.flatten(start_dim=2)
        v = v.flatten(start_dim=2)

        return self.to_out[0](attn_fn(q, k, v, self.heads))


def _split_rows(p):
    # shared modulation rows: (t = 0 row for text and references, sampled-t rows for the target)
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def _modulated_norm(norm, x, scale, prefix_len, zero):
    # LayerNorm * (1 + scale), fused over every row with the target scale; the prefix rows are then redone with the t = 0 scale
    s_prefix, s_target = scale
    if _ck is not None and x.is_cuda:
        out = _ck.adaln(x, s_target, zero, norm.eps)
        if prefix_len:
            out[:, :prefix_len] = _ck.adaln(x[:, :prefix_len], s_prefix, zero, norm.eps)
        return out
    out = norm(x)
    return torch.cat([out[:, :prefix_len] * (1 + s_prefix), out[:, prefix_len:] * (1 + s_target)], dim=1)


def _gated_residual(x, y, gate, prefix_len):
    g_prefix, g_target = gate
    x[:, prefix_len:].addcmul_(y[:, prefix_len:], g_target)
    if prefix_len:
        x[:, :prefix_len].addcmul_(y[:, :prefix_len], g_prefix)
    return x


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3, eps=1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    def forward(self, x, mod, pe, attn_fn, prefix_len):
        scale1, gate1, scale2, gate2, zero = mod
        x = _gated_residual(x, self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len, zero), pe, attn_fn, prefix_len), gate1, prefix_len)
        x = _gated_residual(x, self.img_mlp(_modulated_norm(self.img_norm2, x, scale2, prefix_len, zero)), gate2, prefix_len)
        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)
        return x


class LastLayer(nn.Module):
    # scale only, no shift
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        if _ck is not None and x.is_cuda:
            return _ck.adaln(x, scale, torch.zeros_like(scale[:1]), self.norm.eps)
        return self.norm(x) * (1 + scale)


def _block_causal_attention(segments, store=None, block_index=0, prefix_len=0):
    # segments: (start, end, mask); text segments get a causal mask, image blocks attend to everything before their end
    def attn(q, k, v, heads):
        if store is not None:
            # K and V stacked on dim 1 so batch stays first
            store[block_index] = torch.stack([k[:, :prefix_len], v[:, :prefix_len]], dim=1)
        outs = []
        for start, end, mask in segments:
            if mask is None:
                outs.append(attention_function(q[:, start:end], k[:, :end], v[:, :end], heads))
            else:
                outs.append(attention_pytorch(q[:, start:end], k[:, :end], v[:, :end], heads, mask))
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    return attn


def _prefix_cached_attention(prefix_k, prefix_v):
    # target-only queries: block-causal reduces to full attention over [cached prefix, target]
    def attn(q, k, v, heads):
        return attention_function(q, torch.cat([prefix_k, k], dim=1), torch.cat([prefix_v, v], dim=1), heads)
    return attn


def prefix_cache_key(x, context, refs, slots):
    # one fp32 tensor per batch row: lengths and slots, then the prompt embedding and reference latents
    # the target shape is part of it because reference rope ids are centred on the target
    header = [context.shape[1], len(refs)] + list(x.shape[-2:]) + list(slots) + [s for r in refs for s in r.shape[-2:]]
    header = torch.tensor(header, dtype=torch.float32, device=context.device).expand(context.shape[0], -1)
    return torch.cat([header, context.float().flatten(1)] + [r.float().flatten(1) for r in refs], dim=1)


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        image_model=None,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        # one modulation shared by every block
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList([
            QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps)
            for _ in range(num_layers)
        ])

        self.norm_out = LastLayer(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)

        # text + reference K/V are step-independent (t = 0 modulation, causal prefix), cached for one sampling run
        # slot: {"key": fp32 tensor, "blocks": {index: (B, 2, prefix_len, H*D)}}, LRU, oldest first
        self.prefix_cache_enabled = True
        self._cache_slots = []

    def reset_prefix_cache(self, enabled=True):
        for s in self._cache_slots:
            s["blocks"].clear()
        self._cache_slots = []
        self.prefix_cache_enabled = enabled

    def _select_prefix_cache(self, key, cache_bytes, device):
        # returns (slot to read or fill, whether the slot is filled), or (None, False) to recompute
        for s in self._cache_slots:
            if s["key"].shape == key.shape and torch.equal(s["key"], key.to(s["key"].device)):
                self._cache_slots.remove(s)
                self._cache_slots.append(s)
                return s, len(s["blocks"]) >= len(self.transformer_blocks)
        if self._cache_slots:
            # LRU eviction makes room for the other conditioning's slot
            s = self._cache_slots.pop(0)
            s["blocks"].clear()
        if memory_management.get_free_memory(device) < 1.5 * cache_bytes:
            # no room for this slot: recompute rather than evict every step
            return None, False
        slot = {"key": key.clone().to(device), "blocks": {}}
        self._cache_slots.append(slot)
        return slot, False

    def build_sequence(self, x, context, ref_latents, image_slots):
        # text with each reference image spliced in at its slot, target image last
        txt = self.txt_in(context)
        slots = (image_slots + [txt.shape[1]] * len(ref_latents))[:len(ref_latents)]
        bounds = [0] + slots + [txt.shape[1]]

        parts, ids, segments = [], [], []
        pos, length = 0, 0
        for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x]):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
                segments.append((length, length + n, torch.ones((n, length + n), dtype=torch.bool, device=x.device).tril(length)))
                pos += n
                length += n
            h, w = img.shape[-2:]
            parts.append(self.img_in(img.flatten(2).transpose(1, 2)))
            # half a token where a reference grid has the other parity, so it centres on the target
            hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2)
            ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - x.shape[-1] % 2)
            ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
            segments.append((length, length + h * w, None))
            pos += max(h, w)
            length += h * w

        # (B, N, 1, ...): the layout the fused rms_rope wants for (B, N, H, D) queries
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments

    def forward(self, x, timestep, context, ref_latents=None, image_slots=None, **kwargs):
        B, C, H, W = x.shape
        dtype = x.dtype
        ref_latents = [ref.to(x) for ref in (ref_latents or dynamic_args.ref_latents or [])]
        image_slots = list(image_slots or dynamic_args.qwen21_image_slots or [])

        hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, image_slots)
        prefix_len = hidden_states.shape[1] - H * W

        # pipeline rounds t*1000 and t to the compute dtype; text and reference tokens modulate from t = 0
        t = ((timestep * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        zero = torch.zeros_like(scale1[:1, None])
        mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()), zero)

        slot, cached = None, False
        if self.prefix_cache_enabled and prefix_len > 0:
            key = prefix_cache_key(x, context, ref_latents, image_slots)
            cache_bytes = 2 * len(self.transformer_blocks) * B * prefix_len * self.inner_dim * hidden_states.element_size()
            slot, cached = self._select_prefix_cache(key, cache_bytes, x.device)
        if cached:
            # a cached step runs target rows only
            hidden_states, pe = hidden_states[:, prefix_len:], pe[:, prefix_len:]
            target_prefix_len = 0
        elif slot is not None:
            prefix_states, hidden_states = hidden_states[:, :prefix_len], hidden_states[:, prefix_len:]
            prefix_pe, pe = pe[:, :prefix_len], pe[:, prefix_len:]
            target_prefix_len = 0
        else:
            target_prefix_len = prefix_len

        for i, block in enumerate(self.transformer_blocks):
            if slot is not None:
                if not cached:
                    prefix_attn = _block_causal_attention(segments[:-1], slot["blocks"], i, prefix_states.shape[1])
                    prefix_states = block(prefix_states, mod, prefix_pe, prefix_attn, prefix_states.shape[1])
                prefix_k, prefix_v = slot["blocks"][i].unbind(1)
                attn_fn = _prefix_cached_attention(prefix_k, prefix_v)
            else:
                attn_fn = _block_causal_attention(segments)
            hidden_states = block(hidden_states, mod, pe, attn_fn, target_prefix_len)

        hidden_states = self.norm_out(hidden_states if slot is not None else hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
