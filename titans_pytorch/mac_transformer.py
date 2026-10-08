from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# BlockMask construction is a pure function of these four numbers, but it was
# being redone by every Titan layer on every forward. One entry per distinct
# shape is enough; the mask itself is tiny (a [Q/128, KV/128] index table).
_MAC_BLOCK_MASK_CACHE = {}
_MAC_BLOCK_MASK_CACHE_MAX = 8


def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):

    cache_key = (seq_len, window_size, persist_mem_len, bool(sliding))
    cached = _MAC_BLOCK_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem

        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        return is_persist_mem | (~is_persist_mem & causal_mask)

    # _compile = True is load-bearing, not a tuning knob. With _compile = False
    # create_block_mask materializes the DENSE [Q_LEN, KV_LEN] mask and then
    # reduces it to the sparse block form; the reduction promotes to int64, so
    # a 128K prefill asks for 131072^2 x 8 = 128 GiB and OOMs on an 80 GB card.
    # Compiled, the mask_mod and the block reduction fuse and the
    # dense intermediate is never allocated -- which is the whole point of
    # describing a windowed pattern with a block mask in the first place.
    block_mask = create_block_mask(create_mac_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len + persist_mem_len, _compile = True)

    if len(_MAC_BLOCK_MASK_CACHE) >= _MAC_BLOCK_MASK_CACHE_MAX:
        _MAC_BLOCK_MASK_CACHE.pop(next(iter(_MAC_BLOCK_MASK_CACHE)))
    _MAC_BLOCK_MASK_CACHE[cache_key] = block_mask
    return block_mask

# einstein notation related

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# b - batch
# n - sequence
# h - heads
# d - feature dimension

# absolute and relative positions

from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding

# HF-compatible split-half RoPE (matches LLaMA's convention)

def _rotate_half_hf(x):
    """Split-half rotation: pairs (dim_i, dim_{i+d/2})."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_hf(q, k, inv_freq, seq_offset=0):
    """Apply RoPE using HF LLaMA's split-half convention.

    Args:
        q: (batch, heads, seq_len, dim_head)
        k: (batch, heads, seq_len, dim_head)
        inv_freq: (dim_head // 2,) — the teacher's precomputed inverse frequencies
        seq_offset: starting position for RoPE (for chunked prefill)
    Returns:
        rotated (q, k) with same shapes
    """
    seq_len = k.shape[-2]
    positions = torch.arange(seq_len, device=q.device, dtype=inv_freq.dtype) + seq_offset
    freqs = torch.outer(positions, inv_freq)           # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)            # (seq_len, dim)
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)
    # broadcast: (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half_hf(q) * sin)
    k_rot = (k * cos) + (_rotate_half_hf(k) * sin)
    return q_rot, k_rot

def apply_rotary_pos_emb_hf_cached(q, k, cos_cache, sin_cache, seq_offset=0):
    """HF-compatible RoPE using a precomputed position table."""
    seq_len = k.shape[-2]
    cache_end = seq_offset + seq_len
    if seq_offset < 0 or cache_end > cos_cache.shape[0]:
        raise IndexError(
            f"RoPE positions [{seq_offset}, {cache_end}) exceed cached range "
            f"[0, {cos_cache.shape[0]})"
        )
    cos = cos_cache[seq_offset:cache_end].to(dtype=q.dtype)[None, None]
    sin = sin_cache[seq_offset:cache_end].to(dtype=q.dtype)[None, None]
    q_rot = (q * cos) + (_rotate_half_hf(q) * sin)
    k_rot = (k * cos) + (_rotate_half_hf(k) * sin)
    return q_rot, k_rot

# hyper connections / attend from x-transformers, which handles different queries and key lengths better

from x_transformers.attend import Attend

from hyper_connections import get_init_and_expand_reduce_stream_functions

# proposed neural memory

from titans_pytorch.neural_memory import NeuralMemory

# constants

LinearNoBias = partial(Linear, bias = False)

AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    return seq // mult * mult

def repeat_kv(x, num_groups):
    """Expand KV heads to match query heads: (B, H_kv, N, D) -> (B, H_q, N, D)."""
    if num_groups == 1:
        return x
    return x.repeat_interleave(num_groups, dim=1)

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# feedforward and attention

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

class SegmentedAttention(Module):
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        num_kv_heads = None,
        sliding = False,
        accept_value_residual = False,
        attend_kwargs: dict = dict(),
        use_flex_attn = False,
        pre_normed = False,
        rope_theta = 10000,
        rope_freqs = None,
    ):
        super().__init__()
        self.pre_normed = pre_normed
        self.norm = nn.RMSNorm(dim)

        num_kv_heads = num_kv_heads or heads
        assert heads % num_kv_heads == 0, f'heads ({heads}) must be divisible by num_kv_heads ({num_kv_heads})'
        self._num_kv_heads = num_kv_heads
        self._num_kv_groups = heads // num_kv_heads

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * num_kv_heads

        # When rope_freqs are provided (from a pretrained HF model), use
        # the split-half RoPE convention that HF LLaMA was trained with.
        # rotary_embedding_torch uses interleaved rotation which is incompatible.
        self._use_hf_rope = rope_freqs is not None
        if self._use_hf_rope:
            self.register_buffer('_rope_inv_freq', rope_freqs.float().clone(), persistent=False)
            # Pre-compute RoPE cos/sin lookup table for fast inference
            max_pos = 131072
            positions = torch.arange(max_pos, dtype=rope_freqs.dtype, device=rope_freqs.device)
            freqs = torch.outer(positions, rope_freqs.float())
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('_rope_cos_cached', emb.cos(), persistent=False)
            self.register_buffer('_rope_sin_cached', emb.sin(), persistent=False)
            # Still create rotary_emb for API compat but it won't be used
            self.rotary_emb = RotaryEmbedding(dim_head, theta=rope_theta)
        else:
            self.rotary_emb = RotaryEmbedding(dim_head, theta=rope_theta)

        self.attend = Attend(causal = True, **attend_kwargs)

        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_kv_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, num_kv_heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens

        total_segment_len = segment_len + num_longterm_mem_tokens
        self.total_segment_len = total_segment_len

        self.sliding = sliding

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_kv_heads = Rearrange('b n (h d) -> b h n d', h = num_kv_heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nn.Parameter(torch.zeros(2, num_kv_heads, num_persist_mem_tokens, dim_head))

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self._num_heads = heads
        self._dim_head = dim_head

        # Static KV cache for inference (lazily allocated on first decode step)
        self._static_k = None
        self._static_v = None
        self._cache_pos = 0
        self._absolute_cache_pos = 0
        self._static_prefix_len = 0
        self._static_prefix_signature = None
        self._static_prefix_source = None

        # CUDA-graph-safe decode (opt-in; default keeps the original path).
        #
        # A captured graph replays a fixed kernel sequence against fixed
        # addresses, so nothing that varies per token may live in a Python int:
        # whatever Python holds at capture time is frozen into the recording.
        # When this is on, _fast_forward_inference keeps the write index and the
        # RoPE position in device tensors and attends over the whole
        # preallocated buffer under a mask, so every step is byte-identical in
        # shape and launch order.
        self.graph_safe_decode = False

        # Window decode attention to total_segment_len, matching what prefill
        # and training actually do (see the fold at forward()'s eager path).
        #
        # Default ON. With it off, a decode query attends from position 0 over
        # the whole KV cache while prefill folds that same query into
        # [W*(p//W), p]: at the first token of a new segment that is the entire
        # context instead of a single key, so decode computes a different
        # function from the one the model was trained on. Measured at position
        # 2048 with segment_len 512 and the memory contribution zeroed, the
        # off-path diverges from prefill by relative 1.0 at every layer from
        # Titan layer 1 onward; windowed matches prefill to 3.8e-3, which is
        # BF16 noise. See debug_layer_divergence.py.
        self.windowed_decode = True
        # Set in _init_static_cache: windowed + graph-safe decode together allow
        # a one-window ring buffer instead of a full-length KV cache.
        self._ring_decode = False
        # Registered (non-persistent) so inductor tracks the in-place increments
        # as buffer mutations rather than opaque attribute writes. They stay None
        # until the static cache is allocated on the first decode step.
        self.register_buffer('_cache_pos_tensor', None, persistent=False)
        self.register_buffer('_absolute_cache_pos_tensor', None, persistent=False)
        self.register_buffer('_buffer_positions', None, persistent=False)

        # Pre-extract v_mix linear weight/bias for fast path (avoids nn.Sequential dispatch)
        self._v_mix_weight = None
        self._v_mix_bias = None
        if accept_value_residual:
            self._v_mix_weight = self.to_learned_v_mix[0].weight
            self._v_mix_bias = self.to_learned_v_mix[0].bias

        # Fast decode is enabled after projection setup.
        self._fast_path_ready = False
        self._fuse_qkv_decode = False
        self._fused_qkv_weight = None
        self._fused_qkv_bias = None
        self._fused_qkv_signature = None


    def _setup_fast_path(self):
        """Prepare the fast decode path."""
        self._fast_path_ready = True

    def _get_fused_qkv_decode_params(self):
        """Lazily pack Q and KV projections after checkpoint loading.

        The packed tensors are inference-only derived state and deliberately
        stay out of state_dict. Parameter storage/version changes invalidate
        them, which also makes moving or reloading the model safe.
        """
        q_weight = self.to_q.weight
        kv_weight = self.to_kv.weight
        q_bias = self.to_q.bias
        kv_bias = self.to_kv.bias
        signature = (
            q_weight.data_ptr(),
            q_weight._version,
            kv_weight.data_ptr(),
            kv_weight._version,
            None if q_bias is None else (q_bias.data_ptr(), q_bias._version),
            None if kv_bias is None else (kv_bias.data_ptr(), kv_bias._version),
        )
        if signature != self._fused_qkv_signature:
            self._fused_qkv_weight = torch.cat((q_weight, kv_weight), dim=0)
            if q_bias is None and kv_bias is None:
                self._fused_qkv_bias = None
            elif q_bias is not None and kv_bias is not None:
                self._fused_qkv_bias = torch.cat((q_bias, kv_bias), dim=0)
            else:
                raise RuntimeError("Q and KV projections must either both have bias or neither have bias")
            self._fused_qkv_signature = signature
        return self._fused_qkv_weight, self._fused_qkv_bias

    def project_qkv(self, seq):
        """Apply norm (if not pre-normed) + to_q/to_kv; return pre-RoPE q, k, v.

        q has shape (B, heads, N, dim_head); k, v have (B, num_kv_heads, N, dim_head).
        Callers that want the *same* q/k/v consumed by attention should pass the
        returned tensors back via ``precomputed_qkv=`` so projection isn't redone.
        """
        if not self.pre_normed:
            seq = self.norm(seq)
        q = self.split_heads(self.to_q(seq))
        kv = self.to_kv(seq)
        k, v = kv.chunk(2, dim=-1)
        k, v = map(self.split_kv_heads, (k, v))
        return q, k, v

    # --- shared attention math -------------------------------------------
    #
    # Every backend (eager, flex, decode) performs the same sequence of steps:
    #
    #   1. project Q/K/V            (or accept them precomputed)
    #   2. mix the value residual   -> _mix_value_residual
    #   3. apply rope               -> apply_rope
    #   4. prepend the prefix       (persistent memory + neural-memory K/V)
    #   5. attend                   <- the only step that differs per backend
    #   6. blend the memory read    -> _memory_to_heads / _blend_memory
    #   7. merge heads, project out, apply the output gate
    #
    # Steps 2 and 6 carry the semantics that must not drift between backends, so
    # they live here rather than being restated in each forward body.

    def _mix_value_residual(self, seq, v, value_residual):
        """Blend this layer's V toward the value produced by the first layer.

        ``seq`` may have been padded up to a segment multiple by the caller, in
        which case the residual arriving from the previous layer is shorter than
        V and is padded to match.
        """
        if not exists(self.to_learned_v_mix):
            return v

        mix = self.to_learned_v_mix(seq)
        if value_residual.shape[-2] < v.shape[-2]:
            value_residual = pad_at_dim(
                value_residual, (0, v.shape[-2] - value_residual.shape[-2]), dim = -2
            )
        return v.lerp(value_residual, mix)

    def _memory_to_heads(self, memory_values, memory_mix, batch, tokens):
        """Reshape the neural-memory read into attention head layout."""
        if not exists(memory_values):
            return None, None

        memory_out = memory_values.reshape(
            batch, tokens, self._num_heads, self._dim_head
        ).transpose(1, 2)

        history_mass = None
        if exists(memory_mix):
            history_mass = memory_mix.reshape(
                batch, tokens, self._num_heads, self._dim_head
            ).transpose(1, 2)

        return memory_out, history_mass

    @staticmethod
    def _blend_memory(out, memory_out, history_mass):
        """Combine exact local attention with the long-term memory read.

        Odyssey treats the neural memory as a parametric long-term attention map.
        M(Q) already lives in per-query-head value space, so it combines with
        local attention before the shared output projection: additively, or as a
        per-head/per-channel lerp when a history mass is supplied.
        """
        if not exists(memory_out):
            return out
        if not exists(history_mass):
            return out + memory_out
        return out.lerp(memory_out, history_mass)

    def _select_backend(self, seq, cache, disable_flex_attn):
        """Choose an attention backend. Policy only -- no attention math here."""
        if exists(cache) and seq.shape[-2] == 1:
            return 'decode'
        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return 'flex'
        return 'eager'

    def apply_rope(self, q, k, seq_offset = 0):
        """Rotate q/k for absolute positions starting at ``seq_offset``.

        Exposed so a caller can rotate once and hand the rotated tensors to
        both attention and the neural memory, instead of each rotating its own
        copy.  Only meaningful for the HF split-half convention; the
        rotary_embedding_torch path needs the cached-key form applied inside
        attention and is left to do its own rotation.
        """
        if not self._use_hf_rope:
            raise RuntimeError("apply_rope requires the HF rope convention")
        return apply_rotary_pos_emb_hf(q, k, self._rope_inv_freq, seq_offset = seq_offset)

    def project_memory_tokens(self, memory_tokens):
        """Project fixed hidden-space memory tokens once for cached decoding."""
        if memory_tokens.ndim == 4:
            b, w, m, _ = memory_tokens.shape
            mem_flat = rearrange(memory_tokens, 'b w m d -> (b w) m d')
            mem_kv = self.to_kv(mem_flat)
            mem_k, mem_v = mem_kv.chunk(2, dim=-1)
            mem_k, mem_v = map(self.split_kv_heads, (mem_k, mem_v))
            return (
                rearrange(mem_k, '(b w) h m d -> b w h m d', b=b, w=w),
                rearrange(mem_v, '(b w) h m d -> b w h m d', b=b, w=w),
            )
        if memory_tokens.ndim == 3:
            mem_kv = self.to_kv(memory_tokens)
            mem_k, mem_v = mem_kv.chunk(2, dim=-1)
            return tuple(map(self.split_kv_heads, (mem_k, mem_v)))
        raise ValueError(
            f"memory_tokens must be (B,M,D) or (B,W,M,D), got {tuple(memory_tokens.shape)}"
        )

    def reset_static_cache(self):
        """Free static KV cache buffers. Call between sequences."""
        self._static_k = None
        self._static_v = None
        self._cache_pos = 0
        self._absolute_cache_pos = 0
        self._static_prefix_len = 0
        self._static_prefix_signature = None
        self._static_prefix_source = None
        self._max_new_tokens = None
        self._cache_pos_tensor = None
        self._absolute_cache_pos_tensor = None
        self._buffer_positions = None
        self._ring_decode = False

    def _decode_prefix_signature(self, memory_kv):
        signature = []
        if memory_kv is not None:
            mem_k, mem_v = memory_kv
            signature.extend((mem_k.data_ptr(), mem_v.data_ptr(), tuple(mem_k.shape)))
        if self.num_persist_mem_tokens > 0:
            signature.extend((self.persistent_memory.data_ptr(), self.persistent_memory._version))
        return tuple(signature)

    def _decode_prefix(self, memory_kv, batch_size):
        """Return a decode prefix and a cheap identity used to detect refreshes."""
        prefix_k = []
        prefix_v = []
        signature = self._decode_prefix_signature(memory_kv)

        if memory_kv is not None:
            mem_k, mem_v = memory_kv
            if mem_k.dim() == 5:
                mem_k = rearrange(mem_k, 'b nc h m d -> b h (nc m) d')
                mem_v = rearrange(mem_v, 'b nc h m d -> b h (nc m) d')
            prefix_k.append(mem_k)
            prefix_v.append(mem_v)

        if self.num_persist_mem_tokens > 0:
            pmk, pmv = self.persistent_memory.unsqueeze(1).expand(-1, batch_size, -1, -1, -1).unbind(0)
            prefix_k.append(pmk)
            prefix_v.append(pmv)

        if not prefix_k:
            return None, None, signature
        if len(prefix_k) == 1:
            return prefix_k[0], prefix_v[0], signature
        return cat(prefix_k, dim=-2), cat(prefix_v, dim=-2), signature

    def _init_static_cache(self, ck, cv, max_new_tokens=None, absolute_pos=None, memory_kv=None):
        """Allocate one static [prefix | token cache] buffer for cached decoding."""
        B, H, S, D = ck.shape
        prefix_k, prefix_v, prefix_signature = self._decode_prefix(memory_kv, B)
        prefix_len = 0 if prefix_k is None else prefix_k.shape[-2]
        # Windowed decode never reads outside the current window, so the buffer
        # only has to be one window wide: slot = absolute_position % window. It
        # is then simultaneously fixed-width (what CUDA graph capture needs) and
        # narrow (what makes the attention cheap) -- without the ring those two
        # requirements pull against each other, and the mask ends up restricting
        # a full-length operand the kernel still has to process.
        #
        # The width is self.total_segment_len throughout: a config value, not a
        # constant. 512, 1024 or anything else follows segment_len.
        self._ring_decode = bool(self.windowed_decode and self.graph_safe_decode)

        if self._ring_decode:
            window = self.total_segment_len
            max_len = prefix_len + window
        else:
            # Only allocate what we actually need: prompt + expected generation
            if max_new_tokens is None:
                max_new_tokens = 4096  # reasonable default
            max_len = prefix_len + S + max_new_tokens + 64  # small safety margin

        self._static_k = torch.zeros(B, H, max_len, D, device=ck.device, dtype=ck.dtype)
        self._static_v = torch.zeros(B, H, max_len, D, device=cv.device, dtype=cv.dtype)
        if prefix_len:
            self._static_k[:, :, :prefix_len, :].copy_(prefix_k)
            self._static_v[:, :, :prefix_len, :].copy_(prefix_v)

        absolute_end = S if absolute_pos is None else int(absolute_pos)
        if self._ring_decode:
            # Seed only the current window's tail. Earlier tokens are exactly the
            # ones windowed decode may never look at again.
            filled = absolute_end % window
            if filled:
                self._static_k[:, :, prefix_len:prefix_len + filled, :].copy_(ck[:, :, -filled:, :])
                self._static_v[:, :, prefix_len:prefix_len + filled, :].copy_(cv[:, :, -filled:, :])
        else:
            self._static_k[:, :, prefix_len:prefix_len + S, :].copy_(ck)
            self._static_v[:, :, prefix_len:prefix_len + S, :].copy_(cv)
        self._static_prefix_len = prefix_len
        self._static_prefix_signature = prefix_signature
        # Keep the immutable source alive so CUDA's caching allocator cannot
        # reuse its address for a different recurrence prefix.
        self._static_prefix_source = memory_kv
        # _cache_pos counts the tokens this buffer stands for, which is not the
        # same as the length of the tensor we were seeded from. On the ring path
        # the two deliberately diverge: the seed is trimmed to one window because
        # that is all ring decode can ever read (see
        # TitanDecoderLayer._trim_decode_cache), while the counter still has to
        # track the real token count. Ring decode addresses slots as
        # `absolute % window` and never consults _cache_pos, so taking it from the
        # absolute position keeps it a meaningful counter instead of an artifact
        # of how much K/V the caller happened to hand over. Off the ring path the
        # seed IS the buffer contents and S remains correct.
        self._cache_pos = absolute_end if self._ring_decode else S
        self._absolute_cache_pos = absolute_end

        if self.graph_safe_decode:
            # Graph-safe decode skips _refresh_static_prefix, so a prefix that
            # can change identity mid-run would silently go stale. Persistent
            # memory is a Parameter and stable in inference; a recurrence prefix is
            # not, so refuse it rather than produce wrong output.
            if memory_kv is not None:
                raise RuntimeError(
                    "graph_safe_decode cannot be used with a memory_kv prefix: "
                    "the prefix must be fixed for the lifetime of the capture"
                )
            # Shape [1] so they can index_copy_ directly. This runs once, on the
            # first (eager) decode step, before any capture.
            device = self._static_k.device
            self._cache_pos_tensor = torch.tensor(
                [self._cache_pos], device=device, dtype=torch.long
            )
            self._absolute_cache_pos_tensor = torch.tensor(
                [self._absolute_cache_pos], device=device, dtype=torch.long
            )
            self._buffer_positions = torch.arange(max_len, device=device)
            if not torch._dynamo.is_compiling():
                for buffer in (
                    self._static_k,
                    self._static_v,
                    self._cache_pos_tensor,
                    self._absolute_cache_pos_tensor,
                ):
                    torch._dynamo.mark_static_address(buffer)

    def _refresh_static_prefix(self, memory_kv, batch_size):
        """Refresh a recurrence prefix without rebuilding it on every token."""
        signature = self._decode_prefix_signature(memory_kv)
        if signature == self._static_prefix_signature:
            return
        prefix_k, prefix_v, signature = self._decode_prefix(memory_kv, batch_size)

        new_prefix_len = 0 if prefix_k is None else prefix_k.shape[-2]
        old_prefix_len = self._static_prefix_len
        if new_prefix_len != old_prefix_len:
            old_k, old_v = self._static_k, self._static_v
            capacity = old_k.shape[-2] - old_prefix_len
            B, H, _, D = old_k.shape
            self._static_k = old_k.new_zeros(B, H, new_prefix_len + capacity, D)
            self._static_v = old_v.new_zeros(B, H, new_prefix_len + capacity, D)
            self._static_k[:, :, new_prefix_len:new_prefix_len + self._cache_pos].copy_(
                old_k[:, :, old_prefix_len:old_prefix_len + self._cache_pos]
            )
            self._static_v[:, :, new_prefix_len:new_prefix_len + self._cache_pos].copy_(
                old_v[:, :, old_prefix_len:old_prefix_len + self._cache_pos]
            )

        if new_prefix_len:
            self._static_k[:, :, :new_prefix_len].copy_(prefix_k)
            self._static_v[:, :, :new_prefix_len].copy_(prefix_v)
        self._static_prefix_len = new_prefix_len
        self._static_prefix_signature = signature
        self._static_prefix_source = memory_kv

    def _window_bounds_unsupported(self, prefix_len):
        """Windowed decode with a prefix would need a gather, not a slice."""
        if prefix_len:
            raise RuntimeError(
                "windowed_decode does not support a decode prefix "
                f"(persistent/recurrence memory occupies {prefix_len} positions). "
                "Prefill makes the prefix visible to every window, so the decode "
                "mask would have to select the prefix plus one window, which a "
                "contiguous slice cannot express."
            )

    def _window_start(self, prefix_len):
        """First buffer slot the current window may attend to (Python ints).

        Windows are defined on *absolute* positions, exactly as prefill's fold
        is: a query at absolute position p sees [W*(p//W), p]. The buffer is
        indexed by cache position, so shift by the constant offset between the
        two counters.
        """
        self._window_bounds_unsupported(prefix_len)
        window = self.total_segment_len
        current_absolute = self._absolute_cache_pos - 1
        window_start_absolute = (current_absolute // window) * window
        absolute_offset = self._absolute_cache_pos - self._cache_pos
        return max(prefix_len, prefix_len + window_start_absolute - absolute_offset)

    def _window_start_tensor(self, prefix_len):
        """Device-tensor form of ``_window_start`` for the graph-safe path."""
        self._window_bounds_unsupported(prefix_len)
        window = self.total_segment_len
        current_absolute = self._absolute_cache_pos_tensor - 1
        window_start_absolute = torch.div(
            current_absolute, window, rounding_mode = 'floor'
        ) * window
        absolute_offset = self._absolute_cache_pos_tensor - self._cache_pos_tensor
        return (window_start_absolute - absolute_offset + prefix_len).clamp(min = prefix_len)

    def _graph_safe_decode_attend(self, q, k, v, precomputed_qkv):
        """One decode step with nothing that varies living in a Python value.

        Differences from the default path, each forced by CUDA graph capture:

        * the write index and the RoPE position are device tensors incremented
          in place, so a replay advances instead of rewriting the capture-time
          slot;
        * the new K/V go in via ``index_copy_`` rather than integer indexing;
        * attention runs over the whole preallocated buffer under a mask
          instead of a slice that grows by one every token, so the operand
          shapes are identical on every step.

        The mask admits exactly ``[0, prefix_len + cache_pos)`` after the
        increment -- the same positions the default path selects with
        ``[:, :, :full_cache_pos, :]``.

        ``_refresh_static_prefix`` is deliberately not called here: it branches
        on ``data_ptr()`` identities, which cannot be captured. Graph-safe decode
        therefore requires a fixed prefix, checked once in ``_init_static_cache``.
        """
        prefix_len = self._static_prefix_len

        if not exists(precomputed_qkv):
            # apply_rotary_pos_emb_hf builds positions as `arange(n) + offset`,
            # which broadcasts a tensor offset without any host round trip.
            q, k = apply_rotary_pos_emb_hf(
                q, k, self._rope_inv_freq, seq_offset=self._absolute_cache_pos_tensor
            )

        if self._ring_decode:
            # slot = absolute_position % window. Wrapping is safe precisely
            # because windowed decode may never read the tokens being
            # overwritten: they belong to the previous window.
            window = self.total_segment_len
            slot = torch.remainder(self._absolute_cache_pos_tensor, window)
            write_index = slot + prefix_len
        else:
            write_index = self._cache_pos_tensor + prefix_len

        self._static_k.index_copy_(2, write_index, k)
        self._static_v.index_copy_(2, write_index, v)
        self._cache_pos_tensor += 1
        self._absolute_cache_pos_tensor += 1

        # True == attend. Broadcasts over [batch, heads, 1, buffer width].
        if self._ring_decode:
            # Everything written so far in this window, i.e. slots [0, slot],
            # plus the prefix.
            valid = self._buffer_positions <= write_index
        else:
            valid = self._buffer_positions < (self._cache_pos_tensor + prefix_len)
            if self.windowed_decode:
                valid = valid & (
                    self._buffer_positions >= self._window_start_tensor(prefix_len)
                )
        out = F.scaled_dot_product_attention(
            q,
            self._static_k,
            self._static_v,
            attn_mask=valid.view(1, 1, 1, -1),
            enable_gqa=self._num_kv_groups > 1,
        )

        # The full buffers: a tensor-bounded slice is not expressible, and the
        # caller only needs these to be non-None (backend selection) since
        # graph-safe decode skips the external HF cache mirror.
        return q, k, v, out, (self._static_k, self._static_v)

    def _fast_forward_inference(
        self,
        token,
        value_residual = None,
        output_gating = None,
        memory_kv = None,
        memory_values = None,
        memory_mix = None,
        precomputed_qkv = None,
        seq_offset = None,
    ):
        """Optimized single-token decode: inlined ops, no einops, no nn.Sequential dispatch.

        ``precomputed_qkv`` lets the caller supply q/k/v that are already
        projected *and* already rotated, so a decoder layer that needed the same
        tensors for its neural memory does not pay for a second projection and a
        second rotation here.
        """
        B = token.shape[0]
        H = self._num_heads
        H_kv = self._num_kv_heads
        D = self._dim_head

        # --- Fused QKV projection ---
        # The inference path packs the pretrained Q and packed-KV
        # matrices once, saving one GEMM dispatch per Titan layer and token.
        if exists(precomputed_qkv):
            q, k, v = precomputed_qkv
        elif self._fuse_qkv_decode and not False and not False:
            fused_weight, fused_bias = self._get_fused_qkv_decode_params()
            qkv = F.linear(token, fused_weight, fused_bias)
            q_width = H * D
            q, kv = qkv.split((q_width, 2 * H_kv * D), dim=-1)
        else:
            q = F.linear(token, self.to_q.weight, self.to_q.bias)
            kv = F.linear(token, self.to_kv.weight, self.to_kv.bias)

        if not exists(precomputed_qkv):
            k, v = kv.chunk(2, dim=-1)
            # Inline split_heads: [B, 1, H*D] → [B, H, 1, D]
            q = q.view(B, 1, H, D).transpose(1, 2)
            k = k.view(B, 1, H_kv, D).transpose(1, 2)
            v = v.view(B, 1, H_kv, D).transpose(1, 2)

        orig_v = v

        # --- Value residual ---
        # Same rule as _mix_value_residual, but with the Sequential dispatch
        # inlined (Linear -> permute -> sigmoid -> lerp): decode is latency
        # bound and runs this once per token per layer.
        if self._v_mix_weight is not None:
            mix = F.linear(token, self._v_mix_weight, self._v_mix_bias)  # [B, 1, H]
            mix = torch.sigmoid(mix).transpose(1, 2).unsqueeze(-1)       # [B, H, 1, 1]
            v = v.lerp(value_residual, mix)

        # --- RoPE + static cache (static cache is guaranteed initialized by caller) ---
        if self.graph_safe_decode:
            q, k, v, out, next_cache = self._graph_safe_decode_attend(
                q, k, v, precomputed_qkv
            )
        else:
            cache_len = self._cache_pos
            absolute_pos = self._absolute_cache_pos
            self._refresh_static_prefix(memory_kv, B)
            prefix_len = self._static_prefix_len

            if exists(precomputed_qkv):
                # Already rotated by the caller. The caller's offset must agree
                # with this module's own decode counter, otherwise the cache
                # would be written with keys rotated for the wrong position.
                if exists(seq_offset) and int(seq_offset) != absolute_pos:
                    raise RuntimeError(
                        f"pre-rotated decode q/k were rotated at position {int(seq_offset)} "
                        f"but the static cache is at position {absolute_pos}"
                    )
            elif absolute_pos + k.shape[-2] <= self._rope_cos_cached.shape[0]:
                q, k = apply_rotary_pos_emb_hf_cached(
                    q,
                    k,
                    self._rope_cos_cached,
                    self._rope_sin_cached,
                    seq_offset=absolute_pos,
                )
            else:
                # Preserve the previous unbounded behavior past table capacity.
                q, k = apply_rotary_pos_emb_hf(
                    q, k, self._rope_inv_freq, seq_offset=absolute_pos
                )

            physical_cache_pos = prefix_len + cache_len
            self._static_k[:, :, physical_cache_pos, :] = k.squeeze(-2)
            self._static_v[:, :, physical_cache_pos, :] = v.squeeze(-2)
            self._cache_pos = cache_len + 1
            self._absolute_cache_pos = absolute_pos + 1

            full_cache_pos = prefix_len + self._cache_pos
            attend_from = self._window_start(prefix_len) if self.windowed_decode else 0
            sk = self._static_k[:, :, attend_from:full_cache_pos, :]
            sv = self._static_v[:, :, attend_from:full_cache_pos, :]
            # Prefix positions are an internal implementation detail. HF's cache
            # must expose only autoregressive token K/V so its length is exact.
            next_cache = (
                self._static_k[:, :, prefix_len:full_cache_pos, :],
                self._static_v[:, :, prefix_len:full_cache_pos, :],
            )

            # --- SDPA ---
            # Match native HF SDPA: retain compact KV heads and let the fused
            # kernel perform grouped-query attention rather than repeat_kv.
            out = F.scaled_dot_product_attention(
                q,
                sk,
                sv,
                enable_gqa=self._num_kv_groups > 1,
            )

        out = self._blend_memory(
            out, *self._memory_to_heads(memory_values, memory_mix, B, 1)
        )

        # --- Merge heads + output projection (inline) ---
        out = out.transpose(1, 2).reshape(B, 1, H * D)

        out = F.linear(out, self.to_out.weight, self.to_out.bias)

        if output_gating is not None:
            out = out * output_gating

        # next_cache = (self._static_k[:, :, :self._cache_pos, :],
        #               self._static_v[:, :, :self._cache_pos, :])
        return out, AttnIntermediates(orig_v, next_cache)

    def forward_inference(
        self,
        token,
        cache,
        value_residual = None,
        output_gating = None,
        memory_kv = None,
        memory_values = None,
        memory_mix = None,
    ):
        """Original decode path — used as fallback when HF rope is not active."""
        batch = token.shape[0]

        if not self.pre_normed:
            token = self.norm(token)

        q = self.split_heads(self.to_q(token))
        kv = self.to_kv(token)
        k, v = kv.chunk(2, dim=-1)
        k, v = map(self.split_kv_heads, (k, v))

        # value residual

        orig_v = v

        v = self._mix_value_residual(token, v, value_residual)

        # RoPE + caching

        ck, cv = cache
        cache_len = ck.shape[-2]

        # non-HF rope path (original behaviour)
        k = cat((ck, k), dim = -2)
        v = cat((cv, v), dim = -2)
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        next_cache = (k, v)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = batch)

        # prepend neural memory kv + persistent memory

        prefix_k = [pmk]
        prefix_v = [pmv]
        if exists(memory_kv):
            mem_k, mem_v = memory_kv
            prefix_k.insert(0, mem_k)
            prefix_v.insert(0, mem_v)

        k = cat((*prefix_k, k), dim = -2)
        v = cat((*prefix_v, v), dim = -2)

        # scaled dot-product attention (single query token → no causal mask needed)

        out = F.scaled_dot_product_attention(
            q,
            repeat_kv(k, self._num_kv_groups),
            repeat_kv(v, self._num_kv_groups),
        )

        out = self._blend_memory(
            out, *self._memory_to_heads(memory_values, memory_mix, batch, 1)
        )

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        output_gating = None,
        cache = None,
        memory_kv = None,
        memory_values = None,
        memory_mix = None,
        return_cache = True,
        seq_offset = 0,
        precomputed_qkv = None,
        qkv_already_rotated = False,
    ):

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len = seq.shape[:2]

        # attention

        if exists(precomputed_qkv):
            q, k, v = precomputed_qkv
        else:
            if not self.pre_normed:
                seq = self.norm(seq)
            q = self.split_heads(self.to_q(seq))
            kv = self.to_kv(seq)
            k, v = kv.chunk(2, dim=-1)
            k, v = map(self.split_kv_heads, (k, v))

        # value residual

        orig_v = v

        v = self._mix_value_residual(seq, v, value_residual)

        # relative positions and caching (same order change as forward())

        if self._use_hf_rope:
            if not qkv_already_rotated:
                q, k = apply_rotary_pos_emb_hf(q, k, self._rope_inv_freq, seq_offset=seq_offset)
            next_cache = (k, v) if return_cache else None
        else:
            next_cache = (k, v) if return_cache else None
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = batch)

        # prepend neural memory kv + persistent memory

        num_mem_kv = memory_kv[0].shape[-2] if exists(memory_kv) else 0
        total_prefix = self.num_persist_mem_tokens + num_mem_kv

        prefix_k = [pmk]
        prefix_v = [pmv]
        if exists(memory_kv):
            mem_k, mem_v = memory_kv
            prefix_k.insert(0, mem_k)
            prefix_v.insert(0, mem_v)

        k = cat((*prefix_k, k), dim = -2)
        v = cat((*prefix_v, v), dim = -2)

        # prep flex attention

        if not exists(flex_attn_fn):
            block_mask = create_mac_block_mask(seq_len, self.total_segment_len, total_prefix, self.sliding)

            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # attention — expand KV heads for flex_attention (GQA via repeat)

        out = flex_attn_fn(q, repeat_kv(k, self._num_kv_groups), repeat_kv(v, self._num_kv_groups))

        out = self._blend_memory(
            out, *self._memory_to_heads(memory_values, memory_mix, batch, seq_len)
        )

        out = self.merge_heads(out)

        out = self.to_out(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

    def forward(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False,
        output_gating = None,
        cache = None,
        memory_kv = None,
        memory_tokens = None,
        memory_values = None,
        memory_mix = None,
        return_cache = True,
        seq_offset = 0,
        precomputed_qkv = None,
        qkv_already_rotated = False,
    ):
        if exists(memory_tokens):
            if exists(memory_kv):
                raise ValueError("pass either memory_tokens or memory_kv, not both")
            # Hidden-space virtual tokens use the exact same K/V projection as
            # real tokens.  Their query/output half is intentionally elided:
            # those positions carry no LM loss and only serve as a fixed prefix.
            memory_kv = self.project_memory_tokens(memory_tokens)

            # The flex path builds a global prefix mask and cannot express a
            # different hidden-token prefix for each folded window.
            disable_flex_attn = True

        backend = self._select_backend(seq, cache, disable_flex_attn)

        if backend == 'decode':
            if not self._use_hf_rope:
                return self.forward_inference(seq, cache, value_residual, output_gating = output_gating, memory_kv = memory_kv, memory_values = memory_values, memory_mix = memory_mix)

            # First decode step: init static cache via eager path, then use fast path
            if self._static_k is None:
                ck, cv = cache
                self._init_static_cache(
                    ck,
                    cv,
                    max_new_tokens=getattr(self, '_max_new_tokens', None),
                    absolute_pos=seq_offset if seq_offset > 0 else None,
                    memory_kv=memory_kv,
                )
            return self._fast_forward_inference(
                seq,
                value_residual = value_residual,
                output_gating = output_gating,
                memory_kv = memory_kv,
                memory_values = memory_values,
                memory_mix = memory_mix,
                precomputed_qkv = precomputed_qkv if qkv_already_rotated else None,
                seq_offset = seq_offset,
            )

        if backend == 'flex':
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating = output_gating, cache = cache, memory_kv = memory_kv, memory_values = memory_values, memory_mix = memory_mix, return_cache = return_cache, seq_offset = seq_offset, precomputed_qkv = precomputed_qkv, qkv_already_rotated = qkv_already_rotated)

        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len = seq.shape[:2]

        # auto pad to multiple

        seq, inverse_segment = pad_and_segment_with_inverse(seq, total_segment_len, fold_into_batch = False)

        memory_heads = None
        history_mix = None
        if exists(memory_values):
            assert memory_values.shape[:2] == (batch, seq_len)
            assert memory_values.shape[-1] == self._num_heads * self._dim_head
            if memory_values.shape[1] < seq.shape[1]:
                memory_values = pad_at_dim(
                    memory_values, (0, seq.shape[1] - memory_values.shape[1]), dim=-2
                )
            memory_heads = rearrange(
                memory_values,
                'b (w n) (h d) -> (b w) h n d',
                n=total_segment_len,
                h=self._num_heads,
                d=self._dim_head,
            )
            if exists(memory_mix):
                if memory_mix.shape[1] < seq.shape[1]:
                    memory_mix = pad_at_dim(
                        memory_mix,
                        (0, seq.shape[1] - memory_mix.shape[1]),
                        dim=1,
                    )
                history_mix = rearrange(
                    memory_mix,
                    'b (w n) h d -> (b w) h n d',
                    n=total_segment_len,
                )

        # attention

        if exists(precomputed_qkv):
            # Caller already provided pre-RoPE q, k, v in attention head layout.
            q, k, v = precomputed_qkv
            if q.shape[-2] != seq.shape[-2]:
                assert q.shape[-2] < seq.shape[-2], (
                    f"precomputed_qkv seq_len {q.shape[-2]} exceeds padded seq_len {seq.shape[-2]}"
                )
                qkv_padding = seq.shape[-2] - q.shape[-2]
                q, k, v = tuple(pad_at_dim(t, (0, qkv_padding), dim = -2) for t in (q, k, v))
        else:
            if not self.pre_normed:
                seq = self.norm(seq)
            q = self.split_heads(self.to_q(seq))
            kv = self.to_kv(seq)
            k, v = kv.chunk(2, dim=-1)
            k, v = map(self.split_kv_heads, (k, v))

        # value residual

        orig_v = v

        v = self._mix_value_residual(seq, v, value_residual)

        # relative positions and caching
        # HF rope: rotate before caching so cache stores pre-rotated keys
        # (avoids O(n^2) re-rotation of full cache during inference)

        if self._use_hf_rope:
            if not qkv_already_rotated:
                q, k = apply_rotary_pos_emb_hf(q, k, self._rope_inv_freq, seq_offset=seq_offset)
            next_cache = (inverse_segment(k), inverse_segment(v)) if return_cache else None
        else:
            next_cache = tuple(map(inverse_segment, (k, v))) if return_cache else None
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # fold — Q uses query heads, K/V use KV heads

        q = rearrange(q, 'b h (w n) d -> (b w) h n d', n = total_segment_len)
        k, v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_segment_len) for t in (k, v))

        # determine total prefix length (neural mem kv + persistent mem)

        num_mem_kv = memory_kv[0].shape[-2] if exists(memory_kv) else 0
        total_prefix = self.num_persist_mem_tokens + num_mem_kv

        # maybe sliding for cpu

        attend_kwargs = dict()

        if self.sliding:
            k, v = tuple(rearrange(t, '(b w) ... -> b w ...', b = batch) for t in (k, v))
            k, v = tuple(pad_at_dim(t, (1, 0), value = 0., dim = 1) for t in (k, v))
            k = cat((k[:, :-1], k[:, 1:]), dim = -2)
            v = cat((v[:, :-1], v[:, 1:]), dim = -2)
            k, v = tuple(rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))

            # take care of masking

            idx = torch.arange(seq.shape[-2], device = seq.device)
            q_idx = rearrange(idx, '(w n) -> w n', n = total_segment_len)
            k_idx = pad_at_dim(q_idx, (1, 0), dim = 0, value = -1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim = -1)

            q_idx = rearrange(q_idx, 'w i -> w i 1')
            k_idx = rearrange(k_idx, 'w j -> w 1 j')

            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (total_prefix, 0), value = True)

            sliding_mask = repeat(sliding_mask, 'w i j -> (b w) 1 i j', b = batch)
            attend_kwargs.update(mask = sliding_mask)

        # take care of persistent memory key / values

        pmk, pmv = repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        # prepend neural memory kv + persistent memory

        prefix_k = [pmk]
        prefix_v = [pmv]
        if exists(memory_kv):
            mem_k, mem_v = memory_kv
            num_windows = q.shape[0] // batch
            if mem_k.dim() == 5:
                # Per-chunk memory: (b, n_chunks, h, m, d)
                # c windows per chunk: window i -> chunk min(i // c, n_chunks - 1)
                n_chunks = mem_k.shape[1]
                windows_per_chunk = ceil(num_windows / n_chunks)
                chunk_idx = (torch.arange(num_windows, device=mem_k.device) // windows_per_chunk).clamp(max=n_chunks - 1)
                mem_k = mem_k[:, chunk_idx]  # (b, num_windows, h, m, d)
                mem_v = mem_v[:, chunk_idx]
                mem_k = rearrange(mem_k, 'b w h m d -> (b w) h m d')
                mem_v = rearrange(mem_v, 'b w h m d -> (b w) h m d')
            else:
                # Global memory: (b, h, m, d) -> broadcast to all windows
                mem_k = repeat(mem_k, 'b h m d -> (b w) h m d', w = num_windows)
                mem_v = repeat(mem_v, 'b h m d -> (b w) h m d', w = num_windows)
            prefix_k.insert(0, mem_k)
            prefix_v.insert(0, mem_v)

        k = cat((*prefix_k, k), dim = -2)
        v = cat((*prefix_v, v), dim = -2)

        # attention — expand KV heads for Attend (does not support GQA natively)

        out, _ = self.attend(q, repeat_kv(k, self._num_kv_groups), repeat_kv(v, self._num_kv_groups), **attend_kwargs)

        # already folded into windows above, so blend directly
        out = self._blend_memory(out, memory_heads, history_mix)


        out = self.merge_heads(out)

        out = self.to_out(out)

        out = rearrange(out, '(b w) n d -> b (w n) d', b = batch)

        out = inverse_segment(out)

        if exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)

# MAC transformer

class MemoryAsContextTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        segment_len,
        neural_memory_segment_len = None,
        neural_mem_gate_attn_output = False,
        neural_memory_add_value_residual = False,
        num_longterm_mem_tokens = 0,
        num_persist_mem_tokens = 0,
        neural_memory_batch_size = None,
        neural_memory_qkv_receives_diff_views = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_residual_streams = 4,
        neural_memory_model: Module | None = None,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        use_flex_attn = False,
        sliding_window_attn = False,
        neural_mem_weight_residual = False,
        token_emb: Module | None = None,
    ):
        super().__init__()

        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)

        self.token_emb = token_emb

        # absolute positions

        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(dim = dim, num_axial_dims = 2)

        # long term mem tokens

        self.segment_len = segment_len

        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        has_longterm_mems = num_longterm_mem_tokens > 0

        self.longterm_mems = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim) * 0.02)

        # maybe sliding window attn

        self.sliding_window_attn = sliding_window_attn
        self.attn_window_size = segment_len + num_longterm_mem_tokens

        # hyper connection

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        self.neural_memory_segment_len = default(neural_memory_segment_len, num_longterm_mem_tokens + segment_len)

        layers = tuple(range(1, depth + 1))

        neural_memory_layers = default(neural_memory_layers, layers)

        # weight residual related

        self.neural_mem_weight_residual = neural_mem_weight_residual
        is_first_neural_mem = True

        # mem, attn, and feedforward layers

        for layer in layers:
            is_first = layer == 1

            # attention and feedforward

            attn = SegmentedAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                segment_len = segment_len,
                use_flex_attn = use_flex_attn,
                accept_value_residual = not is_first,
                num_longterm_mem_tokens = num_longterm_mem_tokens,
                num_persist_mem_tokens = num_persist_mem_tokens,
                sliding = sliding_window_attn
            )

            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None

            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual = not neural_mem_gate_attn_output)

                if not is_first and neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1 # for each layer, have memory input select from attn inp, attn out, ff inp, and ff out - plus one for the current point in the residual stream (memory input)

                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(dim),
                        nn.Linear(dim, 3 * num_layer_choices),
                        Rearrange('... (views layers) -> views ... layers', views = 3),
                        nn.Softmax(dim = -1)
                    )

                mem = NeuralMemory(
                    dim = dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = deepcopy(neural_memory_model),
                    qkv_receives_diff_views = True,
                    accept_weight_residual = neural_mem_weight_residual and not is_first_neural_mem,
                    **neural_memory_kwargs
                )

                is_first_neural_mem = False

            ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([
                mem_hyper_conn,
                init_hyper_conn(),
                init_hyper_conn(),
                mem_qkv_layer_selector,
                mem,
                attn,
                ff,
            ]))

        self.norm = nn.RMSNorm(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

        # whether to gate the attention output with the retrieved memories

        self.gate_attn_output = neural_mem_gate_attn_output

        # zero for maybe aux loss + device

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # flex attn related

        assert not (use_flex_attn and not exists(flex_attention)), 'you need to be on the latest pytorch with a cuda device available'
        self.use_flex_attn = use_flex_attn

        self.num_persist_mem_tokens = num_persist_mem_tokens

    def seq_index_is_longterm(
        self,
        seq_index
    ):
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(
        self,
        seq_len
    ):
        assert seq_len > 0

        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    def forward(
        self,
        x,
        return_loss = False,
        return_loss_breakdown = False,
        disable_flex_attn = False,
        cache = None,
        return_cache = False,
        factorized_pos_emb = None
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # math

        batch, seq_len, neural_mem_segment_len, segment_len, num_longterm_mem_tokens, attn_window_size = *x.shape, self.neural_memory_segment_len, self.segment_len, self.num_longterm_mem_tokens, self.attn_window_size

        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)

        # token embedding

        x = self.token_emb(x)

        # intersperse longterm memory

        x, inverse_segment = pad_and_segment_with_inverse(x, segment_len, inverse_remove_pad = False)

        mems = repeat(self.longterm_mems, 'n d -> b n d', b = x.shape[0])
        x, inverse_pack_mems = pack_with_inverse((x, mems), 'b * d')

        x = inverse_segment(x)

        # splice out unneeded tokens from padding for longterm mems

        x = x[:, :seq_len_with_mem]

        # apply axial positional embedding
        # so intra and inter segment can be more easily discerned by the network

        pos_emb = self.axial_pos_emb.forward_with_seq_len(seq_len_with_mem, (neural_mem_segment_len,), factorized = factorized_pos_emb)

        x = x + pos_emb

        # prep flex attention

        use_flex_attn = x.is_cuda and self.use_flex_attn and not disable_flex_attn

        flex_attn_fn = None

        if use_flex_attn:
            block_mask = create_mac_block_mask(seq_len_with_mem, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        # kv caching

        is_inferencing = exists(cache)

        if not exists(cache):
            cache = (seq_len_with_mem - 1, None, None)

        inference_seq_index, kv_caches, neural_mem_caches = cache

        kv_caches = iter(default(kv_caches, []))
        neural_mem_caches = iter(default(neural_mem_caches, []))

        next_kv_caches = []
        next_neural_mem_caches = []

        # value residual

        value_residual = None

        # neural mem weight residual

        mem_weight_residual = None

        # layers for the neural mem to select the qkv inputs from

        mem_input_layers = []

        # when inferencing, only do one token at a time

        if is_inferencing:
            ind = inference_seq_index
            x = x[:, ind:(ind + 1)]

        # expand and reduce streams for hyper connections

        x = self.expand_streams(x)

        for mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff in self.layers:

            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None

            # maybe neural memory

            if exists(mem):

                mem_input, add_residual = mem_hyper_conn(x)

                if not exists(mem_qkv_layer_selector):
                    qkv_mem_input = stack((mem_input, mem_input, mem_input))
                else:
                    layers_to_choose_from = stack((mem_input, *mem_input_layers))

                    # let the current `mem_input` select the 3 layers for qkv

                    selected = mem_qkv_layer_selector(mem_input)

                    qkv_mem_input = einsum(layers_to_choose_from, selected, 'l b n d, v b n l -> v b n d')

                retrieved, next_neural_mem_cache = mem.forward_sequence(
                    qkv_mem_input,
                    state = next(neural_mem_caches, None),
                    prev_weights = mem_weight_residual
                )

                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates

                if self.gate_attn_output:
                    attn_out_gates = retrieved.sigmoid()
                else:
                    x = add_residual(retrieved)

            # attention

            attn_in, add_residual = attn_hyper_conn(x)

            mem_input_layers.append(attn_in)

            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual = value_residual,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn_fn,
                output_gating = attn_out_gates,
                cache = next(kv_caches, None)
            )

            mem_input_layers.append(attn_out)

            value_residual = default(value_residual, values)

            x = add_residual(attn_out)

            # caches

            next_kv_caches.append(next_kv_cache)
            next_neural_mem_caches.append(next_neural_mem_cache)

            # feedforward

            ff_in, add_ff_residual = ff_hyper_conn(x)

            mem_input_layers.append(ff_in)

            ff_out = ff(ff_in)

            mem_input_layers.append(ff_out)

            x = add_ff_residual(ff_out)

        # taking care of cache first
        # for early return when processing long term mem tokens during inference

        if return_cache:
            next_kv_caches = stack([stack(kv_cache) for kv_cache in next_kv_caches])

            # handle kv cache length depending on local attention type

            next_kv_caches = next_kv_caches[..., -attn_window_size:, :]

            kv_cache_length = next_kv_caches.shape[-2]

            if not self.sliding_window_attn and divisible_by(kv_cache_length, attn_window_size):
                next_kv_caches = next_kv_caches[..., 0:0, :]

            next_cache = (
                inference_seq_index + 1,
                next_kv_caches,
                next_neural_mem_caches
            )

            is_longterm_mem = self.seq_index_is_longterm(inference_seq_index)

            if is_inferencing and is_longterm_mem:
                return None, next_cache

        # hyper connection reducing of streams

        x = self.reduce_streams(x)

        # excise out the memories

        if not is_inferencing:

            x, inverse_segment = pad_and_segment_with_inverse(x, attn_window_size, inverse_remove_pad = False)

            x, _ = inverse_pack_mems(x)

            x = inverse_segment(x)

            x = x[:, :seq_len]

        # to logits

        x = self.norm(x)

        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, next_cache

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
