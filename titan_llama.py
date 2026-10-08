import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import Optional, Tuple, Union
import math
from dataclasses import dataclass, fields, replace

from transformers.cache_utils import DynamicCache

from titans_pytorch import (
    NeuralMemory,
    MemoryMLP,
)
from titans_pytorch.mac_transformer import SegmentedAttention

try:
    from transformers.cache_utils import DynamicLayer as _HFDynamicLayer
except ImportError:  # transformers < 4.57 exposes no per-layer cache classes
    _HFDynamicLayer = None
import transformers.integrations.sdpa_attention as _hf_sdpa

_hf_use_gqa_in_sdpa = _hf_sdpa.use_gqa_in_sdpa


def _use_gqa_in_sdpa_with_fused_kernel(attention_mask, key):
    # FP32 has no fused SDPA kernel for enable_gqa=True, so it falls back to the
    # math path and materializes [B, H, N, N] scores. Repeating K/V instead lets
    # the memory-efficient kernel run. Other dtypes keep Transformers' choice.
    return key.dtype != torch.float32 and _hf_use_gqa_in_sdpa(attention_mask, key)


_hf_sdpa.use_gqa_in_sdpa = _use_gqa_in_sdpa_with_fused_kernel


# Register one paired attention/mask implementation.  Assigning this backend
# to the HF backbone makes both training and backbone.generate() use the same
# window; no dense user-provided 4D mask or generation-only patch is needed.


# Snapshotted at import so the hot path can be guarded by a module-level
# constant. torch.compile constant-folds a branch on this and never traces the
# f-string behind it; reading os.environ per call instead forced dynamo to
# specialize forward_core on self.layer_idx, recompiling once per Titan layer.
# The cost is that setting TITAN_CUDA_DEBUG_SYNC after import has no effect.
_DEBUG_CUDA_SYNC = os.environ.get("TITAN_CUDA_DEBUG_SYNC") == "1"


def _debug_cuda_sync(tag: str) -> None:
    """Synchronize only when TITAN_CUDA_DEBUG_SYNC=1 to localize async CUDA faults."""
    if not _DEBUG_CUDA_SYNC:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"CUDA failed while synchronizing after: {tag}") from exc


# ---------------------------------------------------------------------------
# Config (unchanged)
# ---------------------------------------------------------------------------

# Fields populated from a base LlamaConfig; everything else is Titan-specific.
# Used to split overrides in from_llama_config / from_pretrained_llama / from_pretrained
# so the field list lives in exactly one place (the dataclass below).
_LLAMA_DERIVED_FIELDS = frozenset({
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
    "rms_norm_eps", "rope_theta",
})


if _HFDynamicLayer is not None:

    class TitanWindowCacheLayer(_HFDynamicLayer):
        """External KV-cache layer for a Titan layer: keeps one segment window.

        A Titan layer never reads this mirror -- ``_get_cache`` returns its own
        ``state.attention_kv`` first -- and its segmented attention only ever
        attends inside the current window. The mirror exists solely because
        Hugging Face derives sequence length and causal-mask geometry from the
        cache, so it must keep *reporting* the full length while storing only
        what a window needs.

        Without this, every Titan layer costs a full-length KV tensor in the
        external cache on top of whatever the layer itself retains.
        """

        is_sliding = False

        def __init__(self, window: int):
            super().__init__()
            self.window = int(window)
            self.logical_length = 0

        def update(self, key_states, value_states, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(key_states)
            self.logical_length += key_states.shape[-2]
            keys = torch.cat([self.keys, key_states], dim=-2)
            values = torch.cat([self.values, value_states], dim=-2)
            if keys.shape[-2] > self.window:
                # .contiguous() so the trimmed slice stops referencing the
                # full-length storage; a view would keep every byte alive.
                keys = keys[..., -self.window:, :].contiguous()
                values = values[..., -self.window:, :].contiguous()
            self.keys, self.values = keys, values
            return self.keys, self.values

        def get_seq_length(self) -> int:
            """Full logical length, not the number of tokens actually stored."""
            return self.logical_length

        def crop(self, max_length: int) -> None:
            """Rolling the sequence back is not expressible on a windowed mirror.

            This layer holds tokens ``[L - window, L)``.  Cropping to ``M`` needs
            ``[M - window, M)``, which is only still in the buffer when
            ``M >= L`` -- i.e. when the crop is a no-op.  Anything shorter would
            return the wrong tokens while continuing to report the old length,
            so it raises instead of corrupting silently.  ``crop`` is reached by
            assisted/speculative decoding and some rollback paths; ordinary
            greedy and sampling generation never calls it.
            """
            if max_length < 0:
                max_length = self.logical_length + max_length
            if max_length >= self.logical_length:
                return
            raise RuntimeError(
                "TitanWindowCacheLayer cannot crop to "
                f"{max_length} from logical length {self.logical_length}: it "
                f"retains only the last {self.window} tokens. Pass an ordinary "
                "DynamicCache for generation paths that roll the cache back "
                "(assisted/speculative decoding)."
            )


else:  # pragma: no cover - older transformers
    TitanWindowCacheLayer = None


@dataclass
class TitanLLaMAConfig:
    """Configuration for Titan-LLaMA model with segmented attention and neural memory."""

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # Titan-specific parameters
    segment_len: int = 512
    neural_memory_layers: Tuple[int, ...] = (8, 16, 24)
    # Layers converted to segmented attention WITHOUT a neural memory. Used for
    # the "same architecture, no NMM" ablation. None keeps the historical
    # behaviour in which every converted layer also carries an NMM.
    segmented_attention_layers: Optional[Tuple[int, ...]] = None
    neural_memory_segment_len: int = 512
    neural_memory_batch_size: int = 8
    neural_memory_depth: int = 2
    neural_memory_expansion_factor: float = 1.
    neural_memory_activation: str = "gelu"
    neural_memory_inner_learning_rate: float = 0.0025  # max adaptive fast-weight update step
    detach_inner_grads: bool = True   # if False, allow outer loss to backprop through inner loop
    use_flex_attn: bool = True
    use_flash_attn: bool = False
    # Initialize every MemoryMLP matrix to identity instead of fitting the last
    # one to the pretrained K->V map. M(K) ~ K, so with lambda ~ 0 the layer
    # starts as the unmodified pretrained attention block.
    memory_identity_init: bool = False
    memory_qk_rope: bool = True
    history_blend: bool = False
    history_mix_init: float = 0.1
    # Pretrained backbone support
    use_pretrained_backbone: bool = False
    base_model_name_or_path: Optional[str] = None
    freeze_backbone: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.neural_memory_segment_len != self.segment_len:
            raise ValueError(
                "neural_memory_segment_len must equal the segmented-attention recurrence "
                f"length ({self.segment_len}), got {self.neural_memory_segment_len}"
            )
        if self.neural_memory_expansion_factor != 1.:
            raise ValueError(
                "projected per-head memory requires neural_memory_expansion_factor=1.0"
            )
        if self.neural_memory_depth < 2:
            raise ValueError("projected per-head memory requires neural_memory_depth >= 2")
        if self.history_blend:
            if not 0.0 < self.history_mix_init < 1.0:
                raise ValueError(
                    "history_mix_init must lie strictly between 0 and 1, got "
                    f"{self.history_mix_init}"
                )


    @classmethod
    def titan_field_names(cls) -> set:
        """Names of all Titan-specific fields (everything not derived from LlamaConfig)."""
        return {f.name for f in fields(cls)} - _LLAMA_DERIVED_FIELDS


    def get_titan_layer_indices(self) -> set:
        """Return the set of layer indices that need Titan treatment (segmented attn or neural memory)."""
        return set(self.neural_memory_layers) | set(self.segmented_attention_layers or ())

    @classmethod
    def from_llama_config(cls, llama_config, **overrides):
        titan_specific_keys = cls.titan_field_names()
        titan_kwargs = {k: v for k, v in overrides.items() if k in titan_specific_keys}
        return cls(
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_hidden_layers=llama_config.num_hidden_layers,
            num_attention_heads=llama_config.num_attention_heads,
            num_key_value_heads=getattr(llama_config, "num_key_value_heads", llama_config.num_attention_heads),
            max_position_embeddings=getattr(llama_config, "max_position_embeddings", 2048),
            rms_norm_eps=getattr(llama_config, "rms_norm_eps", 1e-6),
            rope_theta=getattr(llama_config, "rope_theta", 10000.0),
            **titan_kwargs,
        )


@dataclass
class TitanLayerState:
    memory: object = None
    attention_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # Absolute index of the first token in the current call. Carried in state
    # so forward_core needs no positional arguments beyond it.
    position: int = 0


@dataclass
class TitanForwardState:
    value_residual: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TitanDecoderLayer: HF-compatible drop-in replacement
# ---------------------------------------------------------------------------

class TitanDecoderLayer(nn.Module):
    """
    Drop-in replacement for HF LlamaDecoderLayer at specific layer indices.

    Reuses the original HF layer's MLP and RMSNorms. Builds a SegmentedAttention path with a NeuralMemory sidecar.

    forward() matches the HF LlamaDecoderLayer signature: accepts the same args,
    returns only hidden_states (a single tensor). KV cache is stored in HF's
    DynamicCache (when available) so that get_seq_length() works correctly,
    with a fallback to internal state for standalone usage.

    NOTE: We inherit from nn.Module (not LlamaDecoderLayer) to avoid creating
    unwanted default submodules. If output_hidden_states is needed (e.g. for
    activation recording), see _patch_hidden_state_recording().
    """

    def __init__(
        self,
        hf_layer: nn.Module,
        titan_config: TitanLLaMAConfig,
        layer_idx: int,
        shared_state: TitanForwardState,
        rope_inv_freq: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = titan_config
        self._titan_shared_state = shared_state
        # Reuse HF layer's MLP and norms (same nn.Module objects, no weight copy needed)
        self.mlp = hf_layer.mlp
        self.input_layernorm = hf_layer.input_layernorm
        self.post_attention_layernorm = hf_layer.post_attention_layernorm
        hf_attn = hf_layer.self_attn


        # Build SegmentedAttention
        hidden_size = titan_config.hidden_size
        num_heads = titan_config.num_attention_heads
        head_dim = hidden_size // num_heads

        # Only accept value residuals if a *previous* Titan layer exists to produce them
        titan_indices = titan_config.get_titan_layer_indices()
        is_first_titan_layer = not titan_indices or layer_idx == min(titan_indices)

        num_kv_heads = titan_config.num_key_value_heads

        self.segmented_attn = SegmentedAttention(
            dim=hidden_size,
            segment_len=titan_config.segment_len,
            num_persist_mem_tokens=0,
            num_longterm_mem_tokens=0,
            dim_head=head_dim,
            heads=num_heads,
            num_kv_heads=num_kv_heads,
            sliding=False,
            accept_value_residual=not is_first_titan_layer,
            attend_kwargs=dict(flash=True) if titan_config.use_flash_attn else dict(),
            use_flex_attn=titan_config.use_flex_attn,
            pre_normed=True,
            rope_theta=titan_config.rope_theta,
            rope_freqs=rope_inv_freq,
        )

        # Copy Q/K/V/O weights from HF attention (native GQA — no weight expansion)
        with torch.no_grad():
            self.segmented_attn.to_q.weight.copy_(hf_attn.q_proj.weight)
            kv_w = torch.cat([hf_attn.k_proj.weight, hf_attn.v_proj.weight], dim=0)
            self.segmented_attn.to_kv.weight.copy_(kv_w)
            self.segmented_attn.to_out.weight.copy_(hf_attn.o_proj.weight)

        # Initialize value residual mix to zero (sigmoid(-10) ≈ 0 → no mixing at start)
        v_mix = self.segmented_attn.to_learned_v_mix
        if v_mix is not None:
            nn.init.zeros_(v_mix[0].weight)
            nn.init.constant_(v_mix[0].bias, -10.0)


        self.segmented_attn._setup_fast_path()

        # A layer listed only in segmented_attention_layers keeps the identical
        # segmented attention but runs no neural memory: the "same architecture,
        # no NMM" ablation.  The module is still constructed so that state dicts
        # and attribute accesses stay uniform; forward() skips it.
        self.has_neural_memory = layer_idx in set(titan_config.neural_memory_layers)

        neural_memory_model = MemoryMLP(
            dim=head_dim,
            depth=titan_config.neural_memory_depth,
            expansion_factor=titan_config.neural_memory_expansion_factor,
            activation=titan_config.neural_memory_activation,
        )
        self.neural_memory = NeuralMemory(
            dim=num_kv_heads * head_dim,
            chunk_size=titan_config.neural_memory_segment_len,
            batch_size=titan_config.neural_memory_batch_size,
            dim_head=head_dim,
            heads=num_kv_heads,
            retrieve_heads=num_heads,
            model=neural_memory_model,
            qkv_receives_diff_views=True,
            accept_weight_residual=False,
            max_grad_norm=1.0,
            default_step_transform_max_lr=titan_config.neural_memory_inner_learning_rate,
            detach_inner_grads=titan_config.detach_inner_grads,
            bypass_projections=True,
            mem_model_norm_add_residual=False,
            heterogeneous_qkv=True,
            history_context_dim=(hidden_size if titan_config.history_blend else None),
            history_mix_init=titan_config.history_mix_init,
        )
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._num_heads = num_heads
        k_weight, v_weight = self.segmented_attn.to_kv.weight.chunk(2, dim=0)
        self.neural_memory.initialize_kv_memory(
            k_weight,
            v_weight,
            identity=titan_config.memory_identity_init,
        )

        # Per-layer recurrent state (reset between independent sequences).
        self.state = TitanLayerState()
        # Host-side mirror of state.position, used only by graph-safe decode:
        # there the position is a device tensor and must not be carried across
        # steps as a graph output. Advanced by the (host-known) token count.
        self._host_position = 0
        # Counts decode tokens only, to drive graph-safe store-chunk flushes.
        self._gs_token_count = 0
        # Escape hatch so a test can A/B the trim against the untrimmed path.
        self._trim_decode_cache_enabled = True
        self._titan_config = titan_config


    # -- KV cache helpers (DynamicCache integration) -----------------------

    def _get_cache(self, past_key_values):
        """Retrieve this layer's KV cache across old and new HF cache APIs."""
        # The custom segmented-attention cache is the source of truth.  In
        # transformers >= 4.57, LlamaModel no longer forwards ``use_cache`` to
        # decoder layers, so this internal state is also what lets the layer
        # recognize subsequent one-token decode calls.
        if self.state.attention_kv is not None:
            return self.state.attention_kv

        # transformers <= 4.56 DynamicCache representation.
        if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
            if self.layer_idx < len(past_key_values.key_cache):
                ck = past_key_values.key_cache[self.layer_idx]
                if isinstance(ck, torch.Tensor) and ck.dim() > 1:
                    return (ck, past_key_values.value_cache[self.layer_idx])

        # transformers >= 4.57 Cache interface.  Individual cache layers are
        # exposed through ``cache[layer_idx]`` rather than ``cache.key_cache``.
        if past_key_values is not None and hasattr(past_key_values, '__getitem__'):
            try:
                ck, cv = past_key_values[self.layer_idx]
            except (IndexError, KeyError, TypeError):
                pass
            else:
                if isinstance(ck, torch.Tensor) and ck.dim() > 1 and ck.numel() > 0:
                    return (ck, cv)

        return None

    def _trim_decode_cache(self, cache_kv):
        """Drop prefill K/V that the decode path can never read.

        Ring decode seeds its buffer from the TAIL of this cache
        (``ck[..., -filled:, :]`` with ``filled = absolute_pos % window``) and
        thereafter attends only inside the current window, so everything older
        than one window is dead the instant prefill ends. It stayed alive only
        because the layer parked the whole prefill K/V in
        ``state.attention_kv``. Measured at 128K: 7.15 GiB resident after
        prefill against a full-attention baseline's 4.00.

        Truncating is safe here because ``_init_static_cache`` takes the
        absolute position from ``seq_offset``, not from this tensor's length,
        so shortening it cannot shift the position counters.

        Deliberately restricted to the ring path. Without ring, ``_window_start``
        derives its buffer offset from ``_absolute_cache_pos - _cache_pos`` and
        ``_cache_pos`` is set from this tensor's length, so shortening it there
        would silently move every window bound. Training and grad-enabled calls
        are excluded outright: the backward pass needs what the forward built.
        """
        attn = self.segmented_attn
        if (
            cache_kv is None
            or not self._trim_decode_cache_enabled
            or not (attn.windowed_decode and attn.graph_safe_decode)
            or self.training
            or torch.is_grad_enabled()
        ):
            return cache_kv

        k, v = cache_kv
        window = attn.total_segment_len
        if k.shape[-2] <= window:
            return cache_kv
        # .contiguous() so the slices stop referencing the full-length storage;
        # a view would keep every byte alive and save nothing.
        return (k[..., -window:, :].contiguous(), v[..., -window:, :].contiguous())

    def _trim_memory_updates(self, memory_state):
        """Keep the last chunk of the neural memory's update timeline, not all of it.

        ``forward_sequence`` returns one cumulative update per store chunk, so the
        timeline grows with the prefill: at 128K over 16 layers it is 0.75 GiB of
        state whose only surviving reader takes ``updates[:, -1:]`` -- the next
        single-token retrieve reads exactly that, and the store half ignores
        ``state.updates`` entirely.

        Trimmed here rather than inside NeuralMemory because the timeline has
        other, legitimate consumers of its full length: MemoryAsContextTransformer
        feeds it to the next layer as ``prev_weights``, and the retrieval
        tooling hooks ``NeuralMemory.forward`` to capture every chunk. Both read
        the value NeuralMemory returns; this trims only what THIS layer carries
        into the next step.

        Inference only, for the same reason as _trim_decode_cache: backward needs
        the entries this drops.
        """
        if (
            memory_state is None
            or getattr(memory_state, "updates", None) is None
            or self.training
            or torch.is_grad_enabled()
        ):
            return memory_state
        updates = memory_state.updates
        if not isinstance(updates, (tuple, list)) or not len(updates):
            return memory_state
        if all(not torch.is_tensor(u) or u.shape[1] <= 1 for u in updates):
            return memory_state
        return memory_state._replace(
            updates=tuple(one_update[:, -1:].contiguous() for one_update in updates)
        )

    def _set_cache(self, past_key_values, cache_kv, cache_position=None):
        """Return the K/V this layer should retain, and mirror the external cache.

        The retained value is *returned* rather than written to ``self.state``:
        ``forward`` overwrites ``self.state`` wholesale immediately afterwards,
        so anything assigned here would be discarded and the trim below would be
        dead code holding the entire prefill K/V alive.
        """
        # Trim only what THIS layer retains. The external mirror is left at full
        # length on purpose: HF derives sequence length and causal-mask geometry
        # from the cache, so shortening it there would corrupt the whole model
        # whenever a Titan layer is the one queried for length.
        retained = self._trim_decode_cache(cache_kv)
        self._mirror_external_cache(past_key_values, cache_kv, cache_position)
        return retained

    def _mirror_external_cache(self, past_key_values, cache_kv, cache_position):
        """Bring the external HF cache in line with this layer's full K/V."""
        # Under graph-safe decode `cache_kv` is the whole preallocated buffer,
        # not the valid region, so the suffix arithmetic below would mirror the
        # wrong slots. Titan reads its own cache first (see _get_cache), and the
        # StaticCache this mode requires reports mask sizes from its capacity
        # rather than its contents, so the external mirror is not needed. Prefill
        # still mirrors: the static buffer only exists from the first decode on.
        if self.segmented_attn.graph_safe_decode and self.segmented_attn._static_k is not None:
            return

        # transformers <= 4.56 DynamicCache representation.
        if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
            k, v = cache_kv
            while len(past_key_values.key_cache) <= self.layer_idx:
                past_key_values.key_cache.append(torch.empty(0))
                past_key_values.value_cache.append(torch.empty(0))
            past_key_values.key_cache[self.layer_idx] = k
            past_key_values.value_cache[self.layer_idx] = v
            return

        # transformers >= 4.57.  SegmentedAttention returns the full cache, but
        # Cache.update() expects only newly appended states.  Determine how much
        # the external layer already holds and append exactly the suffix.
        if past_key_values is not None and hasattr(past_key_values, 'update'):
            k, v = cache_kv
            full_len = k.shape[-2]

            # ``cache_position`` holds one entry per token in this forward, so
            # its length is exactly the suffix the external cache is missing.
            # Prefer it over asking the cache how long it is: StaticCache's
            # get_seq_length() is a device-side reduction over the whole buffer,
            # and int() on it forces a GPU->CPU sync every layer, every token.
            if cache_position is not None:
                new_len = int(cache_position.shape[0])
                if new_len > full_len:
                    raise RuntimeError(
                        f"layer {self.layer_idx} was given {new_len} cache positions "
                        f"but its Titan cache holds only {full_len} tokens"
                    )
            else:
                external_len = 0
                layers = getattr(past_key_values, 'layers', None)
                if layers is not None and self.layer_idx < len(layers):
                    cache_layer = layers[self.layer_idx]
                    if hasattr(cache_layer, 'get_seq_length'):
                        external_len = int(cache_layer.get_seq_length())

                if external_len > full_len:
                    raise RuntimeError(
                        f"external cache for layer {self.layer_idx} is longer than Titan cache "
                        f"({external_len} > {full_len})"
                    )
                new_len = full_len - external_len

            if new_len == 0:
                return

            new_k = k[..., full_len - new_len:, :]
            new_v = v[..., full_len - new_len:, :]
            cache_kwargs = None
            if cache_position is not None:
                cache_kwargs = {'cache_position': cache_position}
            past_key_values.update(
                new_k,
                new_v,
                self.layer_idx,
                cache_kwargs,
            )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # transformers >= 4.57 consumes ``use_cache`` in LlamaModel.forward and
        # does not pass it to decoder layers.  A non-null cache object therefore
        # is the authoritative signal that this layer must produce/update K/V.
        cache_enabled = bool(use_cache or past_key_values is not None)

        # path. In particular, do not run/update the NMM or touch Titan's
        # private KV cache.  We do retain the native projected V as shared
        # bookkeeping: an immediately following active layer has a learned
        # value-residual mixer and therefore requires the previous layer's V,
        # regardless of whether that previous layer is native or segmented.

        # Everything below is state/cache adaptation around one pure call.
        seq_offset = self.state.position
        if position_ids is not None and position_ids.numel() > 0:
            if self.segmented_attn.graph_safe_decode:
                # .item() is a hard GPU->CPU sync, once per Titan layer per
                # token, and is illegal inside a captured CUDA graph. Keep the
                # position on device; apply_rotary_pos_emb_hf broadcasts a
                # tensor offset, and the graph-safe decode path never needs its
                # host value.
                seq_offset = position_ids.reshape(-1)[:1]
            else:
                seq_offset = int(position_ids.reshape(-1)[0].item())

        state = TitanLayerState(
            memory=self.state.memory,
            attention_kv=self._get_cache(past_key_values),
            position=seq_offset,
        )

        if self.has_neural_memory and self.neural_memory.graph_safe_decode and hidden_states.shape[1] == 1:
            # Decide here, outside the compiled region, whether this token
            # completes a store chunk. The neural memory reads it as a bool and
            # dynamo specializes into two graphs; maintaining the counter inside
            # the region instead makes dynamo guard on its exact value and
            # recompile once per slot.
            self._gs_token_count += 1
            self.neural_memory._gs_will_flush = (
                self._gs_token_count % self.neural_memory.store_chunk_size == 0
            )

        hidden_states, next_state, value_residual = self.forward_core(
            hidden_states,
            state,
            self._titan_shared_state.value_residual,
        )

        if self.segmented_attn.graph_safe_decode:
            # next_state.position is `state.position + n`, i.e. a tensor produced
            # *inside* the compiled region. A CUDA graph reuses its output
            # buffers on every replay, so carrying that tensor to the next step
            # reads memory this step already overwrote. The advance is a host
            # value anyway -- it is just the token count -- so recompute it here,
            # outside the graph, with no sync.
            next_state = replace(
                next_state,
                position=self._host_position + hidden_states.shape[1],
            )
            self._host_position = next_state.position

        next_state = replace(
            next_state, memory=self._trim_memory_updates(next_state.memory)
        )

        if cache_enabled:
            # Retain what _set_cache hands back, not what forward_core produced:
            # on the ring path those differ by the whole prefill (one window kept
            # instead of every token), and `self.state = next_state` below would
            # otherwise put the untrimmed tensor straight back.
            next_state = replace(
                next_state,
                attention_kv=self._set_cache(
                    past_key_values,
                    next_state.attention_kv,
                    cache_position=cache_position,
                ),
            )
        else:
            # Do not retain K/V the caller did not ask to keep.
            next_state = replace(next_state, attention_kv=None)

        self.state = next_state
        self._titan_shared_state.value_residual = value_residual
        return hidden_states

    def forward_core(
        self,
        x: torch.Tensor,
        state: TitanLayerState,
        value_residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, TitanLayerState, Optional[torch.Tensor]]:
        """The Titan decoder block as a pure function of its inputs and state.

        Holds no HF concerns: no cache objects, no mutation of module attributes. ``forward`` adapts this to the HF
        decoder-layer signature, which keeps the production path free of the
        adapter's branching.
        """
        residual = x
        x = self.input_layernorm(x)

        q, k, v, memory_q, memory_k, memory_v, rotated = self._project_and_rotate(
            x, state.position
        )


        if self.has_neural_memory:
            memory_values, memory_mix, memory_state = self.neural_memory(
                memory_q,
                memory_k,
                memory_v,
                state=state.memory,
                context=x,
                detach_mem_state=True,
                seq_offset=state.position,
            )
        else:
            # SegmentedAttention._memory_to_heads short-circuits on None, so the
            # layer is exactly its segmented attention with no memory read/write.
            memory_values = memory_mix = None
            memory_state = state.memory
        if _DEBUG_CUDA_SYNC:
            _debug_cuda_sync(f"layer {self.layer_idx} neural memory")

        if self.segmented_attn.to_learned_v_mix is None:
            value_residual = None

        attn_output, attn_intermediates = self.segmented_attn(
            x,
            value_residual=value_residual,
            cache=state.attention_kv,
            memory_values=memory_values,
            memory_mix=memory_mix,
            return_cache=True,
            seq_offset=state.position,
            precomputed_qkv=(q, k, v),
            qkv_already_rotated=rotated,
        )
        if _DEBUG_CUDA_SYNC:
            _debug_cuda_sync(f"layer {self.layer_idx} segmented attention")

        x = residual + attn_output
        x = x + self.mlp(self.post_attention_layernorm(x))

        next_state = TitanLayerState(
            memory=memory_state,
            attention_kv=attn_intermediates.cached_key_values,
            position=state.position + residual.shape[1],
        )
        return x, next_state, attn_intermediates.value_residual


    def _project_and_rotate(self, normed_hidden_states: torch.Tensor, position: int):
        """Project Q/K/V once and rotate once, for both attention and memory.

        Attention and the neural memory previously rotated independent copies of
        Q/K. Under the HF rope convention they now share a single rotation, and
        ``rotated`` tells attention to skip its internal one. When
        ``memory_qk_rope`` is disabled the memory still receives unrotated Q/K.

        The rotary_embedding_torch path is left untouched: it applies a
        cached-key rotation inside attention that has no offset-only equivalent,
        so there attention keeps rotating its own copy.
        """
        attention = self.segmented_attn
        q_raw, k_raw, v = attention.project_qkv(normed_hidden_states)

        rotated = attention._use_hf_rope
        if rotated:
            q, k = attention.apply_rope(q_raw, k_raw, seq_offset=position)
        else:
            q, k = q_raw, k_raw

        if not self._titan_config.memory_qk_rope:
            return q, k, v, q_raw, k_raw, v, rotated

        if rotated:
            return q, k, v, q, k, v, rotated

        memory_q = attention.rotary_emb.rotate_queries_or_keys(q_raw, offset=position)
        memory_k = attention.rotary_emb.rotate_queries_or_keys(k_raw, offset=position)
        return q, k, v, memory_q, memory_k, v, rotated

# ---------------------------------------------------------------------------
# TitanLLaMAForCausalLM: wrapper around HF backbone
# ---------------------------------------------------------------------------

class TitanLLaMAForCausalLM(nn.Module):
    """
    Titan-LLaMA model for causal language modeling.

    Wraps an HF LlamaForCausalLM backbone. Only layers that need Titan features
    (segmented attention, neural memory) are replaced with TitanDecoderLayer.
    All other layers remain native HF for maximum speed.
    """

    def __init__(self, config: TitanLLaMAConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.backbone = None  # Set by from_pretrained_llama / from_pretrained
        self._titan_shared_state = TitanForwardState()
        self.padding_idx = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        lm_logit_chunk_size: int = 0,
    ):
        # Reset transient Titan state
        self._titan_shared_state.value_residual = None
        if use_cache is None:
            use_cache = False

        # Collect hidden states via hooks since transformers >=4.57
        # no longer does this inside LlamaModel.forward().
        all_hidden_states = []
        hooks = []
        if output_hidden_states:
            def _embed_hook(module, input, output):
                all_hidden_states.append(output)
            hooks.append(self.backbone.model.embed_tokens.register_forward_hook(_embed_hook))

            for layer in self.backbone.model.layers:
                def _layer_hook(module, input, output, _states=all_hidden_states):
                    h = output[0] if isinstance(output, tuple) else output
                    _states.append(h)
                hooks.append(layer.register_forward_hook(_layer_hook))

        # `logits_to_keep=0` (the HF default) computes logits for EVERY
        # position: at a 128K prompt that is 131072 x 128256 x 2 bytes = 33.6 GB
        # of which inference uses one row. Generation only needs the last
        # position, so callers pass 1. Training must leave it at 0 -- the
        # shifted cross-entropy below reads every position.
        if logits_to_keep != 0 and labels is not None:
            raise ValueError(
                "logits_to_keep must be 0 when labels are supplied: the shifted "
                "cross-entropy loss reads logits at every position"
            )

        past_key_values = self._default_kv_cache(past_key_values, use_cache)

        _debug_cuda_sync("before HF backbone")
        try:
            if labels is not None and lm_logit_chunk_size > 0:
                # Avoid materializing [B, N, vocab] logits and converting the
                # entire tensor to fp32. At 16K with Llama-3.2's vocabulary the
                # fp32 transient is about 7.8 GiB even for B=1. Checkpoint each
                # token chunk so its logits are freed until backward recompute.
                outputs = self.backbone.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=False,
                    cache_position=cache_position,
                    return_dict=True,
                )
            else:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=False,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                )
            _debug_cuda_sync("after HF backbone")

            for h in hooks:
                h.remove()

            hidden_states = tuple(all_hidden_states) if all_hidden_states else None

        finally:
            # The value-residual is layer-to-layer scratch, but it is a
            # full-length V: 0.25 GiB after a 128K prefill, held from the last
            # layer until the next forward clears it at the top. Drop it as soon
            # as the stack is done rather than carrying it across the gap.
            self._titan_shared_state.value_residual = None

        chunked_lm = labels is not None and lm_logit_chunk_size > 0
        logits = None if chunked_lm else outputs.logits
        loss = None if chunked_lm else outputs.loss
        ppl = None
        accuracy = None

        if chunked_lm:
            final_hidden = outputs.last_hidden_state
            if final_hidden.shape[-2] != labels.shape[-1]:
                raise ValueError(
                    "block-native backbone must preserve one hidden state per label: "
                    f"hidden={final_hidden.shape[-2]}, labels={labels.shape[-1]}"
                )
            shifted_labels = labels[..., 1:].contiguous()
            valid = shifted_labels != -100
            if attention_mask is not None:
                valid = valid & attention_mask[..., 1:].to(dtype=torch.bool)
            elif self.padding_idx is not None:
                valid = valid & (shifted_labels != self.padding_idx)
            metric_labels = shifted_labels.masked_fill(~valid, -100)
            if not bool(valid.any()):
                supervised = (labels != -100).nonzero(as_tuple=False)
                raise RuntimeError(
                    "block-native loss has no supervised token after alignment: "
                    f"labels={tuple(labels.shape)}, hidden={tuple(final_hidden.shape)}, "
                    f"attention_mask={None if attention_mask is None else tuple(attention_mask.shape)}, "
                    f"supervised_positions={supervised[:8].detach().cpu().tolist()}"
                )
            valid_count = valid.sum().clamp_min(1)
            loss_sum = final_hidden.new_zeros((), dtype=torch.float32)
            correct_sum = final_hidden.new_zeros((), dtype=torch.float32)
            chunk_size = max(int(lm_logit_chunk_size), 1)
            num_chunks = -(-shifted_labels.shape[-1] // chunk_size)
            # Transfer only one boolean per projection chunk to the host.  A
            # Python ``bool(valid[..., start:end].any())`` inside the loop
            # would synchronize once per chunk (1,024 times for 512K / 512).
            padded_valid = F.pad(
                valid,
                (0, num_chunks * chunk_size - shifted_labels.shape[-1]),
            )
            active_chunks = padded_valid.reshape(
                valid.shape[0], num_chunks, chunk_size
            ).any(dim=(0, 2)).cpu().tolist()

            def _chunked_projection_loss(hidden_chunk, label_chunk):
                # Keep the large vocabulary projection in the model dtype
                # (bf16 for these runs). The scalar loss is accumulated in
                # fp32 outside this function; no full fp32 logits are formed.
                chunk_logits = self.backbone.lm_head(hidden_chunk)
                flat_logits = chunk_logits.reshape(-1, chunk_logits.shape[-1])
                flat_labels = label_chunk.reshape(-1)
                chunk_loss = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction="sum",
                )
                with torch.no_grad():
                    chunk_correct = (
                        (flat_logits.argmax(dim=-1) == flat_labels)
                        & (flat_labels != -100)
                    ).sum().to(dtype=torch.float32)
                return chunk_loss, chunk_correct

            for chunk_index, start in enumerate(
                range(0, shifted_labels.shape[-1], chunk_size)
            ):
                end = min(start + chunk_size, shifted_labels.shape[-1])
                # Long-context continuation runs often score only a suffix so
                # that the prefix can serve purely as memory warmup.  Do not
                # project an all-ignored chunk through the 128K-vocabulary LM
                # head: it contributes exactly zero to both the loss and its
                # gradient, while at 512K it is otherwise the dominant cost.
                if not active_chunks[chunk_index]:
                    continue
                chunk_loss, chunk_correct = checkpoint(
                    _chunked_projection_loss,
                    final_hidden[:, start:end, :],
                    metric_labels[:, start:end],
                    use_reentrant=False,
                )
                loss_sum = loss_sum + chunk_loss
                correct_sum = correct_sum + chunk_correct
            loss = loss_sum / valid_count.to(dtype=loss_sum.dtype)
            ppl = torch.exp(loss)
            accuracy = correct_sum / valid_count.to(dtype=correct_sum.dtype)
        elif labels is not None:
            ppl = torch.exp(loss) if loss is not None else None

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Match the causal-LM loss mask: cached datasets use -100 for both
            # padding and intentionally unscored prompt tokens.  The tokenizer's
            # pad_token_id must not replace this check because -100 labels would
            # then be counted as incorrect predictions.
            mask = shift_labels != -100
            if attention_mask is not None:
                mask = mask & attention_mask[..., 1:].to(dtype=torch.bool)
            elif self.padding_idx is not None:
                mask = mask & (shift_labels != self.padding_idx)

            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels) & mask
            total_valid = mask.sum().float()
            accuracy = correct.sum().float() / total_valid if total_valid > 0 else torch.tensor(0.0, device=logits.device)

        result = {
            'loss': loss,
            'ppl': ppl,
            'logits': logits,
            'past_key_values': outputs.past_key_values,
            'attentions': outputs.attentions,
        }
        if output_hidden_states:
            result['hidden_states'] = hidden_states
        if accuracy is not None:
            result['correct'] = accuracy

        return result

    def generate(self, *args, **kwargs):
        """Generate with a fresh NMM state and cached, one-token decoding."""
        if kwargs.get("use_cache", True) is False:
            raise ValueError(
                "Stateful Titan generation requires use_cache=True so the prompt is "
                "stored once and only newly decoded tokens update the NMM."
            )
        self.reset_memory_states()
        self._titan_shared_state.value_residual = None
        kwargs["use_cache"] = True
        cache = self._default_kv_cache(
            kwargs.get("past_key_values"), True, kwargs.get("generation_config")
        )
        if cache is not None:
            kwargs["past_key_values"] = cache
        return self.backbone.generate(*args, **kwargs)

    @torch.no_grad()
    def generate_with_titan_memory(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        reset_memory: bool = True,
        use_cache: bool = True,
        **extra_gen_kwargs,
    ):
        """Stateful generation: store the prompt once, then update from new tokens."""
        if not use_cache:
            raise ValueError(
                "Stateful Titan generation requires use_cache=True so the prompt is "
                "not repeatedly written into neural memory."
            )
        if reset_memory:
            self.reset_memory_states()
        # Tell each segmented attention layer how big the cache needs to be
        for layer in self.backbone.model.layers:
            if isinstance(layer, TitanDecoderLayer) and hasattr(layer, 'segmented_attn'):
                layer.segmented_attn._max_new_tokens = max_new_tokens
        self._titan_shared_state.value_residual = None

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )
        if temperature > 0 and do_sample:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        if not do_sample:
            gen_kwargs['temperature'] = None
        gen_kwargs.update(extra_gen_kwargs)

        cache = self._default_kv_cache(
            gen_kwargs.get("past_key_values"), True, gen_kwargs.get("generation_config")
        )
        if cache is not None:
            gen_kwargs["past_key_values"] = cache

        return self.backbone.generate(input_ids=input_ids, **gen_kwargs)

    def _default_kv_cache(self, past_key_values, use_cache, generation_config=None):
        """Supply the non-duplicating cache unless the caller asked for another.

        Converted layers otherwise mirror their full-length KV into the external
        cache on top of what the layer already holds, which costs a second copy
        per Titan layer and made a converted model hold MORE KV than the plain
        full-attention backbone it replaced.

        Two things are deliberately left alone: a cache the caller passed in
        (their intent wins), and any explicit ``cache_implementation`` -- the
        compiled graph-safe decode path requires a StaticCache, and it already
        skips the external mirror on its own.
        """
        if past_key_values is not None or not use_cache:
            return past_key_values
        config = generation_config if generation_config is not None else getattr(
            self, "generation_config", None
        )
        if getattr(config, "cache_implementation", None):
            return past_key_values
        return self.build_kv_cache()

    def build_kv_cache(self):
        """Build an external KV cache that does not duplicate Titan-layer KV.

        Converted layers get a window-sized mirror; every other layer keeps the
        ordinary growing cache.  Pass the result as ``past_key_values`` (forward
        or ``generate``).  ``forward`` builds one automatically when caching is
        requested and none was supplied.
        """
        from transformers.cache_utils import Cache as HFCache
        from transformers.cache_utils import DynamicLayer

        if TitanWindowCacheLayer is None:
            return None

        layers = []
        titan_indices = self.config.get_titan_layer_indices()
        for layer_index, layer in enumerate(self.backbone.model.layers):
            attention = getattr(layer, "segmented_attn", None)
            if isinstance(layer, TitanDecoderLayer) and attention is not None:
                layers.append(TitanWindowCacheLayer(attention.total_segment_len))
            else:
                layers.append(DynamicLayer())
        return HFCache(layers=layers)


    def reset_memory_states(self):
        """Reset all Titan layer state (neural memory + KV caches)."""
        for layer in self.backbone.model.layers:
            if isinstance(layer, TitanDecoderLayer):
                layer.state = TitanLayerState()
                layer._host_position = 0
                layer._gs_token_count = 0
                layer.segmented_attn.reset_static_cache()
                layer.neural_memory.reset_graph_safe_store()
        self._titan_shared_state.value_residual = None

    # -- Offline NMM (memory build + query-only read) ------------------------

    def freeze_backbone(self):
        """Freeze all pretrained weights. Keep Titan-specific params trainable."""
        nm_total = 0

        print("\n[freeze_backbone] ***** BEGIN *****")

        for name, param in self.named_parameters():
            # NeuralMemory keeps an unreplicated module as the structural
            # template passed to torch.func.functional_call.  The actual
            # learned initial fast weights live in memory_model_parameters
            # (replicated once per memory head), so optimizing this template
            # only inflates the reported parameter count and optimizer input.
            if (
                ".neural_memory.memory_model." in name
                and ".neural_memory.memory_model_parameters." not in name
            ):
                param.requires_grad = False
                continue
            # Persistent-attention tokens are not part of Odyssey training.
            # In the common zero-token configuration this is a zero-sized
            # structural placeholder; leaving it trainable can make DDP wait
            # for a gradient that cannot exist (especially for output-only
            # auxiliary losses computed from hooks).
            if "persistent_memory" in name:
                param.requires_grad = False
                continue
            if "to_learned_v_mix" in name:
                param.requires_grad = True
                continue
            if "neural_memory" in name:
                param.requires_grad = True
                nm_total += param.numel()
                continue
            param.requires_grad = False

        print(f"[freeze_backbone] NM trainable:     {nm_total:,}")
        print("[freeze_backbone] ***** END *****\n")

    def freeze_for_inference(self):
        """Freeze all parameters including neural memory. Safe because the TTT inner
        loop uses functional_call with explicit weight dicts, not module.parameters()."""
        total_freed = 0
        for param in self.parameters():
            if param.requires_grad:
                total_freed += param.numel()
                param.requires_grad = False
        print(f"[freeze_for_inference] Disabled grad for {total_freed:,} params")

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.backbone.prepare_inputs_for_generation(*args, **kwargs)

    @classmethod
    def from_pretrained_llama(
        cls,
        base_model_name_or_path: str,
        titan_config: Optional[TitanLLaMAConfig] = None,
        freeze_backbone: bool = True,
        dtype: Optional[torch.dtype] = None,
        device_map = None,
        **from_pretrained_kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoConfig

        # 1) Build Titan config
        base_cfg = AutoConfig.from_pretrained(base_model_name_or_path, **from_pretrained_kwargs)

        titan_kwargs = {}
        if titan_config is not None:
            # Copy every Titan-specific field from the source config. The three
            # backbone fields below are passed explicitly to from_llama_config,
            # so exclude them here to avoid duplicate keyword arguments.
            copy_keys = TitanLLaMAConfig.titan_field_names() - {
                'use_pretrained_backbone', 'base_model_name_or_path', 'freeze_backbone',
            }
            titan_kwargs = {attr: getattr(titan_config, attr) for attr in copy_keys}

        titan_cfg = TitanLLaMAConfig.from_llama_config(
            base_cfg,
            use_pretrained_backbone=True,
            base_model_name_or_path=base_model_name_or_path,
            freeze_backbone=freeze_backbone,
            **titan_kwargs,
        )

        # 2) Load HF backbone
        extra_model_kwargs = {}
        if titan_config is not None and titan_config.use_flash_attn:
            extra_model_kwargs["attn_implementation"] = "sdpa"

        model_kwargs = {}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            **model_kwargs,
            **extra_model_kwargs,
            **from_pretrained_kwargs,
        )
        if titan_config is not None and titan_config.use_flash_attn:
            native_backend = getattr(backbone.config, "_attn_implementation", None)
            if native_backend != "sdpa":
                raise RuntimeError(
                    "--use_flash_attn requested the PyTorch SDPA/FlashAttention path "
                    f"for native Llama layers, but Transformers selected {native_backend!r}"
                )
            print(
                "[attention] Native non-Titan Llama layers use PyTorch SDPA "
                "(FlashAttention kernel on eligible CUDA/BF16 inputs)."
            )

        # 3) Extract RoPE frequencies from backbone
        rope_inv_freq = None
        if hasattr(backbone.model, 'rotary_emb') and hasattr(backbone.model.rotary_emb, 'inv_freq'):
            rope_inv_freq = backbone.model.rotary_emb.inv_freq.float().clone()

        # 4) Create wrapper
        model = cls(titan_cfg)
        model.backbone = backbone
        model.padding_idx = getattr(base_cfg, 'pad_token_id', None)

        # 5) Replace Titan layers
        titan_indices = titan_cfg.get_titan_layer_indices()
        print(f"[from_pretrained_llama] Replacing layers {sorted(titan_indices)} with TitanDecoderLayer")

        for idx in sorted(titan_indices):
            if idx >= len(backbone.model.layers):
                print(f"[warn] Skipping layer {idx} (model only has {len(backbone.model.layers)} layers)")
                continue
            original_layer = backbone.model.layers[idx]
            titan_layer = TitanDecoderLayer(
                hf_layer=original_layer,
                titan_config=titan_cfg,
                layer_idx=idx,
                shared_state=model._titan_shared_state,
                rope_inv_freq=rope_inv_freq,
            )
            # Move to same device/dtype as the layer being replaced.
            layer_param = next(original_layer.parameters())
            titan_layer = titan_layer.to(device=layer_param.device, dtype=layer_param.dtype)
            backbone.model.layers[idx] = titan_layer


        # vanilla-Llama ablations with no TitanDecoderLayer replacements.
        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        base_model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B",
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        strict: bool = False,
        config_overrides: Optional[dict] = None,
    ):
        """Load a TitanLLaMA model from a saved checkpoint."""
        dtype = dtype or torch.bfloat16

        # Titan checkpoints are trusted local artifacts and may use legacy tar
        # serialization, unsupported by PyTorch >=2.6's weights_only default.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]
        train_cfg = ckpt.get("config", {}) or {}

        # Load-time fallbacks for keys an old checkpoint may not have saved, where
        # the desired fallback differs from the field's constructor default.
        checkpoint_load_defaults = {
        }

        # Pull every Titan-specific field from the saved config, falling back to
        # the field default (or the override above) when a key is absent. Backbone
        # identity fields are set explicitly below, not read from the checkpoint.
        field_defaults = {f.name: f.default for f in fields(TitanLLaMAConfig)}
        legacy_config_aliases = {
            "memory_identity_init": "oddysey_q_identity_init",
            "memory_qk_rope": "neural_memory_qk_rope",
            "history_blend": "odyssey_history_blend",
            "history_mix_init": "odyssey_history_lambda_init",
        }
        titan_kwargs = {}
        for name in TitanLLaMAConfig.titan_field_names() - {
            "use_pretrained_backbone", "base_model_name_or_path", "freeze_backbone",
        }:
            default = checkpoint_load_defaults.get(name, field_defaults[name])
            legacy_name = legacy_config_aliases.get(name)
            titan_kwargs[name] = train_cfg.get(
                name,
                train_cfg.get(legacy_name, default) if legacy_name else default,
            )

        # Layer-index fields are serialized as lists; restore them as tuples.
        for name in ("neural_memory_layers",):
            if isinstance(titan_kwargs.get(name), list):
                titan_kwargs[name] = tuple(titan_kwargs[name])

        if config_overrides:
            unknown_overrides = set(config_overrides) - set(titan_kwargs)
            if unknown_overrides:
                raise ValueError(f"Unknown Titan config overrides: {sorted(unknown_overrides)}")
            titan_kwargs.update(config_overrides)

        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained(base_model_name_or_path)

        titan_cfg = TitanLLaMAConfig.from_llama_config(
            base_cfg,
            use_pretrained_backbone=True,
            base_model_name_or_path=base_model_name_or_path,
            freeze_backbone=True,
            **titan_kwargs,
        )

        # First, build the hybrid model with HF backbone + Titan layer replacements
        model = cls.from_pretrained_llama(
            base_model_name_or_path=base_model_name_or_path,
            titan_config=titan_cfg,
            freeze_backbone=True,
            dtype=dtype,
            device_map={"": device} if device is not None else None,
        )

        # Remap old-style state dict keys to new backbone-based keys
        remapped_state_dict = _remap_checkpoint_keys(state_dict, model)

        load_info = model.load_state_dict(remapped_state_dict, strict=strict)
        if load_info.missing_keys:
            print(f"[from_pretrained] Missing keys: {load_info.missing_keys}")
        if load_info.unexpected_keys:
            print(f"[from_pretrained] Unexpected keys: {load_info.unexpected_keys}")

        # layers that was active when they were saved. Model construction makes
        # every Titan path active by default, so restore the architectural stage
        # explicitly after loading the trained projections.
        return model


def _remap_checkpoint_keys(old_state_dict: dict, model: nn.Module) -> dict:
    """
    Remap state dict keys from old format (model.layers.X.*) to new format
    (backbone.model.layers.X.*).

    Old Titan checkpoints saved keys like:
      model.layers.0.self_attn.segmented_attn.to_qkv.weight
      model.layers.0.mlp.gate_proj.weight
      model.embed_tokens.weight
      model.norm.weight
      lm_head.weight

    New format uses:
      backbone.model.layers.0.segmented_attn.to_qkv.weight  (for Titan layers)
      backbone.model.layers.0.mlp.gate_proj.weight
      backbone.model.embed_tokens.weight
      backbone.model.norm.weight
      backbone.lm_head.weight
    """
    new_state_dict = {}
    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    for old_key, value in old_state_dict.items():
        # Try direct backbone prefix mapping
        if old_key.startswith("model."):
            new_key = "backbone." + old_key
        elif old_key.startswith("lm_head."):
            new_key = "backbone." + old_key
        else:
            new_key = old_key

        # Remap self_attn.segmented_attn.* -> segmented_attn.* for Titan layers
        new_key = new_key.replace(".self_attn.segmented_attn.", ".segmented_attn.")
        new_key = new_key.replace(
            ".odyssey_history_lambda", ".neural_memory.history_mix"
        )
        new_key = new_key.replace(
            ".odyssey_history_gate_proj.", ".neural_memory.history_mix_proj."
        )

        if new_key in model_keys:
            target = model_state[new_key]
            # Legacy Odyssey checkpoints learned one history-mix scalar per
            # query head. The current representation keeps one value per head
            # dimension. Broadcasting the scalar parameter and each projection
            # row is function-preserving: every dimension receives exactly the
            # same logit that the old implementation broadcast at runtime.
            if (
                new_key.endswith(".neural_memory.history_mix")
                and value.ndim == 1
                and target.ndim == 2
                and value.shape[0] == target.shape[0]
            ):
                value = value[:, None].expand_as(target).clone()
                print(f"[from_pretrained] Expanded legacy scalar history mix: {new_key}")
            elif (
                new_key.endswith(".neural_memory.history_mix_proj.weight")
                and value.ndim == 2
                and target.ndim == 2
                and target.shape[0] % value.shape[0] == 0
                and value.shape[1] == target.shape[1]
            ):
                repeats = target.shape[0] // value.shape[0]
                value = value.repeat_interleave(repeats, dim=0)
                print(f"[from_pretrained] Expanded legacy scalar history projection: {new_key}")
            new_state_dict[new_key] = value
        elif old_key in model_keys:
            new_state_dict[old_key] = value
        else:
            # Try without any prefix change
            new_state_dict[old_key] = value

    return new_state_dict
