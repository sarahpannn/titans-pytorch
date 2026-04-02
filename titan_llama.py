import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Callable, Sequence
import math
from functools import partial

from titans_pytorch import (
    MemoryAsContextTransformer,
    NeuralMemory,
    MemoryMLP
)
from titans_pytorch.mac_transformer import SegmentedAttention, create_mac_block_mask
try:
    from torch.nn.attention.flex_attention import flex_attention
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    flex_attention = None


# ---------------------------------------------------------------------------
# Config (unchanged)
# ---------------------------------------------------------------------------

class TitanLLaMAConfig:
    """Configuration for Titan-LLaMA model with segmented attention and neural memory."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        # Titan-specific parameters
        segment_len: int = 512,
        num_persist_mem_tokens: int = 4,
        num_longterm_mem_tokens: int = 4,
        neural_memory_layers: Tuple[int, ...] = (8, 16, 24),
        neural_memory_segment_len: int = 16,
        neural_memory_batch_size: int = 8,
        neural_memory_depth: int = 2,
        use_flex_attn: bool = True,
        use_flash_attn: bool = False,
        sliding_window_attn: bool = True,
        neural_mem_gate_attn_output: bool = True,
        neural_mem_weight_residual: bool = True,
        neural_mem_qkv_receives_diff_view: bool = True,
        num_neural_mem_kv_tokens: int = 4,
        zero_init_mem_to_kv: bool = True,
        use_value_residual: bool = True,
        segmented_attention_layers: Optional[Tuple[int, ...]] = None,  # None = all layers segmented
        # Pretrained backbone support
        use_pretrained_backbone: bool = False,
        base_model_name_or_path: Optional[str] = None,
        freeze_backbone: bool = True,
        # LoRA parameters for attention adaptation
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_layers_after_memory: int = 1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.neural_memory_layers = neural_memory_layers
        self.neural_memory_segment_len = neural_memory_segment_len
        self.neural_memory_batch_size = neural_memory_batch_size
        self.neural_memory_depth = neural_memory_depth
        self.use_flex_attn = use_flex_attn
        self.use_flash_attn = use_flash_attn
        self.sliding_window_attn = sliding_window_attn
        self.neural_mem_gate_attn_output = neural_mem_gate_attn_output
        self.neural_mem_weight_residual = neural_mem_weight_residual
        self.neural_mem_qkv_receives_diff_view = neural_mem_qkv_receives_diff_view
        self.num_neural_mem_kv_tokens = num_neural_mem_kv_tokens
        self.zero_init_mem_to_kv = zero_init_mem_to_kv
        self.use_value_residual = use_value_residual
        self.segmented_attention_layers = segmented_attention_layers
        self.use_pretrained_backbone = use_pretrained_backbone
        self.base_model_name_or_path = base_model_name_or_path
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_layers_after_memory = lora_layers_after_memory

    def get_lora_layer_indices(self) -> set:
        if not self.use_lora:
            return set()
        if not self.neural_memory_layers:
            return set(range(self.num_hidden_layers))
        lora_layers = set()
        for mem_layer in self.neural_memory_layers:
            lora_layers.add(mem_layer)
            for offset in range(1, self.lora_layers_after_memory + 1):
                if mem_layer + offset < self.num_hidden_layers:
                    lora_layers.add(mem_layer + offset)
        return lora_layers

    def get_titan_layer_indices(self) -> set:
        """Return the set of layer indices that need Titan treatment (segmented attn or neural memory)."""
        indices = set()
        if self.segmented_attention_layers is not None:
            indices.update(self.segmented_attention_layers)
        else:
            # All layers segmented
            indices.update(range(self.num_hidden_layers))
        indices.update(self.neural_memory_layers)
        return indices

    @classmethod
    def from_llama_config(cls, llama_config, **overrides):
        titan_specific_keys = {
            'segment_len', 'num_persist_mem_tokens', 'num_longterm_mem_tokens',
            'neural_memory_layers', 'neural_memory_segment_len', 'neural_memory_batch_size',
            'neural_memory_depth', 'use_flex_attn', 'use_flash_attn', 'sliding_window_attn',
            'neural_mem_gate_attn_output', 'neural_mem_weight_residual',
            'neural_mem_qkv_receives_diff_view', 'num_neural_mem_kv_tokens',
            'zero_init_mem_to_kv', 'use_value_residual', 'segmented_attention_layers',
            'use_pretrained_backbone', 'base_model_name_or_path', 'freeze_backbone',
            'use_lora', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_layers_after_memory',
        }
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


# ---------------------------------------------------------------------------
# LoRA (unchanged)
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.base_linear = base_linear
        self.lora = LoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank, alpha=alpha, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + self.lora(x)


# ---------------------------------------------------------------------------
# Helper: repeat KV weights for GQA expansion
# ---------------------------------------------------------------------------

def _repeat_kv_weights(weight: torch.Tensor, repeat_factor: int, head_dim: int = 64) -> torch.Tensor:
    if repeat_factor == 1:
        return weight
    out_dim, hidden = weight.shape
    kv_heads = out_dim // head_dim
    weight = weight.view(kv_heads, head_dim, hidden)
    weight = weight.repeat_interleave(repeat_factor, dim=0)
    return weight.reshape(kv_heads * repeat_factor * head_dim, hidden)


# ---------------------------------------------------------------------------
# TitanDecoderLayer: HF-compatible drop-in replacement
# ---------------------------------------------------------------------------

class TitanDecoderLayer(nn.Module):
    """
    Drop-in replacement for HF LlamaDecoderLayer at specific layer indices.

    Reuses the original HF layer's MLP and RMSNorms. Replaces attention with
    SegmentedAttention. Optionally adds NeuralMemory sidecar and LoRA.

    forward() matches the HF LlamaDecoderLayer signature: accepts the same args,
    returns only hidden_states (a single tensor). KV cache is stored in HF's
    DynamicCache (when available) so that get_seq_length() works correctly,
    with a fallback to internal state for standalone usage.

    NOTE: We inherit from nn.Module (not LlamaDecoderLayer) to avoid creating
    unwanted default submodules. If output_hidden_states is needed (e.g. for
    attention distillation), see _patch_hidden_state_recording().
    """

    def __init__(
        self,
        hf_layer: nn.Module,
        titan_config: TitanLLaMAConfig,
        layer_idx: int,
        shared_state: dict,
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

        # Build SegmentedAttention
        hidden_size = titan_config.hidden_size
        num_heads = titan_config.num_attention_heads
        head_dim = hidden_size // num_heads

        # Only accept value residuals if a *previous* Titan layer exists to produce them
        titan_indices = sorted(titan_config.get_titan_layer_indices())
        is_first_titan_layer = (len(titan_indices) == 0 or layer_idx == titan_indices[0])

        self.segmented_attn = SegmentedAttention(
            dim=hidden_size,
            segment_len=titan_config.segment_len,
            num_persist_mem_tokens=titan_config.num_persist_mem_tokens,
            num_longterm_mem_tokens=titan_config.num_longterm_mem_tokens,
            dim_head=head_dim,
            heads=num_heads,
            sliding=titan_config.sliding_window_attn,
            accept_value_residual=titan_config.use_value_residual and not is_first_titan_layer,
            attend_kwargs=dict(flash=True) if titan_config.use_flash_attn else dict(),
            use_flex_attn=titan_config.use_flex_attn,
            pre_normed=True,
            rope_theta=titan_config.rope_theta,
            rope_freqs=rope_inv_freq,
        )

        # Copy Q/K/V/O weights from HF attention into SegmentedAttention
        num_kv_groups = max(1, num_heads // titan_config.num_key_value_heads)
        with torch.no_grad():
            hf_attn = hf_layer.self_attn
            q_w = hf_attn.q_proj.weight
            k_w = _repeat_kv_weights(hf_attn.k_proj.weight, num_kv_groups, head_dim)
            v_w = _repeat_kv_weights(hf_attn.v_proj.weight, num_kv_groups, head_dim)
            self.segmented_attn.to_qkv.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            self.segmented_attn.to_out.weight.copy_(hf_attn.o_proj.weight)

        # Initialize value residual mix to zero (sigmoid(-10) ≈ 0 → no mixing at start)
        v_mix = self.segmented_attn.to_learned_v_mix
        if v_mix is not None:
            nn.init.zeros_(v_mix[0].weight)
            nn.init.constant_(v_mix[0].bias, -10.0)

        # Delete the original HF attention to free memory
        del hf_layer.self_attn

        # LoRA wrappers (optional)
        self.has_lora = titan_config.use_lora and layer_idx in titan_config.get_lora_layer_indices()
        if self.has_lora:
            self.segmented_attn.to_qkv = LoRALinear(
                base_linear=self.segmented_attn.to_qkv,
                rank=titan_config.lora_rank,
                alpha=titan_config.lora_alpha,
                dropout=titan_config.lora_dropout,
            )
            self.segmented_attn.to_out = LoRALinear(
                base_linear=self.segmented_attn.to_out,
                rank=titan_config.lora_rank,
                alpha=titan_config.lora_alpha,
                dropout=titan_config.lora_dropout,
            )

        # Neural Memory sidecar (optional)
        self.has_neural_memory = layer_idx in titan_config.neural_memory_layers
        if self.has_neural_memory:
            neural_memory_model = MemoryMLP(
                dim=hidden_size,
                depth=titan_config.neural_memory_depth,
            )
            self.neural_memory = NeuralMemory(
                dim=hidden_size,
                chunk_size=titan_config.neural_memory_segment_len,
                batch_size=titan_config.neural_memory_batch_size,
                model=neural_memory_model,
                qkv_receives_diff_views=titan_config.neural_mem_qkv_receives_diff_view,
                accept_weight_residual=False,
                max_grad_norm=1.0,
            )
            dim_inner = num_heads * head_dim
            self.mem_to_kv = nn.Linear(hidden_size, 2 * dim_inner, bias=False)
            if titan_config.zero_init_mem_to_kv:
                nn.init.zeros_(self.mem_to_kv.weight)
            self.num_neural_mem_kv_tokens = titan_config.num_neural_mem_kv_tokens
            self._num_heads = num_heads
            self._head_dim = head_dim

        # Per-layer state (persists across generation steps, reset between sequences)
        self.memory_state = None
        self._titan_kv_cache = None

    # -- KV cache helpers (DynamicCache integration) -----------------------

    def _get_cache(self, past_key_values):
        """Retrieve KV cache from HF DynamicCache if available, else internal."""
        if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
            if self.layer_idx < len(past_key_values.key_cache):
                ck = past_key_values.key_cache[self.layer_idx]
                if isinstance(ck, torch.Tensor) and ck.dim() > 1:
                    return (ck, past_key_values.value_cache[self.layer_idx])
        return self._titan_kv_cache

    def _set_cache(self, past_key_values, cache_kv):
        """Store KV cache in HF DynamicCache (so get_seq_length works) and internally."""
        self._titan_kv_cache = cache_kv
        if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
            k, v = cache_kv
            while len(past_key_values.key_cache) <= self.layer_idx:
                past_key_values.key_cache.append(torch.empty(0))
                past_key_values.value_cache.append(torch.empty(0))
            past_key_values.key_cache[self.layer_idx] = k
            past_key_values.value_cache[self.layer_idx] = v

    # -- forward -----------------------------------------------------------

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
        residual = hidden_states

        # --- Neural Memory (before attention) ---
        memory_kv = None
        if self.has_neural_memory:
            memory_input = torch.stack([hidden_states, hidden_states, hidden_states])

            if not torch.isfinite(hidden_states).all():
                print(f"[NaN] hidden_states before memory at layer {self.layer_idx}")
                raise RuntimeError

            retrieved_memory, self.memory_state = self.neural_memory(
                memory_input,
                state=self.memory_state,
                prev_weights=None,
                detach_mem_state=True,
            )

            if not torch.isfinite(retrieved_memory).all():
                retrieved_memory = torch.zeros_like(hidden_states)

            M = self.num_neural_mem_kv_tokens
            pooled = F.adaptive_avg_pool1d(
                retrieved_memory.transpose(1, 2), M
            ).transpose(1, 2)

            kv = self.mem_to_kv(pooled)
            mem_k, mem_v = kv.chunk(2, dim=-1)
            mem_k = mem_k.reshape(-1, self._num_heads, M, self._head_dim)
            mem_v = mem_v.reshape(-1, self._num_heads, M, self._head_dim)
            memory_kv = (mem_k, mem_v)

        # --- Self Attention ---
        hidden_states = self.input_layernorm(hidden_states)

        # Only pass value_residual when this layer's attention actually accepts it
        value_residual = self._titan_shared_state.get('value_residual', None)
        if self.segmented_attn.to_learned_v_mix is None:
            value_residual = None

        titan_cache = self._get_cache(past_key_values)

        attn_output, attn_intermediates = self.segmented_attn(
            hidden_states,
            value_residual=value_residual,
            cache=titan_cache,
            memory_kv=memory_kv,
        )

        if use_cache:
            self._set_cache(past_key_values, attn_intermediates.cached_key_values)

        # Update shared value residual for downstream Titan layers
        self._titan_shared_state['value_residual'] = attn_intermediates.value_residual

        hidden_states = residual + attn_output

        # --- MLP ---
        residual_ff = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_ff + hidden_states

        return hidden_states


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
        self._titan_shared_state = {'value_residual': None}
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
    ):
        # Reset transient Titan state
        self._titan_shared_state['value_residual'] = None

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        logits = outputs.logits
        loss = outputs.loss
        ppl = None
        accuracy = None

        if labels is not None:
            ppl = torch.exp(loss) if loss is not None else None

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if self.padding_idx is not None:
                mask = shift_labels != self.padding_idx
            else:
                mask = shift_labels != -100

            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels) & mask
            total_valid = mask.sum().float()
            accuracy = correct.sum().float() / total_valid if total_valid > 0 else torch.tensor(0.0, device=logits.device)

        result = {
            'loss': loss,
            'ppl': ppl,
            'logits': logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
        if accuracy is not None:
            result['correct'] = accuracy

        return result

    def generate(self, *args, **kwargs):
        """Delegate to HF backbone's optimized generate(), resetting Titan state first."""
        self.reset_memory_states()
        self._titan_shared_state['value_residual'] = None
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
    ):
        """Backward-compatible generation method. Delegates to HF generate()."""
        if reset_memory:
            self.reset_memory_states()
        self._titan_shared_state['value_residual'] = None

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        if temperature > 0 and do_sample:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        if not do_sample:
            gen_kwargs['temperature'] = None

        return self.backbone.generate(input_ids=input_ids, **gen_kwargs)

    def reset_memory_states(self):
        """Reset all Titan layer state (neural memory + KV caches)."""
        for layer in self.backbone.model.layers:
            if isinstance(layer, TitanDecoderLayer):
                layer.memory_state = None
                layer._titan_kv_cache = None
        self._titan_shared_state['value_residual'] = None

    def freeze_backbone(self):
        """Freeze all pretrained weights. Keep Titan-specific params trainable."""
        nm_total = 0
        lora_total = 0

        print("\n[freeze_backbone] ***** BEGIN *****")

        for name, param in self.named_parameters():
            if "persistent_memory" in name:
                param.requires_grad = True
                continue
            if ".lora." in name or ".lora_" in name:
                param.requires_grad = True
                lora_total += param.numel()
                continue
            if "to_learned_v_mix" in name:
                param.requires_grad = True
                continue
            if "neural_memory" in name or "mem_to_kv" in name:
                param.requires_grad = True
                nm_total += param.numel()
                continue
            param.requires_grad = False

        print(f"[freeze_backbone] NM trainable:     {nm_total:,}")
        print(f"[freeze_backbone] LoRA trainable:   {lora_total:,}")
        print("[freeze_backbone] ***** END *****\n")

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.backbone.prepare_inputs_for_generation(*args, **kwargs)

    @classmethod
    def from_pretrained_llama(
        cls,
        base_model_name_or_path: str,
        titan_config: Optional[TitanLLaMAConfig] = None,
        freeze_backbone: bool = True,
        dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        **from_pretrained_kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoConfig

        # 1) Build Titan config
        base_cfg = AutoConfig.from_pretrained(base_model_name_or_path, **from_pretrained_kwargs)

        titan_kwargs = {}
        if titan_config is not None:
            for attr in [
                'segment_len', 'num_persist_mem_tokens', 'num_longterm_mem_tokens',
                'neural_memory_layers', 'neural_memory_segment_len', 'neural_memory_batch_size',
                'neural_memory_depth', 'use_flex_attn', 'sliding_window_attn',
                'neural_mem_gate_attn_output', 'neural_mem_weight_residual',
                'neural_mem_qkv_receives_diff_view', 'num_neural_mem_kv_tokens',
                'zero_init_mem_to_kv', 'use_value_residual',
                'use_lora', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_layers_after_memory',
                'segmented_attention_layers', 'use_flash_attn',
            ]:
                titan_kwargs[attr] = getattr(titan_config, attr)

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

        backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map or ("auto" if torch.cuda.is_available() else None),
            **extra_model_kwargs,
            **from_pretrained_kwargs,
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
            # Move to same device/dtype as backbone
            device = next(backbone.parameters()).device
            titan_layer = titan_layer.to(device=device, dtype=dtype)
            backbone.model.layers[idx] = titan_layer

        # 6) Freeze backbone if requested
        if freeze_backbone:
            model.freeze_backbone()

        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        base_model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B",
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        strict: bool = False,
    ):
        """Load a TitanLLaMA model from a saved checkpoint."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or torch.bfloat16

        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        train_cfg = ckpt.get("config", {}) or {}

        def _get(name, default):
            return train_cfg.get(name, default)

        nm_layers = _get("neural_memory_layers", (8, 16, 24))
        if isinstance(nm_layers, list):
            nm_layers = tuple(nm_layers)

        seg_layers = _get("segmented_attention_layers", None)
        if isinstance(seg_layers, list):
            seg_layers = tuple(seg_layers)

        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained(base_model_name_or_path)

        titan_cfg = TitanLLaMAConfig.from_llama_config(
            base_cfg,
            segment_len=_get("segment_len", 512),
            num_persist_mem_tokens=_get("num_persist_mem_tokens", 4),
            num_longterm_mem_tokens=_get("num_longterm_mem_tokens", 4),
            neural_memory_layers=nm_layers,
            neural_memory_segment_len=_get("neural_memory_segment_len", 16),
            neural_memory_batch_size=_get("neural_memory_batch_size", 8),
            neural_memory_depth=_get("neural_memory_depth", 2),
            use_flex_attn=_get("use_flex_attn", True),
            use_flash_attn=_get("use_flash_attn", False),
            sliding_window_attn=_get("sliding_window_attn", True),
            neural_mem_gate_attn_output=_get("neural_mem_gate_attn_output", False),
            neural_mem_weight_residual=_get("neural_mem_weight_residual", True),
            neural_mem_qkv_receives_diff_view=_get("neural_mem_qkv_receives_diff_view", True),
            num_neural_mem_kv_tokens=_get("num_neural_mem_kv_tokens", 4),
            zero_init_mem_to_kv=_get("zero_init_mem_to_kv", True),
            use_value_residual=_get("use_value_residual", True),
            segmented_attention_layers=seg_layers,
            use_pretrained_backbone=True,
            base_model_name_or_path=base_model_name_or_path,
            freeze_backbone=True,
            use_lora=_get("use_lora", False),
            lora_rank=_get("lora_rank", 8),
            lora_alpha=_get("lora_alpha", 16),
            lora_dropout=_get("lora_dropout", 0.0),
            lora_layers_after_memory=_get("lora_layers_after_memory", 1),
        )

        # First, build the hybrid model with HF backbone + Titan layer replacements
        model = cls.from_pretrained_llama(
            base_model_name_or_path=base_model_name_or_path,
            titan_config=titan_cfg,
            freeze_backbone=True,
            dtype=dtype,
            device_map=device,
        )

        # Remap old-style state dict keys to new backbone-based keys
        remapped_state_dict = _remap_checkpoint_keys(state_dict, model)

        load_info = model.load_state_dict(remapped_state_dict, strict=strict)
        if load_info.missing_keys:
            print(f"[from_pretrained] Missing keys: {load_info.missing_keys}")
        if load_info.unexpected_keys:
            print(f"[from_pretrained] Unexpected keys: {load_info.unexpected_keys}")

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
    model_keys = set(model.state_dict().keys())

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

        if new_key in model_keys:
            new_state_dict[new_key] = value
        elif old_key in model_keys:
            new_state_dict[old_key] = value
        else:
            # Try without any prefix change
            new_state_dict[old_key] = value

    return new_state_dict
