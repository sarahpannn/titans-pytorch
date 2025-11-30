import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Callable
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


class TitanLLaMAConfig:
    """Configuration for Titan-LLaMA model with segmented attention and neural memory."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
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
        neural_memory_segment_len: int = 32,
        neural_memory_batch_size: int = 128,
        neural_memory_depth: int = 2,
        use_flex_attn: bool = True,
        sliding_window_attn: bool = True,
        neural_mem_gate_attn_output: bool = False,
        neural_mem_weight_residual: bool = True,
        neural_mem_qkv_receives_diff_view: bool = True,
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
        
        # Titan parameters
        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.neural_memory_layers = neural_memory_layers
        self.neural_memory_segment_len = neural_memory_segment_len
        self.neural_memory_batch_size = neural_memory_batch_size
        self.neural_memory_depth = neural_memory_depth
        self.use_flex_attn = use_flex_attn
        self.sliding_window_attn = sliding_window_attn
        self.neural_mem_gate_attn_output = neural_mem_gate_attn_output
        self.neural_mem_weight_residual = neural_mem_weight_residual
        self.neural_mem_qkv_receives_diff_view = neural_mem_qkv_receives_diff_view


class TitanLLaMARMSNorm(nn.Module):
    """RMSNorm implementation matching LLaMA."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TitanLLaMAMLP(nn.Module):
    """SwiGLU MLP implementation for LLaMA with optional neural memory integration."""
    
    def __init__(self, config: TitanLLaMAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x):
        down_proj = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class TitanLLaMAAttention(nn.Module):
    """
    Titan-LLaMA Attention layer that integrates segmented attention with LLaMA's architecture.
    Uses the segmented attention mechanism from titans-pytorch for improved memory efficiency.
    """
    
    def __init__(self, config: TitanLLaMAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Use SegmentedAttention from titans-pytorch
        self.segmented_attn = SegmentedAttention(
            dim=config.hidden_size,
            segment_len=config.segment_len,
            num_persist_mem_tokens=config.num_persist_mem_tokens,
            num_longterm_mem_tokens=config.num_longterm_mem_tokens,
            dim_head=self.head_dim,
            heads=self.num_heads,
            sliding=config.sliding_window_attn,
            accept_value_residual=layer_idx > 0,  # First layer doesn't get value residual
            use_flex_attn=config.use_flex_attn
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        value_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # Use segmented attention
        attn_output, attn_intermediates = self.segmented_attn(
            hidden_states,
            value_residual=value_residual,
            cache=past_key_value
        )
        
        # Extract attention weights if requested
        attn_weights = None
        if output_attentions:
            # Segmented attention doesn't return weights by default
            # Could be extracted from attn_intermediates if needed
            attn_weights = None

        # Extract new cache
        present_key_value = None
        if use_cache:
            present_key_value = attn_intermediates.cached_key_values

        return attn_output, attn_weights, present_key_value, attn_intermediates.value_residual


class TitanLLaMADecoderLayer(nn.Module):
    """
    Titan-LLaMA Decoder Layer that integrates neural memory with standard transformer components.

    This version **disables cross-layer weight-residual mixing** for the Titans NeuralMemory.
    That avoids the prev_weights + weights_for_surprise shape issues and the XNOR asserts,
    while still letting each layer use its own NeuralMemory as a sidecar adapter.
    """

    def __init__(self, config: TitanLLaMAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Attention and MLP
        self.self_attn = TitanLLaMAAttention(config=config, layer_idx=layer_idx)
        self.mlp = TitanLLaMAMLP(config)

        # Layer norms
        self.input_layernorm = TitanLLaMARMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = TitanLLaMARMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # ----------------------------
        # Neural Memory sidecar
        # ----------------------------
        self.has_neural_memory = layer_idx in config.neural_memory_layers

        if self.has_neural_memory:
            neural_memory_model = MemoryMLP(
                dim=config.hidden_size,
                depth=config.neural_memory_depth,
            )

            # IMPORTANT: force accept_weight_residual = False
            # so self.to_learned_weight_residual_mix is None and
            # prev_weights is always expected to be None inside Titans.
            self.neural_memory = NeuralMemory(
                dim=config.hidden_size,
                chunk_size=config.neural_memory_segment_len,
                batch_size=config.neural_memory_batch_size,
                model=neural_memory_model,
                qkv_receives_diff_views=config.neural_mem_qkv_receives_diff_view,
                accept_weight_residual=False,
            )

            # Per-layer memory state (for TTT / long sequences)
            self.memory_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:

        residual = hidden_states

        # Value residual from previous attention layer
        value_residual: Optional[torch.Tensor] = kwargs.get("value_residual", None)

        # ------------------------------------------------------------------
        # Neural Memory (before attention)
        # ------------------------------------------------------------------
        retrieved_memory: Optional[torch.Tensor] = None

        if self.has_neural_memory:
            # QKV-style input: 3 x B x T x D
            memory_input = torch.stack(
                [hidden_states, hidden_states, hidden_states]
            )

            # NOTE: prev_weights is always None here (no weight-residual path).
            retrieved_memory, self.memory_state = self.neural_memory(
                memory_input,
                state=self.memory_state,
                prev_weights=None,
            )

            # If we're not gating, inject memory directly into the residual stream.
            if not self.config.neural_mem_gate_attn_output and retrieved_memory is not None:
                residual = residual + retrieved_memory

        # ------------------------------------------------------------------
        # Self Attention
        # ------------------------------------------------------------------
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, new_value_residual = \
            self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                value_residual=value_residual,
            )

        # If configured, gate attention output by retrieved memory
        if (
            self.has_neural_memory
            and self.config.neural_mem_gate_attn_output
            and retrieved_memory is not None
        ):
            gate = retrieved_memory.sigmoid()
            hidden_states = hidden_states * gate

        # Standard residual after attention
        hidden_states = residual + hidden_states

        # ------------------------------------------------------------------
        # Feedforward
        # ------------------------------------------------------------------
        residual_ff = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_ff + hidden_states

        # ------------------------------------------------------------------
        # Pack outputs
        # ------------------------------------------------------------------
        outputs: Tuple[torch.Tensor, ...] = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # Value residual for next attention layer
        new_weight_residual = None  # no cross-layer weight residuals in this variant
        outputs += (new_value_residual, new_weight_residual)

        return outputs



class TitanLLaMAModel(nn.Module):
    """
    Titan-LLaMA Model that combines LLaMA architecture with Titans' segmented attention 
    and neural memory for improved long-context performance and test-time adaptation.
    """

    def __init__(self, config: TitanLLaMAConfig):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Transformer layers with titan enhancements
        self.layers = nn.ModuleList(
            [TitanLLaMADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        self.norm = TitanLLaMARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Track value residuals for attention layers and weight residuals for neural memory
        value_residual = None
        prev_weight_residual = None
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                value_residual=value_residual,
                prev_weight_residual=prev_weight_residual,
            )

            hidden_states = layer_outputs[0]
            
            # Extract value residual and weight residual for next layer (last two elements)
            value_residual = layer_outputs[-2]
            prev_weight_residual = layer_outputs[-1]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
        
        # Return dictionary-like object (simplified)
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns,
        }

    def reset_memory_states(self):
        """Reset all neural memory states. Useful for starting fresh sequences."""
        for layer in self.layers:
            if hasattr(layer, 'neural_memory'):
                layer.memory_state = None
                layer.memory_weight_residual = None


class TitanLLaMAForCausalLM(nn.Module):
    """
    Titan-LLaMA model for causal language modeling.
    """

    def __init__(self, config: TitanLLaMAConfig):
        super().__init__()
        self.model = TitanLLaMAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        
        return_dict = return_dict if return_dict is not None else True

        # Forward through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs['last_hidden_state'] if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.get('past_key_values'),
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions'),
        }

    def reset_memory_states(self):
        """Reset all neural memory states in the model."""
        self.model.reset_memory_states()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        """Prepare inputs for generation."""
        # If we have cache, only use the new token
        if past_key_values is not None:
            if isinstance(past_key_values[0], tuple):
                cache_length = past_key_values[0][0].shape[2]
            else:
                cache_length = past_key_values[0].shape[2]
            
            # Only keep the most recent token if we have a cache
            if input_ids.shape[1] > cache_length:
                input_ids = input_ids[:, cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    @torch.no_grad()
    def generate_with_titan_memory(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        reset_memory: bool = True,
    ):
        """
        Generate text using Titan memory capabilities.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling vs greedy decoding
            top_p: Top-p (nucleus) sampling threshold
            reset_memory: Whether to reset memory states before generation
        """
        if reset_memory:
            self.reset_memory_states()
            
        self.eval()
        
        generated_tokens = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated_tokens,
                use_cache=True,
                return_dict=True
            )
            
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            if do_sample:
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Create mask for top-p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            
            # Stop if we generate an EOS token (assuming vocab has one)
            # if next_token.item() == self.config.eos_token_id:
            #     break
                
        return generated_tokens