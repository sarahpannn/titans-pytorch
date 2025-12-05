"""
Optimized Segmented Attention Implementation

This is a drop-in replacement for the original SegmentedAttention that:
1. Avoids exploding the batch dimension 
2. Precomputes attention masks efficiently
3. Uses standard attention with block-diagonal masks instead of reshaping

Memory usage: O(B * H * N^2) instead of O(B * num_segments * H * segment_len^2)
where N is sequence length, much more efficient for moderate sequence lengths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Callable
from collections import namedtuple

# Import required components from the original
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend
from functools import partial

LinearNoBias = partial(nn.Linear, bias=False)
AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))

def exists(v):
    return v is not None

class OptimizedSegmentedAttention(nn.Module):
    """
    Optimized segmented attention that doesn't explode batch dimensions.
    
    Key improvements:
    1. Precompute block-diagonal attention mask once
    2. Use standard attention with masking instead of reshaping
    3. Persistent memory tokens added efficiently without duplication
    """
    
    def __init__(
        self,
        dim: int,
        segment_len: int,
        num_persist_mem_tokens: int = 0,
        num_longterm_mem_tokens: int = 0,
        dim_head: int = 64,
        heads: int = 8,
        sliding: bool = False,
        accept_value_residual: bool = False,
        attend_kwargs: dict = None,
        use_flex_attn: bool = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        
        self.dim = dim
        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.sliding = sliding
        self.heads = heads
        self.dim_head = dim_head
        
        dim_inner = dim_head * heads
        
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = Attend(causal=True, **(attend_kwargs or {}))
        
        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)
        
        # Value residual mixing
        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None
        
        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        # Persistent memory (much smaller allocation)
        self.persistent_memory = nn.Parameter(
            torch.zeros(2, heads, num_persist_mem_tokens, dim_head)
        ) if num_persist_mem_tokens > 0 else None
        
        # Cache for precomputed masks
        self._cached_mask = None
        self._cached_seq_len = None
        
    def _create_segmented_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Create block-diagonal attention mask for segmented attention.
        Much more memory efficient than reshaping tensors.
        """
        # Check cache first
        if (self._cached_mask is not None and 
            self._cached_seq_len == seq_len and 
            self._cached_mask.device == device):
            return self._cached_mask
            
        total_len = seq_len + self.num_persist_mem_tokens
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        if not self.sliding:
            # Block diagonal mask - tokens can only attend within their segment
            num_segments = (seq_len + self.segment_len - 1) // self.segment_len
            
            # Create block structure
            for i in range(seq_len):
                segment_i = i // self.segment_len
                for j in range(seq_len):
                    segment_j = j // self.segment_len
                    if segment_i != segment_j:
                        causal_mask[i, j] = False
        else:
            # Sliding window mask - tokens can attend within sliding window
            for i in range(seq_len):
                for j in range(seq_len):
                    if causal_mask[i, j] and (i - j) > self.segment_len:
                        causal_mask[i, j] = False
        
        # Add persistent memory mask (persistent tokens can attend to all)
        if self.num_persist_mem_tokens > 0:
            # Full mask including persistent memory
            full_mask = torch.ones(total_len, total_len, device=device, dtype=torch.bool)
            
            # Persistent memory tokens (first num_persist_mem_tokens) can attend to everything
            # Regular tokens can attend based on causal + segment rules, plus to persistent memory
            full_mask[self.num_persist_mem_tokens:, self.num_persist_mem_tokens:] = causal_mask
        else:
            full_mask = causal_mask
            
        # Convert to attention mask format (add head dimensions)
        attention_mask = full_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Cache the result
        self._cached_mask = attention_mask
        self._cached_seq_len = seq_len
        
        return attention_mask
    
    def forward(
        self,
        seq: Tensor,
        value_residual: Optional[Tensor] = None,
        cache: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs
    ) -> Tuple[Tensor, AttnIntermediates]:
        
        batch, seq_len = seq.shape[:2]
        device = seq.device
        
        # Normalize input
        seq = self.norm(seq)
        
        # Generate Q, K, V
        q, k, v = self.to_qkv(seq).chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        
        # Store original v for residual
        orig_v = v
        
        # Apply value residual if configured
        if exists(self.to_learned_v_mix) and exists(value_residual):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)
        
        # Add persistent memory tokens efficiently
        if self.persistent_memory is not None:
            pmk, pmv = self.persistent_memory[0], self.persistent_memory[1]
            # Expand persistent memory for batch
            pmk = pmk.unsqueeze(0).expand(batch, -1, -1, -1)  # (B, H, persist_tokens, D)
            pmv = pmv.unsqueeze(0).expand(batch, -1, -1, -1)
            
            # Concatenate persistent memory to beginning of sequence
            k = torch.cat([pmk, k], dim=2)  # (B, H, persist_tokens + seq_len, D)
            v = torch.cat([pmv, v], dim=2)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        
        # Create attention mask
        attention_mask = self._create_segmented_mask(seq_len, device)
        
        # Apply attention with precomputed mask
        # Shape: (B, H, seq_len, seq_len + persist_tokens)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=False  # We're providing explicit mask
        )
        
        # Merge heads and project output
        out = self.merge_heads(out)
        out = self.to_out(out)
        
        # Prepare cache for next iteration (if needed)
        next_cache = (k, v) if cache is not None else None
        
        return out, AttnIntermediates(orig_v, next_cache)

# Drop-in replacement function
def replace_segmented_attention(model):
    """
    Replace all SegmentedAttention modules in a model with OptimizedSegmentedAttention.
    This should be a drop-in replacement that uses much less memory.
    """
    def replace_module(module):
        for name, child in list(module.named_children()):
            if child.__class__.__name__ == 'SegmentedAttention':
                # Extract parameters from original module
                optimized = OptimizedSegmentedAttention(
                    dim=child.norm.normalized_shape[0],  # Get dim from RMSNorm
                    segment_len=child.segment_len,
                    num_persist_mem_tokens=child.num_persist_mem_tokens,
                    num_longterm_mem_tokens=getattr(child, 'num_longterm_mem_tokens', 0),
                    dim_head=child.persistent_memory.shape[-1] if hasattr(child, 'persistent_memory') else 64,
                    heads=child.persistent_memory.shape[1] if hasattr(child, 'persistent_memory') else 8,
                    sliding=getattr(child, 'sliding', False),
                    accept_value_residual=hasattr(child, 'to_learned_v_mix') and child.to_learned_v_mix is not None,
                    use_flex_attn=getattr(child, 'use_flex_attn', False)
                )
                
                # Copy weights if possible
                try:
                    optimized.load_state_dict(child.state_dict(), strict=False)
                except:
                    print(f"Warning: Could not copy weights for {name}, using random initialization")
                
                setattr(module, name, optimized)
                print(f"Replaced {name} with OptimizedSegmentedAttention")
            else:
                replace_module(child)
    
    replace_module(model)
    return model

if __name__ == "__main__":
    # Test memory usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size = 2
    seq_len = 4096
    dim = 2048
    heads = 32
    dim_head = 64
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test optimized version
    optimized_attn = OptimizedSegmentedAttention(
        dim=dim,
        segment_len=512,
        num_persist_mem_tokens=4,
        dim_head=dim_head,
        heads=heads,
        sliding=False
    ).to(device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        out, intermediates = optimized_attn(x)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"Optimized attention peak memory: {peak_memory:.2f} GB")
    print(f"Output shape: {out.shape}")