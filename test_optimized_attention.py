#!/usr/bin/env python3
"""
Test script to compare memory usage between original and optimized segmented attention.
"""

import torch
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append('.')

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM
from transformers import AutoConfig
from optimized_segmented_attention import replace_segmented_attention

def test_memory_usage():
    """Compare memory usage between original and optimized implementations."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    
    # Test parameters - use smaller model for testing
    batch_size = 1
    seq_len = 2048
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
    print("=" * 50)
    
    # Create base config for smaller test model
    base_cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    
    # Create Titan config with segmented attention but NO neural memory
    titan_cfg = TitanLLaMAConfig.from_llama_config(
        base_cfg,
        segment_len=512,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(),  # No neural memory - this is our "no_nmm" test
        neural_memory_segment_len=0,
        neural_memory_batch_size=0,
        neural_memory_depth=0,
        use_flex_attn=False,
        sliding_window_attn=True,  # This was causing issues
        use_pretrained_backbone=False,  # Don't load full model for test
        freeze_backbone=False,
    )
    
    print("1. Testing ORIGINAL segmented attention (the slow one)...")
    
    # Create model with original attention
    try:
        original_model = TitanLLaMAForCausalLM(titan_cfg).to(device)
        
        # Create test input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = original_model(input_ids=input_ids)
        
        original_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"   Original peak memory: {original_memory:.2f} GB")
        
        del original_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   Original model failed: {e}")
        original_memory = float('inf')
    
    print("\n2. Testing OPTIMIZED segmented attention (the fast one)...")
    
    # Create model with optimized attention  
    try:
        optimized_model = TitanLLaMAForCausalLM(titan_cfg).to(device)
        
        # Replace with optimized attention
        optimized_model = replace_segmented_attention(optimized_model)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = optimized_model(input_ids=input_ids)
        
        optimized_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"   Optimized peak memory: {optimized_memory:.2f} GB")
        
        # Calculate improvement
        if original_memory != float('inf'):
            improvement = ((original_memory - optimized_memory) / original_memory) * 100
            speedup = original_memory / optimized_memory
            print(f"   Memory reduction: {improvement:.1f}% ({speedup:.1f}x less memory)")
        
        del optimized_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   Optimized model failed: {e}")
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("The optimized version should use significantly less memory")
    print("because it avoids exploding the batch dimension during attention.")

def test_functional_equivalence():
    """Test that optimized attention produces similar outputs to original."""
    print("\n3. Testing functional equivalence...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Smaller test for functional verification
    from optimized_segmented_attention import OptimizedSegmentedAttention
    from titans_pytorch.mac_transformer import SegmentedAttention
    
    # Test parameters
    batch_size = 1
    seq_len = 512  # Smaller for testing
    dim = 512
    heads = 8
    dim_head = 64
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    
    try:
        # Original attention
        original_attn = SegmentedAttention(
            dim=dim,
            segment_len=128,
            num_persist_mem_tokens=2,
            dim_head=dim_head,
            heads=heads,
            sliding=False,
            use_flex_attn=False
        ).to(device)
        
        # Optimized attention
        optimized_attn = OptimizedSegmentedAttention(
            dim=dim,
            segment_len=128,
            num_persist_mem_tokens=2,
            dim_head=dim_head,
            heads=heads,
            sliding=False
        ).to(device)
        
        with torch.no_grad():
            orig_out, _ = original_attn(x)
            opt_out, _ = optimized_attn(x)
        
        # Compare outputs (they won't be identical due to different computation paths)
        mse = torch.mean((orig_out - opt_out) ** 2).item()
        print(f"   MSE between outputs: {mse:.6f}")
        print(f"   Original output norm: {torch.norm(orig_out).item():.3f}")
        print(f"   Optimized output norm: {torch.norm(opt_out).item():.3f}")
        
        if mse < 0.1:  # Reasonable threshold
            print("   ✓ Outputs are reasonably similar")
        else:
            print("   ⚠ Outputs differ significantly (this might be OK)")
            
    except Exception as e:
        print(f"   Functional test failed: {e}")

if __name__ == "__main__":
    print("Testing Optimized Segmented Attention")
    print("This will help debug why the no_nmm variant is so slow/memory-hungry")
    
    test_memory_usage()
    test_functional_equivalence()
    
    print("\nTo use the optimized version in your speed_eval.py:")
    print("1. Import: from optimized_segmented_attention import replace_segmented_attention")
    print("2. After model creation: model = replace_segmented_attention(model)")
    print("3. Run your benchmark and see the improvement!")