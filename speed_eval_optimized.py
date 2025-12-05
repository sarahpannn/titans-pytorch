#!/usr/bin/env python3
"""
Speed evaluation with optimized segmented attention.

This is a modified version of speed_eval.py that uses the optimized attention
to fix the memory explosion and performance issues with segmented_no_nmm.
"""

import sys
sys.path.append('.')

# Import the original speed_eval functionality
from speed_eval import *
from optimized_segmented_attention import replace_segmented_attention

def build_optimized_titan_variant(
    model_name: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    segment_len: int,
    num_persist: int,
    num_longterm: int,
    neural_memory_layers: Sequence[int],
    use_flex_attn: bool,
    neural_memory_chunk_size: int,
    neural_memory_batch_size: int,
):
    """Build Titan variant with optimized segmented attention."""
    
    # Use the original build function
    model, tokenizer = build_titan_variant(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        segment_len=segment_len,
        num_persist=num_persist,
        num_longterm=num_longterm,
        neural_memory_layers=neural_memory_layers,
        use_flex_attn=use_flex_attn,
        neural_memory_chunk_size=neural_memory_chunk_size,
        neural_memory_batch_size=neural_memory_batch_size,
    )
    
    # Replace with optimized attention if no neural memory
    if not neural_memory_layers:
        print("Replacing with optimized segmented attention...")
        model = replace_segmented_attention(model)
        print("✓ Optimized attention replacement complete")
    
    return model, tokenizer

def main_optimized() -> None:
    """Main function with optimized attention."""
    args = parse_args()
    device = get_device(args.device)
    torch_dtype = get_dtype(args.dtype)

    variants = []
    variants.append(("base_llama", lambda: build_base_llama(args.model, torch_dtype, device)))

    nmm_batch = args.nmm_batch_size if args.nmm_batch_size is not None else args.batch_size
    nmm_chunk = args.nmm_chunk_size
    if nmm_batch % nmm_chunk != 0:
        print(f"[info] nmm-chunk-size {nmm_chunk} does not divide nmm_batch_size {nmm_batch}; using chunk size 1.")
        nmm_chunk = 1

    # Use optimized variant for no_nmm
    variants.append(
        (
            "optimized_segmented_no_nmm",
            lambda: build_optimized_titan_variant(
                args.model,
                torch_dtype,
                device,
                args.segment_len,
                args.num_persist,
                args.num_longterm,
                neural_memory_layers=(),  # No neural memory
                use_flex_attn=args.use_flex_attn,
                neural_memory_chunk_size=nmm_chunk,
                neural_memory_batch_size=nmm_batch,
            ),
        )
    )
    
    # Original with neural memory (shouldn't have the same issue)
    variants.append(
        (
            "segmented_with_nmm",
            lambda: build_titan_variant(
                args.model,
                torch_dtype,
                device,
                args.segment_len,
                args.num_persist,
                args.num_longterm,
                neural_memory_layers=(8, 16, 24),
                use_flex_attn=args.use_flex_attn,
                neural_memory_chunk_size=nmm_chunk,
                neural_memory_batch_size=nmm_batch,
            ),
        )
    )

    results = []
    for name, builder in variants:
        print(f"\n== Benchmarking {name} ==")
        
        # Clear cache before each variant
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        model, tokenizer = builder()
        input_ids, attention_mask, prompt_text, prompt_tokens = build_prompt_batch(
            tokenizer=tokenizer,
            prompt=args.prompt,
            prompt_length=args.prompt_length,
            batch_size=args.batch_size,
            device=device,
        )
        
        # Report memory usage
        if device.type == "cuda":
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak memory after model load: {memory_gb:.2f} GB")
        
        tokens_per_sec, elapsed, generated_tokens = benchmark_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            target_new_tokens=args.target_new_tokens,
            warmup_steps=args.warmup_steps,
            temperature=args.temperature,
            variant_name=name,
        )
        
        # Report final memory usage
        if device.type == "cuda":
            final_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak memory after generation: {final_memory_gb:.2f} GB")
        
        results.append(
            {
                "name": name,
                "tokens_per_sec": tokens_per_sec,
                "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "target_new_tokens": args.target_new_tokens,
                "peak_memory_gb": final_memory_gb if device.type == "cuda" else 0,
            }
        )
        print(
            f"{name}: {tokens_per_sec:.2f} new tokens/sec targeting {args.target_new_tokens} tokens "
            f"(elapsed {elapsed:.2f}s, prompt {prompt_tokens} tok, generated {generated_tokens} tok)"
        )

        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    print("\n=== Summary (higher tokens/sec is better, lower memory is better) ===")
    for entry in results:
        memory_str = f", {entry['peak_memory_gb']:.1f} GB peak" if entry['peak_memory_gb'] > 0 else ""
        print(f"{entry['name']}: {entry['tokens_per_sec']:.2f} tokens/sec (elapsed {entry['elapsed_seconds']:.2f}s{memory_str})")

    if args.save_dir:
        save_results(
            args.save_dir,
            {
                "model": args.model,
                "prompt": args.prompt,
                "prompt_tokens": results[0]["prompt_tokens"] if results else 0,
                "max_new_tokens": args.max_new_tokens,
                "target_new_tokens": args.target_new_tokens,
                "batch_size": args.batch_size,
                "nmm_batch_size": nmm_batch,
                "nmm_chunk_size": nmm_chunk,
                "warmup_steps": args.warmup_steps,
                "dtype": args.dtype,
                "device": str(device),
                "results": results,
            },
        )

    if args.cleanup_cache:
        cleanup_cache(args.model)


if __name__ == "__main__":
    print("Running speed evaluation with OPTIMIZED segmented attention")
    print("This should fix the memory explosion issue with segmented_no_nmm")
    print("=" * 60)
    main_optimized()