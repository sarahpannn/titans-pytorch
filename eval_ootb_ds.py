#!/usr/bin/env python3
"""
Direct evaluation script for segmented LLaMA on multiple-choice benchmarks.
Uses Titan segmented attention with existing evaluation harnesses from simple_eval.py.
"""

import argparse
import torch
import json
import time
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM
from simple_eval import eval_aqua_rat, eval_pubmedqa, eval_casehold




def load_vanilla_llama(model_name, dtype="bfloat16"):
    """Load vanilla LLaMA model from HuggingFace."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    
    print(f"Loading vanilla {model_name} with dtype {dtype}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if not torch.cuda.is_available():
        model.to(device)
    
    return model, tokenizer, device


def load_segmented_llama(model_path, tokenizer_name, segment_len=512, dtype="bfloat16", segmented_layers=None):
    """Load the segmented LLaMA model with Titan attention."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if model_path.endswith('.pth') or model_path.endswith('.pt'):
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config') or checkpoint.get('model_config')
        
        model = TitanLLaMAForCausalLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load from pretrained
        base_cfg = AutoConfig.from_pretrained(model_path)
        # Try to add segmented_attention_layers parameter if supported
        try:
            titan_cfg = TitanLLaMAConfig.from_llama_config(
                base_cfg,
                segment_len=segment_len,
                num_persist_mem_tokens=4,
                num_longterm_mem_tokens=4,
                neural_memory_layers=(),
                segmented_attention_layers=segmented_layers,
            )
        except TypeError:
            # If segmented_attention_layers is not supported, fall back to default
            print("Warning: segmented_attention_layers parameter not supported in current TitanLLaMAConfig")
            print("Using segmented attention on all layers")
            titan_cfg = TitanLLaMAConfig.from_llama_config(
                base_cfg,
                segment_len=segment_len,
                num_persist_mem_tokens=4,
                num_longterm_mem_tokens=4,
                neural_memory_layers=(),
            )
        
        model = TitanLLaMAForCausalLM.from_pretrained_llama(
            base_model_name_or_path=model_path,
            titan_config=titan_cfg,
            freeze_backbone=True,
            dtype=torch_dtype,
            device_map="cuda",
        )
    
    model.to(device)
    
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="Direct evaluation of LLaMA on multiple-choice tasks")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", help="Path to model checkpoint or HF model name")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3.1-8B", help="Tokenizer name")
    parser.add_argument("--datasets", nargs="+", default=["aqua_rat", "pubmedqa", "casehold"], 
                       choices=["aqua_rat", "pubmedqa", "casehold"],
                       help="Datasets to evaluate on")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (recommend 1 for choice tasks)")
    parser.add_argument("--max-batches", type=int, default=100, help="Maximum number of batches to evaluate")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum examples to load")
    parser.add_argument("--no-segmentation", action="store_true", help="Use vanilla LLaMA without segmentation")
    parser.add_argument("--segment-len", type=int, default=512, help="Segment length for attention (ignored if --no-segmentation)")
    parser.add_argument("--segmented-layers", nargs="+", type=int, default=[4, 8, 12, 16, 20], 
                       help="Which layers to apply segmented attention to (ignored if --no-segmentation)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16", 
                       help="Model precision")
    parser.add_argument("--split", default="validation", help="Dataset split to use")
    parser.add_argument("--output", default=None, help="Output file to save results")
    
    args = parser.parse_args()
    
    # Load model (vanilla or segmented)
    if args.no_segmentation:
        print("Using vanilla LLaMA (no segmentation)")
        model, tokenizer, device = load_vanilla_llama(args.model, args.dtype)
    else:
        print(f"Loading model from {args.model}...")
        print(f"Using segmented attention on layers: {args.segmented_layers}")
        model, tokenizer, device = load_segmented_llama(args.model, args.tokenizer, args.segment_len, args.dtype, args.segmented_layers)
    
    print(f"\nModel loaded on device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    print(f"Evaluating on datasets: {args.datasets}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max examples per dataset: {args.max_examples or 'all'}")
    
    # Initialize results storage
    all_results = {}
    total_start_time = time.time()
    
    # Run evaluation on each dataset using simple_eval harnesses
    print(f"\n{'='*60}")
    print("RUNNING EVALUATIONS")
    print(f"{'='*60}")
    
    for dataset_name in args.datasets:
        print(f"\n--- Evaluating {dataset_name.upper()} ---")
        start_time = time.time()
        
        try:
            if dataset_name == "aqua_rat":
                result = eval_aqua_rat(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=args.batch_size,
                    max_input_len=args.max_length,
                    max_examples=args.max_examples,
                    split=args.split
                )
            elif dataset_name == "pubmedqa":
                result = eval_pubmedqa(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=args.batch_size,
                    max_input_len=args.max_length,
                    max_examples=args.max_examples
                )
            elif dataset_name == "casehold":
                result = eval_casehold(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=args.batch_size,
                    max_input_len=args.max_length,
                    max_examples=args.max_examples,
                    split=args.split
                )
            
            eval_time = time.time() - start_time
            all_results[dataset_name] = result
            all_results[dataset_name]["eval_time"] = eval_time
            
            # Print results for this dataset
            for metric, value in result.items():
                print(f"{metric}: {value:.4f}")
            print(f"Evaluation time: {eval_time:.2f} seconds")
            
        except Exception as e:
            eval_time = time.time() - start_time
            error_msg = f"Error evaluating {dataset_name}: {str(e)}"
            print(error_msg)
            all_results[dataset_name] = {"error": str(e), "eval_time": eval_time}
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    if args.no_segmentation:
        print("Mode: Vanilla LLaMA (no segmentation)")
    else:
        print(f"Mode: Segmented attention")
        print(f"Segmented layers: {args.segmented_layers}")
        print(f"Segment length: {args.segment_len}")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print()
    
    for dataset_name, result in all_results.items():
        print(f"{dataset_name.upper()}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            # Print all metrics from the result
            for metric, value in result.items():
                if metric != "eval_time" and isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
            print(f"  Time: {result['eval_time']:.2f}s")
        print()
    
    # Save results if output file specified
    if args.output:
        evaluation_params = {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "max_examples": args.max_examples,
            "dtype": args.dtype,
            "no_segmentation": args.no_segmentation,
        }
        
        # Add segmentation params only if using segmentation
        if not args.no_segmentation:
            evaluation_params.update({
                "segment_len": args.segment_len,
                "segmented_layers": args.segmented_layers,
            })
        
        output_data = {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "datasets": args.datasets,
            "split": args.split,
            "evaluation_params": evaluation_params,
            "results": all_results,
            "total_eval_time": total_time
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()