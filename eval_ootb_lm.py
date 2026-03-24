#!/usr/bin/env python3
"""
Evaluation script for vanilla LLaMA 3.1-8B on FineWeb-Edu dataset.
Computes perplexity and token accuracy out of the box.
"""

import argparse
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from train_datasets import FineWebEduDataset


def compute_perplexity_and_accuracy(model, dataloader, device, max_batches=None):
    """Compute perplexity and token accuracy on FineWeb-Edu dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating on FineWeb-Edu")):
            if max_batches and batch_idx >= max_batches:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Flatten for loss computation
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            flat_mask = shift_mask.view(-1)
            
            # Compute loss only on valid tokens
            valid_indices = flat_mask.bool()
            if valid_indices.sum() > 0:
                valid_logits = flat_logits[valid_indices]
                valid_labels = flat_labels[valid_indices]
                
                # Cross-entropy loss
                loss = F.cross_entropy(valid_logits, valid_labels, reduction='sum')
                total_loss += loss.item()
                
                # Token accuracy
                predictions = torch.argmax(valid_logits, dim=-1)
                correct = (predictions == valid_labels).sum().item()
                total_correct += correct
                total_tokens += valid_indices.sum().item()
            
            num_batches += 1
    
    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')  # Avoid overflow
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'num_batches': num_batches,
        'avg_loss': avg_loss
    }


def load_vanilla_llama(model_name, dtype="bfloat16"):
    """Load vanilla LLaMA model from HuggingFace."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    
    print(f"Loading {model_name} with dtype {dtype}...")
    
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate vanilla LLaMA 3.1-8B on FineWeb-Edu")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", help="HuggingFace model name")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (recommend 1 for 8B model)")
    parser.add_argument("--max-batches", type=int, default=100, help="Maximum number of batches to evaluate")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16", 
                       help="Model precision")
    parser.add_argument("--num-proc", type=int, default=8, help="Number of processes for data loading")
    parser.add_argument("--streaming", action="store_true", default=True, 
                       help="Use streaming dataset (recommended for large datasets)")
    parser.add_argument("--split", default="train", choices=("train", "validation"), 
                       help="Dataset split to evaluate on")
    parser.add_argument("--output", default=None, help="Output file to save results")
    
    args = parser.parse_args()
    
    # Load vanilla LLaMA model
    model, tokenizer, device = load_vanilla_llama(args.model, args.dtype)
    
    print(f"\nModel loaded on device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    
    # Create FineWeb-Edu dataset
    print(f"\nLoading FineWeb-Edu dataset (split: {args.split})...")
    try:
        fineweb_dataset = FineWebEduDataset(
            dataset_name="HuggingFaceFW/fineweb-edu",
            tokenizer_name=args.model,
            max_length=args.max_length,
            streaming=args.streaming,
            split=args.split,
            num_proc=args.num_proc,
        )
        
        fineweb_loader = DataLoader(
            fineweb_dataset, 
            batch_size=args.batch_size, 
            shuffle=False
        )
        
        print(f"Dataset created successfully")
        print(f"Batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_length}")
        print(f"Max batches to evaluate: {args.max_batches}")
        
    except Exception as e:
        print(f"Error creating FineWeb-Edu dataset: {e}")
        return
    
    # Run evaluation
    print(f"\n=== Evaluating {args.model} on FineWeb-Edu ===")
    try:
        results = compute_perplexity_and_accuracy(
            model, fineweb_loader, device, args.max_batches
        )
        
        # Print results
        print(f"\n=== Results ===")
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"Token Accuracy: {results['accuracy']:.4f}")
        print(f"Average Loss: {results['avg_loss']:.4f}")
        print(f"Total Tokens: {results['total_tokens']:,}")
        print(f"Batches Processed: {results['num_batches']}")
        
        # Save results if output file specified
        if args.output:
            import json
            output_data = {
                "model": args.model,
                "dataset": "fineweb-edu",
                "split": args.split,
                "evaluation_params": {
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "max_batches": args.max_batches,
                    "dtype": args.dtype,
                },
                "results": results
            }
            
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()