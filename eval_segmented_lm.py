#!/usr/bin/env python3
"""
Evaluation script for segmented LLaMA on SlimPajama and FineWeb datasets.
Computes token accuracy and perplexity.
"""

import argparse
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_datasets import SlimPajamaDataset, FineWebEduDataset
from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM


def compute_perplexity_and_accuracy(model, dataloader, device, max_batches=None):
    """Compute perplexity and token accuracy on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
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
                
                # Accuracy
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
        'num_batches': num_batches
    }


def load_model(model_path, tokenizer_name, segment_len=512, dtype="bfloat16", segmented_layers=None):
    """Load the segmented LLaMA model."""
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
        from transformers import AutoConfig
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
    parser = argparse.ArgumentParser(description="Evaluate segmented LLaMA on SlimPajama and FineWeb")
    parser.add_argument("--model", required=True, help="Path to model checkpoint or HF model name")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3.1-8B", help="Tokenizer name")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=100, help="Maximum number of batches to evaluate")
    parser.add_argument("--segment-len", type=int, default=512, help="Segment length for attention")
    parser.add_argument("--segmented-layers", nargs="+", type=int, default=[4, 8, 12, 16, 20], 
                       help="Which layers to apply segmented attention to (0-indexed)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--dataset", choices=("slimpajama", "fineweb", "both"), default="both")
    parser.add_argument("--num-proc", type=int, default=8, help="Number of processes for data loading")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    print(f"Using segmented attention on layers: {args.segmented_layers}")
    model, _, device = load_model(args.model, args.tokenizer, args.segment_len, args.dtype, args.segmented_layers)
    
    results = {}
    
    if args.dataset in ["slimpajama", "both"]:
        print("\n=== Evaluating on SlimPajama ===")
        try:
            slimpajama_dataset = SlimPajamaDataset(
                tokenizer_name=args.tokenizer,
                max_length=args.max_length,
                streaming=True,
                split="train",  # Use train split for streaming
                num_proc=args.num_proc,
            )
            slimpajama_loader = DataLoader(
                slimpajama_dataset, 
                batch_size=args.batch_size, 
                shuffle=False
            )
            
            slimpajama_results = compute_perplexity_and_accuracy(
                model, slimpajama_loader, device, args.max_batches
            )
            results["slimpajama"] = slimpajama_results
            print(f"SlimPajama - Perplexity: {slimpajama_results['perplexity']:.4f}, "
                  f"Accuracy: {slimpajama_results['accuracy']:.4f}")
        except Exception as e:
            print(f"Error evaluating SlimPajama: {e}")
            results["slimpajama"] = {"error": str(e)}
    
    if args.dataset in ["fineweb", "both"]:
        print("\n=== Evaluating on FineWeb ===")
        try:
            fineweb_dataset = FineWebEduDataset(
                dataset_name="HuggingFaceFW/fineweb-edu",
                tokenizer_name=args.tokenizer,
                max_length=args.max_length,
                streaming=True,
                split="train",  # FineWeb might not have validation split
                num_proc=args.num_proc,
            )
            fineweb_loader = DataLoader(
                fineweb_dataset, 
                batch_size=args.batch_size, 
                shuffle=False
            )
            
            fineweb_results = compute_perplexity_and_accuracy(
                model, fineweb_loader, device, args.max_batches
            )
            results["fineweb"] = fineweb_results
            print(f"FineWeb - Perplexity: {fineweb_results['perplexity']:.4f}, "
                  f"Accuracy: {fineweb_results['accuracy']:.4f}")
        except Exception as e:
            print(f"Error evaluating FineWeb: {e}")
            results["fineweb"] = {"error": str(e)}
    
    # Print summary
    print("\n=== Summary ===")
    for dataset_name, metrics in results.items():
        if "error" in metrics:
            print(f"{dataset_name}: Error - {metrics['error']}")
        else:
            print(f"{dataset_name}:")
            print(f"  Perplexity: {metrics['perplexity']:.4f}")
            print(f"  Token Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Total Tokens: {metrics['total_tokens']}")
            print(f"  Batches Processed: {metrics['num_batches']}")


if __name__ == "__main__":
    main()