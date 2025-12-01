#!/usr/bin/env python3
"""
Launcher script for TitanLLaMA training with configurable parameters.
"""

import argparse
import os
import sys
from dataclasses import asdict
import json

from train_titan_llama import TrainingConfig, main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TitanLLaMA on SlimPajama dataset")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="titan-llama-1b", help="Model name")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=22, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    
    # Titan-specific
    parser.add_argument("--segment_len", type=int, default=512, help="Segment length for attention")
    parser.add_argument("--num_persist_mem", type=int, default=4, help="Number of persistent memory tokens")
    parser.add_argument("--num_longterm_mem", type=int, default=4, help="Number of longterm memory tokens")
    parser.add_argument("--neural_memory_layers", type=str, default="4,8,12,16,20", 
                       help="Comma-separated layer indices for neural memory")
    parser.add_argument("--neural_memory_segment_len", type=int, default=64, help="Neural memory segment length")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Pretrained LLaMA checkpoint for backbone")
    parser.add_argument("--no_pretrained_backbone", action="store_true", help="Train from scratch instead of using a pretrained LLaMA backbone")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Allow backbone weights to update (default keeps them frozen)")
    
    # Training configuration
    parser.add_argument("--total_tokens", type=int, default=1_000_000_000, help="Total tokens to train on")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--neural_mem_lr", type=float, default=1e-2, help="Neural memory learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    
    # Data
    parser.add_argument("--dataset_name", type=str, default="cerebras/SlimPajama-627B", help="Dataset name")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Tokenizer name")
    
    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="./titan_llama_checkpoints", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="titan-llama-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save interval")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    
    # Distributed training
    parser.add_argument("--use_ddp", action="store_true", help="Use distributed training")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    # Other options
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create TrainingConfig from command line arguments."""
    
    # Parse neural memory layers
    neural_memory_layers = tuple(map(int, args.neural_memory_layers.split(',')))
    
    config = TrainingConfig(
        # Model config
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        
        # Titan config
        segment_len=args.segment_len,
        num_persist_mem_tokens=args.num_persist_mem,
        num_longterm_mem_tokens=args.num_longterm_mem,
        neural_memory_layers=neural_memory_layers,
        neural_memory_segment_len=args.neural_memory_segment_len,
        use_pretrained_backbone=not args.no_pretrained_backbone,
        base_model_name=args.base_model_name,
        freeze_backbone=not args.unfreeze_backbone,
        
        # Training config
        total_tokens=args.total_tokens,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        neural_mem_learning_rate=args.neural_mem_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        
        # Data config
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        
        # Logging config
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        
        # Distributed config
        use_ddp=args.use_ddp,
        
        # Resume config
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    # Debug mode adjustments
    if args.debug:
        config.total_tokens = 1_000_000  # 1M tokens for debugging
        config.eval_interval = 100
        config.save_interval = 500
        config.log_interval = 10
    
    return config


def main_with_args():
    """Main function with argument parsing."""
    args = parse_args()
    
    # Disable wandb if requested
    if args.no_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    
    # Create config
    config = create_config_from_args(args)
    
    # Save config
    os.makedirs(config.output_dir, exist_ok=True)
    config_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Training config saved to: {config_path}")
    
    # Print key configuration
    print("\n" + "="*80)
    print("TitanLLaMA Training Configuration")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Backbone: {config.base_model_name} | Pretrained: {config.use_pretrained_backbone} | Frozen: {config.freeze_backbone}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Neural Memory Layers: {config.neural_memory_layers}")
    print(f"Total Tokens: {config.total_tokens:,}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Sequence Length: {config.sequence_length}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Neural Memory LR: {config.neural_mem_learning_rate}")
    print(f"Output Directory: {config.output_dir}")
    print("="*80 + "\n")
    
    # Set up the config globally and run training
    import train_titan_llama
    train_titan_llama.TrainingConfig = lambda: config
    main()


if __name__ == "__main__":
    main_with_args()
