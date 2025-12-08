#!/usr/bin/env python3
"""
Training script for TitanLLaMA on SlimPajama dataset.
Trains on 1B tokens with segmented attention and neural memory.

Attention Distillation Usage:
    # Enable attention distillation with default settings
    python train_titan_llama.py --use-attention-distillation
    
    # Custom distillation weight and layers
    python train_titan_llama.py --use-attention-distillation \
                                --distillation-weight 0.05 \
                                --distillation-layers 4 8 12 16 20
    
The attention distillation loss compares segmented attention outputs with and without 
flex attention optimizations, helping to maintain attention pattern fidelity.
"""

import os
import json
import math
import time
import logging
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm

# Import our TitanLLaMA implementation
from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM
from train_datasets import SlimPajamaDataset, FineWebEduDataset, BoolQDataset, WinograndeDataset, MixedEvalDataset
from cuda_utils import log_cuda_mem
from simple_eval import eval_winogrande_boolq, quick_eval_boolq, should_run_intermittent_eval, log_eval_metrics


def compute_attention_distillation_loss(model, input_ids, attention_mask, distillation_layers, device):
    """
    Compute attention distillation loss by comparing segmented attention outputs
    with and without flex attention optimizations or neural memory features.
    """
    total_distill_loss = 0.0
    num_layers = len(distillation_layers)
    
    if num_layers == 0:
        return torch.tensor(0.0, device=device)
    
    # Get the base model (unwrap from DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model
    if not hasattr(base_model, 'model'):
        return torch.tensor(0.0, device=device)
    
    transformer_layers = base_model.model.layers
    
    # Run a forward pass to get intermediate hidden states
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = base_model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # Store hidden states for distillation layers
        layer_hidden_states = {}
        
        for layer_idx, layer in enumerate(transformer_layers):
            if layer_idx in distillation_layers:
                layer_hidden_states[layer_idx] = hidden_states.clone()
            
            # Forward through layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
    
    # Now compute distillation loss for each specified layer
    for layer_idx in distillation_layers:
        if layer_idx >= len(transformer_layers) or layer_idx not in layer_hidden_states:
            continue
            
        layer = transformer_layers[layer_idx]
        layer_input = layer_hidden_states[layer_idx]
        
        # Get the segmented attention module
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'segmented_attn'):
            attn_module = layer.self_attn.segmented_attn
            
            # Normalize input (as done in the layer)
            normalized_input = layer.input_layernorm(layer_input)
            
            # Target: attention without flex attention (more basic implementation)
            with torch.no_grad():
                attn_target, _ = attn_module(
                    normalized_input,
                    disable_flex_attn=True,
                    value_residual=None
                )
            
            # Student: attention with flex attention enabled
            attn_output, _ = attn_module(
                normalized_input, 
                disable_flex_attn=False,
                value_residual=None
            )
            
            # Compute MSE loss between outputs
            layer_distill_loss = F.mse_loss(attn_output, attn_target.detach())
            total_distill_loss += layer_distill_loss
    
    return total_distill_loss / max(num_layers, 1)


@dataclass
class TrainingConfig:
    """Configuration for TitanLLaMA training."""
    
    # Model configuration
    model_name: str = "titan-llama-1b"
    vocab_size: int = 32000
    hidden_size: int = 2048  # Smaller for 1B param model
    intermediate_size: int = 5504
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 2048
    
    # Titan-specific configuration
    segment_len: int = 512
    num_persist_mem_tokens: int = 4
    num_longterm_mem_tokens: int = 4
    neural_memory_layers: tuple = (4, 8, 12, 16, 20)  # Add memory to multiple layers
    neural_memory_segment_len: int = 64
    neural_memory_batch_size: int = 64
    neural_memory_depth: int = 2
    use_flex_attn: bool = True
    sliding_window_attn: bool = False
    neural_mem_gate_attn_output: bool = False
    neural_mem_weight_residual: bool = True
    
    # Training configuration
    total_tokens: int = 1_000_000_000  # 1B tokens
    batch_size: int = 4
    micro_batch_size: int = 1
    sequence_length: int = 2048
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Memory and neural memory specific
    neural_mem_learning_rate: float = 1e-2  # Higher LR for neural memory
    neural_mem_momentum: float = 0.9
    neural_mem_weight_decay: float = 0.01
    
    # Attention distillation
    use_attention_distillation: bool = False
    distillation_weight: float = 0.1  # Weight for distillation loss vs LM loss
    distillation_layers: tuple = (8, 16, 24)  # Which layers to apply distillation to
    
    # Optimization
    warmup_steps: int = 2000
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    
    # Intermittent evaluation during training
    intermittent_eval_frequency: int = 15  # Evaluate every 15 steps
    intermittent_eval_limit: int = 200     # Test on 200 BoolQ questions
    intermittent_eval_start_step: int = 50  # Start evaluating after 50 steps
    
    # Data
    dataset_name: str = "cerebras/SlimPajama-627B"
    tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B"  # align with pretrained backbone by default
    num_proc: int = 8
    
    # Distributed training
    use_ddp: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Logging and checkpointing
    output_dir: str = "./titan_llama_checkpoints"
    wandb_project: str = "titan-llama-training"
    wandb_run_name: str = None
    log_level: str = "INFO"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Pretrained backbone
    use_pretrained_backbone: bool = True
    base_model_name: Optional[str] = "meta-llama/Meta-Llama-3.1-8B"
    freeze_backbone: bool = True
    
    def __post_init__(self):
        # Calculate total training steps
        self.tokens_per_batch = self.batch_size * self.sequence_length
        self.total_steps = self.total_tokens // self.tokens_per_batch
        
        # Gradient accumulation
        self.gradient_accumulation_steps = max(1, self.batch_size // self.micro_batch_size)
        self.effective_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        
        if self.wandb_run_name is None:
            self.wandb_run_name = f"{self.model_name}-{int(time.time())}"



def setup_logging(config: TrainingConfig):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def setup_distributed(config: TrainingConfig):
    """Set up distributed training if specified."""
    if config.use_ddp:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            config.local_rank = int(os.environ['RANK'])
            config.world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(config.local_rank)
        dist.init_process_group(backend='nccl')
    
    return config.local_rank == 0 or not config.use_ddp


def create_model_and_optimizer(config: TrainingConfig, device):
    """Create TitanLLaMA model and optimizer."""
    
    # Create model config
    model_config = TitanLLaMAConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        segment_len=config.segment_len,
        num_persist_mem_tokens=config.num_persist_mem_tokens,
        num_longterm_mem_tokens=config.num_longterm_mem_tokens,
        neural_memory_layers=config.neural_memory_layers,
        neural_memory_segment_len=config.neural_memory_segment_len,
        neural_memory_batch_size=config.neural_memory_batch_size,
        neural_memory_depth=config.neural_memory_depth,
        use_flex_attn=config.use_flex_attn,
        sliding_window_attn=config.sliding_window_attn,
        neural_mem_gate_attn_output=config.neural_mem_gate_attn_output,
        neural_mem_weight_residual=config.neural_mem_weight_residual,
        use_pretrained_backbone=config.use_pretrained_backbone,
        base_model_name_or_path=config.base_model_name,
        freeze_backbone=config.freeze_backbone,
    )
    
    # Create model
    if config.use_pretrained_backbone and config.base_model_name:
        model = TitanLLaMAForCausalLM.from_pretrained_llama(
            base_model_name_or_path=config.base_model_name,
            titan_config=model_config,
            freeze_backbone=config.freeze_backbone,
            device_map="cpu",
            dtype=torch.bfloat16,
        ).eval()
    else:
        model = TitanLLaMAForCausalLM(model_config)
        # model = model.to(device)
    
    # Align training config with backbone in case it was derived from a pretrained checkpoint
    config.hidden_size = model.model.config.hidden_size
    config.intermediate_size = model.model.config.intermediate_size
    config.num_hidden_layers = model.model.config.num_hidden_layers
    config.num_attention_heads = model.model.config.num_attention_heads
    config.num_key_value_heads = model.model.config.num_key_value_heads
    config.vocab_size = model.model.config.vocab_size
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    
    # Separate neural memory parameters for different optimization
    neural_memory_params = []
    regular_params = []
    
    from collections import defaultdict
    bucket_counts = defaultdict(int)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "neural_memory.memory_model_parameters" in name:
            bucket_counts["nm_inner_model"] += p.numel()
        elif any(x in name for x in [".to_keys", ".to_values", ".to_adaptive_step",
                                    ".to_momentum", ".to_decay_factor",
                                    ".to_layer_modulation", ".to_learned_weight_residual_mix"]):
            bucket_counts["nm_write_side"] += p.numel()
        elif "neural_memory" in name:
            bucket_counts["nm_read_side"] += p.numel()
        elif "persistent_memory" in name:
            bucket_counts["persistent_memory"] += p.numel()
        else:
            bucket_counts["other"] += p.numel()

        print("\n[create_model_and_optimizer] Trainable param breakdown:")
        for k, v in bucket_counts.items():
            print(f"  - {k:18s}: {v:,} ({v/1e6:.2f}M)")

        # Separate neural memory parameters for different optimization
        neural_memory_params = []
        regular_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'neural_memory' in name:
                neural_memory_params.append(param)
            else:
                regular_params.append(param)

        print(f"\n[create_model_and_optimizer] #trainable nm params: {sum(p.numel() for p in neural_memory_params):,}")
        print(f"[create_model_and_optimizer] #trainable non-nm params: {sum(p.numel() for p in regular_params):,}\n")
    
    # Create optimizers
    optimizer_groups = []

    if regular_params:
        optimizer_groups.append({
            'params': regular_params,
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            'betas': (config.beta1, config.beta2)
        })
    
    if neural_memory_params:
        optimizer_groups.append({
            'params': neural_memory_params,
            'lr': config.neural_mem_learning_rate,
            'weight_decay': config.neural_mem_weight_decay,
            'betas': (config.neural_mem_momentum, config.beta2)
        })
        print(f"Neural memory parameters: {sum(p.numel() for p in neural_memory_params):,}")
    
    if not optimizer_groups:
        raise ValueError("No trainable parameters were found. Ensure backbone freezing is configured correctly.")

    optimizer = AdamW(optimizer_groups)

    warmup_steps = int(config.total_steps * 0.1)
    decay_steps = config.total_steps - warmup_steps

    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-10,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=decay_steps, 
        eta_min=config.min_learning_rate
    )

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    
    
    # Wrap with DDP if using distributed training
    if config.use_ddp:
        model = DDP(model, device_ids=[config.local_rank])
    
    return model, optimizer, scheduler


def save_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    config: TrainingConfig, 
    step: int, 
    loss: float,
    is_best: bool = False
):
    """Save model checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.module.state_dict() if config.use_ddp else model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        'config': config.__dict__,
        'step': step,
        'loss': loss,
    }
    
    # Save regular checkpoint
    # checkpoint_path = os.path.join(config.output_dir, f"checkpoint-{step}.pt")
    # torch.save(checkpoint, checkpoint_path)
    
    # # Save best checkpoint
    # if is_best:
    #     best_path = os.path.join(config.output_dir, "best_checkpoint.pt")
    #     torch.save(checkpoint, best_path)
    
    # Save latest checkpoint
    latest_path = os.path.join(config.output_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)
    
    print(f"Checkpoint saved: {latest_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str):
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['step'], checkpoint['loss']


def evaluate_model(model, eval_dataloader, device, max_eval_steps=100):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_steps = 0
    
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if step >= max_eval_steps:
                break
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0)
            if len(labels.shape) == 1: labels = labels.unsqueeze(0)
            
            # Reset memory states for each eval batch
            model.reset_memory_states()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            total_loss += loss.item()
            total_accuracy += outputs['correct']
            num_steps += 1
    
    model.train()

    return {
        'loss': total_loss / num_steps if num_steps > 0 else float('inf'),
        'acc': total_accuracy / num_steps if num_steps > 0 else float('inf'),
        }


def main(config=None):
    """Main training function."""
    
    if config is None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Train TitanLLaMA with optional attention distillation")
        parser.add_argument("--use-attention-distillation", action="store_true", 
                           help="Enable attention distillation loss")
        parser.add_argument("--distillation-weight", type=float, default=0.1,
                           help="Weight for distillation loss vs LM loss (default: 0.1)")
        parser.add_argument("--distillation-layers", nargs="+", type=int, default=[8, 16, 24],
                           help="Which layers to apply distillation to (default: 8 16 24)")
        
        args = parser.parse_args()
        
        # Create config with command-line overrides
        config = TrainingConfig(
            use_attention_distillation=args.use_attention_distillation,
            distillation_weight=args.distillation_weight,
            distillation_layers=tuple(args.distillation_layers)
        )
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(
        f"Backbone: {config.base_model_name} | use_pretrained={config.use_pretrained_backbone} | "
        f"freeze_backbone={config.freeze_backbone}"
    )
    
    # Log attention distillation settings
    if config.use_attention_distillation:
        logger.info(
            f"Attention Distillation ENABLED | Weight: {config.distillation_weight} | "
            f"Layers: {config.distillation_layers}"
        )
    else:
        logger.info("Attention Distillation DISABLED")
    
    # Set up distributed training
    is_main_process = setup_distributed(config)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    if is_main_process and wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__
        )

    # Create datasets
    logger.info("Creating datasets...")

    if 'pajama' in config.dataset_name:
        train_dataset = SlimPajamaDataset(
            dataset_name=config.dataset_name,
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            streaming=True,
            split="train",
            num_proc=config.num_proc
        )
        
        # For validation, we'll use a small subset
        eval_dataset = SlimPajamaDataset(
            dataset_name=config.dataset_name,
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            streaming=True,
            split="validation" if "validation" in ["train"] else "train",  # Use train split for now
            num_proc=config.num_proc
        )
    
    elif 'fineweb' in config.dataset_name:
        train_dataset = FineWebEduDataset(
            dataset_name=config.dataset_name,
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            streaming=True,
            split="train",
            num_proc=config.num_proc,
        )
        
        # For validation, we'll use a small subset
        eval_dataset = FineWebEduDataset(
            dataset_name=config.dataset_name,
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            streaming=True,
            split="validation" if "validation" in ["train"] else "train",  # Use train split for now
            num_proc=config.num_proc,
        )

    elif 'mixed' in config.dataset_name:
        train_dataset = MixedEvalDataset(
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            boolq_split="train", 
            winogrande_split="train",
        )

        eval_dataset = MixedEvalDataset(
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            boolq_split="validation" if "validation" in ["train"] else "train",  # Use train split for now 
            winogrande_split="validation" if "validation" in ["train"] else "train",  # Use train split for now
        )

    else: raise RuntimeError("Could not infer dataset")

    print("THIS IS THE MICRO BS FROM CFG: ", config.micro_batch_size)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,  # Streaming dataset
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with HF datasets
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with HF datasets
        pin_memory=True
    )

    # Create model and optimizer
    logger.info("Creating model and optimizer...")
    model, optimizer, scheduler = create_model_and_optimizer(config, device)
    
    # Create tokenizer for evaluation
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    # Resume from checkpoint if specified
    start_step = 0
    best_eval_loss = float('inf')
    
    if config.resume_from_checkpoint:
        start_step, _ = load_checkpoint(model, optimizer, scheduler, config.resume_from_checkpoint)
        logger.info(f"Resumed training from step {start_step}")
    
    # Training loop
    logger.info(f"Starting training for {config.total_steps} steps...")
    
    model.train()
    running_loss = 0.0
    running_lm_loss = 0.0
    running_distill_loss = 0.0
    running_accuracy = 0.0
    running_ppl = 0.0
    log_steps = 0
    
    data_iter = iter(train_dataloader)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        log_cuda_mem("start")
    
    for step in tqdm(range(start_step, config.total_steps), desc="Training", disable=not is_main_process):
        # if step == 0:  # just log for the first step
        #     log_cuda_mem("before forward")

        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_accuracy = 0.0
        epoch_ppl = 0.0
        
        # Gradient accumulation loop
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset iterator if we've exhausted the dataset
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
            
            input_ids = batch['input_ids'].to(device)
            if len(input_ids.shape) == 1: input_ids = input_ids.unsqueeze(0)
            labels = batch['labels'].to(device) 
            if len(labels.shape) == 1: labels = labels.unsqueeze(0)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

            # print("TRAIN SIZE: ", input_ids.shape)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            lm_loss = outputs['loss'] / config.gradient_accumulation_steps
            
            # Compute attention distillation loss if enabled
            distill_loss = torch.tensor(0.0, device=device)
            if config.use_attention_distillation:
                distill_loss = compute_attention_distillation_loss(
                    model, input_ids, attention_mask, config.distillation_layers, device
                )
                distill_loss = distill_loss / config.gradient_accumulation_steps
            
            # Combine losses
            total_loss = lm_loss + config.distillation_weight * distill_loss
            loss = total_loss

            # if step == 0:
            #     log_cuda_mem("after forward")
            
            # Backward pass
            loss.backward()
            epoch_loss += loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_distill_loss += distill_loss.item()
            epoch_ppl += outputs['ppl'].item()
            epoch_accuracy += outputs['correct'].item()

            # if step == 0:
            #     log_cuda_mem("after backward")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # if step == 0:
        #     log_cuda_mem("after optimizer.step")
        #     print("[CUDA] peak allocated:", torch.cuda.max_memory_allocated()/1024**3, "GiB")
        #     break
        
        # Reset memory states periodically to prevent memory leaks
        if step % 100 == 0:
            model.reset_memory_states()
        
        # Logging
        running_loss += epoch_loss
        running_lm_loss += epoch_lm_loss
        running_distill_loss += epoch_distill_loss
        running_accuracy += epoch_accuracy
        running_ppl += epoch_ppl
        log_steps += 1
        
        if step % config.log_interval == 0 and is_main_process:
            avg_loss = running_loss / log_steps
            avg_lm_loss = running_lm_loss / log_steps
            avg_distill_loss = running_distill_loss / log_steps
            avg_acc = running_accuracy / log_steps
            avg_ppl = running_ppl / log_steps
            lr = scheduler.get_last_lr()[0]
            
            log_msg = (
                f"Step {step}/{config.total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LM: {avg_lm_loss:.4f}"
            )
            
            if config.use_attention_distillation:
                log_msg += f" | Distill: {avg_distill_loss:.4f}"
                
            log_msg += f" | LR: {lr:.2e} | Tokens: {step * config.tokens_per_batch:,}"
            logger.info(log_msg)
            
            # Log to wandb
            if wandb and wandb.run:
                log_dict = {
                    'train/loss': avg_loss,
                    'train/lm_loss': avg_lm_loss,
                    'train/ppl': avg_ppl / config.gradient_accumulation_steps,
                    'train/accuracy': avg_acc / config.gradient_accumulation_steps,
                    'train/learning_rate': lr,
                    'train/step': step,
                    'train/tokens_processed': step * config.tokens_per_batch
                }
                
                if config.use_attention_distillation:
                    log_dict['train/distillation_loss'] = avg_distill_loss
                    
                wandb.log(log_dict)
            
            running_loss = 0.0
            running_lm_loss = 0.0
            running_distill_loss = 0.0
            running_accuracy = 0.0
            running_ppl = 0.0
            log_steps = 0
        
        # Evaluation
        if step % config.eval_interval == 0 and step > 0 and is_main_process:
            logger.info("Running evaluation...")
            ret_dict = evaluate_model(model, eval_dataloader, device)
            eval_loss, eval_acc = ret_dict['loss'], ret_dict['acc']
            
            logger.info(f"Eval loss: {eval_loss:.4f}")
            
            # Log eval results
            if wandb and wandb.run:
                wandb.log({
                    'eval/loss': eval_loss,
                    'eval/step': step
                })
            
            # Save best checkpoint
            is_best = eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = eval_loss
                logger.info(f"New best eval loss: {best_eval_loss:.4f}")
            
            if step % config.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, config, step, eval_loss, is_best)
        
        # Intermittent evaluation on BoolQ
        if should_run_intermittent_eval(step, config.intermittent_eval_frequency, config.intermittent_eval_start_step) and is_main_process:
            logger.info(f"Running intermittent evaluation at step {step}...")

            model.eval() 
            
            # Reset model memory states BEFORE evaluation
            try:
                if hasattr(model, 'reset_memory_states'):
                    model.reset_memory_states()
                elif hasattr(model, 'module') and hasattr(model.module, 'reset_memory_states'):
                    model.module.reset_memory_states()
            except Exception as e:
                logger.warning(f"Failed to reset memory states before eval: {str(e)}")

            eval_metrics = eval_winogrande_boolq(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_examples=config.intermittent_eval_limit,
                boolq_batch_size=config.micro_batch_size,
            )

            logger.info(
                f"[eval] step {step} "
                f"BoolQ acc={eval_metrics['boolq_acc']:.3f}, "
                f"Winogrande acc={eval_metrics['winogrande_acc']:.3f}"
            )
            log_eval_metrics(eval_metrics, step, logger, wandb)

            # Complete cleanup and state restoration
            # Force complete CUDA cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset memory states after evaluation
            if hasattr(model, 'reset_memory_states'):
                model.reset_memory_states()
            elif hasattr(model, 'module') and hasattr(model.module, 'reset_memory_states'):
                model.module.reset_memory_states()
                
            # Final memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            model.train()
        
        # Regular checkpointing
        elif step % config.save_interval == 0 and step > 0 and is_main_process:
            save_checkpoint(model, optimizer, scheduler, config, step, epoch_loss, False)
    
    # Final checkpoint
    if is_main_process:
        save_checkpoint(model, optimizer, scheduler, config, config.total_steps, epoch_loss, False)
        logger.info("Training completed!")
        
        # Final evaluation
        ret_dict = evaluate_model(model, eval_dataloader, device)
        final_eval_loss, final_eval_acc = ret_dict['loss'], ret_dict['acc']
        logger.info(f"Final eval loss: {final_eval_loss:.4f}")
        
        if wandb and wandb.run:
            wandb.log({
                'eval/final_loss': final_eval_loss,
                'eval/final_acc': final_eval_acc,
                'train/final_step': config.total_steps
            })
            wandb.finish()


if __name__ == "__main__":
    main()
