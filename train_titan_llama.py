#!/usr/bin/env python3
"""
Training script for TitanLLaMA on SlimPajama dataset.
Trains on 1B tokens with segmented attention and neural memory.
"""

import os
import json
import math
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm

# Import our TitanLLaMA implementation
from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM
from train_datasets import SlimPajamaDataset, FineWebEduDataset
from cuda_utils import log_cuda_mem


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
    
    # Optimization
    warmup_steps: int = 2000
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    
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
    
    print(f"Checkpoint saved: {checkpoint_path}")


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


def main():
    """Main training function."""
    
    # Parse config (in real script, you'd use argparse or hydra)
    config = TrainingConfig()
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(
        f"Backbone: {config.base_model_name} | use_pretrained={config.use_pretrained_backbone} | "
        f"freeze_backbone={config.freeze_backbone}"
    )
    
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

    # Create model and optimizer
    logger.info("Creating model and optimizer...")
    model, optimizer, scheduler = create_model_and_optimizer(config, device)
    
    
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

    else: raise RuntimeError("Could not infer dataset")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,  # Streaming dataset
        num_workers=4,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
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
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss'] / config.gradient_accumulation_steps

            # if step == 0:
            #     log_cuda_mem("after forward")
            
            # Backward pass
            loss.backward()
            epoch_loss += loss.item()
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
        running_accuracy += epoch_accuracy
        running_ppl += epoch_ppl
        log_steps += 1
        
        if step % config.log_interval == 0 and is_main_process:
            avg_loss = running_loss / log_steps
            avg_acc = running_accuracy / log_steps
            avg_ppl = running_ppl / log_steps
            lr = scheduler.get_last_lr()[0]
            
            logger.info(
                f"Step {step}/{config.total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tokens: {step * config.tokens_per_batch:,}"
            )
            
            # Log to wandb
            if wandb and wandb.run:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/ppl': avg_ppl / config.gradient_accumulation_steps,
                    'train/accuracy': avg_acc / config.gradient_accumulation_steps,
                    'train/learning_rate': lr,
                    'train/step': step,
                    'train/tokens_processed': step * config.tokens_per_batch
                })
            
            running_loss = 0.0
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
