"""Training-only Titan-Llama route for the archived 512/1024/2048 runs."""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from fp32_adamw import FP32AdamW
from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM, _remap_checkpoint_keys
from train_datasets import MixedLongAlignFineWebPretrainDataset, MixedLongAlignLoongRLSFTDataset


@dataclass
class TrainingConfig:
    model_name: str = 'titan-llama-1b'
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 2048
    segment_len: int = 512
    neural_memory_layers: tuple = (4, 8, 12, 16, 20)
    neural_memory_segment_len: int = 512
    neural_memory_batch_size: int = 512
    neural_memory_depth: int = 2
    neural_memory_expansion_factor: float = 1.0
    neural_memory_activation: str = 'gelu'
    detach_inner_grads: bool = True
    use_flex_attn: bool = True
    use_flash_attn: bool = False
    memory_identity_init: bool = False
    memory_qk_rope: bool = True
    history_blend: bool = False
    history_mix_init: float = 0.1
    crop_trailing_padding: bool = False
    lm_logit_chunk_size: int = 0
    total_tokens: int = 1000000000
    num_epochs: int = 1
    batch_size: int = 4
    micro_batch_size: int = 1
    sequence_length: int = 2048
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0003
    min_learning_rate: float = 0.001 * learning_rate
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    neural_mem_learning_rate: float = 0.01
    min_neural_mem_learning_rate: float = 1e-06
    neural_mem_warmup_start_learning_rate: float = 0.0
    neural_memory_inner_learning_rate: float = 0.0025
    neural_mem_momentum: float = 0.9
    neural_mem_weight_decay: float = 0.01
    neural_mem_schedule: bool = False
    warmup_steps: int = 2000
    save_interval: int = 5000
    log_interval: int = 100
    dataset_name: str = 'zai-org/LongAlign-10k'
    dataset_config_name: str = ''
    dataset_max_examples: Optional[int] = None
    loongrl_memory_forced_query_tokens: int = 0
    longalign_sft_max_examples: Optional[int] = None
    quality_sft_max_examples: Optional[int] = None
    tokenizer_name: str = 'unsloth/Llama-3.2-1B-Instruct'
    longalign_shuffle_seed: int = 42
    fast_resume_data: bool = False
    mixed_pretrain_fineweb_shuffle_buffer_size: int = 1024
    use_ddp: bool = False
    local_rank: int = -1
    global_rank: int = 0
    world_size: int = 1
    output_dir: str = './titan_llama_checkpoints'
    wandb_project: str = 'titan-llama-training'
    wandb_run_name: str = None
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    log_level: str = 'INFO'
    resume_from_checkpoint: Optional[str] = None
    pretrained_from_checkpoint: Optional[str] = None
    use_pretrained_backbone: bool = True
    base_model_name: Optional[str] = 'unsloth/Llama-3.2-1B-Instruct'
    freeze_backbone: bool = True
    model_dtype: str = 'bfloat16'

    def __post_init__(self):
        if self.sequence_length <= 0:
            raise ValueError(f'sequence_length must be positive, got {self.sequence_length}')
        if self.num_epochs < 1:
            raise ValueError(f'num_epochs must be at least 1, got {self.num_epochs}')
        if self.grad_clip < 0:
            raise ValueError(f'grad_clip must be non-negative (0 disables outer clipping), got {self.grad_clip}')
        if self.neural_memory_inner_learning_rate <= 0:
            raise ValueError(f'neural_memory_inner_learning_rate must be positive, got {self.neural_memory_inner_learning_rate}')
        if self.neural_mem_schedule and (not 0 < self.min_neural_mem_learning_rate <= self.neural_mem_learning_rate):
            raise ValueError(f'min_neural_mem_learning_rate must be positive and no greater than neural_mem_learning_rate; got {self.min_neural_mem_learning_rate} and {self.neural_mem_learning_rate}')
        if not 0 <= self.neural_mem_warmup_start_learning_rate <= self.neural_mem_learning_rate:
            raise ValueError(f'neural_mem_warmup_start_learning_rate must be between zero and neural_mem_learning_rate; got {self.neural_mem_warmup_start_learning_rate} and {self.neural_mem_learning_rate}')
        self.tokens_per_batch = self.batch_size * self.sequence_length
        self.total_steps = self.total_tokens // self.tokens_per_batch
        ddp_world = int(os.environ.get('WORLD_SIZE', '1')) if self.use_ddp else 1
        self.gradient_accumulation_steps = max(1, self.batch_size // (self.micro_batch_size * ddp_world))
        self.effective_batch_size = self.micro_batch_size * self.gradient_accumulation_steps * ddp_world
        if self.wandb_run_name is None:
            self.wandb_run_name = f'{self.model_name}-{int(time.time())}'


def setup_logging(config: TrainingConfig):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def _reset_memory_states(model):
    """Reset Titan neural-memory state, unwrapping DDP if present.

    DistributedDataParallel does not forward attribute lookups to the wrapped
    module, so `model.reset_memory_states()` raises AttributeError under DDP.
    No-op for models that do not define it.
    """
    target = model.module if isinstance(model, DDP) else model
    if hasattr(target, 'reset_memory_states'):
        target.reset_memory_states()


def setup_distributed(config: TrainingConfig):
    """Set up distributed training if specified.

    LOCAL_RANK indexes the GPU on this node; RANK is the global rank across all
    nodes.  They coincide on a single node but must not be conflated -- using
    RANK as a device index breaks multi-node runs.
    """
    if config.use_ddp:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            config.global_rank = int(os.environ['RANK'])
            config.local_rank = int(os.environ.get('LOCAL_RANK', os.environ['RANK']))
            config.world_size = int(os.environ['WORLD_SIZE'])

        torch.cuda.set_device(config.local_rank)
        dist.init_process_group(
            backend='nccl', timeout=datetime.timedelta(hours=2)
        )

    return config.global_rank == 0 or not config.use_ddp

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
        neural_memory_layers=config.neural_memory_layers,
        neural_memory_segment_len=config.neural_memory_segment_len,
        neural_memory_batch_size=config.neural_memory_batch_size,
        neural_memory_depth=config.neural_memory_depth,
        neural_memory_expansion_factor=config.neural_memory_expansion_factor,
        neural_memory_activation=config.neural_memory_activation,
        neural_memory_inner_learning_rate=config.neural_memory_inner_learning_rate,
        detach_inner_grads=config.detach_inner_grads,
        use_flex_attn=config.use_flex_attn,
        use_flash_attn=config.use_flash_attn,
        memory_identity_init=config.memory_identity_init,
        memory_qk_rope=config.memory_qk_rope,
        history_blend=config.history_blend,
        history_mix_init=config.history_mix_init,
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
            dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[config.model_dtype],
        )
    else:
        model = TitanLLaMAForCausalLM(model_config)

    model = model.to(device)

    # Align training config with backbone in case it was derived from a pretrained checkpoint
    hf_cfg = model.backbone.config
    config.hidden_size = hf_cfg.hidden_size
    config.intermediate_size = hf_cfg.intermediate_size
    config.num_hidden_layers = hf_cfg.num_hidden_layers
    config.num_attention_heads = hf_cfg.num_attention_heads
    config.num_key_value_heads = hf_cfg.num_key_value_heads
    config.vocab_size = hf_cfg.vocab_size

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
        elif "memory_token_selector" in name or "memory_value_to_hidden" in name:
            bucket_counts["nm_virtual_token_adapter"] += p.numel()
        elif "persistent_memory" in name:
            bucket_counts["persistent_memory"] += p.numel()
        else:
            bucket_counts["other"] += p.numel()

    for k, v in bucket_counts.items():
        print(f"  - {k:18s}: {v:,} ({v/1e6:.2f}M)")

        neural_memory_params = []
        regular_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                'neural_memory' in name
                or 'mem_to_kv' in name
                or 'memory_concat_proj' in name
                or 'memory_token_selector' in name
                or 'memory_value_to_hidden' in name
                or 'to_learned_v_mix' in name
            ):
                neural_memory_params.append(param)
            else:
                regular_params.append(param)

        # print(f"\n[create_model_and_optimizer] #trainable nm params: {sum(p.numel() for p in neural_memory_params):,}")
        # print(f"[create_model_and_optimizer] #trainable non-nm params: {sum(p.numel() for p in regular_params):,}\n")

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

    optimizer = FP32AdamW(optimizer_groups)
    print("AdamW: FP32 master weights and moments; model compute dtype unchanged")

    warmup_steps = int(config.warmup_steps)
    decay_steps = config.total_steps - warmup_steps
    total_steps = config.total_steps

    def regular_lr_lambda(step):
        """Linear warmup then cosine decay for regular parameters."""
        if step < warmup_steps:
            return 1e-10 + (1.0 - 1e-10) * step / warmup_steps
        progress = (step - warmup_steps) / max(decay_steps, 1)
        min_factor = config.min_learning_rate / config.learning_rate
        return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))

    def neural_mem_lr_lambda(step):
        """Linear warmup followed by cosine decay for neural-memory params."""
        if step < warmup_steps:
            start_factor = (
                config.neural_mem_warmup_start_learning_rate
                / config.neural_mem_learning_rate
            )
            return start_factor + (1.0 - start_factor) * step / max(warmup_steps, 1)
        progress = min((step - warmup_steps) / max(decay_steps, 1), 1.0)
        min_factor = config.min_neural_mem_learning_rate / config.neural_mem_learning_rate
        return min_factor + 0.5 * (1.0 - min_factor) * (1 + math.cos(math.pi * progress))

    def constant_lr_lambda(step):
        return 1.0

    # Build per-group lambda list matching optimizer group order
    lr_lambdas = []
    if regular_params:
        lr_lambdas.append(regular_lr_lambda)
    if neural_memory_params:
        lr_lambdas.append(neural_mem_lr_lambda if config.neural_mem_schedule else constant_lr_lambda)

    scheduler = LambdaLR(optimizer, lr_lambdas)
    
    
    # Wrap with DDP if using distributed training
    if config.use_ddp:
        model = DDP(
            model,
            device_ids=[config.local_rank],
            # launcher failed after its first backward because the reducer
            # expected gradients for those parameters.  Always discover the
            # safe; the graph walk is negligible next to a 16k NMM forward.
            find_unused_parameters=True,
        )
    
    return model, optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, config, step, loss):
    """Atomically replace the latest checkpoint in the original format."""
    os.makedirs(config.output_dir, exist_ok=True)
    target = model.module if isinstance(model, DDP) else model
    payload = {
        "model_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
    }
    path = os.path.join(config.output_dir, "latest_checkpoint.pt")
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if isinstance(model, DDP) else model
    state = _remap_checkpoint_keys(checkpoint["model_state_dict"], target)
    missing, unexpected = target.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if ".full_attn." not in key]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing}, unexpected={unexpected}")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    saved_step = int(checkpoint["step"])
    old_total = checkpoint.get("config", {}).get("total_steps")
    next_step = saved_step if old_total is not None and saved_step == int(old_total) else saved_step + 1
    return next_step


def load_pretrained_from_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = _remap_checkpoint_keys(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.getLogger(__name__).warning(
            "weight transfer: %d missing, %d unexpected keys", len(missing), len(unexpected)
        )


def make_dataset(config):
    if config.dataset_name == "longalign_fineweb_pretrain":
        return MixedLongAlignFineWebPretrainDataset(
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            seed=config.longalign_shuffle_seed,
            fineweb_shuffle_buffer_size=config.mixed_pretrain_fineweb_shuffle_buffer_size,
        )
    if config.dataset_name == "quality_mc_sft":
        from sft_kvwrite_quality_mc import QualityMCSFTDataset
        return QualityMCSFTDataset(
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            quality_examples=config.quality_sft_max_examples or 0,
        )
    if config.dataset_name == "longalign_loongrl_sft":
        return MixedLongAlignLoongRLSFTDataset(
            tokenizer_name=config.tokenizer_name,
            max_length=config.sequence_length,
            longalign_examples=config.longalign_sft_max_examples or 9984,
            loongrl_examples=config.dataset_max_examples or 2496,
            loongrl_config_name=config.dataset_config_name or "hotpotqa_qwen_0_2500",
            loongrl_memory_forced_query_tokens=config.loongrl_memory_forced_query_tokens,
            seed=config.longalign_shuffle_seed,
        )
    raise ValueError(f"dataset outside the selected training route: {config.dataset_name}")


def main(config: TrainingConfig):
    logger = setup_logging(config)
    is_main = setup_distributed(config)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device(f"cuda:{config.local_rank}" if config.use_ddp else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if is_main and os.environ.get("WANDB_MODE", "").lower() != "disabled":
        run_id = hashlib.sha1(config.wandb_run_name.encode()).hexdigest()[:32]
        wandb.init(project=config.wandb_project, name=config.wandb_run_name,
                   group=config.wandb_group, job_type=config.wandb_job_type,
                   id=run_id, resume="allow", config=config.__dict__)
    dataset = make_dataset(config)
    if not isinstance(dataset, torch.utils.data.IterableDataset):
        raise TypeError("Selected route expects a streaming dataset")
    loader = DataLoader(dataset, batch_size=config.micro_batch_size,
                        num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True)
    model, optimizer, scheduler = create_model_and_optimizer(config, device)
    start_step = 0
    if config.resume_from_checkpoint:
        start_step = load_checkpoint(model, optimizer, scheduler, config.resume_from_checkpoint)
    elif config.pretrained_from_checkpoint:
        target = model.module if isinstance(model, DDP) else model
        load_pretrained_from_checkpoint(target, config.pretrained_from_checkpoint)
    if start_step and config.fast_resume_data and isinstance(dataset, MixedLongAlignFineWebPretrainDataset):
        seed = config.longalign_shuffle_seed + start_step
        dataset.seed = seed
        dataset.longalign.seed = seed
        dataset.longalign._stream_epoch = 0
        dataset.fineweb.seed = seed
    iterator = iter(loader)
    if start_step and not (config.fast_resume_data and isinstance(dataset, MixedLongAlignFineWebPretrainDataset)):
        for _ in range(start_step * config.gradient_accumulation_steps):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(loader)
                next(iterator)
    model.train()
    optimizer.zero_grad()
    last_loss = float("nan")
    for step in tqdm(range(start_step, config.total_steps), disable=not is_main):
        start = time.perf_counter()
        loss_sum = accuracy_sum = 0.0
        for micro in range(config.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            if config.crop_trailing_padding and attention_mask.shape[1] > config.segment_len:
                if bool((attention_mask[:, 1:] <= attention_mask[:, :-1]).all()):
                    longest = int(attention_mask.sum(dim=1).max())
                    kept = min(attention_mask.shape[1],
                               -(-max(longest, 1) // config.segment_len) * config.segment_len)
                    input_ids = input_ids[:, :kept]
                    labels = labels[:, :kept]
                    attention_mask = attention_mask[:, :kept]
            _reset_memory_states(model)
            sync = nullcontext() if micro == config.gradient_accumulation_steps - 1 or not isinstance(model, DDP) else model.no_sync()
            with sync:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels, output_hidden_states=False,
                                lm_logit_chunk_size=config.lm_logit_chunk_size)
                loss = outputs["loss"] / config.gradient_accumulation_steps
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite loss at step {step}, microbatch {micro}")
                loss.backward()
            loss_sum += float(loss.detach())
            accuracy_sum += float(outputs["correct"].detach())
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        last_loss = loss_sum
        if is_main and step % config.log_interval == 0:
            logger.info("step=%d/%d loss=%.4f accuracy=%.4f lr=%.3e seconds=%.2f",
                        step, config.total_steps, loss_sum,
                        accuracy_sum / config.gradient_accumulation_steps,
                        scheduler.get_last_lr()[0], time.perf_counter() - start)
            if wandb.run:
                wandb.log({"train/loss": loss_sum,
                           "train/accuracy": accuracy_sum / config.gradient_accumulation_steps,
                           "train/step": step})
        if is_main and (step + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, config, step, last_loss)
        if config.use_ddp:
            dist.barrier()
    if is_main:
        save_checkpoint(model, optimizer, scheduler, config, config.total_steps, last_loss)
        if wandb.run:
            wandb.finish()
    if config.use_ddp:
        dist.barrier()
        dist.destroy_process_group()
