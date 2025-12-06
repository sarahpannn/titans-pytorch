#!/usr/bin/env python3
"""
Domain-specific NeuralMemory finetuning.

Trains only the Titans NeuralMemory parameters on a downstream domain dataset
while keeping the frozen LLaMA backbone and segmented-attention adapter fixed.
Useful for experiments to test whether memory modules cnan learn information relevant to specific downstream domains.

"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM


# ---------------------------------------------------------------------------
# Domain templates
# ---------------------------------------------------------------------------

def _join_options(option_dict):
    """Turn a dict of options into a readable A/B/C/D string."""
    letters = "ABCD"
    options = []
    for i, key in enumerate(["option_a", "option_b", "option_c", "option_d", "opa", "opb", "opc", "opd"]):
        if key in option_dict and option_dict[key]:
            options.append(f"({letters[i % len(letters)]}) {option_dict[key]}")
    return " ".join(options)


def medmcqa_template(example: Dict) -> str:
    opts = {
        "option_a": example.get("opa"),
        "option_b": example.get("opb"),
        "option_c": example.get("opc"),
        "option_d": example.get("opd"),
    }
    options = _join_options(opts)
    answer = example.get("answer") or example.get("cop")
    try:
        # Convert numeric label into the matching option text if possible
        idx = int(answer)
        answer_text = list(filter(None, opts.values()))[idx] if idx < len(list(filter(None, opts.values()))) else str(answer)
    except Exception:
        answer_text = str(answer)
    return f"Question: {example.get('question', '')}\nOptions: {options}\nCorrect answer: {answer_text}"


def casehold_template(example: Dict) -> str:
    # LexGLUE case_hold has fields: context, question, candidates (list), label (int)
    context = example.get("context", "")
    question = example.get("question", "")
    candidates = example.get("candidates") or example.get("answers") or []
    label = example.get("label", -1)
    labeled_answer = candidates[label] if isinstance(label, int) and label >= 0 and label < len(candidates) else ""
    option_str = " ".join([f"({chr(65+i)}) {cand}" for i, cand in enumerate(candidates)])
    return f"Case context: {context}\nQuestion: {question}\nOptions: {option_str}\nCorrect answer: {labeled_answer}"


def gsm8k_template(example: Dict) -> str:
    question = example.get("question", "")
    answer = example.get("answer", "")
    return f"Problem: {question}\nSolution: {answer}"


def generic_template(example: Dict) -> str:
    # Fallback if we do not know the schema
    parts = [f"{k}: {v}" for k, v in example.items() if isinstance(v, (str, int, float))]
    return "\n".join(parts)


# Mapping from high-level domain names to datasets and templates
DOMAIN_SPECS: Dict[str, Dict] = {
    "biomed": {
        "dataset": "openlifescienceai/medmcqa",
        "subset": None,
        "template": medmcqa_template,
    },
    "legal": {
        "dataset": "coastalcph/lex_glue",
        "subset": "case_hold",
        "template": casehold_template,
    },
    "math": {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "template": gsm8k_template,
    },
}


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

class DomainSequenceDataset(Dataset):
    """Tokenizes examples for next-token prediction using a domain-specific template."""

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        domain: str,
        split: str = "train",
    ):
        if domain not in DOMAIN_SPECS:
            raise ValueError(f"Unknown domain '{domain}'. Available: {list(DOMAIN_SPECS.keys())}")

        spec = DOMAIN_SPECS[domain]
        self.template_fn: Callable[[Dict], str] = spec["template"]
        self.dataset = load_dataset(spec["dataset"], spec["subset"], split=split)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw = self.dataset[idx]
        text = self.template_fn(raw) if self.template_fn else generic_template(raw)
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training/eval utilities
# ---------------------------------------------------------------------------

@dataclass
class DomainMemoryConfig:
    domain: str
    tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B"
    base_model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    base_checkpoint: Optional[str] = "./titan_llama_checkpoints/best_checkpoint.pt"
    output_dir: str = "./domain_nmm"
    max_length: int = 1024
    batch_size: int = 2
    eval_batch_size: int = 2
    num_epochs: int = 1
    steps_per_epoch: Optional[int] = None  # if set, limit updates per epoch
    learning_rate: float = 1e-2  # higher LR for NeuralMemory
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Titan overrides (only used when no checkpoint is provided)
    segment_len: int = 512
    neural_memory_segment_len: int = 64
    neural_memory_batch_size: Optional[int] = 64


def _titan_config_from_checkpoint(checkpoint_cfg: Dict, cli_cfg: DomainMemoryConfig) -> TitanLLaMAConfig:
    """Rehydrate Titan config from a training checkpoint dict."""
    # Use stored training config when available, otherwise fall back to defaults
    def _get(key, default):
        return checkpoint_cfg.get(key, default)

    return TitanLLaMAConfig(
        vocab_size=_get("vocab_size", 32000),
        hidden_size=_get("hidden_size", 4096),
        intermediate_size=_get("intermediate_size", 11008),
        num_hidden_layers=_get("num_hidden_layers", 32),
        num_attention_heads=_get("num_attention_heads", 32),
        num_key_value_heads=_get("num_key_value_heads", _get("num_attention_heads", 32)),
        max_position_embeddings=_get("max_position_embeddings", 2048),
        segment_len=_get("segment_len", 512),
        num_persist_mem_tokens=_get("num_persist_mem_tokens", 4),
        num_longterm_mem_tokens=_get("num_longterm_mem_tokens", 4),
        neural_memory_layers=tuple(_get("neural_memory_layers", (8, 16, 24))),
        neural_memory_segment_len=_get("neural_memory_segment_len", 16),
        neural_memory_batch_size=_get("neural_memory_batch_size", 8),
        neural_memory_depth=_get("neural_memory_depth", 2),
        use_flex_attn=_get("use_flex_attn", True),
        sliding_window_attn=_get("sliding_window_attn", True),
        neural_mem_gate_attn_output=_get("neural_mem_gate_attn_output", False),
        neural_mem_weight_residual=_get("neural_mem_weight_residual", True),
        use_pretrained_backbone=_get("use_pretrained_backbone", True),
        base_model_name_or_path=_get("base_model_name", cli_cfg.base_model_name),
        freeze_backbone=True,
    )


def load_domain_model(cfg: DomainMemoryConfig) -> TitanLLaMAForCausalLM:
    """
    Load Titan-LLaMA either from a prior checkpoint (preferred) or directly from HF weights.
    Only NeuralMemory parameters will be left trainable.
    """
    model = None

    if cfg.base_checkpoint and os.path.exists(cfg.base_checkpoint):
        print(f"Loading base checkpoint from {cfg.base_checkpoint}")
        ckpt = torch.load(cfg.base_checkpoint, map_location="cpu")
        ckpt_cfg = ckpt.get("config", {})
        titan_cfg = _titan_config_from_checkpoint(ckpt_cfg, cfg)
        model = TitanLLaMAForCausalLM(titan_cfg)
        state = ckpt.get("model_state_dict") or ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading checkpoint: {unexpected}")
    else:
        print("No checkpoint provided; loading frozen backbone directly from HF.")
        base_cfg = AutoConfig.from_pretrained(cfg.base_model_name)
        nm_batch_size = cfg.neural_memory_batch_size
        if isinstance(nm_batch_size, int) and nm_batch_size <= 0:
            nm_batch_size = None

        titan_cfg = TitanLLaMAConfig.from_llama_config(
            base_cfg,
            segment_len=cfg.segment_len,
            neural_memory_segment_len=cfg.neural_memory_segment_len,
            neural_memory_batch_size=nm_batch_size,
        )
        model = TitanLLaMAForCausalLM.from_pretrained_llama(
            base_model_name_or_path=cfg.base_model_name,
            titan_config=titan_cfg,
            freeze_backbone=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        # Reset default device to CPU so dataloaders and torch.randperm stay on CPU
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

    # Freeze everything except NeuralMemory parameters
    for name, param in model.named_parameters():
        param.requires_grad = "neural_memory" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable (NeuralMemory-only): {trainable:,} / {total:,} parameters")

    return model.to(cfg.device)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            model.reset_memory_states()
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            total_acc += outputs.get("correct", 0.0)
            steps += 1
    model.train()
    return {
        "loss": total_loss / max(steps, 1),
        "ppl": math.exp(total_loss / max(steps, 1)) if steps > 0 else float("inf"),
        "acc": (total_acc / max(steps, 1)).item() if steps > 0 else 0.0,
    }


def train_domain_memory(cfg: DomainMemoryConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = load_domain_model(cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_ds = DomainSequenceDataset(
        tokenizer_name=cfg.tokenizer_name,
        max_length=cfg.max_length,
        domain=cfg.domain,
        split="train",
    )

    # Try validation, fall back to test/train as needed
    for candidate_split in ["validation", "test", "train"]:
        try:
            eval_ds = DomainSequenceDataset(
                tokenizer_name=cfg.tokenizer_name,
                max_length=cfg.max_length,
                domain=cfg.domain,
                split=candidate_split,
            )
            break
        except Exception:
            eval_ds = None
            continue
    if eval_ds is None:
        raise RuntimeError(f"Could not create evaluation dataset for domain {cfg.domain}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.eval_batch_size, shuffle=False)

    global_step = 0
    model.train()

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch+1}/{cfg.num_epochs}")
        for step, batch in enumerate(train_loader):
            if cfg.steps_per_epoch and step >= cfg.steps_per_epoch:
                break

            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            model.reset_memory_states()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if global_step % 20 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

            if cfg.eval_every and global_step > 0 and global_step % cfg.eval_every == 0:
                metrics = evaluate(model, eval_loader, cfg.device)
                print(f"[eval] step {global_step} | loss {metrics['loss']:.4f} | ppl {metrics['ppl']:.2f} | acc {metrics['acc']:.4f}")
                save_path = os.path.join(cfg.output_dir, f"{cfg.domain}-nmm-step{global_step}.pt")
                torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, save_path)
                print(f"Saved checkpoint to {save_path}")

            global_step += 1

    # Final checkpoint
    final_path = os.path.join(cfg.output_dir, f"{cfg.domain}-nmm-final.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, final_path)
    print(f"Finished training. Final checkpoint: {final_path}")


def parse_args() -> DomainMemoryConfig:
    parser = argparse.ArgumentParser(description="Train domain-specific NeuralMemory adapter.")
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_SPECS.keys()))
    parser.add_argument("--tokenizer_name", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--base_model_name", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--base_checkpoint", default="./titan_llama_checkpoints/best_checkpoint.pt")
    parser.add_argument("--output_dir", default="./domain_nmm")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--neural_memory_segment_len", type=int, default=64)
    parser.add_argument("--neural_memory_batch_size", type=int, default=64)
    args = parser.parse_args()
    return DomainMemoryConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train_domain_memory(cfg)
