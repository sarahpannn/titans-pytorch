#!/usr/bin/env python3
"""
Domain-specific NeuralMemory finetuning.

Trains only the Titans NeuralMemory parameters on a downstream domain dataset
while keeping the frozen LLaMA backbone and segmented-attention adapter fixed.
Useful for experiments to test whether memory modules can learn information
relevant to specific downstream domains.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any
from inspect import signature

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM


# ---------------------------------------------------------------------------
# Domain templates
# ---------------------------------------------------------------------------

def _join_options(option_dict):
    """Turn a dict of options into a readable A/B/C/D string."""
    letters = "ABCD"
    options = []
    for i, key in enumerate(
        ["option_a", "option_b", "option_c", "option_d", "opa", "opb", "opc", "opd"]
    ):
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
        idx = int(answer)
        vals = list(filter(None, opts.values()))
        answer_text = vals[idx] if idx < len(vals) else str(answer)
    except Exception:
        answer_text = str(answer)
    return (
        f"Question: {example.get('question', '')}\n"
        f"Options: {options}\n"
        f"Correct answer: {answer_text}"
    )


def casehold_template(example: Dict) -> str:
    # LexGLUE case_hold has fields: context, question, candidates (list), label (int)
    context = example.get("context", "")
    question = example.get("question", "")
    candidates = example.get("candidates") or example.get("answers") or []
    label = example.get("label", -1)
    labeled_answer = (
        candidates[label]
        if isinstance(label, int) and 0 <= label < len(candidates)
        else ""
    )
    option_str = " ".join(
        [f"({chr(65 + i)}) {cand}" for i, cand in enumerate(candidates)]
    )
    return (
        f"Case context: {context}\n"
        f"Question: {question}\n"
        f"Options: {option_str}\n"
        f"Correct answer: {labeled_answer}"
    )


def gsm8k_template(example: Dict) -> str:
    question = example.get("question", "")
    answer = example.get("answer", "")
    return f"Problem: {question}\nSolution: {answer}"


def _pubmedqa_context_to_text(ctx: Any) -> str:
    """
    PubMedQA 'context' field is often a dict with 'contexts' list.
    Fall back gracefully if it's already a string.
    """
    if isinstance(ctx, dict):
        contexts = ctx.get("contexts") or []
        if isinstance(contexts, list):
            return " ".join(str(c) for c in contexts)
        else:
            return str(contexts)
    return str(ctx)


def pubmedqa_template(example: Dict) -> str:
    """
    Training template for PubMedQA.

    We include the *final_decision* in the text, since training is LM-style.
    Evaluation for yes/no will instead stop before the answer token.
    """
    question = example.get("question", "")
    ctx_text = _pubmedqa_context_to_text(example.get("context", ""))
    decision = example.get("final_decision", "")
    return (
        f"Context: {ctx_text}\n"
        f"Question: {question}\n"
        f"Answer: {decision}"
    )


def generic_template(example: Dict) -> str:
    # Fallback if we do not know the schema
    parts = [f"{k}: {v}" for k, v in example.items()
             if isinstance(v, (str, int, float))]
    return "\n".join(parts)


# Mapping from high-level domain names to datasets and templates
#
# For PubMedQA we explicitly use:
#   - train   → pqa_artificial
#   - val/test→ pqa_labeled (so eval is on labeled questions)
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
    "pubmedqa": {
        "dataset": "qiaojin/PubMedQA",
        # Config per split
        "subset": {
            "train": "pqa_artificial",
            "validation": "pqa_labeled",
            "test": "pqa_labeled",
        },
        "template": pubmedqa_template,
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
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {list(DOMAIN_SPECS.keys())}"
            )

        spec = DOMAIN_SPECS[domain]
        self.template_fn: Callable[[Dict], str] = spec["template"]

        subset_cfg = spec.get("subset")
        if isinstance(subset_cfg, dict):
            hf_subset = subset_cfg.get(split, None)
        else:
            hf_subset = subset_cfg

        # hf_subset may be None (default config)
        if hf_subset is None:
            self.dataset = load_dataset(spec["dataset"], split=split)
        else:
            self.dataset = load_dataset(spec["dataset"], hf_subset, split=split)

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

    # Optimizer / training
    learning_rate: float = 3e-4          # base LR (non-memory)
    neural_mem_lr: float = 1e-5          # separate LR for NeuralMemory
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Gradient accumulation (per optimizer step)
    gradient_accumulation_steps: int = 1

    # Attention distillation
    use_attention_distillation: bool = False
    distillation_weight: float = 0.0  # 0 disables distillation even if flag is on
    distillation_layers: tuple = (8, 16, 24)

    # Titan overrides (only used when no checkpoint is provided)
    segment_len: int = 512
    neural_memory_segment_len: int = 64
    neural_memory_batch_size: Optional[int] = 64


def _titan_config_from_checkpoint(checkpoint_cfg: Dict, cli_cfg: DomainMemoryConfig) -> TitanLLaMAConfig:
    """
    Rehydrate Titan config from a training checkpoint dict.

    We mostly want to reuse exactly what was used during LM training,
    and only override things that *must* change for this run (like
    base_model_name_or_path and freezing the backbone).
    """
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError(f"Checkpoint config is not a dict – got type {type(checkpoint_cfg)}")

    # Make a shallow copy so we don't mutate the original
    cfg_dict = dict(checkpoint_cfg)

    # --- normalize base model path across old/new checkpoints ---
    # Prefer explicit base_model_name_or_path, else fall back to older fields
    base_name = cfg_dict.get("base_model_name_or_path")
    if base_name is None:
        base_name = (
            cfg_dict.get("base_model_name")
            or cfg_dict.get("model_name")
            or cli_cfg.base_model_name
        )
    cfg_dict["base_model_name_or_path"] = base_name

    # For domain-memory finetuning we always want a frozen backbone
    cfg_dict["freeze_backbone"] = True

    # --- keep only the fields TitanLLaMAConfig actually accepts ---
    allowed_keys = set(signature(TitanLLaMAConfig).parameters.keys())
    allowed_keys.discard("self")

    cfg_filtered = {k: v for k, v in cfg_dict.items() if k in allowed_keys}

    dropped = set(cfg_dict.keys()) - set(cfg_filtered.keys())
    if dropped:
        pass
        # print("Dropping unused config fields from checkpoint:", dropped)

    titan_cfg = TitanLLaMAConfig(**cfg_filtered)
    return titan_cfg


def load_domain_model(cfg: DomainMemoryConfig) -> TitanLLaMAForCausalLM:
    """
    Load Titan-LLaMA either from a prior checkpoint (preferred) or directly from HF weights.
    Only NeuralMemory parameters will be left trainable.
    """
    if cfg.base_checkpoint and os.path.exists(cfg.base_checkpoint):
        # ---- use Titan's built-in loader for LM checkpoints ----
        print(f"Loading base checkpoint from {cfg.base_checkpoint}")
        device = cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # This uses titan_llama.TitanLLaMAForCausalLM.from_pretrained
        # which:
        #   - torch.load(..., map_location=device)
        #   - rebuilds Titan config from ckpt["config"]
        #   - loads model_state_dict
        #   - moves model to (dtype, device)
        model = TitanLLaMAForCausalLM.from_pretrained(
            checkpoint_path=cfg.base_checkpoint,
            base_model_name_or_path=cfg.base_model_name,
            dtype=dtype,
            device=device,
            strict=False,
        )

    else:
        # ---- fallback: build Titan around frozen HF LLaMA directly ----
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
        # Keep dataloader internals on CPU to avoid generator/device mismatch
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

def compute_attention_distillation_loss(
    student_model_hiddens,
    teacher_model,
    input_ids,
    attention_mask,
    distillation_layers,
    device,
):
    """
    Distill from a full-attention *teacher backbone* into a segmented-attention
    *student backbone* by supervising hidden state outputs at selected layers.

    - Teacher: run full-attention model, get hidden_states (no grad).
    - Student: use the hidden_states returned by Titan (with grad).
    - For each layer index ℓ in distillation_layers, add an MSE loss between
      teacher_hidden_states[ℓ + 1] and student_hidden_states[ℓ + 1].

    Returns a scalar loss (torch.Tensor) on `device`.
    """
    if not distillation_layers:
        return torch.tensor(0.0, device=device)

    # Student hiddens come from Titan forward pass
    student_hiddens = student_model_hiddens
    if student_hiddens is None:
        # model wasn't configured to return them
        return torch.tensor(0.0, device=device)

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # ---- Teacher forward (full attention, no grad) ----
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Try both attribute + dict style
    teacher_hiddens = getattr(teacher_outputs, "hidden_states", None)
    if teacher_hiddens is None and isinstance(teacher_outputs, dict):
        teacher_hiddens = teacher_outputs.get("hidden_states", None)

    if teacher_hiddens is None:
        # No teacher hidden states → nothing to distill
        return torch.tensor(0.0, device=device)

    # Both should be tuples: len = num_layers + 1
    total_loss = 0.0
    counted_layers = 0

    for layer_idx in distillation_layers:
        # hidden_states[0] = embeddings, [1] = after layer 0, etc.
        s_idx = layer_idx + 1
        t_idx = layer_idx + 1

        if s_idx >= len(student_hiddens) or t_idx >= len(teacher_hiddens):
            continue

        s_h = student_hiddens[s_idx]       # [B, N, D]
        t_h = teacher_hiddens[t_idx]       # [B, N, D]

        # Match dtype/device
        t_h = t_h.to(device=s_h.device, dtype=s_h.dtype)

        layer_loss = F.mse_loss(s_h, t_h.detach())
        total_loss += layer_loss
        counted_layers += 1

    if counted_layers == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / counted_layers

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if hasattr(model, "reset_memory_states"):
                model.reset_memory_states()
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            total_acc += outputs.get("correct", 0.0)
            steps += 1
    model.train()
    avg_loss = total_loss / max(steps, 1)
    ppl = math.exp(avg_loss) if steps > 0 else float("inf")
    acc = (total_acc / max(steps, 1)).item() if steps > 0 else 0.0
    return {"loss": avg_loss, "ppl": ppl, "acc": acc}


# ---------------------------------------------------------------------------
# PubMedQA yes/no evaluation on pqa_labeled
# ---------------------------------------------------------------------------

def evaluate_pubmedqa_yesno(model, cfg: DomainMemoryConfig) -> float:
    """
    Evaluate yes/no accuracy on PubMedQA pqa_labeled.

    We:
    - load config 'pqa_labeled'
    - filter to final_decision in {yes,no}
    - prompt the model with context + question and ask "Answer (yes or no):"
    - greedy-generate 1 token and map to yes/no.
    """
    print("[PubMedQA] Running yes/no evaluation on pqa_labeled...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print("Dataset loaded")

    def is_yesno(ex):
        return ex.get("final_decision") in ("yes", "no")

    ds = ds.filter(is_yesno)

    model.eval()
    print("Dataset evaluated")
    correct = 0
    total = 0

    for ex in ds:
        ctx_text = _pubmedqa_context_to_text(ex.get("context", ""))
        question = ex.get("question", "")
        gold = ex.get("final_decision")
        prompt = (
            f"Context: {ctx_text}\n"
            f"Question: {question}\n"
            f"Answer (yes or no):"
        )

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_length,
        ).to(cfg.device)

        if hasattr(model, "reset_memory_states"):
            model.reset_memory_states()

        with torch.no_grad():
            if hasattr(model, "generate"):
                # HF-style models (if you ever plug one in)
                out = model.generate(
                    **enc,
                    max_new_tokens=1,
                    do_sample=False,
                )
            else:
                # Titan-LLaMA
                out = model.generate_with_titan_memory(
                    input_ids=enc["input_ids"],
                    max_new_tokens=1,
                    temperature=1.0,
                    do_sample=False,
                    top_p=0.9,
                    reset_memory=False,  # already reset before the loop
                    use_cache=True,
                )

        new_ids = out[0, enc["input_ids"].shape[-1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()

        if text.startswith("yes"):
            pred = "yes"
        elif text.startswith("no"):
            pred = "no"
        else:
            pred = None  # weird response → treat as incorrect

        if pred is not None:
            if pred == gold:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(
        f"[PubMedQA] yes/no accuracy on pqa_labeled: "
        f"{acc:.4f} ({correct}/{total} counted examples)"
    )
    model.train()
    return acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_domain_memory(cfg: DomainMemoryConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load Titan-LLaMA with NeuralMemory (student)
    model = load_domain_model(cfg)

    # Optional: load full-attention teacher (frozen HF LLaMA)
    teacher_model = None
    if cfg.use_attention_distillation and cfg.distillation_weight > 0.0:
        print("[Distillation] Loading full-attention teacher:", cfg.base_model_name)
        dtype = next(model.parameters()).dtype
        teacher_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            torch_dtype=dtype,
        )
        teacher_model.to(cfg.device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    # Split params into neural-memory vs others
    neural_mem_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "neural_memory" in name:
            neural_mem_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append(
            {"params": other_params, "lr": cfg.learning_rate}
        )
    if neural_mem_params:
        param_groups.append(
            {"params": neural_mem_params, "lr": cfg.neural_mem_lr}
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
    )

    train_ds = DomainSequenceDataset(
        tokenizer_name=cfg.tokenizer_name,
        max_length=cfg.max_length,
        domain=cfg.domain,
        split="train",
    )

    # Try validation, fall back to test/train as needed
    eval_ds = None
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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    global_step = 0
    model.train()
    optimizer.zero_grad()

    steps_per_epoch_effective = len(train_loader)

    print(
        f"[Config] batch_size={cfg.batch_size}, "
        f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}, "
        f"effective_global_batch={cfg.batch_size * cfg.gradient_accumulation_steps}"
    )
    if cfg.use_attention_distillation and cfg.distillation_weight > 0.0:
        print(
            f"[Config] attention distillation ENABLED | "
            f"weight={cfg.distillation_weight} | layers={cfg.distillation_layers}"
        )
    else:
        print("[Config] attention distillation DISABLED")

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        for step, batch in enumerate(train_loader):
            # Interpret steps_per_epoch as *optimizer steps*
            if cfg.steps_per_epoch is not None and global_step >= cfg.steps_per_epoch:
                break

            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)

            if hasattr(model, "reset_memory_states"):
                model.reset_memory_states()

            # We only need hidden states when distilling
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=cfg.use_attention_distillation and cfg.distillation_weight > 0.0,
            )

            # LM loss with gradient accumulation normalization
            lm_loss = outputs["loss"] / cfg.gradient_accumulation_steps

            # Optional attention distillation loss
            distill_loss = torch.tensor(0.0, device=cfg.device)
            if cfg.use_attention_distillation and cfg.distillation_weight > 0.0 and teacher_model is not None:
                distill_loss = compute_attention_distillation_loss(
                    student_model_hiddens=outputs.get("hidden_states", None),
                    teacher_model=teacher_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    distillation_layers=cfg.distillation_layers,
                    device=cfg.device,
                )
                distill_loss = distill_loss / cfg.gradient_accumulation_steps

            total_loss = lm_loss + cfg.distillation_weight * distill_loss
            total_loss.backward()

            # Only step optimizer every gradient_accumulation_steps micro-batches
            micro_step = (step + 1) % cfg.gradient_accumulation_steps
            is_accum_step = micro_step == 0 or (step + 1) == steps_per_epoch_effective

            if is_accum_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    print(
                        f"step {global_step} | "
                        f"loss {total_loss.item():.4f} | "
                        f"lm {lm_loss.item():.4f} | "
                        f"distill {distill_loss.item():.4f}"
                    )

                if cfg.eval_every and global_step > 0 and global_step % cfg.eval_every == 0:
                    metrics = evaluate(model, eval_loader, cfg.device)
                    print(
                        f"[eval] step {global_step} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"ppl {metrics['ppl']:.2f} | "
                        f"acc {metrics['acc']:.4f}"
                    )
                    save_path = os.path.join(
                        cfg.output_dir, f"{cfg.domain}-nmm-step{global_step}.pt"
                    )
                    torch.save(
                        {"model_state_dict": model.state_dict(), "config": cfg.__dict__},
                        save_path,
                    )
                    print(f"Saved checkpoint to {save_path}")

    # Final checkpoint
    final_path = os.path.join(cfg.output_dir, f"{cfg.domain}-nmm-final.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, final_path)
    print(f"Finished training. Final checkpoint: {final_path}")

    # Extra: PubMedQA yes/no evaluation if requested domain
    if cfg.domain == "pubmedqa":
        evaluate_pubmedqa_yesno(model, cfg)


def parse_args() -> DomainMemoryConfig:
    parser = argparse.ArgumentParser(
        description="Train domain-specific NeuralMemory adapter."
    )
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_SPECS.keys()))
    parser.add_argument("--tokenizer_name", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--base_model_name", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--base_checkpoint", default="./titan_llama_checkpoints/best_checkpoint.pt")
    parser.add_argument("--output_dir", default="./domain_nmm")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--neural_mem_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=200)

    # Gradient accumulation
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Attention distillation
    parser.add_argument(
        "--use_attention_distillation",
        action="store_true",
        help="Enable attention distillation loss against a full-attention teacher.",
    )
    parser.add_argument(
        "--distillation_weight",
        type=float,
        default=0.0,
        help="Weight for attention distillation loss vs LM loss.",
    )
    parser.add_argument(
        "--distillation_layers",
        nargs="+",
        type=int,
        default=[8, 16, 24],
        help="Layer indices (0-based) used for attention distillation.",
    )

    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--neural_memory_segment_len", type=int, default=64)
    parser.add_argument("--neural_memory_batch_size", type=int, default=64)

    args = parser.parse_args()
    return DomainMemoryConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train_domain_memory(cfg)
