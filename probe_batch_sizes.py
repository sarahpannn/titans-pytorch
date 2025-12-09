#!/usr/bin/env python
"""
Probe Titan-LLaMA+NeuralMemory to find:

  - max *micro_batch_size* for training (forward + backward)
  - max *eval_batch_size* for inference (forward only)

Run with:

    (titan310) python probe_batch_sizes.py
"""

import torch
from train_domain_memory import DomainMemoryConfig, load_domain_model


def try_batch(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    backward: bool,
) -> bool:
    """
    Try a single forward (and optional backward) pass with given batch size.
    Returns True if it fits in memory, False if OOM.
    """
    device = next(model.parameters()).device
    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        # fallback if not set
        vocab_size = 32000

    try:
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        labels = input_ids.clone()

        if hasattr(model, "reset_memory_states"):
            model.reset_memory_states()

        if backward:
            model.train()
        else:
            model.eval()

        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"] if isinstance(out, dict) else out[0]

        if backward:
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        model.zero_grad(set_to_none=True)
        return True

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda error" in msg:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            return False
        # if it's not OOM, re-raise
        raise


def find_max_batch(model, seq_len: int, backward: bool, max_try: int = 1024) -> int:
    """
    Doubling search for the largest batch size that fits.
    If backward=True → train micro-batch.
    If backward=False → eval batch.
    """
    device = next(model.parameters()).device
    kind = "train (forward+backward)" if backward else "eval (forward-only)"
    print(f"\nProbing max batch for {kind}, seq_len={seq_len} on {device}...")

    bs = 1
    last_ok = 1

    while bs <= max_try:
        ok = try_batch(model, bs, seq_len, backward=backward)
        if ok:
            print(f"  ✓ batch_size={bs} OK")
            last_ok = bs
            bs *= 2
        else:
            print(f"  ✗ batch_size={bs} OOM, stopping search")
            break

    print(f"Max batch size that fits for {kind}: {last_ok}")
    return last_ok


def main():
    # Configure for your pubmedqa domain run.
    # This only matters for model construction, not dataset.
    cfg = DomainMemoryConfig(
        domain="pubmedqa",
        base_model_name="meta-llama/Meta-Llama-3.1-8B",
        base_checkpoint="titan_llama_checkpoints/titan_llama_lm_latest.pt",
        output_dir="runs/pubmedqa-nmm-probe",
        max_length=512,
        batch_size=1,
        eval_batch_size=1,
        num_epochs=1,
        steps_per_epoch=None,
        gradient_accumulation_steps=1,
        use_attention_distillation=False,
        distillation_weight=0.0,
    )

    # Load Titan-LLaMA + NMM once
    model = load_domain_model(cfg)
    model.to(cfg.device)

    # 1) micro batch size for training (forward + backward)
    max_micro = find_max_batch(
        model, seq_len=cfg.max_length, backward=True, max_try=512
    )

    # 2) eval batch size for inference (forward only).
    #    Start searching from micro batch size upward.
    max_eval = find_max_batch(
        model, seq_len=cfg.max_length, backward=False, max_try=1024
    )

    print("\n=== Recommended batch sizes ===")
    print(f"Micro batch size (train `--batch_size`): {max_micro}")
    print(f"Eval batch size      (`--eval_batch_size`): {max_eval}")


if __name__ == "__main__":
    main()