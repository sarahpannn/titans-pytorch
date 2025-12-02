#!/usr/bin/env python3
"""
Throughput benchmark comparing:
1) Base LLaMA (HF)
2) Titan segmented attention (no neural memory)
3) Titan segmented attention with neural memory

Reports tokens/sec on simple forward passes over random token batches.
"""

import argparse
import time
from typing import Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark throughput for base vs segmented Titan LLaMA variants.")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", help="HF base model name/path.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic inputs.")
    p.add_argument("--seq-len", type=int, default=512, help="Sequence length for synthetic inputs.")
    p.add_argument("--steps", type=int, default=20, help="Timed steps.")
    p.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps before timing.")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16", help="Computation dtype.")
    p.add_argument("--segment-len", type=int, default=512, help="Segment/window length for segmented attention.")
    p.add_argument("--num-persist", type=int, default=4, help="Persistent memory tokens.")
    p.add_argument("--num-longterm", type=int, default=4, help="Long-term memory tokens.")
    p.add_argument("--use-flex-attn", action="store_true", help="Enable flex attention kernels for Titan variants.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Force device selection.")
    return p.parse_args()


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def get_dtype(arg: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[arg]


def build_base_llama(model_name: str, torch_dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    if device.type != "cuda":
        model.to(device)
    return model, tokenizer


def build_titan_variant(
    model_name: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    segment_len: int,
    num_persist: int,
    num_longterm: int,
    neural_memory_layers: Sequence[int],
    use_flex_attn: bool,
):
    base_cfg = AutoConfig.from_pretrained(model_name)
    titan_cfg = TitanLLaMAConfig.from_llama_config(
        base_cfg,
        segment_len=segment_len,
        num_persist_mem_tokens=num_persist,
        num_longterm_mem_tokens=num_longterm,
        neural_memory_layers=tuple(neural_memory_layers),
        neural_memory_segment_len=16 if neural_memory_layers else 0,
        neural_memory_batch_size=8 if neural_memory_layers else 0,
        neural_memory_depth=2 if neural_memory_layers else 0,
        use_flex_attn=use_flex_attn,
        sliding_window_attn=True,
        use_pretrained_backbone=True,
        base_model_name_or_path=model_name,
        freeze_backbone=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TitanLLaMAForCausalLM.from_pretrained_llama(
        base_model_name_or_path=model_name,
        titan_config=titan_cfg,
        freeze_backbone=True,
        dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    if device.type != "cuda":
        model.to(device)
    return model, tokenizer


def _extract_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise ValueError("Could not find logits in model output")


@torch.inference_mode()
def benchmark(
    model,
    vocab_size: int,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    steps: int,
) -> Tuple[float, float]:
    """Return (tokens_per_sec, elapsed_seconds)."""
    total_steps = warmup_steps + steps
    start_time: Optional[float] = None
    for step_idx in range(total_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Ensure computation actually runs
        _ = _extract_logits(outputs)

        if step_idx == warmup_steps - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    assert start_time is not None, "Timing did not start; check warmup steps."
    elapsed = end_time - start_time
    tokens = steps * batch_size * seq_len
    tps = tokens / elapsed if elapsed > 0 else float("inf")
    return tps, elapsed


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch_dtype = get_dtype(args.dtype)

    variants = []
    variants.append(("base_llama", lambda: build_base_llama(args.model, torch_dtype, device)))
    variants.append(
        (
            "segmented_no_nmm",
            lambda: build_titan_variant(
                args.model,
                torch_dtype,
                device,
                args.segment_len,
                args.num_persist,
                args.num_longterm,
                neural_memory_layers=(),
                use_flex_attn=args.use_flex_attn,
            ),
        )
    )
    variants.append(
        (
            "segmented_with_nmm",
            lambda: build_titan_variant(
                args.model,
                torch_dtype,
                device,
                args.segment_len,
                args.num_persist,
                args.num_longterm,
                neural_memory_layers=(8, 16, 24),  # align with Titan defaults
                use_flex_attn=args.use_flex_attn,
            ),
        )
    )

    results = []
    for name, builder in variants:
        print(f"\n== Benchmarking {name} ==")
        model, tokenizer = builder()
        vocab_size = len(tokenizer)
        tps, elapsed = benchmark(
            model=model,
            vocab_size=vocab_size,
            device=device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
        results.append((name, tps, elapsed))
        print(f"{name}: {tps:.2f} tokens/sec over {args.steps} steps (elapsed {elapsed:.2f}s)")

    print("\n=== Summary (higher is better) ===")
    for name, tps, elapsed in results:
        print(f"{name}: {tps:.2f} tokens/sec (elapsed {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
