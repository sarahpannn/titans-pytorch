#!/usr/bin/env python3
"""
Measure peak memory vs sequence length for:
1) Base LLaMA (HF)
2) Titan segmented attention (no neural memory)
3) Titan segmented attention with neural memory

Reports peak GPU memory (MiB) per sequence length; CPU fallback prints N/A.
"""

import argparse
from typing import Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Peak memory scaling by sequence length across model variants.")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", help="HF base model name/path.")
    p.add_argument("--seq-lens", default="128,256,512,1024", help="Comma-separated sequence lengths to test.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic inputs.")
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
def measure_peak_memory(
    model,
    vocab_size: int,
    device: torch.device,
    batch_size: int,
    seq_len: int,
) -> float:
    """Return peak memory in bytes (GPU) or -1 if unavailable."""
    if device.type != "cuda":
        # GPU-only measurement to keep dependencies light
        _ = model(torch.randint(0, vocab_size, (batch_size, seq_len), device=device))
        return -1.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _ = _extract_logits(outputs)

    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return float(peak_bytes)


def parse_seq_lens(seq_str: str) -> Sequence[int]:
    return [int(s) for s in seq_str.split(",") if s.strip()]


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch_dtype = get_dtype(args.dtype)
    seq_lens = parse_seq_lens(args.seq_lens)

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
                neural_memory_layers=(8, 16, 24),
                use_flex_attn=args.use_flex_attn,
            ),
        )
    )

    results = {name: [] for name, _ in variants}

    for name, builder in variants:
        print(f"\n== Measuring {name} ==")
        model, tokenizer = builder()
        vocab_size = len(tokenizer)
        for seq_len in seq_lens:
            peak_bytes = measure_peak_memory(
                model=model,
                vocab_size=vocab_size,
                device=device,
                batch_size=args.batch_size,
                seq_len=seq_len,
            )
            if peak_bytes < 0:
                print(f"seq_len={seq_len}: peak_mem=N/A (CPU measurement not available)")
                results[name].append((seq_len, None))
            else:
                peak_mib = peak_bytes / (1024 ** 2)
                print(f"seq_len={seq_len}: peak_mem={peak_mib:.1f} MiB")
                results[name].append((seq_len, peak_mib))

    print("\n=== Summary (MiB) ===")
    for name, data in results.items():
        print(f"{name}:")
        for seq_len, peak_mib in data:
            if peak_mib is None:
                print(f"  seq_len={seq_len}: N/A")
            else:
                print(f"  seq_len={seq_len}: {peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
