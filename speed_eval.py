#!/usr/bin/env python3
"""
Generation benchmark comparing:
1) Base LLaMA (HF)
2) Titan segmented attention (no neural memory)
3) Titan segmented attention with neural memory

Measures generated tokens/sec using generation (HF generate or Titan memory path) on a repeated prompt batch,
running until a target number of new tokens have been produced.
"""

import argparse
import json
import math
import os
import shutil
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark generation throughput for base vs segmented Titan LLaMA variants.")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", help="HF base model name/path.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for generation prompts.")
    p.add_argument("--nmm-batch-size", type=int, default=64, help="Batch size for neural memory store (defaults to batch-size).")
    p.add_argument("--prompt", type=str, default="In a distant future, an explorer finds", help="Base prompt text.")
    p.add_argument("--prompt-length", type=int, default=4096, help="Target prompt length in tokens; prompt is repeated until reaching this length (use larger values to stress long-context).")
    p.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens to generate per generation call.")
    p.add_argument("--target-new-tokens", type=int, default=1024, help="Total new tokens to generate during timed runs (will loop generation calls until reached).")
    p.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps before timing.")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature; 0 disables sampling for greedy decode.")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16", help="Computation dtype.")
    p.add_argument("--segment-len", type=int, default=512, help="Segment/window length for segmented attention.")
    p.add_argument("--num-persist", type=int, default=0, help="Persistent memory tokens.")
    p.add_argument("--num-longterm", type=int, default=4, help="Long-term memory tokens.")
    p.add_argument("--use-flex-attn", action="store_true", help="Enable flex attention kernels for Titan variants.")
    p.add_argument("--nmm-chunk-size", type=int, default=64, help="Neural memory chunk size (must divide batch size; will fall back to 1 if not).")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Force device selection.")
    p.add_argument("--cleanup-cache", action="store_true", help="Delete the HF cache entries for the tested model after benchmarking.")
    p.add_argument("--save-dir", type=str, default=None, help="Directory to write a JSON summary of results.")
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


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def build_base_llama(model_name: str, torch_dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ensure_pad_token(tokenizer)

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
    neural_memory_chunk_size: int,
    neural_memory_batch_size: int,
):
    base_cfg = AutoConfig.from_pretrained(model_name)
    titan_cfg = TitanLLaMAConfig.from_llama_config(
        base_cfg,
        segment_len=segment_len,
        num_persist_mem_tokens=num_persist,
        num_longterm_mem_tokens=num_longterm,
        neural_memory_layers=tuple(neural_memory_layers),
        neural_memory_segment_len=neural_memory_chunk_size if neural_memory_layers else 0,
        neural_memory_batch_size=neural_memory_batch_size if neural_memory_layers else 0,
        neural_memory_depth=2 if neural_memory_layers else 0,
        use_flex_attn=use_flex_attn,
        sliding_window_attn=True,
        use_pretrained_backbone=True,
        base_model_name_or_path=model_name,
        freeze_backbone=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ensure_pad_token(tokenizer)

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


def _expand_prompt(tokenizer, prompt: str, target_len: int) -> str:
    prompt = prompt.strip()
    if target_len <= 0:
        return prompt
    base_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if not base_ids:
        raise ValueError("Prompt tokenization produced zero tokens; provide a non-empty prompt.")
    if len(base_ids) >= target_len:
        return prompt
    repeats = math.ceil(target_len / len(base_ids))
    return (" " + prompt).join([prompt] * repeats)


def build_prompt_batch(tokenizer, prompt: str, prompt_length: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
    prompt_text = _expand_prompt(tokenizer, prompt, prompt_length)
    encoded = tokenizer([prompt_text] * batch_size, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask, prompt_text, input_ids.shape[1]

@torch.inference_mode()
def benchmark_single_generation(
    model,
    tokenizer,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_len: int,
    temperature: float,
    variant_name: str,
    do_warmup: bool = True,
) -> Tuple[float, float, int, Optional[int]]:
    """
    Run a single eval-like generation:
      - optional warmup call (not timed)
      - one timed call generating ~gen_len new tokens

    Returns:
        tokens_per_sec, elapsed_seconds, total_new_tokens, peak_mem_bytes
    """
    batch_size = input_ids.shape[0]

    # ----------------- pick the right generate function -----------------
    if hasattr(model, "generate"):
        def run_generate(max_new_tokens: int) -> torch.Tensor:
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
                use_cache=True,
            )
    elif hasattr(model, "generate_with_titan_memory"):
        def run_generate(max_new_tokens: int) -> torch.Tensor:
            return model.generate_with_titan_memory(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                reset_memory=True,
            )
    else:
        raise ValueError(f"Model {variant_name} has no generation method.")
    # -------------------------------------------------------------------

    # --------------- reset CUDA peak memory statistics -----------------
    peak_mem_bytes: Optional[int] = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    # -------------------------------------------------------------------

    # -------------------------- warmup (optional) ----------------------
    if do_warmup:
        _ = run_generate(gen_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # -------------------------------------------------------------------

    # ---------------------- timed generation call ----------------------
    if device.type == "cuda":
        # use events for more accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        outputs = run_generate(gen_len)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed = elapsed_ms / 1000.0
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    else:
        # CPU timing fallback
        start = time.perf_counter()
        outputs = run_generate(gen_len)
        elapsed = time.perf_counter() - start
    # -------------------------------------------------------------------

    # -------------------- token counting sanity check ------------------
    seq_in = input_ids.shape[1]
    seq_out = outputs.shape[1]
    new_tokens = max(seq_out - seq_in, 0)
    total_new_tokens = new_tokens * batch_size

    # Optional: debug print if something weird happens
    if new_tokens != gen_len:
        print(
            f"[warn] requested gen_len={gen_len} but model produced "
            f"{new_tokens} new tokens (in_len={seq_in}, out_len={seq_out}) "
            f"for variant {variant_name}"
        )
    # -------------------------------------------------------------------

    tokens_per_sec = total_new_tokens / elapsed if elapsed > 0 else float("inf")
    return tokens_per_sec, elapsed, total_new_tokens, peak_mem_bytes



@torch.inference_mode()
def benchmark_generation(
    model,
    tokenizer,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    target_new_tokens: int,
    warmup_steps: int,
    temperature: float,
    variant_name: str,
) -> Tuple[float, float, int, Optional[int]]:
    """
    Run generation several times and measure throughput.

    Here we'll usually set target_new_tokens == max_new_tokens so that we
    time a single eval-like call, but the function supports multiple calls.

    Returns:
        tokens_per_sec, elapsed_seconds, total_new_tokens, peak_mem_bytes
    """
    if target_new_tokens <= 0:
        raise ValueError("target_new_tokens must be positive.")
    timed_generations = math.ceil(target_new_tokens / max_new_tokens)
    total_steps = warmup_steps + timed_generations

    # --- CUDA peak memory tracking ---
    peak_mem_bytes: Optional[int] = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    # ---------------------------------

    def _build_generator():
        # This is whatever you had before; leaving structure intact,
        # just showing the skeleton.
        if hasattr(model, "generate"):
            def _run(step_max_new_tokens: int) -> torch.Tensor:
                return model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=step_max_new_tokens,
                    do_sample=False,
                    temperature=temperature,
                    use_cache=True,
                )
        elif hasattr(model, "generate_with_titan_memory"):
            def _run(step_max_new_tokens: int) -> torch.Tensor:
                return model.generate_with_titan_memory(
                    input_ids=input_ids,
                    max_new_tokens=step_max_new_tokens,
                    temperature=temperature,
                    do_sample=False,
                    reset_memory=True,
                )
        else:
            raise ValueError(f"Model {variant_name} has no generation method.")
        return _run

    run_generate = _build_generator()

    start_time: Optional[float] = None
    total_new_tokens = 0

    for step_idx in range(total_steps):
        # Warmup steps are not timed
        if step_idx == warmup_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

        step_max_new_tokens = max_new_tokens
        outputs = run_generate(step_max_new_tokens)

        # outputs: (batch_size, prompt_len + new_len)
        new_tokens = outputs.shape[1] - input_ids.shape[1]
        total_new_tokens += new_tokens * input_ids.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    end_time = time.perf_counter()

    if start_time is None:
        raise RuntimeError("Timing did not start; check warmup_steps.")

    elapsed = (end_time - start_time) / 1000
    tokens_per_sec = total_new_tokens / elapsed if elapsed > 0 else float("inf")
    return tokens_per_sec, elapsed, total_new_tokens, peak_mem_bytes



def save_results(save_dir: str, payload: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "speed_eval_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[info] wrote results to {path}")


def cleanup_cache(model_name: str) -> None:
    safe_name = model_name.replace("/", "--")
    candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.extend([hf_home, os.path.join(hf_home, "hub")])
    candidates.append(os.path.expanduser("~/.cache/huggingface"))
    candidates.append(os.path.expanduser("~/.cache/huggingface/hub"))

    seen = set()
    removed = []
    for cache_dir in candidates:
        if not cache_dir or cache_dir in seen or not os.path.isdir(cache_dir):
            continue
        seen.add(cache_dir)
        for entry in os.listdir(cache_dir):
            if safe_name in entry:
                path = os.path.join(cache_dir, entry)
                try:
                    shutil.rmtree(path)
                    removed.append(path)
                except OSError as err:
                    print(f"[warn] failed to remove cache path {path}: {err}")
    if removed:
        print(f"[info] removed cached entries: {removed}")


def main() -> None:
    """
    Fix prompt length, vary generation length.

    For each variant (base_llama, segmented_no_nmm, segmented_with_nmm) and
    each generation length G in [64, 128, 256, 512, 1024, 2048, 4096],
    we:

      1. Build a batch of prompts of length `args.prompt_length`.
      2. Call generate ONCE to produce G new tokens.
      3. Measure wall-clock time, tokens/sec, and peak GPU mem.

    This matches a realistic eval step: given a fixed context, generate a chunk
    of new tokens, and lets us see how throughput changes with G.
    """
    args = parse_args()
    device = get_device(args.device)
    torch_dtype = get_dtype(args.dtype)

    # --- NMM knobs (keep your defaults, e.g. 64/64) ---
    nmm_batch = args.nmm_batch_size
    nmm_chunk = args.nmm_chunk_size
    if nmm_batch % nmm_chunk != 0:
        print(
            f"[info] nmm-chunk-size {nmm_chunk} does not divide nmm_batch_size {nmm_batch}; "
            "falling back to chunk size 1."
        )
        nmm_chunk = 1
    # --------------------------------------------------

    # Build the three variants
    variants: List[Tuple[str, Callable[[], Tuple[torch.nn.Module, Any]]]] = []
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
                neural_memory_chunk_size=nmm_chunk,
                neural_memory_batch_size=nmm_batch,
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
                neural_memory_chunk_size=nmm_chunk,
                neural_memory_batch_size=nmm_batch,
            ),
        )
    )

    # Optional: you can reorder if you want segmented first
    variants.reverse()

    # Fixed prompt length for the whole sweep
    prompt_len = args.prompt_length
    print(f"[info] using prompt_length = {prompt_len} tokens")

    # Generation lengths to sweep
    gen_lengths = [64, 128, 256, 512, 1024, 2048]

    all_results: List[dict] = []

    for name, builder in variants:
        print(f"\n=== Variant: {name} ===")
        model, tokenizer = builder()

        # Build the prompt batch once per variant
        input_ids, attention_mask, prompt_text, prompt_tokens = build_prompt_batch(
            tokenizer=tokenizer,
            prompt=args.prompt,
            prompt_length=prompt_len,
            batch_size=args.batch_size,
            device=device,
        )

        # for gen_len in gen_lengths:
        #     # Skip if you ever want to cap generation length via CLI
        #     # (optional; for now we just always run all)
        #     print(f"\n  -- gen_len = {gen_len} --")

        #     tokens_per_sec, elapsed, generated_tokens, peak_mem_bytes = benchmark_single_generation(
        #         model=model,
        #         tokenizer=tokenizer,
        #         device=device,
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         max_new_tokens=gen_len,
        #         target_new_tokens=gen_len,  # single eval-like call
        #         warmup_steps=args.warmup_steps,
        #         temperature=args.temperature,
        #         variant_name=name,
        #     )

        #     peak_mem_gib = None
        #     if peak_mem_bytes is not None:
        #         peak_mem_gib = peak_mem_bytes / (1024 ** 3)

        #     if peak_mem_gib is not None:
        #         print(
        #             f"    {name}: {tokens_per_sec:.2f} tok/s | "
        #             f"{generated_tokens} new tokens in {elapsed:.2f}s "
        #             f"(prompt_tokens={prompt_tokens}, peak GPU mem={peak_mem_gib:.2f} GiB)"
        #         )
        #     else:
        #         print(
        #             f"    {name}: {tokens_per_sec:.2f} tok/s | "
        #             f"{generated_tokens} new tokens in {elapsed:.2f}s "
        #             f"(prompt_tokens={prompt_tokens})"
        #         )

        #     all_results.append(
        #         {
        #             "variant": name,
        #             "prompt_length": int(prompt_tokens),
        #             "gen_len": int(gen_len),
        #             "tokens_per_sec": float(tokens_per_sec),
        #             "elapsed": float(elapsed),
        #             "generated_tokens": int(generated_tokens),
        #             "peak_mem_bytes": int(peak_mem_bytes) if peak_mem_bytes is not None else None,
        #             "peak_mem_gib": float(peak_mem_gib) if peak_mem_gib is not None else None,
        #         }
        #     )

        for gen_len in gen_lengths:
            print(f"\n  -- gen_len = {gen_len} --")

            tokens_per_sec, elapsed, generated_tokens, peak_mem_bytes = benchmark_single_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gen_len=gen_len,
                temperature=args.temperature,
                variant_name=name,
                do_warmup=True,  # or False if you want raw cold numbers
            )

            peak_mem_gib = None
            if peak_mem_bytes is not None:
                peak_mem_gib = peak_mem_bytes / (1024 ** 3)

            if peak_mem_gib is not None:
                print(
                    f"    {name}: {tokens_per_sec:.2f} tok/s | "
                    f"{generated_tokens} new tokens in {elapsed:.2f}s "
                    f"(prompt_tokens={prompt_tokens}, peak GPU mem={peak_mem_gib:.2f} GiB)"
                )
            else:
                print(
                    f"    {name}: {tokens_per_sec:.2f} tok/s | "
                    f"{generated_tokens} new tokens in {elapsed:.2f}s "
                    f"(prompt_tokens={prompt_tokens})"
                )

            all_results.append(
                {
                    "variant": name,
                    "prompt_length": int(prompt_tokens),
                    "gen_len": int(gen_len),
                    "tokens_per_sec": float(tokens_per_sec),
                    "elapsed": float(elapsed),
                    "generated_tokens": int(generated_tokens),
                    "peak_mem_bytes": int(peak_mem_bytes) if peak_mem_bytes is not None else None,
                    "peak_mem_gib": float(peak_mem_gib) if peak_mem_gib is not None else None,
                }
            )


        # Free model before building the next variant
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save to JSON for plotting, if requested
    if args.save_dir:
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "batch_size": args.batch_size,
            "prompt_length": prompt_len,
            "gen_lengths": gen_lengths,
            "nmm_batch_size": nmm_batch,
            "nmm_chunk_size": nmm_chunk,
            "dtype": args.dtype,
            "device": str(device),
            "segment_len": args.segment_len,
            "num_persist": args.num_persist,
            "num_longterm": args.num_longterm,
            "warmup_steps": args.warmup_steps,
            "results": all_results,
        }
        save_results(args.save_dir, payload)

    if args.cleanup_cache:
        cleanup_cache(args.model)

if __name__ == "__main__":
    main()
