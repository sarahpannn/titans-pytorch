#!/usr/bin/env python

import argparse
import json
import os

import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        raise ValueError("No 'results' key or empty results in JSON.")
    return data, results


def plot_throughput(json_path: str, output_path: str | None = None):
    data, results = load_results(json_path)

    # Prompt length (optional, for title)
    prompt_lengths = {r.get("prompt_length") for r in results if "prompt_length" in r}
    prompt_len = None
    if len(prompt_lengths) == 1:
        prompt_len = next(iter(prompt_lengths))
        print(f"Detected prompt_length = {prompt_len}")

    # Single y-axis: just plot all variants together
    variants = sorted({r["variant"] for r in results})

    fig_thr, ax = plt.subplots(figsize=(6, 4), dpi=120)

    for variant in variants:
        subset = [r for r in results if r["variant"] == variant]
        subset = sorted(subset, key=lambda r: r["gen_len"])

        gen_lens = [r["gen_len"] for r in subset]
        throughputs = [r["tokens_per_sec"] for r in subset]

        ax.plot(gen_lens, throughputs, marker="o", label=variant)

    ax.set_xlabel("Generation length (tokens)")
    ax.set_ylabel("Throughput (tokens/sec)")
    title = "Throughput vs generation length"
    if prompt_len is not None:
        title += f"\n(prompt length = {prompt_len})"
    ax.set_title(title)

    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if variants:
        ax.legend(loc="best")

    fig_thr.tight_layout()

    if output_path is None:
        base = os.path.splitext(os.path.basename(json_path))[0]
        output_path = base + "_throughput.png"

    print(f"Saving throughput figure to {output_path}")
    fig_thr.savefig(output_path, bbox_inches="tight")


def plot_memory(json_path: str, output_path: str | None = None):
    data, results = load_results(json_path)
    variants = sorted({r["variant"] for r in results})

    # Prompt length (optional, for title)
    prompt_lengths = {r.get("prompt_length") for r in results if "prompt_length" in r}
    prompt_len = None
    if len(prompt_lengths) == 1:
        prompt_len = next(iter(prompt_lengths))
        print(f"Detected prompt_length = {prompt_len}")

    fig_mem, ax_mem = plt.subplots(figsize=(6, 4), dpi=120)

    for variant in variants:
        subset = [r for r in results if r["variant"] == variant]
        subset = sorted(subset, key=lambda r: r["gen_len"])

        gen_lens = [r["gen_len"] for r in subset]
        peak_mem_gib = [r.get("peak_mem_gib") for r in subset]

        mem_x = [g for g, m in zip(gen_lens, peak_mem_gib) if m is not None]
        mem_y = [m for m in peak_mem_gib if m is not None]

        if mem_x:
            ax_mem.plot(mem_x, mem_y, marker="o", label=variant)

    ax_mem.set_xlabel("Generation length (tokens)")
    ax_mem.set_ylabel("Peak GPU memory (GiB)")
    title = "Peak GPU memory vs generation length"
    if prompt_len is not None:
        title += f"\n(prompt length = {prompt_len})"
    ax_mem.set_title(title)

    ax_mem.set_xscale("log", base=2)
    ax_mem.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_mem.legend()

    fig_mem.tight_layout()

    if output_path is None:
        base = os.path.splitext(os.path.basename(json_path))[0]
        output_path = base + "_memory.png"

    print(f"Saving memory figure to {output_path}")
    fig_mem.savefig(output_path, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize speed_eval JSON results: throughput & memory vs generation length."
    )
    parser.add_argument("json_path", type=str, help="Path to JSON results file from speed_eval.py")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Prefix for output PNGs (default: derived from JSON basename)",
    )
    args = parser.parse_args()

    if args.out_prefix is None:
        base = os.path.splitext(os.path.basename(args.json_path))[0]
        thr_out = base + "_throughput.png"
        mem_out = base + "_memory.png"
    else:
        thr_out = args.out_prefix + "_throughput.png"
        mem_out = args.out_prefix + "_memory.png"

    plot_throughput(args.json_path, thr_out)
    plot_memory(args.json_path, mem_out)


if __name__ == "__main__":
    main()
