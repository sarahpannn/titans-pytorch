#!/usr/bin/env python3
"""
Evaluate a Titan-LLaMA model (segmented attention + neural memory)
from a training checkpoint produced by `train_fw.sh`.

This reuses the LM-eval wrapper (and greedy generate) from baseline_eval.py.
"""

import argparse
import torch
from typing import Iterable, List, Sequence, Tuple
from transformers import AutoConfig, AutoTokenizer

from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM

# Reuse LM wrapper + default tasks from baseline_eval
from baseline_eval import TitanSegmentedLM

DEFAULT_TASKS: Sequence[str] = (
    # "mmlu",
    # "mmlu_pro",
    # "agieval",
    # "commonsense_qa",
    # "winogrande",
    # "bbh",
    # "arc_challenge",
    # "triviaqa_wiki",
    # "squadv2", # limit 100
    # "quac",
    "boolq",
    # "drop", # limit 100
)

def load_titan_from_checkpoint(
    checkpoint_path: str,
    base_model_name: str,
    dtype: str = "bfloat16",
):
    """
    Rebuild TitanLLaMAForCausalLM and load weights from the saved checkpoint.

    Expected checkpoint structure:

        checkpoint = {
            'model_state_dict': model.module.state_dict() if config.use_ddp else model.state_dict(),
            'config': config.__dict__,
            'step': step,
            'loss': loss,
        }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("TO DEVICE")
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    # 1) Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    train_cfg = ckpt.get("config", {}) or {}

    print('checkpoint done')
    # 2) Base HF config (for d_model, num_layers, etc.)
    base_cfg = AutoConfig.from_pretrained(base_model_name)

    # 3) Titan-specific overrides from training config when available
    def _get(name, default):
        return train_cfg.get(name, default)

    nm_layers = _get("neural_memory_layers", (8, 16, 24))
    if isinstance(nm_layers, list):
        nm_layers = tuple(nm_layers)

    titan_cfg = TitanLLaMAConfig.from_llama_config(
        base_cfg,
        segment_len=_get("segment_len", 512),
        num_persist_mem_tokens=_get("num_persist_mem_tokens", 4),
        num_longterm_mem_tokens=_get("num_longterm_mem_tokens", 4),
        neural_memory_layers=nm_layers,
        neural_memory_segment_len=_get("neural_memory_segment_len", 16),
        neural_memory_batch_size=_get("neural_memory_batch_size", 8),
        neural_memory_depth=_get("neural_memory_depth", 2),
        use_flex_attn=_get("use_flex_attn", True),
        sliding_window_attn=_get("sliding_window_attn", True),
        neural_mem_gate_attn_output=_get("neural_mem_gate_attn_output", False),
        neural_mem_weight_residual=_get("neural_mem_weight_residual", True),
        neural_mem_qkv_receives_diff_view=_get("neural_mem_qkv_receives_diff_view", True),
        use_pretrained_backbone=False,      # we load *all* weights from ckpt
        base_model_name_or_path=base_model_name,
        freeze_backbone=True,
    )

    print('loading titan model class')

    # 4) Instantiate Titan model and load weights
    model = TitanLLaMAForCausalLM(titan_cfg)
    load_info = model.load_state_dict(state_dict, strict=False)

    print('done loading titan model class')

    if load_info.missing_keys:
        print("[load_titan_from_checkpoint] Missing keys:", load_info.missing_keys)
    if load_info.unexpected_keys:
        print("[load_titan_from_checkpoint] Unexpected keys:", load_info.unexpected_keys)

    model.to(dtype=torch_dtype)

    # 5) Tokenizer from base HF model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Titan-LLaMA from training checkpoint (reusing baseline_eval LM wrapper)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file produced by train_fw.sh.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Base HF model name (tokenizer + base config).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="LM Eval task names to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for LM Eval.",
    )
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=128,
        help="Max new tokens for generation tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Per-task example limit (for a quick smoke test).",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model dtype.",
    )
    parser.add_argument(
        "--eval_out",
        type=str,
        default="titan_eval_from_checkpoint.txt",
        help="Where eval summary text is saved.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="Bootstrap iterations for CI (set >0 if you care about CIs).",
    )
    return parser.parse_args()


def main() -> None:
    from lm_eval import evaluator, tasks

    args = parse_args()

    print('loading checkpoint...')

    # 1) Build Titan from checkpoint
    model, tokenizer = load_titan_from_checkpoint(
        checkpoint_path=args.checkpoint,
        base_model_name=args.model,
        dtype=args.dtype,
    )

    # print('finished loading checkpoint')

    # 2) Wrap with LM-eval interface from baseline_eval
    lm = TitanSegmentedLM(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_gen_toks=args.max_gen_toks,
    )

    # 3) Run LM Eval Harness
    task_dict = tasks.get_task_dict(args.tasks)
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
    )

    with open(args.eval_out, "w") as f:
        f.write(str(results))


if __name__ == "__main__":
    main()
