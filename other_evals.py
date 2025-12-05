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
    Load TitanLLaMA model using the new from_pretrained method.
    Much cleaner than the previous implementation.
    """
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    print('loading checkpoint...')
    
    # Use the new from_pretrained method
    model = TitanLLaMAForCausalLM.from_pretrained(
        checkpoint_path=checkpoint_path,
        base_model_name_or_path=base_model_name,
        dtype=torch_dtype,
        strict=False,
    )

    print('done loading titan model')

    # Tokenizer from base HF model
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
