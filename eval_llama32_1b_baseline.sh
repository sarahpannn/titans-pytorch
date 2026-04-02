#!/bin/bash
# Baseline evals for unsloth/Llama-3.2-1B-Instruct
# Run 1: Vanilla (full attention)
# Run 2: Segmented attention with 128-token window
#
# To get a GPU allocation first:
#   salloc -p mit_normal_gpu --gres=gpu:l40s:1 -c 8 --mem=32G --time=05:00:00 --exclude=node3405

set -euo pipefail

MODEL="unsloth/Llama-3.2-1B-Instruct"
TASKS="hellaswag winogrande boolq"

echo "========================================="
echo "Run 1: Vanilla LLaMA (full attention)"
echo "========================================="
python3 eval_vanilla_llama.py \
  --model "$MODEL" \
  --tasks $TASKS \
  --batch-size 1 \
  --output "eval_vanilla_llama32_1b.txt"

echo ""
echo "========================================="
echo "Run 2: Segmented attention (window=128)"
echo "========================================="
python3 baseline_eval.py \
  --model "$MODEL" \
  --tasks $TASKS \
  --batch-size 1 \
  --segment-len 128 \
  --eval_out "eval_segmented128_llama32_1b.txt"

echo ""
echo "Done. Results saved to:"
echo "  eval_vanilla_llama32_1b.txt"
echo "  eval_segmented128_llama32_1b.txt"
