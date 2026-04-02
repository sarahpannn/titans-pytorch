#!/bin/bash
# Speed eval: vanilla vs segmented-128 for Llama 3.2 1B Instruct
#
# To get a GPU allocation first:
#   salloc -p mit_normal_gpu --gres=gpu:l40s:1 -c 8 --mem=32G --time=05:00:00 --exclude=node3405
# - --skip-base --skip-nmm → only segmented (no NMM)
# - --skip-segmented --skip-nmm → only base LLaMA
# - --skip-base --skip-segmented → only NMM variant

set -euo pipefail

python3 speed_eval.py \
  --model unsloth/Llama-3.2-1B-Instruct \
  --prompt-length 256 \
  --batch-size 1 \
  --skip-segmented --skip-nmm \
  --use-flash-attn \
  --warmup-steps 1 \
  --num-trials 1 \
  --max-gen-len 16384 \
  --save-dir speed_results_llama32_1b

# python3 speed_eval.py \
#   --model unsloth/Llama-3.2-1B-Instruct \
#   --prompt-length 256 \
#   --batch-size 1 \
#   --skip-base --skip-nmm \
#   --segment-len 128 \
#   --segmented-layers 3 4 5 \
#   --use-flash-attn \
#   --warmup-steps 1 \
#   --num-trials 1 \
#   --max-gen-len 16384 \
#   --save-dir speed_results_llama32_1b
