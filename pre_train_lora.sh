#!/bin/bash
# Pre-training with LoRA adapters on attention layers
# LoRA helps attention adapt to the neural memory's modified hidden state distribution

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python3 run_training.py \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --segment_len 32 \
    --sequence_length 128 \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_persist_mem 0 \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --total_tokens 100000000 \
    --eval_interval 50 \
    --save_interval 2000 \
    --learning_rate 1e-4 \
    --min_learning_rate 1e-9 \
    --neural_mem_lr 5e-3 \
    --log_interval 1 \
    --intermittent_eval_frequency 100 \
    --intermittent_eval_limit 256 \
    --intermittent_eval_start_step 10 \
    --distillation_weight 0.0 \
    --distillation_layers "4,8,12,16,20" \
    --neural_memory_layers "4,8,12,16,20" \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_layers_after_memory 1 \
    --lora_lr 1e-4 \
    --output_dir "./titan_llama_checkpoints_lora" \
    --wandb_run_name "titan-lora-pretrain"
