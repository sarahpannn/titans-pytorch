#!/bin/bash
# LoRA experiment on SlimPajama with Llama-3.2-1B-Instruct backbone
# Tests whether LoRA adapters help the transformer process neural memory residuals

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python3 run_training.py \
    --base_model_name unsloth/Llama-3.2-1B-Instruct \
    --tokenizer_name unsloth/Llama-3.2-1B-Instruct \
    --dataset_name rokset3/slim_pajama_chunk1 \
    --num_layers 16 \
    --num_heads 32 \
    --hidden_size 2048 \
    --segment_len 512 \
    --sequence_length 1024 \
    --batch_size 64 \
    --micro_batch_size 8 \
    --num_persist_mem 0 \
    --num_longterm_mem 4 \
    --neural_memory_layers "3,7,11,15" \
    --distillation_layers "3,4,7,8,11,12,15,16" \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --total_tokens 10000000 \
    --learning_rate 3e-4 \
    --min_learning_rate 5e-6 \
    --neural_mem_lr 1e-2 \
    --warmup_steps 2000 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --log_interval 1 \
    --intermittent_eval_frequency 100 \
    --intermittent_eval_limit 256 \
    --intermittent_eval_start_step 50 \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_layers_after_memory 1 \
    --use_attention_distillation \
    --distillation_weight 0.3 \
    --lora_lr 1e-4 \
    --output_dir "./titan_llama_lora_slimpajama" \
    --wandb_project "titan-llama-lora" \
    --wandb_run_name "lora-slimpajama-llama3.2-1b"
