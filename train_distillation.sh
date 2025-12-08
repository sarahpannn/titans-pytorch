export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python3 run_training.py \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --segment_len 512 \
    --sequence_length 2048 \
    --batch_size 32 \
    --micro_batch_size 1 \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --total_tokens 10000000 \
    --eval_interval 500 \
    --save_interval 2000 \
    --log_interval 50 \
    --intermittent_eval_frequency 100 \
    --intermittent_eval_limit 128 \
    --intermittent_eval_start_step 100 \
    --use_attention_distillation \
    --distillation_weight 0.1 \
    --distillation_layers "4,8,12,16,20" \
    --neural_memory_layers "4,8,12,16,20" \
    --no_wandb \
    --debug
