export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python3 run_training.py \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --segment_len 512 \
    --sequence_length 2048 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_persist_mem 0 \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --total_tokens 1000000000 \
    --eval_interval 500 \
    --save_interval 2000 \
    --log_interval 1 \
    --intermittent_eval_frequency 30 \
    --intermittent_eval_limit 128 \
    --intermittent_eval_start_step 10 \
    --use_attention_distillation \
    --distillation_weight 0.7 \
    --distillation_layers "4,8,12,16,20" \
    --neural_memory_layers "4,8,12,16,20"