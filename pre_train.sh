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
    --total_tokens 1000000000 \
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
    --save_interval 100