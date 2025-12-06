CUDA_VISIBLE_DEVICES=0 python3 run_training.py \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --segment_len 512 \
    --sequence_length 8192 \
    --log_interval 1 \
    --batch_size 256 \
    --micro_batch_size 256 \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --total_tokens 1000000000 \
    --intermittent_eval_frequency 10 \
    --intermittent_eval_limit 1000 \
    --intermittent_eval_start_step 0
    

CUDA_VISIBLE_DEVICES=0 python3 other_evals.py \
    --checkpoint titan_llama_checkpoints/latest_checkpoint.pt \
    --limit 10