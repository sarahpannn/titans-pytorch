python3 run_training.py \
    --dataset_name rokset3/slim_pajama_chunk1 \
    --segment_len 512 \
    --log_interval 1 \
    --batch_size 32 \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --no_wandb
    