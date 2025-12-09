export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python3 run_training.py \
    --pretrained_from_checkpoint titan_llama_checkpoints/latest_checkpoint.pt \
    --dataset_name pubmed \
    --segment_len 128 \
    --sequence_length 512 \
    --batch_size 64 \
    --micro_batch_size 8 \
    --num_persist_mem 0 \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --num-epochs 1 \
    --eval_interval 30 \
    --save_interval 2000 \
    --learning_rate 3e-4 \
    --neural_mem_lr 1e-3 \
    --log_interval 1 \
    --intermittent_eval_frequency 10 \
    --intermittent_eval_limit 256 \
    --intermittent_eval_start_step 0 \
    --use_attention_distillation \
    --distillation_weight 0.85 \
    --distillation_layers "4,8,12,16,20" \
    --neural_memory_layers "4,8,12,16,20" \
    --save_interval 40 \
    --no_wandb