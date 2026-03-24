export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1

python3 run_training.py \
    --pretrained_from_checkpoint titan_llama_checkpoints/latest_checkpoint.pt \
    --dataset_name casehold \
    --segment_len 64 \
    --sequence_length 512 \
    --batch_size 128 \
    --micro_batch_size 8 \
    --num_persist_mem 0 \
    --neural_memory_segment_len 64 \
    --neural_memory_batch_size 64 \
    --num-epochs 3 \
    --eval_interval 200 \
    --save_interval 2000 \
    --learning_rate 1e-7 \
    --neural_mem_lr 5e-3 \
    --log_interval 1 \
    --intermittent_eval_frequency 50 \
    --intermittent_eval_limit 256 \
    --intermittent_eval_start_step 10 \
    --use_attention_distillation \
    --distillation_weight 0.2 \
    --distillation_layers "4,8,12,16,20" \
    --neural_memory_layers "4,8,12,16,20" \
    --save_interval 60000