export MAX_PIXELS=262144
export MASTER_PORT=27500
swift sft \
    --model /path/to/Qwen2___5-VL-7B-Instruct \
    --dataset /path/to/processed_datafile1k.jsonl \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --eval_steps 200 \
    --save_steps 1000 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --max_length 1024 \
    --output_dir output_sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --system 'You are a helpful assistant.' \
    --report_to 'tensorboard'
