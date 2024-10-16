accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero1.yaml \
    --num_processes=8 \
    pipeline/train/training.py \
    --batch_size 4 \
    --total_training_steps 17400 \
    --external_save_dir <"your_save_dir"> \
    --run_name <"your_run_name"> \
    --workers 8 \
    --lr_scheduler cosine \
    --learning_rate 2e-5 \
    --warmup_steps_ratio 0.01 \
    --save_hf_model \
    --resume_from_checkpoint \
    --data_dir <"your_data_folder"> \
    --image_dir <"your_image_folder"> \
    --together_hq_datasets <"your_datasets"> \
    --logging_steps 1000 \
    --max_seq_length 4096 \
    --checkpointing_steps 5000 \
    --image_encoder_name_or_path openai/clip-vit-large-patch14-336 \
    --text_pretrained_model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --mm_tune_vision_encoder \
    --tune_vision_embed_tokens_only \
    --data_cache_dir <"your_hf_cache_datasets">

    
