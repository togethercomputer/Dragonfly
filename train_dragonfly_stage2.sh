accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero1.yaml \
    --num_processes=8 \
    pipeline/train/training.py \
    --batch_size 1 \
    --total_training_steps 507798 \
    --external_save_dir <"your_save_dir"> \
    --run_name <"your_run_name"> \
    --workers 8 \
    --lr_scheduler cosine \
    --learning_rate 2e-6 \
    --warmup_steps_ratio 0.01 \
    --save_hf_model \
    --resume_from_checkpoint \
    --data_dir <"your_data_folder"> \
    --image_dir <"your_image_folder"> \
    --together_hq_datasets <"your_vit_datasets"> \
    --together_text_datasets <"your_text_datasets"> \
    --together_math_datasets <"your_math_datasets"> \
    --text_dataset_prob 0.1 \
    --logging_steps 100 \
    --max_seq_length 4096 \
    --checkpointing_steps 10000 \
    --save_hf_checkpoints \
    --total_hf_checkpoint_limits 8 \
    --hf_checkpointing_steps 100000 \
    --seed 42 \
    --image_encoder_name_or_path openai/clip-vit-large-patch14-336 \
    --text_pretrained_model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --pretrained_model_name_or_path <"your_stage1_checkpoint">  \
    --gradient_checkpointing \
    --data_cache_dir <"your_hf_cache_datasets">

    
