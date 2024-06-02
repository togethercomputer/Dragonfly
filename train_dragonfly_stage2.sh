export HF_DATASETS_CACHE=/scratch/kezhen/cache
export NCCL_SOCKET_IFNAME=ens7

accelerate launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero1.yaml \
    --num_processes=8 \
    pipeline/train/training.py \
    --batch_size 1 \
    --total_training_steps 507798 \
    --external_save_dir /scratch/kezhen/checkpoints/test_run \
    --run_name test_run_stage2 \
    --workers 8 \
    --lr_scheduler cosine \
    --learning_rate 2e-6 \
    --warmup_steps_ratio 0.01 \
    --save_hf_model \
    --resume_from_checkpoint \
    --data_dir /data/kezhen/multi-modality/merged_dataset/stage2_3_mixture \
    --image_dir /data/kezhen/multi-modality/images \
    --together_hq_datasets mixture10_vit \
    --together_text_datasets textonly_instruct \
    --together_math_datasets math_instruct \
    --text_dataset_prob 0.1 \
    --logging_steps 100 \
    --max_seq_length 2048 \
    --checkpointing_steps 10000 \
    --save_hf_checkpoints \
    --total_hf_checkpoint_limits 8 \
    --hf_checkpointing_steps 100000 \
    --seed 42 \
    --image_encoder_name_or_path openai/clip-vit-base-patch32 \
    --text_pretrained_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --pretrained_model_name_or_path /data/kezhen/multi-modality/checkpoints/final_runs_v3/raccoon_stage1_zoom_select/raccoon_stage1_absolute_resolution_imgselect_new  \
    --gradient_checkpointing

    
