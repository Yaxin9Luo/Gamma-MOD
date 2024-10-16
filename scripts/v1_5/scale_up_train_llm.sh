#!/bin/bash
# deepspeed llava_hr/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /data/vicuna/vicuna-10b-v1.5 \
#     --version plain \
#     --data_path /data/data/blip_laion_cc_sbu_558k.json \
#     --image_folder /data/data/images \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-7b-pretrain-384/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter False \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/scale_up_llm_10b_2 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --is_multipath_encoder False \
#     --freeze_mm_mlp_adapter True \
#     --scale_up_train_llm True \
#     --freeze_backbone True \
#     --input_image_size 384 

deepspeed llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/luogen_code/LLaVA-HR-OCR/checkpoints/vicuna-10b-llava-hr \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-7b-pretrain-384/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-hr-10b-v4-full-weights \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2496 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
    --freeze_vision False \
    --input_image_size 1024

# bash scripts/v1_5/eval.sh ./checkpoints/llava-hr-10b-v4-full-weights 2>&1 | tee log-llava-hr-10b-v4-full-weights.txt