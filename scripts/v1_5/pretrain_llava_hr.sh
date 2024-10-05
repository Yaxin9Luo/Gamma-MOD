#!/bin/bash
deepspeed llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/luogen_code/LLaVA-HR-OCR/checkpoints/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path /data/data/blip_laion_cc_sbu_558k.json \
    --image_folder /data/data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-hr-llama3-7b-pretrain-384 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
    --input_image_size 384
    
# deepspeed llava_hr/train/train_mem.py \
#     --capacity_factor 0.5 \
#     --mod_enable True  \
#     --mod_mode 'sparse' \
#     --router_aux_loss_coef 0.01 \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /data/luogen_code/LLaVA-HR-OCR/checkpoints/Meta-Llama-3-8B-Instruct \
#     --version plain \
#     --data_path /data/data/blip_laion_cc_sbu_558k.json \
#     --image_folder /data/data/images \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-hr-llama3-7b-pretrain-384 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
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
#     --is_multipath_encoder True \
#     --input_image_size 384