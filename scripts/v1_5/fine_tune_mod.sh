#!/bin/bash
# deepspeed llava_hr/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /data/vicuna/vicuna-7b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-7b-pretrain-384/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-hr-7b-stage1-visualize\
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4\
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2496 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --is_multipath_encoder True \
#     --freeze_vision False \
#     --input_image_size 1024


# deepspeed llava_hr/train/train_mem.py \
#     --capacity_factor 0.5 \
#     --mod_enable True  \
#     --mod_mode 'arank_mod' \
#     --router_aux_loss_coef 0.01 \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /data/vicuna/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/modified_llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --vision_tower_slow convnext_xxlarge.clip_laion2b_soup  \
#     --pretrain_mm_mlp_adapter /data/luogen_code/upload_hf_weights/llava-hr-13b-x-pretrain-384/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/rank-mod-llava-hr-13b-img_txt_0.5\
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4\
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2496 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --is_multipath_encoder True \
#     --freeze_vision False \
#     --input_image_size 1024


# deepspeed llava_hr/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /data/vicuna/vicuna-13b-v1.5 \
#     --version v1 \
#     --data_path ./playground/data/modified_llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320  \
#     --pretrain_mm_mlp_adapter /data/luogen_code/LLaVA-HR-OCR/checkpoints/llava-hr-13b-pretrain-384/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-hr-test-time\
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4\
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2496 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --is_multipath_encoder True \
#     --freeze_vision False \
#     --input_image_size 1024
    
deepspeed llava_hr/train/train_mem.py \
    --capacity_factor 0.5 \
    --mod_enable True  \
    --mod_mode 'arank_mod' \
    --router_aux_loss_coef 0.01 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/vicuna/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/modified_llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320  \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-hr-7b-pretrain-384/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/mod-llava-hr-7b-route-0.34-only-img\
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
    --is_multipath_encoder True \
    --freeze_vision False \
    --input_image_size 1024


# bash scripts/v1_5/eval.sh /data/luogen_code/LLaVA-HR-OCR/checkpoints/rank-mod-llava-hr-13b-img_txt_0.34 2>&1 | tee ./experiments_logs/13b-mod-results.out
