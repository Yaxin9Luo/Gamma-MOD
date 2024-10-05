import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from gamma_mod.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from gamma_mod.conversation import conv_templates, SeparatorStyle
from gamma_mod.model.builder import load_pretrained_model
from gamma_mod.utils import disable_torch_init
from gamma_mod.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria,process_images
import requests
import time
from PIL import Image
import math
from transformers import AutoProcessor,LlavaForConditionalGeneration
path = "/data/luogen_code/LLaVA-HR-OCR/checkpoints/llava-v1.5-7b/"
processor = AutoProcessor.from_pretrained(path)

model_path = "/data/luogen_code/LLaVA-HR-OCR/checkpoints/llava-hr-7b-sft-1024"
model_name = "llava-hr-7b-sft-1024"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
device = torch.device("cuda:0")
model.to(device)
model.eval()
# processor._tokenizer.add_tokens(["<image>", "<pad>"])
# model.resize_token_embeddings(len(tokenizer))

tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
all_comsumed_time = 0
total_num_dec_tokens = 0
for i in range(2):
    image = Image.open(f'/data/luogen_code/LLaVA-robust/playground/data/eval/llava-bench-in-the-wild/images/{str(i+1).zfill(3)}.jpg')
    prompt = "<image>\nDescribe this photo in detail."
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    # image_tensor = process_images([image], image_processor, model.config).half().to(device)
    start_time = time.time()
    for _ in range(10):
        generate_ids = model.generate(**inputs, max_new_tokens=10)
        # generate_ids = model.generate(input_ids, images=image_tensor, max_new_tokens=10).to(device)
        num_dec_tokens = len(generate_ids[0]) - input_ids.shape[1]
        if i>0:
            total_num_dec_tokens += num_dec_tokens
    if i>0:
        total_time = time.time() - start_time
        all_comsumed_time += total_time

print('total time: ', all_comsumed_time)
print('total tokens: ', total_num_dec_tokens)
print('tokens per sec: ', float(total_num_dec_tokens) / all_comsumed_time)