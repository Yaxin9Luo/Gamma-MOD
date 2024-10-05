import transformers
import json
from copy import deepcopy
import torch
from gamma_mod.model.language_model.llava_llama_moe import MoELLaVALlamaForCausalLM
from gamma_mod.model import *
from gamma_mod.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from gamma_mod.model.builder import load_pretrained_model

model_path = '/data/luogen_code/LLaVA-HR-OCR/checkpoints/llava-hr-mod-7b-last_two_thirds-img-answer-0.5'
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,None,model_name)


vicuna = transformers.AutoModelForCausalLM.from_pretrained('/data/vicuna/vicuna-7b-v1.5')

import torch.nn as nn
import torch.nn.init as init
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        # Initialize weights using Xavier/Glorot initialization
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)
    if isinstance(module, nn.LayerNorm):
        init.constant_(module.weight, 1.0)
        init.constant_(module.bias, 0.0)

new_backbone = torch.nn.ModuleList()
for i in range(24):
    new_backbone.append(deepcopy(model.model.layers[i]))
for i in range(24):
    new_backbone.append(deepcopy(model.model.layers[8 + i]))
vicuna.model.layers = new_backbone

print(vicuna)
# target_tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/store/Chinese-Llama-2-7b-plus/Llama-2-8b-hf-expand')
# origin_tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/store/Llama-2-13b-hf')
# origin_tokenizer_8b = transformers.AutoTokenizer.from_pretrained('/mnt/store/llama2-checkpoints-plus-longer/checkpoint-27000')

# for key in target_tokenizer.vocab.keys():
#     if key in origin_tokenizer.vocab.keys():
#         model.lm_head.weight.data[target_tokenizer.vocab[key], :] = origin_lm_head.weight.data[origin_tokenizer.vocab[key], :]
#         model.model.embed_tokens.weight.data[target_tokenizer.vocab[key], :] = origin_embed.weight.data[origin_tokenizer.vocab[key], :]
#     else:
#         model.lm_head.weight.data[target_tokenizer.vocab[key], :] = lm_head_8b[origin_tokenizer_8b.vocab[key], :] @ proj_lm_head
#         model.model.embed_tokens.weight.data[target_tokenizer.vocab[key], :] = embed_8b[origin_tokenizer_8b.vocab[key], :] @ proj_embed

total=0.

for name, param in model.named_parameters():
    total += param.nelement()
print('  + Number of trainable params: %.2fM' % (total / 1e6))
vicuna.save_pretrained('/data/luogen_code/LLaVA-HR-OCR/checkpoints/mod-vicuna-10b-stage2_3')