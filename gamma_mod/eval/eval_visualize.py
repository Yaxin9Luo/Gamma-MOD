import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt
import seaborn as sns
from gamma_mod.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from gamma_mod.conversation import conv_templates, SeparatorStyle
from gamma_mod.model.builder import load_pretrained_model
from gamma_mod.utils import disable_torch_init
from gamma_mod.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria,process_images
from deepspeed.profiling.flops_profiler import get_model_profile
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import math
import time  # Added for measuring inference speed
def visualize_route_prob(route_prob, layer_num):
    plt.figure(figsize=(12, 6))
    sns.histplot(route_prob[layer_num][0].cpu().numpy(), kde=True, bins=50)
    plt.title(f"Distribution of Route Probabilities for Layer {layer_num}")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    output_path = f'/data/luogen_code/LLaVA-HR-OCR/visualize/qualitative_results/route_prob_distribution_layer_{layer_num}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Route probability distribution saved to: {output_path}")
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def tokens_to_string(tokens):
    return ''.join([t.replace('▁', ' ') for t in tokens]).strip()
def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def visualize_routed_image(image, route_index, num_patches,layer_num):
    orig_width, orig_height = image.size
    # print(f"Original image size: {orig_width}x{orig_height}")
    input_tokens_len = 37
    # LLaVA typically uses a 32x32 patch grid
    grid_size = 32
    model_input_size = grid_size * 32  # Assuming 32x32 pixel patches, adjust if different

    # Resize image to model input size while maintaining aspect ratio
    ratio = min(model_input_size / orig_width, model_input_size / orig_height)
    new_size = (int(orig_width * ratio), int(orig_height * ratio))
    resized_image = image.resize(new_size, Image.LANCZOS)
    # print(f"Resized image size: {new_size}")

    # Create a blank image with the model input size
    model_input_image = Image.new('RGB', (model_input_size, model_input_size), color='gray')
    
    # Paste the resized image onto the center of the blank image
    paste_x = (model_input_size - new_size[0]) // 2
    paste_y = (model_input_size - new_size[1]) // 2
    model_input_image.paste(resized_image, (paste_x, paste_y))
    # print(f"Model input image size: {model_input_size}x{model_input_size}")

    # Create a mask for the model input image
    mask = np.ones((model_input_size, model_input_size), dtype=np.float32)

    patch_size = model_input_size // grid_size
    # print(f"Patch size: {patch_size}x{patch_size}")

    # Filter route_index to only include image tokens
    image_route_index = [idx for idx in route_index if input_tokens_len <= idx < input_tokens_len + grid_size * grid_size]
    # print(f"Processing {len(image_route_index)} routed image tokens")
    print(f"Layer {layer_num} routed image tokens proportion: {len(image_route_index) / 1024 }")
    # for idx in route_index:
    #     # if input_tokens_len <= idx < input_tokens_len + grid_size * grid_size:  # Process all image tokens
    #         image_token_idx = idx - input_tokens_len
    #         row = image_token_idx // grid_size
    #         col = image_token_idx % grid_size
            
    #         start_x = col * patch_size
    #         start_y = row * patch_size
    #         end_x = start_x + patch_size
    #         end_y = start_y + patch_size
            
    #         # print(f"\nToken {idx}: Grid position ({row}, {col})")
    #         # print(f"Image area: ({start_x}, {start_y}) to ({end_x}, {end_y})")
            
    #         mask[start_y:end_y, start_x:end_x] = 0.5

    # # Apply mask to model input image
    # masked_image = np.array(model_input_image) * mask[:,:,np.newaxis]

    # # Crop the masked image back to the original aspect ratio
    # crop_x = paste_x
    # crop_y = paste_y
    # crop_width = new_size[0]
    # crop_height = new_size[1]
    # masked_image = masked_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    # # print(f"\nCreating and saving visualization")
    # plt.figure(figsize=(10, 10 * orig_height / orig_width))
    # plt.imshow(masked_image.astype(np.uint8))
    # plt.axis('off')
    
    # output_path = f'/data/luogen_code/LLaVA-HR-OCR/visualize/qualitative_results/route_visualize_layer_{layer_num}.png'
    # plt.savefig(output_path, dpi=400, bbox_inches='tight')
    # plt.close()
    # print(f"Visualization saved to: {output_path}")
def safe_convert_ids_to_tokens(tokenizer, ids):
    tokens = []
    for id in ids:
        try:
            tokens.append(tokenizer._tokenizer.id_to_token(id.item()))
        except OverflowError:
            tokens.append(f"[UNKNOWN_{id.item()}]")
    return tokens
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    if args.custom_example:
        custom_image = Image.open(args.custom_image_path)
        image_width, image_height = custom_image.size
        print(f"Image resolution: {image_width}x{image_height}")
        custom_question = args.custom_question
        images = process_images([custom_image], image_processor, model.config).half().cuda()
        if getattr(model.config, 'mm_use_im_start_end', False):
            print("Using im_start and im_end")
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + custom_question
        else:
            print("Using separate image token")
            qs = DEFAULT_IMAGE_TOKEN + '\n' + custom_question
        cur_prompt = '<image>' + '\n' + custom_question
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                output_attentions=True
            )
            ########## only for visualization purpose ##########
            output = model(input_ids, images=images,return_dict=True)
        ########### codes for visualization ###########
        for i in range(28):
            print(f"layer_num: {i} !!!!!!!!!!!!!!!!!! /n")
            layer_num = i
            route_index = output.route_index
            route_prob = route_index[0]
            # print(f"route_prob: {route_prob[layer_num]}")
            # visualize_route_prob(route_prob, layer_num)
            route_index = torch.where(route_prob[layer_num][0] < 0.005)[0]
            # values_list = route_prob[layer_num].squeeze().tolist()
            # for i, value in enumerate(values_list):
            #     print(f"Index {i}: {value}")
            # print(f"route_index: {route_index}")
            # print(f"shape of route_index: {route_index.shape}")        
            input_tokens = safe_convert_ids_to_tokens(tokenizer, input_ids[0])
            # print(f"input_tokens: {input_tokens}")
            # print(f"Input text: {tokens_to_string(input_tokens)}")

            routed_tokens = [input_tokens[i] for i in route_index if i < 45 ]
            # print(f"Routed question tokens: {routed_tokens}")
            # print(f"Routed question text: {tokens_to_string(routed_tokens)}")
            print(f"Layer {layer_num} Question tokens routing proportions: {len(routed_tokens) / (len(input_tokens)-37)}")
            # Visualize routed image parts
            num_patches = 1024  # Adjust based on your model's configuration
            visualize_routed_image(custom_image, route_index, num_patches,layer_num)
        ############ codes for generation ############

        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        num_output_tokens = output_ids.shape[1] - input_ids.shape[1]
        print(f"Number of output tokens: {num_output_tokens}")
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print(f"Question: {custom_question}")
        print(f"Answer: {outputs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--custom-example", action="store_true", help="Use a custom example for visualization")
    parser.add_argument("--custom-image-path", type=str, help="Path to the custom image")
    parser.add_argument("--custom-question", type=str, help="Custom question for the image")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()

    eval_model(args)