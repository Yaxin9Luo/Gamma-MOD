import torch
import os
import random
import time  # Added for time measurement

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from gamma_mod.model.builder import load_pretrained_model
from gamma_mod.mm_utils import get_model_name_from_path,tokenizer_image_token, process_images,KeywordsStoppingCriteria
from gamma_mod.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from gamma_mod.conversation import conv_templates, SeparatorStyle
from argparse import ArgumentParser

from gamma_mod.eval.mmmu_utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from gamma_mod.eval.mmmu_utils.model_utils import call_llava_engine_df, llava_image_processor
from gamma_mod.eval.mmmu_utils.eval_utils import parse_multi_choice_response, parse_open_response


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    total_inference_time = 0
    total_num_tokens = 0
    total_samples = 0
    max_samples = 10000
    start_time = time.time()  # Start measuring total time
    with torch.no_grad():
        for sample in tqdm(samples):
            total_samples += 1
            if total_samples >= max_samples:
                break
            qs=sample['final_input_prompt']
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            if sample['image'] is not None:
                with torch.inference_mode():
                    start_time = time.time()  # Start measuring inference time
                    output_ids = model.generate(
                        input_ids,
                        images=sample['image'].unsqueeze(0).half().cuda(),
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1,
                        output_attentions=True,
                        use_cache=True)
                    inference_time = time.time() - start_time  # Calculate inference time
                    total_inference_time += inference_time  # Add to total inference time

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                num_tokens = output_ids.shape[1] - input_token_len
                total_num_tokens += num_tokens

                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                response = outputs.strip()

            else:  # multiple images actually
                if sample['question_type'] == 'multiple-choice':
                    all_choices = sample['all_choices']
                    response = random.choice(all_choices)
                else:
                    response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    end_time = time.time()  # End measuring total time
    total_time = end_time - start_time
    print('Total inference time: ', total_inference_time)
    print('Total tokens: ', total_num_tokens)
    print('Tokens per second: ', float(total_num_tokens) / total_inference_time)
    print('Total samples: ', total_samples)
    print('Total time: ', total_time)
    print('Samples per second: ', float(total_samples) / total_inference_time)
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

llava_config={
    'multi_choice_example_format':"{}\n{}\nAnswer with the option's letter from the given choices directly.",
    'short_ans_example_format':"{}\nAnswer the question using a single word or phrase.",
    'task_instructions':'',
    'temperature':0.
}

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
                        help='name of saved json')
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # set_seed(args.seed)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor


    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, llava_config)
        if sample['image']:
            sample['image_size']=sample['image'].size
            sample['image'] =  process_images([sample['image'].convert('RGB')], image_processor, model.config)[0]#vis_process_func(sample['image'], vis_processors).to(device)
            #print(sample['final_input_prompt'])
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()