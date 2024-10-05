#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path /data/luogen_code/LLaVA-HR-OCR/checkpoints/llava-hr-10b-sft-1024-2 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-10b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-10b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-10b_result.json
