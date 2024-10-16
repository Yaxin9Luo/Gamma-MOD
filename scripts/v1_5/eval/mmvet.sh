#!/bin/bash
MODEL_PATH=$1
MODEL_BASE=$2
python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-10b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-10b.json
