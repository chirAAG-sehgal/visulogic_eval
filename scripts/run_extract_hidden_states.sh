#!/bin/bash
# Extract final-layer hidden states from Qwen3-VL for each sample.
#
# Usage:
#   bash scripts/run_extract_hidden_states.sh
#   bash scripts/run_extract_hidden_states.sh Qwen_Qwen3-VL-4B-Instruct
#
# Args:
#   $1 — model folder name under weights/ (default: Qwen_Qwen3-VL-2B-Instruct)

MODEL_NAME=${1:-"Qwen/Qwen2.5-VL-7B-Instruct"}

python scripts/extract_hidden_states.py \
    --input_file data/visulogic_benchmark/data.jsonl \
    --model_path "${MODEL_NAME}" \
    --gpu_ids ${GPU_IDS:-"0,1"} \
    --user_prompt ${USER_PROMPT:-"sft"}

MODEL_NAME=${1:-"Qwen/Qwen3-VL-2B-Instruct"}

python scripts/extract_hidden_states.py \
    --input_file data/visulogic_benchmark/data.jsonl \
    --model_path "${MODEL_NAME}" \
    --gpu_ids ${GPU_IDS:-"0,1"} \
    --user_prompt ${USER_PROMPT:-"sft"}
