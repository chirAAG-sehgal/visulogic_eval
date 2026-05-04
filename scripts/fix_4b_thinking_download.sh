#!/bin/bash
# One-off: robustly download + extract Qwen3-VL-4B-Thinking after stalled HF download.
set -e
cd /home/uasdtu/Documents/chirag/visulogic_eval

NAME="Qwen3-VL-4B-Thinking"

echo "[fix] downloading via hf cli with up to 5 retries"
for try in 1 2 3 4 5; do
    if hf download "Qwen/${NAME}" \
         --include "*.safetensors" "*.json" "*.txt" "tokenizer*" \
         --max-workers 4; then
        echo "[fix] download attempt $try succeeded"
        break
    fi
    echo "[fix] attempt $try failed; sleeping 15s before retry"
    sleep 15
done

echo "[fix] launching extraction (train)"
python scripts/extract_hidden_states_v2.py \
    --input_file data/visulogic_train/visulogic_train_qwen.jsonl \
    --model_path "Qwen/${NAME}" \
    --output_dir "outputs/hidden_states/${NAME}" \
    --gpu_ids 0 \
    --user_prompt sft \
    --last_only > "logs/extract_${NAME}_train.log" 2>&1

echo "[fix] launching extraction (val)"
python scripts/extract_hidden_states_v2.py \
    --input_file data/visulogic_benchmark/data.jsonl \
    --model_path "Qwen/${NAME}" \
    --output_dir "outputs/hidden_states/val/${NAME}" \
    --gpu_ids 0 \
    --user_prompt sft \
    --last_only > "logs/extract_${NAME}_val.log" 2>&1

echo "[fix] DONE. Freeing VLM weights cache."
rm -rf "$HOME/.cache/huggingface/hub/models--Qwen--${NAME//\//-}" || true
df -h . | tail -1
