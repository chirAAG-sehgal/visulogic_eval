#!/bin/bash
# Robust download (no --include filter) + extract for Qwen3-VL-4B-Thinking.
set -e
cd /home/uasdtu/Documents/chirag/visulogic_eval
NAME="Qwen3-VL-4B-Thinking"

echo "[fresh] downloading full repo via hf cli (resumes shard 1)"
for try in 1 2 3 4 5; do
    if hf download "Qwen/${NAME}" --max-workers 4; then
        echo "[fresh] download attempt $try succeeded"
        break
    fi
    echo "[fresh] attempt $try failed; sleeping 15s"
    sleep 15
done

echo "[fresh] snapshot contents:"
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Thinking/snapshots/*/

echo "[fresh] extraction (train)"
python scripts/extract_hidden_states_v2.py \
    --input_file data/visulogic_train/visulogic_train_qwen.jsonl \
    --model_path "Qwen/${NAME}" \
    --output_dir "outputs/hidden_states/${NAME}" \
    --gpu_ids 0 --user_prompt sft --last_only \
    > "logs/extract_${NAME}_train.log" 2>&1

echo "[fresh] extraction (val)"
python scripts/extract_hidden_states_v2.py \
    --input_file data/visulogic_benchmark/data.jsonl \
    --model_path "Qwen/${NAME}" \
    --output_dir "outputs/hidden_states/val/${NAME}" \
    --gpu_ids 0 --user_prompt sft --last_only \
    > "logs/extract_${NAME}_val.log" 2>&1

echo "[fresh] DONE. Freeing weights cache."
rm -rf "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Thinking" || true
df -h . | tail -1
