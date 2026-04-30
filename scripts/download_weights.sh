#!/bin/bash
# Download model weights locally using huggingface-cli
# Usage: bash scripts/download_weights.sh [model_name]
# If no model_name is given, all models are downloaded.
#
# Prerequisites:
#   pip install huggingface_hub[cli]
#   huggingface-cli login  (if gated models require authentication)

WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

download_model() {
    local model_id="$1"
    local local_dir="$WEIGHTS_DIR/$(echo $model_id | tr '/' '_')"
    if [ -d "$local_dir" ] && [ "$(ls -A $local_dir)" ]; then
        echo "Model $model_id already exists at $local_dir, skipping."
    else
        echo "Downloading $model_id to $local_dir ..."
        huggingface-cli download "$model_id" --local-dir "$local_dir"
    fi
}

MODELS=(
    "Qwen/Qwen2.5-VL-72B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-2B-Thinking"
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-4B-Thinking"
)

if [ -n "$1" ]; then
    download_model "$1"
else
    for model in "${MODELS[@]}"; do
        download_model "$model"
    done
fi
