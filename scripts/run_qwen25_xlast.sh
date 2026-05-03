#!/bin/bash
# Pipeline: extract Qwen2.5-VL-7B-Instruct hidden states for both train and val,
# then train the TRM x_last_only ablation (mlp_ratio=2, learnable y_init/z_init,
# fp32 full precision) on those hidden states.
#
# Usage:
#   bash scripts/run_qwen25_xlast.sh
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct GPU_IDS="0,1" bash scripts/run_qwen25_xlast.sh
#   bash scripts/run_qwen25_xlast.sh extract   # only extraction
#   bash scripts/run_qwen25_xlast.sh train     # only training (assumes hidden states exist)

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-VL-7B-Instruct"}
MODEL_NAME=$(basename "$MODEL")     # -> Qwen2.5-VL-7B-Instruct
GPU_IDS=${GPU_IDS:-"0,1"}            # GPUs for extraction (uses both for device_map=auto)
TRAIN_GPU=${TRAIN_GPU:-0}            # single GPU for TRM training
USER_PROMPT=${USER_PROMPT:-"sft"}
RUN_TAG=${RUN_TAG:-"qwen25_7b_mlp2_xlast"}

TRAIN_HIDDEN_DIR="outputs/hidden_states/${MODEL_NAME}"
VAL_HIDDEN_DIR="outputs/hidden_states/val/${MODEL_NAME}"

extract() {
    echo "=== Extracting train hidden states -> $TRAIN_HIDDEN_DIR ==="
    python scripts/extract_hidden_states_v2.py \
        --input_file data/visulogic_train/visulogic_train_qwen.jsonl \
        --model_path "$MODEL" \
        --output_dir "$TRAIN_HIDDEN_DIR" \
        --gpu_ids "$GPU_IDS" \
        --user_prompt "$USER_PROMPT" \
        --last_only

    echo "=== Extracting val hidden states -> $VAL_HIDDEN_DIR ==="
    python scripts/extract_hidden_states_v2.py \
        --input_file data/visulogic_benchmark/data.jsonl \
        --model_path "$MODEL" \
        --output_dir "$VAL_HIDDEN_DIR" \
        --gpu_ids "$GPU_IDS" \
        --user_prompt "$USER_PROMPT" \
        --last_only
}

train_one() {
    # Args: $1=mlp_ratio  $2=gpu_id  $3=run_tag
    local MR=$1 GPU=$2 TAG=$3
    echo "[GPU $GPU] Training TRM x_last_only on $MODEL_NAME (fp32, mlp_ratio=$MR, tag=$TAG)"
    python -m trm.train \
        --model_name "$MODEL_NAME" \
        --train_hidden_dir "$TRAIN_HIDDEN_DIR" \
        --val_hidden_dir "$VAL_HIDDEN_DIR" \
        --train_labels data/visulogic_train/visulogic_train_qwen.jsonl \
        --val_labels data/visulogic_benchmark/data.jsonl \
        --train_label_key answer \
        --val_label_key label \
        --dim 3584 \
        --n_heads 28 \
        --mlp_ratio "$MR" \
        --x_last_only \
        --no_fp16 \
        --gpu "$GPU" \
        --run_tag "$TAG" \
        --batch_size 32 \
        --grad_accum_steps 2 \
        --lr 1e-4 \
        --epochs 50 \
        --n_latent_steps 6 \
        --n_deep_passes 3 \
        --n_sup_steps 8 \
        --patience 20 \
        --iter_log_every 20
}

train() {
    # MLflow tracking (matches our existing runs so they show up alongside in the UI)
    export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
    export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
    export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=10
    mkdir -p logs

    # Ablation A: GPU 0, mlp_ratio=2
    train_one 2 0 "qwen25_7b_mlp2_xlast" > "logs/trm_qwen25_7b_mlp2_xlast.log" 2>&1 &
    PID1=$!
    # Ablation B: GPU 1, mlp_ratio=4
    train_one 4 1 "qwen25_7b_mlp4_xlast" > "logs/trm_qwen25_7b_mlp4_xlast.log" 2>&1 &
    PID2=$!
    echo "Training PIDs: mlp2=$PID1, mlp4=$PID2"
    echo "Tail:  tail -f logs/trm_qwen25_7b_mlp2_xlast.log logs/trm_qwen25_7b_mlp4_xlast.log"
    wait $PID1 $PID2
    echo "Both Qwen2.5-VL-7B x_last ablations complete."
}

cleanup_weights() {
    # Hidden states are already on disk; VLM weights are not needed for TRM training.
    # HF cache layout: models--<org>--<repo>  (slashes -> double-dash)
    local SAFE
    SAFE=$(echo "$MODEL" | sed 's|/|--|g')
    local CACHE="$HOME/.cache/huggingface/hub/models--$SAFE"
    if [ -d "$CACHE" ]; then
        echo "=== Deleting VLM cache to free space: $CACHE ==="
        du -sh "$CACHE" || true
        rm -rf "$CACHE"
        df -h . | tail -1
    else
        echo "[cleanup_weights] cache dir not found, skipping: $CACHE"
    fi
}

case "${1:-all}" in
    extract) extract ;;
    train)   train ;;
    all)     extract; cleanup_weights; train ;;
    *)       echo "Usage: $0 [extract|train|all]"; exit 1 ;;
esac
