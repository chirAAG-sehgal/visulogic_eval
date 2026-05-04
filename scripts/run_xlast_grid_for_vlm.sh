#!/bin/bash
# Run the x_last_only no_rope dim grid for one VLM, sequentially on one GPU.
#
# Usage:
#   bash scripts/run_xlast_grid_for_vlm.sh MODEL_NAME INPUT_DIM GPU "DIM1 DIM2 ..."
#
# Example:
#   bash scripts/run_xlast_grid_for_vlm.sh Qwen3-VL-4B-Instruct 2560 0 "128 256 512 1024 2048 2560"
#
# Recipe (matches scripts/run_xlast_dim_grid.sh):
#   --x_last_only --no_rope --block_type attn --no_fp16 --mlp_ratio 2
#   --batch_size 32 --grad_accum_steps 2 --lr 1e-4 --epochs 50
#   --n_latent_steps 6 --n_deep_passes 3 --n_sup_steps 16 --patience 20

set -e

MODEL=$1
IDIM=$2
GPU=$3
DIMS=$4
# Optional 5th positional or BLOCK_TYPE env var: 'attn' (default) or 'mixer'
BLOCK_TYPE=${5:-${BLOCK_TYPE:-attn}}
TAG_SUFFIX=${TAG_SUFFIX:-norope}  # appended to run_tag, default 'norope'

if [ -z "$MODEL" ] || [ -z "$IDIM" ] || [ -z "$GPU" ] || [ -z "$DIMS" ]; then
    echo "Usage: $0 MODEL_NAME INPUT_DIM GPU \"DIM1 DIM2 ...\" [BLOCK_TYPE]"
    echo "  BLOCK_TYPE = attn (default) or mixer"
    exit 1
fi

export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=10

mkdir -p logs

# Short tag prefix for log filenames: 2b/4b/7b/2bt/4bt etc, derived from MODEL.
case "$MODEL" in
    *2B-Instruct*) TAGPRE="2b" ;;
    *4B-Instruct*) TAGPRE="4b" ;;
    *7B-Instruct*) TAGPRE="7b" ;;
    *2B-Thinking*) TAGPRE="2bt" ;;
    *4B-Thinking*) TAGPRE="4bt" ;;
    *) TAGPRE="x" ;;
esac

for D in $DIMS; do
    NHEADS=$(( D / 64 ))
    [ "$NHEADS" -lt 1 ] && NHEADS=1
    # Match Qwen native head count when D equals VLM hidden_size (3584/56, 2560/32, 2048/16).
    case "$D-$IDIM" in
        3584-3584) NHEADS=56 ;;
        2560-2560) NHEADS=32 ;;
        2048-2048) NHEADS=32 ;;  # keep d/64 for native 2048 too
    esac
    # Tag includes block type for clarity; "norope" suffix preserved by default for back-compat.
    if [ "$BLOCK_TYPE" = "mixer" ]; then
        TAG="xlast_d${D}_mixer"
        LOG="logs/trm_${TAGPRE}_xlast_d${D}_mixer.log"
    else
        TAG="xlast_d${D}_${TAG_SUFFIX}"
        LOG="logs/trm_${TAGPRE}_xlast_d${D}_${TAG_SUFFIX}.log"
    fi
    echo "[GPU $GPU] $MODEL  input_dim=$IDIM  dim=$D  n_heads=$NHEADS  block=$BLOCK_TYPE  tag=$TAG  log=$LOG"
    python -m trm.train \
        --model_name "$MODEL" \
        --train_hidden_dir "outputs/hidden_states/${MODEL}" \
        --val_hidden_dir "outputs/hidden_states/val/${MODEL}" \
        --train_labels data/visulogic_train/visulogic_train_qwen.jsonl \
        --val_labels data/visulogic_benchmark/data.jsonl \
        --train_label_key answer --val_label_key label \
        --input_dim "$IDIM" --dim "$D" --n_heads "$NHEADS" --mlp_ratio 2 \
        --block_type "$BLOCK_TYPE" --no_rope --x_last_only --no_fp16 \
        --gpu "$GPU" --run_tag "$TAG" \
        --batch_size 32 --grad_accum_steps 2 --lr 1e-4 \
        --epochs 50 --n_latent_steps 6 --n_deep_passes 3 --n_sup_steps 16 \
        --patience 20 --iter_log_every 20 \
        > "$LOG" 2>&1
    # Free latest.pt to keep disk in check (best.pt stays).
    rm -f "checkpoints/trm/${MODEL}/${TAG}/latest.pt"
done

echo "[GPU $GPU] Grid complete for $MODEL: dims = $DIMS"
