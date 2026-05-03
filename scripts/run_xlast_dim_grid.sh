#!/bin/bash
# Dim ablation grid for x_last_only TRM ablation.
#
# For each VLM, train TRMs with internal dim ∈ {128, 256, 512, 1024, 2048} using
# an input projection from VLM hidden size to TRM dim. RoPE is disabled (paper §4.5:
# unnecessary when L <= D; here L=3) and N_sup is bumped to 16 (paper value).
#
# GPU 0 runs the Qwen3-VL-2B grid, GPU 1 runs the Qwen2.5-VL-7B grid in parallel.
# Each grid runs its 5 dims sequentially within its GPU.
#
# Usage:
#   bash scripts/run_xlast_dim_grid.sh                 # both GPUs in parallel (default)
#   bash scripts/run_xlast_dim_grid.sh 2b              # only the 2B grid (GPU 0)
#   bash scripts/run_xlast_dim_grid.sh 7b              # only the 7B grid (GPU 1)

set -e

export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=10

mkdir -p logs

DIMS=(128 256 512 1024 2048)

run_one() {
    # $1=model_name  $2=input_dim  $3=trm_dim  $4=gpu
    local MODEL=$1 IDIM=$2 DIM=$3 GPU=$4
    # head_dim ~= 64; minimum 1 head; for dim < 64 default to 1 head.
    local NHEADS=$(( DIM / 64 ))
    [ "$NHEADS" -lt 1 ] && NHEADS=1
    local TAG="xlast_d${DIM}_norope"

    echo "[GPU $GPU] $MODEL  input_dim=$IDIM  dim=$DIM  n_heads=$NHEADS  tag=$TAG"
    python -m trm.train \
        --model_name "$MODEL" \
        --train_hidden_dir "outputs/hidden_states/${MODEL}" \
        --val_hidden_dir "outputs/hidden_states/val/${MODEL}" \
        --train_labels data/visulogic_train/visulogic_train_qwen.jsonl \
        --val_labels data/visulogic_benchmark/data.jsonl \
        --train_label_key answer \
        --val_label_key label \
        --input_dim "$IDIM" \
        --dim "$DIM" \
        --n_heads "$NHEADS" \
        --mlp_ratio 2 \
        --block_type attn \
        --no_rope \
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
        --n_sup_steps 16 \
        --patience 20 \
        --iter_log_every 20
    # Free latest.pt right after to keep disk usage in check (best.pt stays).
    local CKPT_DIR="checkpoints/trm/${MODEL}/${TAG}/latest.pt"
    [ -f "$CKPT_DIR" ] && rm -f "$CKPT_DIR"
}

run_grid_2b() {
    local MODEL="Qwen3-VL-2B-Instruct"
    local IDIM=2048 GPU=0
    for D in "${DIMS[@]}"; do
        run_one "$MODEL" "$IDIM" "$D" "$GPU" \
            > "logs/trm_2b_xlast_d${D}_norope.log" 2>&1
    done
}

run_grid_7b() {
    local MODEL="Qwen2.5-VL-7B-Instruct"
    local IDIM=3584 GPU=1
    for D in "${DIMS[@]}"; do
        run_one "$MODEL" "$IDIM" "$D" "$GPU" \
            > "logs/trm_7b_xlast_d${D}_norope.log" 2>&1
    done
}

case "${1:-both}" in
    2b|2B) run_grid_2b ;;
    7b|7B) run_grid_7b ;;
    both)
        run_grid_2b &
        PID1=$!
        run_grid_7b &
        PID2=$!
        echo "Grid PIDs: 2b=$PID1, 7b=$PID2"
        echo "Tail:  tail -f logs/trm_2b_xlast_d*.log logs/trm_7b_xlast_d*.log"
        wait $PID1 $PID2
        echo "Both grids complete."
        ;;
    *) echo "Usage: $0 [2b|7b|both]"; exit 1 ;;
esac
