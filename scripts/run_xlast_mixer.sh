#!/bin/bash
# Run the MLP-Mixer variant (paper §4.5) at the best dim found by the dim grid,
# one run per VLM. Set BEST_DIM_2B and BEST_DIM_7B to the winning dims from
# `bash scripts/run_xlast_dim_grid.sh` before invoking this.
#
# Usage:
#   BEST_DIM_2B=512 BEST_DIM_7B=512 bash scripts/run_xlast_mixer.sh
#   BEST_DIM_2B=512 bash scripts/run_xlast_mixer.sh 2b      # only the 2B mixer run

set -e

export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=10

mkdir -p logs

: "${BEST_DIM_2B:=512}"
: "${BEST_DIM_7B:=512}"

run_one() {
    # $1=model_name  $2=input_dim  $3=trm_dim  $4=gpu
    local MODEL=$1 IDIM=$2 DIM=$3 GPU=$4
    local TAG="xlast_d${DIM}_mixer"
    echo "[GPU $GPU] MIXER  $MODEL  input_dim=$IDIM  dim=$DIM  tag=$TAG"
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
        --mlp_ratio 2 \
        --block_type mixer \
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
    local CKPT="checkpoints/trm/${MODEL}/${TAG}/latest.pt"
    [ -f "$CKPT" ] && rm -f "$CKPT"
}

run_2b() { run_one "Qwen3-VL-2B-Instruct" 2048 "$BEST_DIM_2B" 0 \
    > "logs/trm_2b_xlast_d${BEST_DIM_2B}_mixer.log" 2>&1 ; }
run_7b() { run_one "Qwen2.5-VL-7B-Instruct" 3584 "$BEST_DIM_7B" 1 \
    > "logs/trm_7b_xlast_d${BEST_DIM_7B}_mixer.log" 2>&1 ; }

case "${1:-both}" in
    2b|2B) run_2b ;;
    7b|7B) run_7b ;;
    both)
        run_2b &
        PID1=$!
        run_7b &
        PID2=$!
        echo "Mixer PIDs: 2b=$PID1, 7b=$PID2"
        wait $PID1 $PID2
        echo "Both mixer runs complete."
        ;;
    *) echo "Usage: BEST_DIM_2B=N BEST_DIM_7B=N $0 [2b|7b|both]"; exit 1 ;;
esac
