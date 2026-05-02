#!/bin/bash
# Train TRM on Qwen3-VL hidden states.
#
# Two parallel ablations on the 2B hidden states:
#   GPU 0: mlp_ratio=2, no max_seq_len cap (smaller net, full context)
#   GPU 1: mlp_ratio=4, max_seq_len=384 pruning (paper-size net, capped context)
#
# Usage:
#   bash scripts/run_trm_train.sh                # Both GPUs in parallel (default)
#   bash scripts/run_trm_train.sh mlp2           # Only GPU 0 ablation
#   bash scripts/run_trm_train.sh mlp4           # Only GPU 1 ablation
#   bash scripts/run_trm_train.sh 4b             # 4B model on GPU 1 (legacy)

set -e

# Use the project-local SQLite tracking store so mlflow ui reads the same data.
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
# System metrics (CPU/GPU/RAM) — also enabled programmatically inside train.py.
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=10

MODEL_2B="Qwen3-VL-2B-Instruct"
MODEL_4B="Qwen3-VL-4B-Instruct"

# Effective batch size = batch_size * grad_accum_steps = 1 * 16 = 16.
# Paper used 64; sqrt-scaled LR ~ 5e-5 (down from 1e-4).
COMMON_ARGS="
    --train_labels data/visulogic_train/visulogic_train_qwen.jsonl
    --val_labels data/visulogic_benchmark/data.jsonl
    --train_label_key answer
    --val_label_key label
    --batch_size 1
    --grad_accum_steps 16
    --lr 5e-5
    --epochs 50
    --n_latent_steps 6
    --n_deep_passes 3
    --n_sup_steps 8
    --patience 10
    --iter_log_every 20
"

# Ablation A: smaller MLP, no sequence cap.
run_mlp2() {
    echo "[GPU 0] TRM ablation: mlp_ratio=2, no seq cap on $MODEL_2B"
    python -m trm.train \
        --model_name $MODEL_2B \
        --dim 2048 \
        --mlp_ratio 2 \
        --gpu 0 \
        --run_tag mlp2_fullseq \
        $COMMON_ARGS
}

# Ablation B: paper MLP, prune long sequences.
run_mlp4() {
    echo "[GPU 1] TRM ablation: mlp_ratio=4, max_seq_len=384 on $MODEL_2B"
    python -m trm.train \
        --model_name $MODEL_2B \
        --dim 2048 \
        --mlp_ratio 4 \
        --max_seq_len 384 \
        --gpu 1 \
        --run_tag mlp4_seqcap384 \
        $COMMON_ARGS
}

# Legacy: train on 4B hidden states (kept for reference).
run_4b() {
    echo "[GPU 1] TRM training on $MODEL_4B"
    python -m trm.train \
        --model_name $MODEL_4B \
        --dim 2048 \
        --gpu 1 \
        $COMMON_ARGS
}

mkdir -p logs

case "${1:-both}" in
    mlp2) run_mlp2 ;;
    mlp4) run_mlp4 ;;
    4b|4B) run_4b ;;
    both)
        run_mlp2 > logs/trm_mlp2_fullseq.log 2>&1 &
        PID1=$!
        run_mlp4 > logs/trm_mlp4_seqcap384.log 2>&1 &
        PID2=$!
        echo "Training PIDs: mlp2=$PID1 (logs/trm_mlp2_fullseq.log), mlp4=$PID2 (logs/trm_mlp4_seqcap384.log)"
        echo "Tail logs:  tail -f logs/trm_mlp2_fullseq.log logs/trm_mlp4_seqcap384.log"
        wait $PID1 $PID2
        echo "Both ablation runs complete."
        ;;
    *) echo "Usage: $0 [mlp2|mlp4|4b|both]"; exit 1 ;;
esac
