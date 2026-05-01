#!/bin/bash
# Train TRM on Qwen3-VL hidden states.
#
# Usage:
#   bash scripts/run_trm_train.sh                    # Train both models in parallel
#   bash scripts/run_trm_train.sh 2B                 # Train only 2B
#   bash scripts/run_trm_train.sh 4B                 # Train only 4B

set -e

COMMON_ARGS="
    --train_labels data/visulogic_train/visulogic_train_qwen.jsonl
    --val_labels data/visulogic_benchmark/data.jsonl
    --train_label_key answer
    --val_label_key label
    --batch_size 16
    --grad_accum_steps 4
    --epochs 50
    --n_latent_steps 6
    --n_deep_passes 3
    --n_sup_steps 8
    --patience 10
"

run_2b() {
    echo "Starting TRM training for Qwen3-VL-2B-Instruct on GPU 0..."
    python -m trm.train \
        --model_name Qwen3-VL-2B-Instruct \
        --dim 2048 \
        --gpu 0 \
        $COMMON_ARGS
}

run_4b() {
    echo "Starting TRM training for Qwen3-VL-4B-Instruct on GPU 1..."
    python -m trm.train \
        --model_name Qwen3-VL-4B-Instruct \
        --dim 2048 \
        --gpu 1 \
        $COMMON_ARGS
}

case "${1:-both}" in
    2B|2b) run_2b ;;
    4B|4b) run_4b ;;
    both)
        run_2b &
        PID1=$!
        run_4b &
        PID2=$!
        echo "Training PIDs: 2B=$PID1, 4B=$PID2"
        wait $PID1 $PID2
        echo "Both training runs complete."
        ;;
    *) echo "Usage: $0 [2B|4B|both]"; exit 1 ;;
esac
