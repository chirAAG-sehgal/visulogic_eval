#!/bin/bash
# Run all 4 Qwen3-VL evals in parallel across 7x 12GB GPUs.
#
# GPU allocation (each 12GB):
#   GPU 0     -> Qwen3-VL-2B-Instruct  (~4.5GB)
#   GPU 1     -> Qwen3-VL-2B-Thinking  (~4.5GB)
#   GPU 2     -> Qwen3-VL-4B-Instruct  (~8.5GB)
#   GPU 3     -> Qwen3-VL-4B-Thinking  (~8.5GB)
#   GPUs 4-6  -> free for other work / training
#
# Adjust BATCH_SIZE based on available VRAM headroom.
# 2B models: batch_size=4 uses ~10GB, 4B models: batch_size=2 uses ~11GB.
# Reduce if you hit OOM.
#
# Usage:
#   bash scripts/eval_all_qwen3vl_parallel.sh
#   BATCH_SIZE=2 bash scripts/eval_all_qwen3vl_parallel.sh

set -e
mkdir -p outputs/

echo "Starting all Qwen3-VL evaluations in parallel..."

# 2B models — fit comfortably, larger batch
GPU_IDS="0" BATCH_SIZE=${BATCH_SIZE:-4} bash scripts/eval_qwen3vl_2b_instruct.sh &
PID1=$!

GPU_IDS="1" BATCH_SIZE=${BATCH_SIZE:-4} bash scripts/eval_qwen3vl_2b_thinking.sh &
PID2=$!

# 4B models — tighter fit, smaller batch
GPU_IDS="2" BATCH_SIZE=${BATCH_SIZE_4B:-2} bash scripts/eval_qwen3vl_4b_instruct.sh &
PID3=$!

GPU_IDS="3" BATCH_SIZE=${BATCH_SIZE_4B:-2} bash scripts/eval_qwen3vl_4b_thinking.sh &
PID4=$!

echo "PIDs: 2B-Instruct=$PID1 2B-Thinking=$PID2 4B-Instruct=$PID3 4B-Thinking=$PID4"
echo "Waiting for all jobs to complete..."

FAIL=0
wait $PID1 || { echo "2B-Instruct FAILED"; FAIL=1; }
wait $PID2 || { echo "2B-Thinking FAILED"; FAIL=1; }
wait $PID3 || { echo "4B-Instruct FAILED"; FAIL=1; }
wait $PID4 || { echo "4B-Thinking FAILED"; FAIL=1; }

if [ $FAIL -eq 0 ]; then
    echo "All evaluations completed successfully!"
else
    echo "Some evaluations failed. Check logs above."
    exit 1
fi
