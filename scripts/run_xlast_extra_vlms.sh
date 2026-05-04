#!/bin/bash
# Master orchestrator for the next wave of x_last_only dim grids:
#   - Qwen3-VL-4B-Instruct  (input_dim=2560, dims = 128 256 512 1024 2048 2560)
#   - Qwen3-VL-2B-Thinking  (input_dim=2048, dims = 128 256 512 1024 2048)
#   - Qwen3-VL-4B-Thinking  (input_dim=2560, dims = 128 256 512 1024 2048 2560)
#
# Behaviour:
#   1. Wait for both GPUs to become free (the in-flight 2B@2048 and 7B@3584 runs
#      need to finish first). Polls every 60s.
#   2. Extract last-token hidden states for the two Thinking VLMs (in parallel
#      across GPUs). Skipped automatically per sample if .pt already exists.
#   3. Launch in parallel: 4B-Inst grid on GPU 0, 2B-Thinking grid on GPU 1.
#   4. Once 2B-Thinking grid finishes (5 dims, smaller VLM = faster) launch
#      4B-Thinking grid on GPU 1. 4B-Inst grid keeps running on GPU 0.
#   5. Done.
#
# Usage:
#   nohup bash scripts/run_xlast_extra_vlms.sh > logs/run_xlast_extra_vlms.log 2>&1 &
#   disown

set -e

MODEL_4BI="Qwen3-VL-4B-Instruct"
MODEL_2BT="Qwen3-VL-2B-Thinking"
MODEL_4BT="Qwen3-VL-4B-Thinking"

DIMS_4BI="128 256 512 1024 2048 2560"
DIMS_2BT="128 256 512 1024 2048"
DIMS_4BT="128 256 512 1024 2048 2560"

mkdir -p logs

gpu_free() {
    # Returns 0 if `nvidia-smi` reports no compute apps on $1, else 1.
    local g=$1
    local n
    n=$(CUDA_VISIBLE_DEVICES= nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -c 100)
    # The above lists all GPUs' apps; instead probe per-GPU memory.
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    [ "${mem:-9999}" -lt 1000 ]
}

wait_for_both_gpus_free() {
    echo "[wait] waiting for GPU 0 and GPU 1 to become free (poll every 60s)..."
    local n=0
    while ! ( gpu_free 0 && gpu_free 1 ); do
        n=$((n+1))
        sleep 60
        if [ $((n % 10)) -eq 0 ]; then
            nvidia-smi --query-gpu=index,memory.used --format=csv | tail -2
        fi
    done
    echo "[wait] both GPUs free."
}

extract_thinking() {
    # Skip if hidden states already exist (the script itself also skips per-id).
    local NAME=$1 GPU=$2
    local OUT_TRAIN="outputs/hidden_states/${NAME}"
    local OUT_VAL="outputs/hidden_states/val/${NAME}"
    if [ -d "$OUT_TRAIN" ] && [ "$(ls "$OUT_TRAIN"/*.pt 2>/dev/null | wc -l)" -ge 4290 ] \
       && [ -d "$OUT_VAL" ]   && [ "$(ls "$OUT_VAL"/*.pt 2>/dev/null | wc -l)" -ge 990 ]; then
        echo "[extract] $NAME already extracted, skipping."
        return 0
    fi
    echo "[extract] $NAME on GPU $GPU"
    python scripts/extract_hidden_states_v2.py \
        --input_file data/visulogic_train/visulogic_train_qwen.jsonl \
        --model_path "Qwen/${NAME}" \
        --output_dir "$OUT_TRAIN" \
        --gpu_ids "$GPU" \
        --user_prompt sft \
        --last_only > "logs/extract_${NAME}_train.log" 2>&1
    python scripts/extract_hidden_states_v2.py \
        --input_file data/visulogic_benchmark/data.jsonl \
        --model_path "Qwen/${NAME}" \
        --output_dir "$OUT_VAL" \
        --gpu_ids "$GPU" \
        --user_prompt sft \
        --last_only > "logs/extract_${NAME}_val.log" 2>&1
    # Optional: drop the VLM weights cache after extraction to reclaim disk.
    local SAFE
    SAFE=$(echo "Qwen/${NAME}" | sed 's|/|--|g')
    local CACHE="$HOME/.cache/huggingface/hub/models--$SAFE"
    if [ -d "$CACHE" ]; then
        echo "[extract] freeing $CACHE"
        rm -rf "$CACHE"
    fi
    df -h . | tail -1
}

# === Step 1: wait for current GPU work to finish ===
wait_for_both_gpus_free

# === Step 2: extract Thinking-model hidden states in parallel ===
extract_thinking "$MODEL_2BT" 0 &
PID_E1=$!
extract_thinking "$MODEL_4BT" 1 &
PID_E2=$!
echo "Extract PIDs: 2B-T=$PID_E1, 4B-T=$PID_E2"
wait $PID_E1 $PID_E2
echo "[extract] both extractions done."

# === Step 3: launch 4B-Instruct (GPU 0) and 2B-Thinking (GPU 1) grids in parallel ===
bash scripts/run_xlast_grid_for_vlm.sh "$MODEL_4BI" 2560 0 "$DIMS_4BI" &
PID_4BI=$!
bash scripts/run_xlast_grid_for_vlm.sh "$MODEL_2BT" 2048 1 "$DIMS_2BT" &
PID_2BT=$!
echo "Grid PIDs: 4B-Inst=$PID_4BI, 2B-Thinking=$PID_2BT"

# === Step 4: when 2B-Thinking finishes, start 4B-Thinking on GPU 1 ===
wait $PID_2BT
echo "[grid] 2B-Thinking finished; launching 4B-Thinking on GPU 1..."
bash scripts/run_xlast_grid_for_vlm.sh "$MODEL_4BT" 2560 1 "$DIMS_4BT" &
PID_4BT=$!

# === Step 5: wait for the remaining grids ===
wait $PID_4BI
echo "[grid] 4B-Instruct finished."
wait $PID_4BT
echo "[grid] 4B-Thinking finished."

echo "All extra-VLM grids complete. Inspect:"
echo "  sqlite3 mlflow.db \"SELECT r.name, MAX(m.value) FROM runs r JOIN metrics m USING(run_uuid) WHERE m.key='val_acc' AND r.name LIKE '%xlast_d%norope%' GROUP BY r.name ORDER BY MAX(m.value) DESC;\""
