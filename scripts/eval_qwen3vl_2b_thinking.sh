mkdir -p outputs/
python evaluation/eval_model.py \
    --input_file data/visulogic_benchmark/data.jsonl \
    --output_file outputs/qwen3_vl_2b_thinking.jsonl \
    --model_path weights/Qwen_Qwen3-VL-2B-Thinking \
    --user_prompt "rl_cot" \
    --batch_size ${BATCH_SIZE:-4} \
    --gpu_ids ${GPU_IDS:-"1"}
