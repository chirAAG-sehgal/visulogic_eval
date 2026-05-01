mkdir -p outputs/
python evaluation/eval_model.py \
    --input_file data/visulogic_benchmark/data.jsonl \
    --output_file outputs/qwen3_vl_4b_instruct.jsonl \
    --model_path weights/Qwen_Qwen3-VL-4B-Instruct \
    --batch_size ${BATCH_SIZE:-1} \
    --gpu_ids ${GPU_IDS:-"0,1"}
