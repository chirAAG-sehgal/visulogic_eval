mkdir -p outputs/
python evaluation/eval_model.py \
    --input_file data/visulogic_benchmark/data.jsonl \
    --output_file outputs/qwen3_vl_4b_instruct.jsonl \
    --model_path Qwen/Qwen3-VL-4B-Instruct
