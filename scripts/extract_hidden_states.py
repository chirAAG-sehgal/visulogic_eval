import sys
sys.path.append(".")
import argparse
import os

# Parse --gpu_ids early so CUDA_VISIBLE_DEVICES is set before torch is imported
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu_ids', type=str, default=None)
_early_args, _ = _parser.parse_known_args()
if _early_args.gpu_ids is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.gpu_ids

import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from models.prompts import COT_PROMPT, RL_COT_PROMPT, SFT_PROMPT


def load_data(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def prepare_input(processor, image_path, text, user_prompt):
    image = Image.open(image_path).convert("RGB")
    input_text = text + user_prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": input_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Extract final layer hidden states from Qwen3-VL")
    parser.add_argument('--input_file', type=str, default='data/visulogic_train/data.jsonl')
    parser.add_argument('--model_path', type=str, default='weights/Qwen_Qwen3-VL-2B-Instruct')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. Defaults to outputs/hidden_states/{model_name}/')
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--user_prompt', type=str, default='sft',
                        choices=['sft', 'cot', 'rl_cot'],
                        help='Prompt type to append to questions')

    args = parser.parse_args()

    # Resolve prompt
    prompt_map = {'sft': SFT_PROMPT, 'cot': COT_PROMPT, 'rl_cot': RL_COT_PROMPT}
    user_prompt = prompt_map[args.user_prompt]

    # Resolve output dir
    model_name = os.path.basename(args.model_path)
    output_dir = args.output_dir or os.path.join('outputs', 'hidden_states', model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load data
    base_dir = args.input_file.replace("data.jsonl", "")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} samples. Saving to {output_dir}")

    for item in tqdm(data, desc="Extracting hidden states"):
        sample_id = item['id']
        out_path = os.path.join(output_dir, f"{sample_id}.pt")

        # Skip if already extracted
        if os.path.exists(out_path):
            continue

        image_path = os.path.join(base_dir, item['image_path'])
        inputs = prepare_input(processor, image_path, item['question'], user_prompt)
        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 = embedding output, index -1 = final transformer layer output
        # Shape: (1, seq_len, hidden_dim)
        final_hidden = outputs.hidden_states[-1].cpu()

        torch.save(final_hidden, out_path)

    print(f"Done. Hidden states saved to {output_dir}")


if __name__ == '__main__':
    main()
