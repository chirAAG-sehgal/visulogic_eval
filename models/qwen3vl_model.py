from typing import Dict
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from models.base_model import BaseModel


class Qwen3VisionModel(BaseModel):
    def __init__(self, model_path: str, user_prompt: str = None, max_image_size: int = -1):
        """
        Initialize the Qwen3 Vision Model.
        Args:
            model_path: HuggingFace model path (e.g., Qwen/Qwen3-VL-2B-Instruct).
            user_prompt: Prompt template to append to questions.
            max_image_size: Max image dimension (-1 for no resize).
        """
        self.model_path = model_path
        self.user_prompt = user_prompt
        self.max_image_size = max_image_size
        self.is_thinking = "thinking" in model_path.lower()

        device_map = "auto"
        dtype = "auto"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    @property
    def name(self) -> str:
        return self.model.config._name_or_path

    def _prepare_input(self, image_path: str, text: str):
        input_image = Image.open(image_path).convert("RGB")
        input_text = text + self.user_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_image},
                    {"type": "text", "text": input_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs

    def predict(self, input_data: Dict) -> str:
        inputs = self._prepare_input(
            image_path=input_data["image_path"],
            text=input_data["text"],
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""
