from __future__ import annotations
from typing import List
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class MedGemmaService:
    _instance = None
    _processor = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedGemmaService, cls).__new__(cls)

            model_id = "google/medgemma-4b-it"

            cls._processor = AutoProcessor.from_pretrained(model_id)
            cls._model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        return cls._instance

    @classmethod
    async def analyze_image(cls, image: Image.Image, prompt: str) -> str:
        if cls._model is None:
            cls()  # Initialize the service

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]

        # Получаем текстовый промпт (ещё не токенизированный)
        chat = cls._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors=None,
        )

        # Объединяем текст и картинку → словарь (BatchEncoding)
        inputs = cls._processor(
            text=chat,
            images=image,
            return_tensors="pt"
        ).to(cls._model.device)

        with torch.inference_mode():
            generation = cls._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
            decoded_generation = cls._processor.decode(
                generation[0], skip_special_tokens=False
            )

        # Извлекаем только ответ модели
        response_start = decoded_generation.find("<start_of_turn>model\n") + len(
            "<start_of_turn>model\n"
        )
        response_end = decoded_generation.find("<end_of_turn>", response_start)
        response = decoded_generation[response_start:response_end].strip()

        return response


medgemma_service = MedGemmaService()
