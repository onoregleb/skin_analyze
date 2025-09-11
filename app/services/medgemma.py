from __future__ import annotations
from typing import List
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from app.utils.logging import get_logger

logger = get_logger("medgemma")


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
            cls()

        logger.info(f"[MedGemma] Prompt: {prompt}")
        logger.info(f"[MedGemma] Image size: {image.size}")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert dermatologist."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        inputs = cls._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(cls._model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = cls._model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generation = generation[0][input_len:]

        response = cls._processor.decode(generation, skip_special_tokens=True).strip()
        logger.info(f"[MedGemma] Output: {response}")
        return response


medgemma_service = MedGemmaService()