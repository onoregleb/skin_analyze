from __future__ import annotations
from typing import List
from PIL import Image
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from app.utils.logging import get_logger

logger = get_logger("medgemma")


class MedGemmaService:
    _instance = None
    _pipe = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedGemmaService, cls).__new__(cls)

            model_id = "google/medgemma-4b-it"

            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_id)

            cls._pipe = pipeline(
                "image-text-to-text",
                model=model,
                processor=processor,
                do_sample=False,
            )

        return cls._instance

    @classmethod
    async def analyze_image(cls, image: Image.Image, prompt: str) -> str:
        if cls._pipe is None:
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

        output = cls._pipe(text=messages, max_new_tokens=512)
        response = output[0]["generated_text"][-1]["content"].strip()
        logger.info(f"[MedGemma] Output: {response}")
        return response


medgemma_service = MedGemmaService()