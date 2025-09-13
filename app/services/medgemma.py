from __future__ import annotations
from typing import List
from PIL import Image
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from app.utils.logging import get_logger
from app.config import settings
import torch

logger = get_logger("medgemma")


class MedGemmaService:
    _instance = None
    _pipe = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedGemmaService, cls).__new__(cls)

            model_id = "google/medgemma-4b-it"

            # Map dtype from settings
            dtype_map = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(settings.torch_dtype.lower(), "auto")

            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=settings.device_map,
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
                "content": [{"type": "text", "text": """You are an expert dermatologist. 
Provide a detailed analysis of the skin condition using professional terminology. 
Focus on:
- Skin type and texture
- Hydration levels and barrier function
- Sebum production and pore condition
- Presence of any lesions, inflammation, or acne
- Pigmentation and color uniformity
- Signs of aging or photodamage
- Visible blood vessels or redness
- Any abnormal formations or concerning features"""}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """Please analyze this skin image in detail. 
Describe all visible characteristics and potential concerns.
Include both surface-level observations and potential underlying conditions.
Use medical terminology where appropriate, but ensure the description remains understandable.
Be specific about locations and severity of any issues observed."""},
                    {"type": "image", "image": image}
                ]
            }
        ]

        output = cls._pipe(text=messages, max_new_tokens=settings.medgemma_max_new_tokens)
        response = output[0]["generated_text"][-1]["content"].strip()
        logger.info(f"[MedGemma] Output: {response}")
        return response


medgemma_service = MedGemmaService()