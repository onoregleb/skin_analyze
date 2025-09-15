from __future__ import annotations
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
            torch_dtype = dtype_map.get(settings.dtype.lower(), "auto")

            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
            )
            # Try to use fast processor if available
            try:
                processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            except TypeError:
                processor = AutoProcessor.from_pretrained(model_id)

            cls._pipe = pipeline(
                "image-text-to-text",
                model=model,
                processor=processor,
                device=0,
                do_sample=False,
            )

        return cls._instance

    @classmethod
    async def analyze_image(cls, image: Image.Image, mode: str = "extended") -> str:
        if cls._pipe is None:
            cls()

        mode_norm = (mode or "extended").strip().lower()
        if mode_norm not in {"basic", "extended"}:
            mode_norm = "extended"

        # Build messages based on requested mode
        if mode_norm == "basic":
            prompt_system = (
                "You are a professional dermatologist. Provide a concise, user-friendly assessment.\n"
                "Return exactly two labeled sections in English:\n"
                "Summary: a brief 1-2 sentence overview of the skin condition.\n"
                "Description (basic): a short paragraph (3-6 sentences) focusing on key observations and main concerns."
            )
            prompt_user = (
                "Please analyze this skin image. Keep it concise and approachable.\n"
                "Respond using the two sections: 'Summary:' and 'Description (basic):'."
            )
        else:  # extended
            prompt_system = (
                """You are an expert dermatologist. 
Provide a detailed analysis of the skin condition using professional terminology. 
Focus on:
- Skin type and texture
- Hydration levels and barrier function
- Sebum production and pore condition
- Presence of any lesions, inflammation, or acne
- Pigmentation and color uniformity
- Signs of aging or photodamage
- Visible blood vessels or redness
- Any abnormal formations or concerning features"""                          
            )
            prompt_user = (
                """Please analyze this skin image in detail. 
Describe all visible characteristics and potential concerns.
Include both surface-level observations and potential underlying conditions.
Use medical terminology where appropriate, but ensure the description remains understandable.
Be specific about locations and severity of any issues observed.
Respond using the two sections: 'Summary:' and 'Description (extended):'.""" )

        logger.info(f"[MedGemma] Mode: {mode_norm}")

        logger.info(f"[MedGemma] Image size: {image.size}")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_system}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_user},
                    {"type": "image", "image": image}
                ]
            }
        ]

        output = cls._pipe(text=messages, max_new_tokens=settings.medgemma_max_new_tokens)
        generated = output[0].get("generated_text")
        response = ""
        # Case 1: generated_text is a plain string
        if isinstance(generated, str):
            response = generated.strip()
        # Case 2: list-of-dicts with content fields
        elif isinstance(generated, list):
            # Find last text chunk or join all text contents
            texts = []
            for part in generated:
                if isinstance(part, dict):
                    if "content" in part and isinstance(part["content"], str):
                        texts.append(part["content"]) 
                    elif "text" in part and isinstance(part["text"], str):
                        texts.append(part["text"]) 
            response = "\n".join([t.strip() for t in texts if t and t.strip()])
        else:
            response = ""
        logger.info(f"[MedGemma] Output: {response}")
        return response


medgemma_service = MedGemmaService()