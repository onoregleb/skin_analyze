from __future__ import annotations
from typing import Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from app.config import settings


_MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"


class MedGemmaService:
	_processor: AutoProcessor | None = None
	_model: AutoModelForCausalLM | None = None

	@classmethod
	def load(cls) -> None:
		if cls._processor is None:
			cls._processor = AutoProcessor.from_pretrained(_MEDGEMMA_MODEL_ID, token=settings.hf_token)
		if cls._model is None:
			dtype = torch.bfloat16 if settings.torch_dtype.lower() == "bf16" else torch.float16
			cls._model = AutoModelForCausalLM.from_pretrained(
				_MEDGEMMA_MODEL_ID,
				torch_dtype=dtype,
				low_cpu_mem_usage=True,
				device_map=settings.device_map,
				trust_remote_code=True,
			)

	@classmethod
	@torch.inference_mode()
	def analyze_image(cls, image: Image.Image, user_text: str | None) -> str:
		if cls._processor is None or cls._model is None:
			cls.load()
		prompt = "Analyze the skin condition on this photo. Describe key findings and possible issues."
		if user_text:
			prompt += f" User note: {user_text}"
		inputs = cls._processor(text=prompt, images=image, return_tensors="pt").to(cls._model.device)
		generate_ids = cls._model.generate(**inputs, max_new_tokens=256)
		output = cls._processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
		return output.strip()
