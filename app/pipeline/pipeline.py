from __future__ import annotations
from typing import Dict, Any, List
import json
from PIL import Image

from app.services.medgemma import MedGemmaService
from app.services.qwen_client import QwenClient


async def analyze_skin_pipeline(image: Image.Image, user_text: str | None) -> Dict[str, Any]:
	# Step 1: Visual analysis via MedGemma
	medgemma_prompt = "Please analyze the skin condition in the image and describe it in detail."
	visual_summary = await MedGemmaService.analyze_image(image, medgemma_prompt)

	# Step 2: Planning with Qwen using tool calling
	qwen = QwenClient()
	planning = await qwen.plan_with_tool(visual_summary, user_text)

	# Step 3: Finalize answer with Qwen (products already fetched via tool if needed)
	products = planning.get("products") or []
	final_text = qwen.finalize_with_products(json.dumps(planning, ensure_ascii=False), json.dumps(products, ensure_ascii=False))
	try:
		final = json.loads(final_text)
	except Exception:
		final = {
			"diagnosis": planning.get("diagnosis", visual_summary[:200]),
			"skin_type": planning.get("skin_type", "unknown"),
			"explanation": "Heuristic selection due to JSON parse fail.",
			"products": products[:5],
		}
	final["products"] = (final.get("products") or [])[:5]
	return final
