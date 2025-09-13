from __future__ import annotations
from typing import Dict, Any
import json
from PIL import Image
import time

from app.services.medgemma import MedGemmaService
from app.services.qwen_client import QwenClient
from app.utils.logging import get_logger

logger = get_logger("pipeline")


async def analyze_skin_pipeline(image: Image.Image, user_text: str | None) -> Dict[str, Any]:
    timings = {} 

    # Step 1: Visual analysis via MedGemma
    medgemma_prompt = "Please analyze the skin condition in the image and describe it in detail."
    start_time = time.perf_counter()
    visual_summary = await MedGemmaService.analyze_image(image, medgemma_prompt)
    medgemma_time = time.perf_counter() - start_time
    timings["medgemma_seconds"] = round(medgemma_time, 2)
    logger.info(f"[STEP 1] MedGemma visual_summary: {visual_summary}")
    logger.info(f"[TIMING] MedGemma took {timings['medgemma_seconds']} seconds")

    # Step 2: Planning with Qwen using tool calling
    qwen = QwenClient()
    start_time = time.perf_counter()
    planning = await qwen.plan_with_tool(visual_summary, user_text)
    qwen_plan_time = time.perf_counter() - start_time
    timings["qwen_plan_seconds"] = round(qwen_plan_time, 2)
    logger.info(f"[STEP 2] Qwen planning result: {json.dumps(planning, ensure_ascii=False)}")
    logger.info(f"[TIMING] Qwen planning took {timings['qwen_plan_seconds']} seconds")

    # Step 3: Finalize answer with Qwen
    # Prefer raw tool results from Qwen planning
    products = planning.get("tool_products") or []
    start_time = time.perf_counter()
    final_text = qwen.finalize_with_products(
        json.dumps(planning, ensure_ascii=False),
        json.dumps(products, ensure_ascii=False)
    )

    qwen_finalize_time = time.perf_counter() - start_time
    timings["qwen_finalize_seconds"] = round(qwen_finalize_time, 2)
    logger.info(f"[STEP 3] Qwen finalize raw output: {final_text}")
    logger.info(f"[TIMING] Qwen finalize took {timings['qwen_finalize_seconds']} seconds")

    try:
        final = json.loads(final_text)
    except Exception as e:
        logger.warning(f"[STEP 3] JSON parse failed: {e}, using fallback")
        final = {
            "diagnosis": planning.get("diagnosis", visual_summary[:200]),
            "skin_type": planning.get("skin_type", "unknown"),
            "explanation": "Heuristic selection due to JSON parse fail.",
            "products": products[:5],
        }

    final["products"] = (final.get("products") or [])[:5]
    # Add intermediate visibility fields
    final["medgemma_summary"] = visual_summary
    final["tool_products"] = products[:5]
    final["timings"] = timings 
    
    logger.info(f"[STEP 3] Final response: {json.dumps(final, ensure_ascii=False)}")
    logger.info(f"[OVERALL TIMING] Total pipeline time: {sum(timings.values()):.2f} seconds")

    return final