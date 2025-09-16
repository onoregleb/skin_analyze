from __future__ import annotations
from typing import Dict, Any
import json
from PIL import Image
import time

from app.services.medgemma import MedGemmaService
from app.services.gemini_client import GeminiClient
from app.utils.logging import get_logger

logger = get_logger("pipeline")


async def analyze_skin_pipeline(image: Image.Image, user_text: str | None) -> Dict[str, Any]:
    timings = {}

    # Step 1: Visual analysis via MedGemma
    start_time = time.perf_counter()
    visual_summary = await MedGemmaService.analyze_image(image, mode="extended")
    medgemma_time = time.perf_counter() - start_time
    timings["medgemma_seconds"] = round(medgemma_time, 2)
    logger.info(f"[STEP 1] MedGemma visual_summary: {visual_summary}")
    logger.info(f"[TIMING] MedGemma took {timings['medgemma_seconds']} seconds")

    # Step 2: Planning with Gemini using tool calling
    gemini = GeminiClient()
    start_time = time.perf_counter()
    products = []
    planning_result = await gemini.plan_with_tool(visual_summary, user_text)
    if isinstance(planning_result, tuple) and len(planning_result) == 2:
        planning, products = planning_result
    else:
        planning = planning_result or {}
        products = products or []
    gemini_plan_time = time.perf_counter() - start_time
    timings["gemini_plan_seconds"] = round(gemini_plan_time, 2)
    logger.info(f"[STEP 2] Gemini planning result: {json.dumps(planning, ensure_ascii=False)}")
    logger.info(f"[TIMING] Gemini planning took {timings['gemini_plan_seconds']} seconds")

    # Step 3: Finalize answer with Gemini
    # ✅ Передаём JSON-строки, получаем готовый dict
    start_time = time.perf_counter()
    final_gemini = gemini.finalize_with_products(
        json.dumps(planning, ensure_ascii=False),
        json.dumps(products, ensure_ascii=False)
    )
    gemini_finalize_time = time.perf_counter() - start_time
    timings["gemini_finalize_seconds"] = round(gemini_finalize_time, 2)
    logger.info(f"[STEP 3] Gemini finalize raw output: {json.dumps(final_gemini, ensure_ascii=False)}")

    # ✅ НЕ ДЕЛАЕМ json.loads() — final_gemini уже dict!
    # Проверяем, не вернул ли Gemini fallback
    if (
        final_gemini.get("diagnosis") == "Analysis failed"
        or not final_gemini.get("products")
    ):
        logger.warning("[STEP 3] Using fallback values due to Gemini failure")
        final_gemini = {
            "diagnosis": planning.get("diagnosis", visual_summary[:200]),
            "skin_type": planning.get("skin_type", "unknown"),
            "explanation": "Heuristic selection due to Gemini failure.",
            "routine_steps": [],
            "products": (products or [])[:5],
            "additional_recommendations": "",
            "medgemma_summary": visual_summary,
        }

    # Construct the final response with the desired order
    products_list = (final_gemini.get("products") or products or [])[:5]
    final_response = {
        "medgemma_summary": visual_summary,
        "products": products_list,
    }
    # Add remaining fields from Gemini output except products
    final_response.update({k: v for k, v in final_gemini.items() if k != "products"})
    final_response["timings"] = timings

    logger.info(f"[STEP 3] Final response: {json.dumps(final_response, ensure_ascii=False)}")
    logger.info(f"[OVERALL TIMING] Total pipeline time: {timings['medgemma_seconds'] + timings['gemini_plan_seconds'] + timings['gemini_finalize_seconds']:.2f} seconds")

    return final_response