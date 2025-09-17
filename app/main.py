from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Any
import io
from PIL import Image
import httpx
import asyncio
import json

from app.services.medgemma import MedGemmaService
from app.schemas import AnalyzeResponse
from app.utils.logging import get_logger
from app.services.job_manager import job_manager, JobStatus

app = FastAPI(title="Skin Analyze API", version="0.1.1")
logger = get_logger("app")


class SkinAnalysisRequest(BaseModel):
    image_url: str
    text: str | None = None
    mode: str = "basic"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    logger.info("API starting up")
    try:
        # Warm up MedGemma model to reduce first-request latency
        MedGemmaService()
        logger.info("MedGemma warmed up")
    except Exception as e:
        logger.warning(f"MedGemma warmup failed: {e}")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("API shutting down")


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _bytes_to_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")


async def _fetch_image_from_url(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image_url: {e}")


async def _run_analysis_job(job_id: str, image: Image.Image, user_text: str | None, mode: str = "basic") -> None:
    timings: dict[str, float] = {}
    try:
        # Step 1: MedGemma
        mode_norm = (mode or "basic").strip().lower()
        if mode_norm not in {"basic", "extended"}:
            mode_norm = "basic"
        start_time = asyncio.get_running_loop().time()
        visual_summary = await MedGemmaService.analyze_image(image, mode=mode_norm)
        medgemma_time = asyncio.get_running_loop().time() - start_time
        timings["medgemma_seconds"] = round(medgemma_time, 2)
        job_manager.update_progress(job_id, {"medgemma_summary": visual_summary, "timings": timings})

        # Step 2: Gemini planning with tool
        from app.services.gemini_client import GeminiClient  # local import to avoid startup latency
        gemini = GeminiClient()
        start_time = asyncio.get_running_loop().time()
        products: list[dict[str, Any]] = []  # ensure defined
        planning_result = await gemini.plan_with_tool(visual_summary, user_text)
        if isinstance(planning_result, tuple) and len(planning_result) == 2:
            planning, products = planning_result
        else:
            planning = planning_result or {}
            products = products or []
        gemini_plan_time = asyncio.get_running_loop().time() - start_time
        timings["gemini_plan_seconds"] = round(gemini_plan_time, 2)
        job_manager.update_progress(job_id, {"planning": planning, "timings": timings})

        # Step 3: Finalization
        start_time = asyncio.get_running_loop().time()
        final_text = gemini.finalize_with_products(
            json.dumps(planning, ensure_ascii=False),
            json.dumps(products, ensure_ascii=False),
        )
        gemini_finalize_time = asyncio.get_running_loop().time() - start_time
        timings["gemini_finalize_seconds"] = round(gemini_finalize_time, 2)

        try:
            final = json.loads(final_text)
        except Exception:
            final = {
                "diagnosis": planning.get("diagnosis", visual_summary[:200]),
                "skin_type": planning.get("skin_type", "unknown"),
                "explanation": "Heuristic selection due to JSON parse fail.",
                "products": (products or [])[:5],
            }

        # Normalize output like pipeline
        final["products"] = (final.get("products") or [])[:5]
        final["medgemma_summary"] = visual_summary
        final["timings"] = timings

        job_manager.complete(job_id, final)
    except Exception as e:
        job_manager.fail(job_id, str(e))


@app.post("/v1/skin-analysis")
async def skin_analysis_start(body: SkinAnalysisRequest):
    """
    Начать анализ кожи
    
    Принимает изображение по URL и запускает фоновую задачу анализа.
    
    Args:
        body: JSON тело запроса с параметрами:
            - image_url: URL изображения (обязательный параметр)
            - text: Дополнительное описание (опционально)
            - mode: Режим анализа ("basic" или "extended", по умолчанию "basic")
    
    Returns:
        job_id: ID задачи для отслеживания статуса
        status: Текущий статус задачи
        mode: Выбранный режим анализа
    """
    try:
        # Нормализуем режим
        mode_norm = (body.mode or "basic").strip().lower()
        if mode_norm not in {"basic", "extended"}:
            mode_norm = "basic"

        # Получаем изображение по URL
        image_bytes = await _fetch_image_from_url(body.image_url)
        pil_image = await _bytes_to_image(image_bytes)

        # Создаем background job
        job = job_manager.create()
        asyncio.create_task(_run_analysis_job(job.id, pil_image, body.text, mode_norm))
        
        return {
            "job_id": job.id, 
            "status": JobStatus.in_progress, 
            "mode": mode_norm
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /v1/skin-analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/skin-analysis/status/{job_id}")
async def get_analysis_status(job_id: str):
    """
    Получить статус анализа кожи
    
    Args:
        job_id: ID задачи, полученный из /v1/skin-analysis
    
    Returns:
        Информация о статусе задачи включая прогресс и время выполнения
    """
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "updated_at": job.updated_at,
        "created_at": job.created_at,
    }


@app.get("/v1/skin-analysis/result/{job_id}")
async def get_analysis_result(job_id: str):
    """
    Получить результат анализа кожи
    
    Args:
        job_id: ID задачи, полученный из /v1/skin-analysis
    
    Returns:
        Полный результат анализа кожи или информацию о статусе, если анализ еще не завершен
    """
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    
    if job.status == JobStatus.in_progress:
        return JSONResponse(
            status_code=202, 
            content={"status": job.status, "progress": job.progress}
        )
    
    if job.status == JobStatus.failed:
        return JSONResponse(
            status_code=500, 
            content={"status": job.status, "error": job.error}
        )
    
    # Задача завершена успешно
    try:
        validated = AnalyzeResponse(**(job.result or {}))
    except ValidationError as ve:
        logger.error(f"Stored job result schema error: {ve}")
        raise HTTPException(status_code=500, detail="Stored result schema error")
    
    return JSONResponse(content=validated.model_dump())