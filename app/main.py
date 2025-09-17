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

from app.services.supabase_service import (
    supabase_service, 
    SkinAnalysisJobCreate, 
    SkinAnalysisJobUpdate,
    SkinAnalysisResult,
    RecommendedProduct
)


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
        # Обновляем статус в обеих системах - job_manager и Supabase
        await supabase_service.update_job(job_id, SkinAnalysisJobUpdate(
            status="in_progress",
            progress={"step": "starting_analysis"}
        ))
        
        # Step 1: MedGemma
        mode_norm = (mode or "basic").strip().lower()
        if mode_norm not in {"basic", "extended"}:
            mode_norm = "basic"
            
        start_time = asyncio.get_running_loop().time()
        visual_summary = await MedGemmaService.analyze_image(image, mode=mode_norm)
        medgemma_time = asyncio.get_running_loop().time() - start_time
        timings["medgemma_seconds"] = round(medgemma_time, 2)
        
        # Обновляем прогресс
        job_manager.update_progress(job_id, {"medgemma_summary": visual_summary, "timings": timings})
        await supabase_service.update_job(job_id, SkinAnalysisJobUpdate(
            progress={"step": "medgemma_completed", "medgemma_summary": visual_summary},
            timings=timings
        ))

        # Step 2: Gemini planning with tool
        from app.services.gemini_client import GeminiClient
        gemini = GeminiClient()
        start_time = asyncio.get_running_loop().time()
        products: list[dict[str, Any]] = []
        planning_result = await gemini.plan_with_tool(visual_summary, user_text)
        
        if isinstance(planning_result, tuple) and len(planning_result) == 2:
            planning, products = planning_result
        else:
            planning = planning_result or {}
            products = products or []
            
        gemini_plan_time = asyncio.get_running_loop().time() - start_time
        timings["gemini_plan_seconds"] = round(gemini_plan_time, 2)
        
        job_manager.update_progress(job_id, {"planning": planning, "timings": timings})
        await supabase_service.update_job(job_id, SkinAnalysisJobUpdate(
            progress={"step": "planning_completed", "planning": planning},
            timings=timings
        ))

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

        # Normalize output
        final["products"] = (final.get("products") or [])[:5]
        final["medgemma_summary"] = visual_summary
        final["timings"] = timings

        # Сохраняем результат анализа в Supabase
        await supabase_service.save_analysis_result(SkinAnalysisResult(
            job_id=job_id,
            diagnosis=final.get("diagnosis"),
            skin_type=final.get("skin_type"),
            explanation=final.get("explanation"),
            medgemma_summary=visual_summary,
            planning_data=planning,
            final_result=final
        ))

        # Сохраняем продукты в Supabase
        if final.get("products"):
            recommended_products = []
            for product_data in final["products"]:
                recommended_products.append(RecommendedProduct(
                    job_id=job_id,
                    product_name=product_data.get("name"),
                    brand=product_data.get("brand"),
                    description=product_data.get("description"),
                    price=product_data.get("price"),
                    product_url=product_data.get("url"),
                    image_url=product_data.get("image"),
                    category=product_data.get("category"),
                    benefits=product_data.get("benefits", []),
                    suitable_for_skin_type=final.get("skin_type")
                ))
            
            if recommended_products:
                await supabase_service.save_recommended_products(recommended_products)

        # Обновляем статус на завершенный
        job_manager.complete(job_id, final)
        await supabase_service.update_job(job_id, SkinAnalysisJobUpdate(
            status="completed",
            timings=timings
        ))

    except Exception as e:
        error_msg = str(e)
        job_manager.fail(job_id, error_msg)
        await supabase_service.update_job(job_id, SkinAnalysisJobUpdate(
            status="failed",
            error_message=error_msg
        ))


@app.post("/v1/skin-analysis")
async def skin_analysis_start(body: SkinAnalysisRequest):
    """
    Начать анализ кожи с сохранением в Supabase
    """
    try:
        # Нормализуем режим
        mode_norm = (body.mode or "basic").strip().lower()
        if mode_norm not in {"basic", "extended"}:
            mode_norm = "basic"

        # Получаем изображение по URL
        image_bytes = await _fetch_image_from_url(body.image_url)
        pil_image = await _bytes_to_image(image_bytes)

        # Создаем background job в job_manager
        job = job_manager.create()
        
        # Создаем запись в Supabase
        await supabase_service.create_job(SkinAnalysisJobCreate(
            job_id=job.id,
            image_url=body.image_url,
            user_text=body.text,
            mode=mode_norm
        ))
        
        # Запускаем фоновую задачу
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