from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import base64
import io
from PIL import Image
import httpx
import asyncio
import json

from app.pipeline.pipeline import analyze_skin_pipeline
from app.services.medgemma import MedGemmaService
from app.schemas import AnalyzeResponse, SkinAnalysisResponse
from app.utils.logging import get_logger
from app.services.job_manager import job_manager, JobStatus

app = FastAPI(title="Skin Analyze API", version="0.1.1")
logger = get_logger("app")


class AnalyzeRequest(BaseModel):
	image_url: str | None = None
	text: str | None = None


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


def _parse_medgemma_sections(text: str, mode: str) -> tuple[str, str]:
	"""Parse 'Summary:' and 'Description (...)' sections from the model output.
	Fallback: first sentence as summary, remainder as description.
	"""
	summary = ""
	description = text.strip() if text else ""
	if not text:
		return summary, description

	lines = [l.strip() for l in text.splitlines()]
	current = None
	collected = {"summary": [], "description": []}
	desc_label_basic = "description (basic)"
	desc_label_ext = "description (extended)"

	for ln in lines:
		low = ln.lower()
		if low.startswith("summary:"):
			current = "summary"
			content = ln.split(":", 1)[1].strip()
			if content:
				collected[current].append(content)
			continue
		if low.startswith(desc_label_basic + ":") or low.startswith(desc_label_ext + ":"):
			current = "description"
			content = ln.split(":", 1)[1].strip()
			if content:
				collected[current].append(content)
			continue
		if current:
			collected[current].append(ln)

	summary = " ".join([s for s in collected["summary"] if s]).strip()
	description = "\n".join([s for s in collected["description"] if s]).strip() or description

	if not summary:
		# Fallback heuristic: split by common sentence boundaries
		for sep in [". ", "\n", "! ", "? "]:
			idx = text.find(sep)
			if idx != -1:
				summary = text[: idx + len(sep)].strip()
				description = text[idx + len(sep):].strip()
				break
		if not summary:
			summary = text.strip()
			description = text.strip()
	return summary, description


@app.post("/analyze")
async def analyze(
	image: UploadFile | None = File(default=None),
	image_b64: str | None = Form(default=None),
	text: str | None = Form(default=None),
	body: AnalyzeRequest | None = Body(default=None),
):
	try:
		image_bytes: bytes | None = None
		user_text: str | None = text

		# Priority: file > image_b64 > body.image_url
		if image is not None:
			image_bytes = await image.read()
		elif image_b64 is not None:
			image_bytes = base64.b64decode(image_b64)
		elif body and body.image_url:
			image_bytes = await _fetch_image_from_url(body.image_url)
			user_text = body.text
		else:
			raise HTTPException(status_code=400, detail="Provide image (file), image_b64, or body.image_url")

		pil_image = await _bytes_to_image(image_bytes)
		result_dict = await analyze_skin_pipeline(pil_image, user_text)

		# validate strict output
		try:
			validated = AnalyzeResponse(**result_dict)
		except ValidationError as ve:
			logger.error(f"Response validation failed: {ve}")
			raise HTTPException(status_code=500, detail="Internal response schema error")

		return JSONResponse(content=validated.model_dump())
	except HTTPException:
		raise
	except Exception as e:
		logger.exception("Unexpected error in /analyze")
		raise HTTPException(status_code=500, detail=str(e))


# Simple MedGemma-only endpoint with mode selector
@app.get("/v1/skin-analysis")
async def skin_analysis(image_url: str, mode: str = "extended"):
	try:
		mode_norm = (mode or "extended").strip().lower()
		if mode_norm not in {"basic", "extended"}:
			mode_norm = "extended"
		image_bytes = await _fetch_image_from_url(image_url)
		pil_image = await _bytes_to_image(image_bytes)
		start = asyncio.get_running_loop().time()
		medgemma_text = await MedGemmaService.analyze_image(pil_image, mode=mode_norm)
		elapsed = asyncio.get_running_loop().time() - start
		summary, description = _parse_medgemma_sections(medgemma_text, mode_norm)
		resp = SkinAnalysisResponse(
			mode=mode_norm,
			summary=summary,
			description=description,
			timings={"medgemma_seconds": round(elapsed, 2)},
		)
		return JSONResponse(content=resp.model_dump())
	except HTTPException:
		raise
	except Exception as e:
		logger.exception("Unexpected error in /v1/skin-analysis")
		raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Background job version
# -----------------------------

async def _run_analysis_job(job_id: str, image: Image.Image, user_text: str | None) -> None:
    timings: dict[str, float] = {}
    try:
        # Step 1: MedGemma
        start_time = asyncio.get_running_loop().time()
        visual_summary = await MedGemmaService.analyze_image(image, mode="extended")
        medgemma_time = asyncio.get_running_loop().time() - start_time
        timings["medgemma_seconds"] = round(medgemma_time, 2)
        job_manager.update_progress(job_id, {"medgemma_summary": visual_summary, "timings": timings})

        # Step 2: Qwen planning with tool
        from app.services.qwen_client import QwenClient  # local import to avoid startup latency
        qwen = QwenClient()
        start_time = asyncio.get_running_loop().time()
        planning = await qwen.plan_with_tool(visual_summary, user_text)
        qwen_plan_time = asyncio.get_running_loop().time() - start_time
        timings["qwen_plan_seconds"] = round(qwen_plan_time, 2)
        job_manager.update_progress(job_id, {"planning": planning, "timings": timings})

        # Step 3: Finalization
        products = planning.get("tool_products") or []
        start_time = asyncio.get_running_loop().time()
        final_text = qwen.finalize_with_products(
            json.dumps(planning, ensure_ascii=False),
            json.dumps(products, ensure_ascii=False),
        )
        qwen_finalize_time = asyncio.get_running_loop().time() - start_time
        timings["qwen_finalize_seconds"] = round(qwen_finalize_time, 2)

        try:
            final = json.loads(final_text)
        except Exception:
            final = {
                "diagnosis": planning.get("diagnosis", visual_summary[:200]),
                "skin_type": planning.get("skin_type", "unknown"),
                "explanation": "Heuristic selection due to JSON parse fail.",
                "products": products[:5],
            }

        # Normalize output like pipeline
        final["products"] = (final.get("products") or [])[:5]
        final["medgemma_summary"] = visual_summary
        final["tool_products"] = products[:5]
        final["timings"] = timings

        job_manager.complete(job_id, final)
    except Exception as e:
        job_manager.fail(job_id, str(e))


class AnalyzeStartRequest(BaseModel):
    image_url: str | None = None
    text: str | None = None


@app.post("/analyze/start")
async def analyze_start(
    image: UploadFile | None = File(default=None),
    image_b64: str | None = Form(default=None),
    text: str | None = Form(default=None),
    body: AnalyzeStartRequest | None = Body(default=None),
):
    try:
        # Same input handling as /analyze
        image_bytes: bytes | None = None
        user_text: str | None = text
        if image is not None:
            image_bytes = await image.read()
        elif image_b64 is not None:
            image_bytes = base64.b64decode(image_b64)
        elif body and body.image_url:
            image_bytes = await _fetch_image_from_url(body.image_url)
            user_text = body.text
        else:
            raise HTTPException(status_code=400, detail="Provide image (file), image_b64, or body.image_url")

        pil_image = await _bytes_to_image(image_bytes)

        # Create job and schedule background task
        job = job_manager.create()
        asyncio.create_task(_run_analysis_job(job.id, pil_image, user_text))
        return {"job_id": job.id, "status": JobStatus.in_progress}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /analyze/start")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/status/{job_id}")
async def analyze_status(job_id: str):
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


@app.get("/analyze/result/{job_id}")
async def analyze_result(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job.status == JobStatus.in_progress:
        return JSONResponse(status_code=202, content={"status": job.status, "progress": job.progress})
    if job.status == JobStatus.failed:
        return JSONResponse(status_code=500, content={"status": job.status, "error": job.error})
    # done
    try:
        validated = AnalyzeResponse(**(job.result or {}))
    except ValidationError as ve:
        logger.error(f"Stored job result schema error: {ve}")
        raise HTTPException(status_code=500, detail="Stored result schema error")
    return JSONResponse(content=validated.model_dump())
