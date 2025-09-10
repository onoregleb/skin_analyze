from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import base64
import io
from PIL import Image
import httpx

from app.pipeline.pipeline import analyze_skin_pipeline
from app.schemas import AnalyzeResponse
from app.utils.logging import get_logger

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
