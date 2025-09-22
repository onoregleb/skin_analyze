from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from app.tools.search_products import search_products
from app.config import settings
from app.utils.logging import get_logger

app = FastAPI(title="Product Search API", version="0.1.0")
logger = get_logger("product_search")

class ProductSearchRequest(BaseModel):
    query: str
    num: int = 10

class ProductResponse(BaseModel):
    name: str
    url: str
    snippet: str
    image_url: str | None
    price: str | None = None
    brand: str | None = None
    category: str | None = None
    rating: float | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    logger.info("Product Search API starting up")
    if not settings.google_cse_api_key or not settings.google_cse_cx:
        logger.warning("Google CSE API keys not configured")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Product Search API shutting down")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/search-products", response_model=List[ProductResponse])
async def search_products_endpoint(request: ProductSearchRequest):
    """
    Поиск продуктов для ухода за кожей

    Args:
        request: JSON тело запроса с параметрами:
            - query: Поисковый запрос
            - num: Количество результатов (по умолчанию 10, максимум 10)

    Returns:
        Список найденных продуктов
    """
    try:
        if not settings.google_cse_api_key or not settings.google_cse_cx:
            raise HTTPException(
                status_code=503,
                detail="Product search service not configured"
            )

        # Выполняем поиск
        results = await search_products(request.query, request.num)

        # Преобразуем результаты в нужный формат
        products = []
        for result in results:
            products.append(ProductResponse(
                name=result.get("name", "Unknown Product"),
                url=result.get("url", ""),
                snippet=result.get("snippet", ""),
                image_url=result.get("image_url"),
                price=result.get("price"),
                brand=result.get("brand"),
                category=result.get("category"),
                rating=result.get("rating")
            ))

        return products

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in product search")
        raise HTTPException(status_code=500, detail=str(e))
