from __future__ import annotations
from typing import List, Dict, Any
import httpx
import os
from app.config import settings
from app.utils.logging import get_logger


logger = get_logger("search")
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


async def search_products(query: str, num: int = 10) -> List[Dict[str, Any]]:
    if not settings.google_cse_api_key or not settings.google_cse_cx:
        logger.warning("Google CSE keys missing; returning empty products")
        return []
    
    params = {
        "key": settings.google_cse_api_key,
        "cx": settings.google_cse_cx,
        "q": query,
        "num": min(max(num, 1), 10),
    }
    
    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(GOOGLE_SEARCH_URL.strip(), params=params)  # Убрал лишний пробел
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                results: List[Dict[str, Any]] = []
                for it in items:
                    p = {
                        "name": it.get("title", "Unknown Product"),
                        "url": it.get("link", ""),
                        "snippet": it.get("snippet", ""),
                        "image_url": (it.get("pagemap", {}).get("cse_image", [{}])[0].get("src") if it.get("pagemap") else None),
                    }
                    # Фильтруем пустые результаты
                    if p["name"] and p["url"]:
                        results.append(p)
                return results if results else []  # Возвращаем пустой список если ничего нет
        except Exception as e:
            logger.warning(f"Google CSE attempt {attempts} failed: {e}")
    logger.error("Google CSE all attempts failed; returning empty list")
    return []
