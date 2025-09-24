from typing import List, Dict, Any
import httpx
from app.config import settings
from app.utils.logging import get_logger


logger = get_logger("product_search_client")


class ProductSearchClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.product_search_base_url).rstrip("/")
        self.timeout = httpx.Timeout(15.0)
        logger.info(f"[DEBUG] Initialized client with base_url: {self.base_url}")

    async def search_products(self, query: str, num: int = 10) -> List[Dict[str, Any]]:
        """
        Поиск продуктов через внешний Product Search Service

        Args:
            query: Поисковый запрос
            num: Количество результатов

        Returns:
            Список продуктов в формате:
            [
                {
                    "name": str,
                    "url": str,
                    "snippet": str,
                    "image_url": str | None
                },
                ...
            ]
        """
        url = f"{self.base_url}/v1/search-products"
        logger.info(f"[DEBUG] Attempting to connect to: {url}")
        logger.info(f"[DEBUG] Query: {query}, num: {num}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"[DEBUG] Created httpx client, making request...")
                response = await client.post(
                    url,
                    json={"query": query, "num": num},
                    headers={"Content-Type": "application/json"}
                )
                logger.info(f"[DEBUG] Got response: {response.status_code}")
                response.raise_for_status()
                data = response.json()
                logger.info(f"[DEBUG] Successfully parsed JSON, got {len(data)} items")
                return data

        except httpx.TimeoutException as e:
            logger.error(f"[DEBUG] Timeout calling product search service: {url}, error: {e}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"[DEBUG] HTTP error calling product search service: {e}, response: {e.response.text if e.response else 'No response'}")
            return []
        except Exception as e:
            logger.error(f"[DEBUG] Error calling product search service: {e}, type: {type(e)}")
            import traceback
            logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return []
