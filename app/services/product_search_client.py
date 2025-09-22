from typing import List, Dict, Any
import httpx
from app.config import settings
from app.utils.logging import get_logger


logger = get_logger("product_search_client")


class ProductSearchClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.product_search_base_url).rstrip("/")
        self.timeout = httpx.Timeout(15.0)

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

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json={"query": query, "num": num},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                return data

        except httpx.TimeoutException:
            logger.error(f"Timeout calling product search service: {url}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling product search service: {e}")
            return []
        except Exception as e:
            logger.error(f"Error calling product search service: {e}")
            return []
