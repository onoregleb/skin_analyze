from __future__ import annotations
from typing import List, Dict, Any, Optional
import httpx
import os
import re
import json
from bs4 import BeautifulSoup
from app.config import settings
from app.utils.logging import get_logger


logger = get_logger("search")
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


async def extract_price_from_url(url: str) -> Optional[str]:
    """
    Извлекает цену с страницы продукта через парсинг HTML

    Args:
        url: URL страницы продукта

    Returns:
        Цена в виде строки или None если не найдена
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Популярные селекторы для цен
            price_selectors = [
                # Общие селекторы
                '[data-testid="price"]',
                '.price',
                '.product-price',
                '.product-price-current',
                '.price-current',
                '[data-cy="price"]',
                '.price-tag',
                '.price-value',
                '.product-price-value',
                # Ulta Beauty
                '.ProductCard__price',
                '.Price',
                # Sephora
                '.css-1f35b1j',
                '.css-14hdny2',
                # Amazon
                '.a-price-whole',
                '.a-price-fraction',
                # Common patterns
                'span[class*="price"]',
                'div[class*="price"]',
                'meta[property="product:price:amount"]',
            ]

            for selector in price_selectors:
                price_elements = soup.select(selector)
                if price_elements:
                    for elem in price_elements:
                        text = elem.get_text(strip=True)
                        # Ищем числа с валютными символами
                        price_pattern = r'(\$|€|£|¥|₽)?\s*(\d+(?:\.\d{2})?)'
                        match = re.search(price_pattern, text)
                        if match:
                            currency = match.group(1) or '$'
                            price = match.group(2)
                            return f"{currency}{price}"

                        # Если не нашли паттерн, возвращаем весь текст
                        if text and any(char.isdigit() for char in text):
                            return text

            # Если не нашли с селекторами, ищем в JSON-LD
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get('@type') == 'Product':
                        offers = data.get('offers', {})
                        if isinstance(offers, dict):
                            price = offers.get('price')
                            if price:
                                return str(price)
                        elif isinstance(offers, list) and offers:
                            price = offers[0].get('price')
                            if price:
                                return str(price)
                except (json.JSONDecodeError, AttributeError):
                    continue

            return None

    except Exception as e:
        logger.warning(f"Failed to extract price from {url}: {e}")
        return None


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
                resp = await client.get(GOOGLE_SEARCH_URL.strip(), params=params)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                results: List[Dict[str, Any]] = []
                for it in items:
                    # Извлекаем больше данных из pagemap
                    pagemap = it.get("pagemap", {})

                    # Цена (может быть в разных форматах)
                    price = None
                    if pagemap.get("product"):
                        product_data = pagemap["product"][0]
                        # Проверяем offers.price
                        if product_data.get("offers", {}).get("price"):
                            price = product_data["offers"]["price"]
                        # Проверяем product.price
                        elif product_data.get("price"):
                            price = product_data["price"]
                        # Проверяем offers.pricecurrency
                        elif product_data.get("offers", {}).get("pricecurrency"):
                            price = product_data["offers"]["pricecurrency"]

                    # Альтернативные источники цены
                    if not price:
                        # Из metatags (Open Graph, Twitter cards)
                        if pagemap.get("metatags"):
                            for tag in pagemap["metatags"]:
                                if tag.get("product:price:amount"):
                                    price = tag["product:price:amount"]
                                    break
                                elif tag.get("twitter:data1"):
                                    # Twitter cards иногда содержат цену
                                    twitter_data = tag["twitter:data1"]
                                    if "$" in twitter_data or "price" in twitter_data.lower():
                                        price = twitter_data
                                        break

                        # Из structured data
                        if pagemap.get("jsonld"):
                            for json_data in pagemap["jsonld"]:
                                if isinstance(json_data, dict):
                                    if json_data.get("@type") == "Product":
                                        if json_data.get("offers", {}).get("price"):
                                            price = json_data["offers"]["price"]
                                            break

                    # Бренд
                    brand = None
                    if pagemap.get("product"):
                        brand = pagemap["product"][0].get("brand")
                    elif pagemap.get("metatags"):
                        for tag in pagemap["metatags"]:
                            if tag.get("og:site_name"):
                                brand = tag["og:site_name"]
                                break

                    # Категория
                    category = None
                    if pagemap.get("product"):
                        category = pagemap["product"][0].get("category")

                    # Рейтинг
                    rating = None
                    if pagemap.get("aggregaterating"):
                        rating = pagemap["aggregaterating"][0].get("ratingvalue")

                    p = {
                        "name": it.get("title", "Unknown Product"),
                        "url": it.get("link", ""),
                        "snippet": it.get("snippet", ""),
                        "image_url": (pagemap.get("cse_image", [{}])[0].get("src") if pagemap.get("cse_image") else None),
                        "price": price,
                        "brand": brand,
                        "category": category,
                        "rating": rating,
                    }

                    # Если цена не найдена в pagemap, пытаемся извлечь с сайта
                    if not p["price"] and p["url"]:
                        try:
                            extracted_price = await extract_price_from_url(p["url"])
                            if extracted_price:
                                p["price"] = extracted_price
                                logger.info(f"Extracted price from {p['url']}: {extracted_price}")
                        except Exception as e:
                            logger.warning(f"Failed to extract price from {p['url']}: {e}")

                    # Фильтруем пустые результаты
                    if p["name"] and p["url"]:
                        results.append(p)
                return results if results else []  # Возвращаем пустой список если ничего нет
        except Exception as e:
            logger.warning(f"Google CSE attempt {attempts} failed: {e}")
    logger.error("Google CSE all attempts failed; returning empty list")
    return []
