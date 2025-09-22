from __future__ import annotations
from typing import List, Dict, Any
import json
import time
import asyncio

from google import genai
from google.genai import types

from app.config import settings
from app.utils.logging import get_logger
from app.services.product_search_client import ProductSearchClient

logger = get_logger("gemini")

SYSTEM_PROMPT_PLAN = (
    "You are an expert dermatology assistant. Your task is to:\n"
    "1. Analyze the provided MedGemma skin analysis\n"
    "2. Call the search_products tool MULTIPLE TIMES (2-3 calls) with SPECIFIC, FOCUSED queries\n"
    "3. Each search query should target ONE specific skin issue or product type\n"
    "4. Use targeted queries like 'acne cleanser', 'anti-aging serum', 'moisturizer dry skin' instead of long combined queries\n\n"
    "IMPORTANT: You MUST call the search_products function MULTIPLE TIMES with different focused queries."
)

SYSTEM_PROMPT_FINAL = (
    "You are an expert dermatologist selecting personalized skin-care products. "
    "Based on the detailed skin analysis and search results, create a comprehensive care plan.\n"
    "Return a JSON with:\n"
    "- diagnosis: detailed skin condition summary\n"
    "- skin_type: specific skin type with characteristics\n"
    "- explanation: thorough explanation of why each product is recommended\n"
    "- routine_steps: recommended skincare routine steps\n"
    "- products: list of up to 5 items {name,url,price?,snippet?,image_url?} with specific purpose for each\n"
    "- additional_recommendations: lifestyle and care tips\n"
    "- medgemma_summary: include the full MedGemma analysis text for reference\n"
    "STRICT OUTPUT REQUIREMENTS: Respond with a single valid JSON object ONLY, no markdown, no explanations, no code fences. "
    "DO NOT wrap the JSON in ```json or any other format. DO NOT add any prefix or suffix. JUST THE JSON."
)

# Function declaration JSON schema used by SDK
SEARCH_PRODUCTS_FUNCTION = types.FunctionDeclaration(
    name="search_products",
    description="Search for skincare products based on skin conditions",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING", "description": "Search query for skincare products"}
        },
        "required": ["query"],
    },
)

class GeminiClient:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set; Gemini calls will fail")

        # Create a central client (sync + async available via client.aio)
        # For Gemini Developer API, use api_key param
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model

        # create a Tool wrapper for function-calling usage
        self.search_tool = types.Tool(function_declarations=[SEARCH_PRODUCTS_FUNCTION])

        # Initialize product search client
        self.product_search_client = ProductSearchClient()

    async def plan_with_tool(self, medgemma_summary: str, user_text: str | None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        First stage: ask model to analyze medgemma_summary and call search_products multiple times.
        We use multiple rounds of conversation to encourage multiple tool calls.
        """
        user_message = (
            f"Based on this MedGemma skin analysis, search for appropriate skincare products.\n\n"
            f"MedGemma Analysis:\n{medgemma_summary}\n\n"
            f"User note: {user_text or 'No additional notes'}\n\n"
            f"Please call the search_products function multiple times (2-3 times) with specific, focused queries. "
            f"Each query should target a specific skin concern or product type (e.g., 'acne cleanser', 'anti-aging serum', 'moisturizer for dry skin')."
        )

        collected_products: List[Dict[str, Any]] = []
        conversation_history = [user_message]
        
        # Пытаемся получить множественные вызовы через многоходовую беседу
        max_rounds = 3
        for round_num in range(max_rounds):
            attempts, response = 0, None
            backoffs = [1, 2, 4]

            while attempts < 3:
                attempts += 1
                try:
                    # async call
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=conversation_history[-1] if round_num == 0 else "Please make another search for a different skin concern from the analysis.",
                        config=types.GenerateContentConfig(
                            tools=[self.search_tool],
                            temperature=0.1,
                            max_output_tokens=1024,
                        ),
                    )
                    break
                except Exception as e:
                    logger.warning(f"[Gemini] planning round {round_num+1} attempt {attempts} failed: {e}")
                    if attempts < 3:
                        time.sleep(backoffs[attempts - 1])
                    else:
                        if round_num == 0:  # Только первый раунд критичен
                            raise
                        else:
                            logger.warning(f"[Gemini] Skipping round {round_num+1} due to errors")
                            break

            if response is None:
                if round_num == 0:
                    raise Exception("Failed to get response from Gemini after 3 attempts")
                continue

            # Process function calls from this round
            function_calls = getattr(response, "function_calls", None)
            if function_calls:
                for fc in function_calls:
                    func_name = getattr(fc, "name", None)
                    args = getattr(fc, "args", {}) or {}
                    if func_name == "search_products":
                        query = args.get("query", "")
                        logger.info(f"[Gemini] Round {round_num+1} Tool call: search_products with query='{query}'")
                        if query:
                            try:
                                products = await self.product_search_client.search_products(query=query, num=3)  # Меньше продуктов на запрос
                                collected_products.extend(products)
                            except Exception as e:
                                logger.warning(f"[Gemini] Tool search_products failed in round {round_num+1}: {e}")

            # Если модель не вызвала функции, добавляем подсказку для следующего раунда
            if not function_calls and round_num < max_rounds - 1:
                conversation_history.append("Please search for skincare products for another skin concern mentioned in the analysis.")

        # Если все же не получили достаточно вызовов, делаем fallback множественные поиски
        if len(collected_products) < 3:
            fallback_queries = self._generate_multiple_fallback_queries(medgemma_summary)
            logger.info(f"[Gemini] Using fallback queries: {fallback_queries}")
            
            # Выполняем параллельно несколько поисков
            search_tasks = [self.product_search_client.search_products(query=q, num=2) for q in fallback_queries[:3]]
            try:
                fallback_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                for i, result in enumerate(fallback_results):
                    if isinstance(result, Exception):
                        logger.warning(f"[Gemini] Fallback search {i+1} failed: {result}")
                    else:
                        collected_products.extend(result)
            except Exception as e:
                logger.warning(f"[Gemini] Fallback searches failed: {e}")

        # Убираем дубликаты по URL
        seen_urls = set()
        unique_products = []
        for product in collected_products:
            if product.get('url') and product['url'] not in seen_urls:
                seen_urls.add(product['url'])
                unique_products.append(product)
        
        collected_products = unique_products[:10]  # Ограничиваем до 10 продуктов

        plan = self._create_plan_from_analysis(medgemma_summary, collected_products, "multiple focused searches")

        logger.info(f"[Gemini] Plan created with {len(collected_products)} products from multiple searches")
        return plan, collected_products

    def _generate_multiple_fallback_queries(self, medgemma_summary: str) -> List[str]:
        """Генерирует несколько целенаправленных запросов на основе анализа кожи"""
        low = medgemma_summary.lower()
        queries = []

        # Основные категории продуктов
        if "blackhead" in low or "comedone" in low or "pore" in low:
            queries.append("pore cleansing treatment")
        if "dehydrat" in low or "dry" in low:
            queries.append("hydrating moisturizer")
        if "acne" in low or "pimple" in low or "pustule" in low:
            queries.append("acne treatment serum")
        if "aging" in low or "fine line" in low or "wrinkle" in low:
            queries.append("anti-aging cream")
        if "hyperpigmentation" in low or "dark spot" in low or "sun damage" in low:
            queries.append("brightening serum")
        if "inflam" in low or "redness" in low or "irritat" in low:
            queries.append("soothing skincare")
        if "oily" in low or "sebum" in low:
            queries.append("oil control products")
        if "sensitive" in low:
            queries.append("gentle skincare")

        # Если ничего специфического не найдено, используем базовые категории
        if not queries:
            queries = ["facial cleanser", "moisturizer", "sunscreen"]

        # Возвращаем максимум 4 запроса
        return queries[:4]

    def _generate_fallback_query(self, medgemma_summary: str) -> str:
        """Оставляем для совместимости, но теперь используется редко"""
        queries = self._generate_multiple_fallback_queries(medgemma_summary)
        return " ".join(queries[:2]) if queries else "skincare products routine"

    def _create_plan_from_analysis(self, medgemma_summary: str, products: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        low = medgemma_summary.lower()
        skin_type = "combination"

        if "oily" in low and "dry" not in low:
            skin_type = "oily"
        elif "dry" in low and "oily" not in low:
            skin_type = "dry"
        elif "normal" in low:
            skin_type = "normal"
        elif "sensitive" in low:
            skin_type = "sensitive"

        concerns = []
        if "blackhead" in low or "comedone" in low:
            concerns.append("blackheads and comedones")
        if "dehydrat" in low:
            concerns.append("dehydration")
        if "acne" in low:
            concerns.append("acne")
        if "hyperpigmentation" in low:
            concerns.append("hyperpigmentation")
        if "aging" in low or "fine line" in low:
            concerns.append("signs of aging")
        if "inflam" in low or "redness" in low:
            concerns.append("inflammation and redness")

        deficiencies, excesses = [], []
        if "dehydrat" in low or "dry" in low:
            deficiencies.append("moisture")
        if "dull" in low:
            deficiencies.append("radiance")
        if "barrier" in low and "compromised" in low:
            deficiencies.append("barrier function")
        if "oily" in low or "sebum" in low:
            excesses.append("sebum production")
        if "comedone" in low:
            excesses.append("clogged pores")

        diagnosis = medgemma_summary.split("\n")[0] if medgemma_summary else "Skin analysis completed"
        if "Summary:" in diagnosis:
            diagnosis = diagnosis.replace("**Summary:**", "").replace("**", "").strip()

        return {
            "skin_type": skin_type,
            "diagnosis": diagnosis[:500],
            "concerns": concerns,
            "deficiencies": deficiencies,
            "excesses": excesses,
            "query": query,
            "need_search": True,
            "medgemma_analysis": medgemma_summary
        }

    def finalize_with_products(self, planning_json: str, products_jsonl: str) -> Dict[str, Any]:
        """
        Second stage: combine planning + product search results and ask model to produce a single JSON.
        This is using the sync client.models.generate_content so it can be called from sync code.
        """
        # Create the response schema (you can keep using dict-based JSON schema; SDK accepts it)
        schema = {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "string"},
                "skin_type": {"type": "string"},
                "explanation": {"type": "string"},
                "routine_steps": {"type": "array", "items": {"type": "string"}},
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "url": {"type": "string"},
                            "price": {"type": "string"},
                            "snippet": {"type": "string"},
                            "image_url": {"type": "string"},
                        },
                        "required": ["name", "url"],
                    },
                },
                "additional_recommendations": {"type": "string"},
                "medgemma_summary": {"type": "string"},
            },
            "required": ["diagnosis", "skin_type", "explanation", "products", "medgemma_summary"],
        }

        attempts, resp = 0, None
        backoffs = [1, 2, 4]

        prompt = f"Plan: {planning_json}\nProducts: {products_jsonl}"

        while attempts < 3:
            attempts += 1
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                break
            except Exception as e:
                logger.warning(f"[Gemini] finalize attempt {attempts} failed: {e}")
                if attempts < 3:
                    time.sleep(backoffs[attempts - 1])
                else:
                    raise

        if not resp:
            return self._fallback_response("No response returned")

        # Try to parse response.text (SDK exposes combined text via .text)
        content_text = getattr(resp, "text", None)
        if not content_text:
            # Try to grab first candidate textual content as fallback
            try:
                candidate = resp.candidates[0]
                content_text = getattr(candidate, "text", "") or ""
            except Exception:
                content_text = ""

        if not content_text.strip():
            return self._fallback_response("Empty text content after parsing parts")

        try:
            return json.loads(content_text)
        except json.JSONDecodeError as e:
            logger.warning(f"[Gemini] finalize JSON parse failed: {e}")
            logger.warning(f"[Gemini] Raw response: {content_text[:1000]}...")
            return self._fallback_response(f"JSON decode error: {str(e)}")

    def _fallback_response(self, reason: str) -> Dict[str, Any]:
        logger.error(f"[Gemini] Fallback finalize response due to: {reason}")
        return {
            "diagnosis": "Analysis failed",
            "skin_type": "unknown",
            "explanation": "",
            "routine_steps": [],
            "products": [],
            "additional_recommendations": "",
            "medgemma_summary": "",
        }