from __future__ import annotations
from typing import List, Dict, Any
import json
import time

# Импорты из нового SDK
from google import genai
from google.genai import types # Для типов из SDK
# errors для обработки ошибок
from google.genai import errors

from app.config import settings
from app.utils.logging import get_logger
from app.tools.search_products import search_products

logger = get_logger("gemini")

# Системные промпты остаются без изменений
SYSTEM_PROMPT_PLAN = (
    "You are an expert dermatology assistant. Your task is to:\n"
    "1. Analyze the provided MedGemma skin analysis\n"
    "2. IMMEDIATELY call the search_products tool with a relevant query based on the skin concerns identified\n"
    "3. The search query should target specific skin issues mentioned in the analysis\n\n"
    "IMPORTANT: You MUST call the search_products function in your first response."
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

# Объявления инструментов теперь используют типы из SDK
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_products",
        description="Search for skincare products based on skin conditions",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="Search query for skincare products"
                )
            },
            required=["query"]
        )
    )
]

class GeminiClient:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set; Gemini calls will fail")
        # Создание клиента с использованием нового SDK
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model

    async def plan_with_tool(self, medgemma_summary: str, user_text: str | None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Используем клиент для доступа к моделям
        model_instance = self.client.models # Это просто ссылка на модуль моделей

        user_message_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=(
                f"Based on this MedGemma skin analysis, search for appropriate skincare products.\n\n"
                f"MedGemma Analysis:\n{medgemma_summary}\n\n"
                f"User note: {user_text or 'No additional notes'}\n\n"
                f"Now call the search_products function with an appropriate query based on the skin issues identified."
            ))]
        )

        attempts, response = 0, None
        backoffs = [1, 2, 4]

        while attempts < 3:
            attempts += 1
            try:
                # Используем generate_content из нового SDK
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=user_message_content,
                    config=types.GenerateContentConfig(
                        tools=TOOL_DECLARATIONS, # Передаем список FunctionDeclaration
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=types.FunctionCallingConfigMode.ANY,
                                allowed_function_names=["search_products"]
                            )
                        ),
                        system_instruction=SYSTEM_PROMPT_PLAN,
                        temperature=0.1,
                        max_output_tokens=256
                    ),
                )
                break
            except errors.APIError as e: # Обработка ошибок из SDK
                 logger.warning(f"[Gemini] planning first turn attempt {attempts} failed with APIError: {e}")
                 if attempts < 3:
                     time.sleep(backoffs[attempts - 1])
                 else:
                     raise
            except Exception as e:
                logger.warning(f"[Gemini] planning first turn attempt {attempts} failed with general error: {e}")
                if attempts < 3:
                    time.sleep(backoffs[attempts - 1])
                else:
                    raise

        if response is None:
            raise Exception("Failed to get response from Gemini after 3 attempts")

        collected_products: List[Dict[str, Any]] = []
        func_calls = []

        # Обработка ответа и вызов инструментов
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Проверяем, является ли часть вызовом функции
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            if fc.name == "search_products":
                                args = dict(fc.args) if fc.args else {}
                                query = args.get("query", "")
                                logger.info(f"[Gemini] Tool call detected: search_products with query='{query}'")

                                if query:
                                    try:
                                        # Выполняем асинхронный вызов внешней функции
                                        products = await search_products(query=query, num=5)
                                        collected_products = products
                                        func_calls.append({
                                            "name": "search_products",
                                            "result": products,
                                        })
                                    except Exception as e:
                                        logger.warning(f"[Gemini] Tool search_products failed: {e}")

        # Если инструмент не был вызван, используем резервный вариант
        if not func_calls:
            fallback_query = self._generate_fallback_query(medgemma_summary)
            logger.info(f"[Gemini] No function call detected, using fallback query: {fallback_query}")
            try:
                products = await search_products(query=fallback_query, num=5)
                collected_products = products
                func_calls.append({
                    "name": "search_products",
                    "result": products,
                })
            except Exception as e:
                logger.warning(f"[Gemini] Fallback search failed: {e}")
                collected_products = []

        plan = self._create_plan_from_analysis(
            medgemma_summary,
            collected_products,
            "" if func_calls else fallback_query,
        )

        logger.info(f"[Gemini] Plan created with {len(collected_products)} products")
        return plan, collected_products

    def _generate_fallback_query(self, medgemma_summary: str) -> str:
        # Логика резервного запроса остается без изменений
        low = medgemma_summary.lower()
        queries = []

        if "blackhead" in low or "comedone" in low:
            queries.append("blackheads comedones treatment")
        if "dehydrat" in low or "dry" in low:
            queries.append("dehydrated skin moisturizer")
        if "acne" in low or "pimple" in low or "pustule" in low:
            queries.append("acne treatment products")
        if "aging" in low or "fine line" in low or "wrinkle" in low:
            queries.append("anti-aging skincare")
        if "hyperpigmentation" in low or "dark spot" in low:
            queries.append("hyperpigmentation treatment")
        if "inflam" in low or "redness" in low:
            queries.append("anti-inflammatory skincare")
        if "oily" in low:
            queries.append("oil control products")

        if queries:
            return " ".join(queries[:2])

        return "skincare products routine"

    def _create_plan_from_analysis(self, medgemma_summary: str, products: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        # Логика создания плана остается без изменений
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
            "query": query or " ".join(concerns[:2]) if concerns else "skincare products",
            "need_search": True,
            "medgemma_analysis": medgemma_summary
        }

    async def finalize_with_products(self, planning_json: str, products_jsonl: str) -> Dict[str, Any]:
        # Определение схемы ответа с использованием типов SDK
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "diagnosis": types.Schema(type=types.Type.STRING),
                "skin_type": types.Schema(type=types.Type.STRING),
                "explanation": types.Schema(type=types.Type.STRING),
                "routine_steps": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "products": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "name": types.Schema(type=types.Type.STRING),
                            "url": types.Schema(type=types.Type.STRING),
                            "price": types.Schema(type=types.Type.STRING),
                            "snippet": types.Schema(type=types.Type.STRING),
                            "image_url": types.Schema(type=types.Type.STRING),
                        },
                        required=["name", "url"],
                    ),
                ),
                "additional_recommendations": types.Schema(type=types.Type.STRING),
                "medgemma_summary": types.Schema(type=types.Type.STRING),
            },
            required=["diagnosis", "skin_type", "explanation", "products", "medgemma_summary"],
        )

        attempts, resp = 0, None
        backoffs = [1, 2, 4]

        while attempts < 3:
            attempts += 1
            try:
                # Используем асинхронный метод generate_content из нового SDK
                resp = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=f"Plan: {planning_json}\nProducts: {products_jsonl}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT_FINAL,
                        temperature=0.2,
                        max_output_tokens=1024,
                        response_mime_type='application/json',
                        response_schema=response_schema,
                    ),
                )
                break
            except errors.APIError as e: # Обработка ошибок из SDK
                logger.warning(f"[Gemini] finalize attempt {attempts} failed with APIError: {e}")
                if attempts < 3:
                    time.sleep(backoffs[attempts - 1])
                else:
                    raise
            except Exception as e:
                logger.warning(f"[Gemini] finalize attempt {attempts} failed with general error: {e}")
                if attempts < 3:
                    time.sleep(backoffs[attempts - 1])
                else:
                    raise

        if not resp or not resp.candidates:
            return self._fallback_response("No candidates returned")

        # Проверяем, не заблокирован ли ответ
        candidate = resp.candidates[0]
        # Используем FinishReason из types
        if candidate.finish_reason != types.FinishReason.STOP:
            # Получаем имя причины завершения
            safety_reason = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
            logger.warning(f"[Gemini] Response blocked due to finish reason: {safety_reason}")
            # Используем SafetyRating из types
            for rating in candidate.safety_ratings:
                if rating.probability != types.HarmProbability.NEARLY_NONE:
                    logger.warning(f"[Gemini] Safety rating: {rating.category} -> {rating.probability}")
            return self._fallback_response(f"Response blocked: {safety_reason}")

        # Теперь проверяем, есть ли части (parts)
        if not candidate.content or not candidate.content.parts:
            return self._fallback_response("No content parts in response")

        # Извлекаем текст — теперь безопасно
        content_text = ""
        for part in candidate.content.parts:
            if hasattr(part, "text"):
                content_text += part.text
            elif hasattr(part, "function_call"):  # На всякий случай — если вернёт функцию (хотя не должно)
                logger.warning("[Gemini] Unexpected function call in finalize phase")

        if not content_text.strip():
            return self._fallback_response("Empty text content after parsing parts")

        try:
            return json.loads(content_text)
        except json.JSONDecodeError as e:
            logger.warning(f"[Gemini] finalize JSON parse failed: {e}")
            logger.warning(f"[Gemini] Raw response: {content_text[:500]}...")
            return self._fallback_response(f"JSON decode error: {str(e)}")

    def _fallback_response(self, reason: str) -> Dict[str, Any]:
        # Резервный ответ остается без изменений
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
