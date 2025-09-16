from __future__ import annotations
from typing import List, Dict, Any
import json
import time

import google.generativeai as genai

from app.config import settings
from app.utils.logging import get_logger
from app.tools.search_products import search_products


logger = get_logger("gemini")

# Упрощенный и более директивный промпт
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
    "STRICT OUTPUT REQUIREMENTS: Respond with a single valid JSON object ONLY, no markdown, no explanations, no code fences."
)

# Упрощенное объявление инструмента
TOOL_DECLARATIONS = [{
    "name": "search_products",
    "description": "Search for skincare products based on skin conditions",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "Search query for skincare products"
            }
        },
        "required": ["query"]
    }
}]


class GeminiClient:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set; Gemini calls will fail")
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model

    async def plan_with_tool(self, medgemma_summary: str, user_text: str | None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Создаем модель с инструментами
        model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            system_instruction=SYSTEM_PROMPT_PLAN
        )

        # Более прямое указание на необходимость вызвать функцию
        user_message = (
            f"Based on this MedGemma skin analysis, search for appropriate skincare products.\n\n"
            f"MedGemma Analysis:\n{medgemma_summary}\n\n"
            f"User note: {user_text or 'No additional notes'}\n\n"
            f"Now call the search_products function with an appropriate query based on the skin issues identified."
        )

        # Первая попытка с явным требованием вызвать функцию
        attempts = 0
        backoffs = [1, 2, 4]
        response = None
        
        while attempts < 3:
            attempts += 1
            try:
                # Используем более явную конфигурацию для вызова функции
                response = model.generate_content(
                    user_message,
                    tool_config={
                        "function_calling_config": {
                            "mode": "ANY",
                            "allowed_function_names": ["search_products"]
                        }
                    },
                    generation_config={
                        "temperature": 0.1,  # Снижаем температуру для более предсказуемого поведения
                        "max_output_tokens": 256  # Уменьшаем, так как нужен только вызов функции
                    },
                )
                break
            except Exception as e:
                logger.warning(f"[Gemini] planning first turn attempt {attempts} failed: {e}")
                if attempts < 3:
                    time.sleep(backoffs[attempts - 1])
                else:
                    raise

        if response is None:
            raise Exception("Failed to get response from Gemini after 3 attempts")

        # Обработка вызовов функций
        collected_products: List[Dict[str, Any]] = []
        func_calls = []
        
        # Проверяем, есть ли вызовы функций
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            fc = part.function_call
                            if fc.name == "search_products":
                                args = dict(fc.args) if fc.args else {}
                                query = args.get("query", "")
                                
                                logger.info(f"[Gemini] Tool call detected: search_products with query='{query}'")
                                
                                if query:
                                    try:
                                        products = await search_products(query=query, num=5)
                                        collected_products = products
                                        func_calls.append({
                                            "name": "search_products",
                                            "result": products,
                                        })
                                    except Exception as e:
                                        logger.warning(f"[Gemini] Tool search_products failed: {e}")

        # Если функция не была вызвана, используем fallback
        if not func_calls:
            # Извлекаем ключевые слова из анализа для формирования запроса
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
                products = []

        # Формируем план на основе анализа и найденных продуктов
        plan = self._create_plan_from_analysis(medgemma_summary, collected_products, fallback_query if not func_calls else "")
        
        logger.info(f"[Gemini] Plan created with {len(collected_products)} products")
        return plan, collected_products

    def _generate_fallback_query(self, medgemma_summary: str) -> str:
        """Генерирует поисковый запрос на основе анализа MedGemma"""
        low = medgemma_summary.lower()
        queries = []
        
        # Проверяем различные проблемы кожи
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
        
        # Если нашли проблемы, объединяем первые 2-3
        if queries:
            return " ".join(queries[:2])
        
        # Дефолтный запрос
        return "skincare products routine"

    def _create_plan_from_analysis(self, medgemma_summary: str, products: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Создает план на основе анализа MedGemma"""
        low = medgemma_summary.lower()
        
        # Определяем тип кожи
        skin_type = "combination"
        if "oily" in low and "dry" not in low:
            skin_type = "oily"
        elif "dry" in low and "oily" not in low:
            skin_type = "dry"
        elif "normal" in low:
            skin_type = "normal"
        elif "sensitive" in low:
            skin_type = "sensitive"
        
        # Собираем проблемы
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
        
        # Определяем недостатки и избытки
        deficiencies = []
        excesses = []
        
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
        
        # Формируем диагноз
        diagnosis = medgemma_summary.split('\n')[0] if medgemma_summary else "Skin analysis completed"
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

    def finalize_with_products(self, planning_json: str, products_jsonl: str) -> str:
        """Финализирует ответ с продуктами"""
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT_FINAL,
        )
        
        attempts = 0
        backoffs = [1, 2, 4]
        resp = None
        
        while attempts < 3:
            attempts += 1
            try:
                resp = model.generate_content(
                    f"Plan: {planning_json}\nProducts: {products_jsonl}",
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 1024,
                    }
                )
                break
            except Exception as e:
                logger.warning(f"[Gemini] finalize attempt {attempts} failed: {e}")
                if attempts < 3:
                    time.sleep(backoffs[attempts - 1])
                else:
                    raise

        if resp is None:
            return "{}"

        # Извлекаем текст из ответа
        content_text = ""
        try:
            content_text = resp.text if hasattr(resp, 'text') else ""
        except Exception:
            if resp and resp.candidates:
                try:
                    content_text = resp.candidates[0].content.parts[0].text
                except Exception:
                    pass

        if not content_text:
            return "{}"

        # Очищаем JSON от markdown
        return self._clean_json_response(content_text)

    def _clean_json_response(self, text: str) -> str:
        """Очищает JSON от markdown и других артефактов"""
        clean_text = text.strip()
        
        # Удаляем markdown code fences
        if clean_text.startswith('```json'):
            clean_text = clean_text[7:]
        elif clean_text.startswith('```'):
            clean_text = clean_text[3:]
        if clean_text.endswith('```'):
            clean_text = clean_text[:-3]
        
        # Находим границы JSON объекта
        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_text = clean_text[start_idx:end_idx + 1]
            # Проверяем валидность JSON
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass
        
        # Если не удалось извлечь валидный JSON, возвращаем исходный текст
        return clean_text