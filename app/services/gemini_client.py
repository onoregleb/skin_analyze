from __future__ import annotations
from typing import List, Dict, Any
import json
import time

import google.generativeai as genai

from app.config import settings
from app.utils.logging import get_logger
from app.tools.search_products import search_products


logger = get_logger("gemini")

SYSTEM_PROMPT_PLAN = (
    "You are an expert dermatology assistant. When analyzing skin conditions:\n"
    "1. First, acknowledge and repeat the detailed MedGemma analysis to preserve all important observations\n"
    "2. Then, provide your structured assessment in these categories:\n"
    "   - Skin type (oily, dry, combination, normal)\n"
    "   - Hydration level\n"
    "   - Oil production\n"
    "   - Texture analysis\n"
    "   - Specific issues (acne, blackheads, etc.)\n"
    "   - Signs of aging or sun damage\n"
    "   - Skin barrier condition\n"
    "   - Sensitivity indicators\n"
    "3. Finally, produce a detailed JSON with keys:\n"
    "   - medgemma_analysis: full text of the original analysis\n"
    "   - skin_type: detailed skin type\n"
    "   - diagnosis: your comprehensive analysis summary\n"
    "   - concerns: list of specific concerns\n"
    "   - deficiencies: list of missing elements\n"
    "   - excesses: list of elements in excess\n"
    "   - query: search query for products\n"
    "   - need_search: boolean\n"
    "STRICT OUTPUT REQUIREMENTS: Respond with a single valid JSON object ONLY, no markdown, no explanations, no code fences."
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
    "STRICT OUTPUT REQUIREMENTS: Respond with a single valid JSON object ONLY, no markdown, no explanations, no code fences."
)

TOOL_DECLARATIONS = [{
    "name": "search_products",
    "description": "Search skincare products on the web via Google Custom Search.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING"},
            "num": {"type": "NUMBER"}
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

    async def plan_with_tool(self, medgemma_summary: str, user_text: str | None) -> Dict[str, Any]:
        # Prepare tool-enabled model for planning
        model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            system_instruction=SYSTEM_PROMPT_PLAN + " Return only a valid JSON object, no markdown."
        )

        user_parts = [
            f"MedGemma Analysis:\n{medgemma_summary}\n\nUser note: {user_text or ''}"
        ]

        # First turn: expect a function_call request
        attempts = 0
        last_error: Exception | None = None
        backoffs = [1, 2, 4]
        while attempts < 3:
            attempts += 1
            try:
                response = model.generate_content(
                    [{"role": "user", "parts": user_parts}],
                    tool_config={"function_calling_config": {"mode": "ANY"}},
                    generation_config={"temperature": 0.3, "max_output_tokens": 1024},
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"[Gemini] planning first turn attempt {attempts} failed: {e}")
                if attempts < 3:
                    try:
                        time.sleep(backoffs[attempts - 1])
                    except Exception:
                        pass
                else:
                    raise

        # Collect and execute tool calls
        collected_products: List[Dict[str, Any]] = []
        func_calls: List[Dict[str, Any]] = []
        try:
            parts = response.candidates[0].content.parts if response.candidates else []
        except Exception:
            parts = []
        for p in parts:
            try:
                fc = getattr(p, "function_call", None)
                if fc and getattr(fc, "name", None) == "search_products":
                    args = dict(fc.args or {})
                    query = args.get("query")
                    num = int(args.get("num", 10))
                    logger.info(f"[Gemini] Executing tool search_products args={{'query': '{query}', 'num': {num}}}")
                    try:
                        products = await search_products(query=query, num=num)
                    except Exception as e:
                        logger.warning(f"[Gemini] Tool search_products failed: {e}")
                        products = []
                    collected_products = products
                    func_calls.append({
                        "name": "search_products",
                        "response": {"name": "search_products", "content": products},
                    })
            except Exception as e:
                logger.warning(f"[Gemini] Parse function_call error: {e}")

        if not func_calls:
            logger.error("[Gemini] No function calls produced in planning phase")
            return {"skin_type": "unknown", "diagnosis": "No tool call produced", "query": "", "need_search": False, "tool_products": []}

        # Build a follow-up turn including the assistant function_call and our tool responses
        tool_parts = [{
            "function_response": {
                "name": fc["name"],
                "response": fc["response"],
            }
        } for fc in func_calls]

        # Continue conversation to get final structured JSON plan
        attempts = 0
        last_error = None
        while attempts < 3:
            attempts += 1
            try:
                followup = model.generate_content(
                    [
                        {"role": "user", "parts": user_parts},
                        response.candidates[0].content,
                        {"role": "tool", "parts": tool_parts},
                    ],
                    generation_config={"temperature": 0.2, "max_output_tokens": 1024},
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"[Gemini] planning follow-up attempt {attempts} failed: {e}")
                if attempts < 3:
                    try:
                        time.sleep(backoffs[attempts - 1])
                    except Exception:
                        pass
                else:
                    raise

        content_text = getattr(followup, "text", None) or ""
        if not content_text:
            try:
                # Some SDK versions require extracting from candidates
                content_text = followup.candidates[0].content.parts[0].text
            except Exception:
                content_text = ""

        try:
            plan = json.loads(content_text)
        except Exception:
            logger.warning("[Gemini] Plan JSON parse failed; returning fallback")
            plan = {"skin_type": "unknown", "diagnosis": content_text[:200], "query": "", "need_search": False}
        plan["tool_products"] = collected_products
        logger.info(f"[Gemini] Plan parsed need_search={plan.get('need_search')} skin_type={plan.get('skin_type')}")
        return plan

    def finalize_with_products(self, planning_json: str, products_jsonl: str) -> str:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT_FINAL,
        )
        attempts = 0
        last_error: Exception | None = None
        backoffs = [1, 2, 4]
        while attempts < 3:
            attempts += 1
            try:
                resp = model.generate_content([
                    {"role": "user", "parts": [f"Plan: {planning_json}\nProducts: {products_jsonl}"]}
                ], generation_config={"temperature": 0.2, "max_output_tokens": 1024})
                break
            except Exception as e:
                last_error = e
                logger.warning(f"[Gemini] finalize attempt {attempts} failed: {e}")
                if attempts < 3:
                    try:
                        time.sleep(backoffs[attempts - 1])
                    except Exception:
                        pass
                else:
                    raise

        content_text = getattr(resp, "text", None) or ""
        if not content_text:
            try:
                content_text = resp.candidates[0].content.parts[0].text
            except Exception:
                content_text = ""
        return content_text
