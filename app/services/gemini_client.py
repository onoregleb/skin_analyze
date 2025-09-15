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
    "3. ALWAYS generate a search query for relevant skincare products based on the analysis\n"
    "4. Produce a detailed JSON with keys:\n"
    "   - medgemma_analysis: full text of the original analysis\n"
    "   - skin_type: detailed skin type\n"
    "   - diagnosis: your comprehensive analysis summary\n"
    "   - concerns: list of specific concerns\n"
    "   - deficiencies: list of missing elements\n"
    "   - excesses: list of elements in excess\n"
    "   - query: search query for products (REQUIRED)\n"
    "   - need_search: true (ALWAYS true)\n"
    "STRICT OUTPUT REQUIREMENTS: Respond with a single valid JSON object ONLY, no markdown, no explanations, no code fences."
    "IMPORTANT: You MUST call the search_products tool to find relevant products."
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
            "num": {"type": "INTEGER"}
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
        response = None
        while attempts < 3:
            attempts += 1
            try:
                response = model.generate_content(
                    [{"role": "user", "parts": user_parts}],
                    tool_config={"function_calling_config": {"mode": "AUTO"}},
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

        if not response:
            raise Exception("Failed to get response from Gemini after 3 attempts")

        # Collect and execute tool calls
        collected_products: List[Dict[str, Any]] = []
        func_calls: List[Dict[str, Any]] = []
        try:
            parts = response.candidates[0].content.parts if response.candidates and response.candidates[0].content else []
        except Exception as e:
            logger.warning(f"[Gemini] Error accessing response parts: {e}")
            parts = []
            
        for p in parts:
            try:
                fc = getattr(p, "function_call", None)
                if fc and getattr(fc, "name", None) == "search_products":
                    args = dict(fc.args or {})
                    query = args.get("query", "")
                    num = int(args.get("num", 5))
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

        # Fallback: if no function calls, create default search
        if not func_calls:
            logger.warning("[Gemini] No function calls produced, creating fallback search...")
            # Try to extract basic info from the response or create default query
            default_query = "skincare products for acne blackheads dehydration"
            try:
                # Try to get text content to extract skin type
                content_text = ""
                try:
                    content_text = response.text
                except:
                    try:
                        content_text = response.candidates[0].content.parts[0].text
                    except:
                        pass
                
                if content_text:
                    # Simple extraction of potential skin concerns
                    if "blackheads" in content_text.lower() or "comedones" in content_text.lower():
                        default_query = "skincare products for blackheads comedones"
                    elif "dehydration" in content_text.lower() or "dry" in content_text.lower():
                        default_query = "skincare products for dehydrated skin"
                    elif "aging" in content_text.lower() or "fine lines" in content_text.lower():
                        default_query = "anti-aging skincare products"
            except Exception as e:
                logger.warning(f"[Gemini] Error extracting content for fallback: {e}")
            
            logger.info(f"[Gemini] Executing fallback search with query: {default_query}")
            try:
                products = await search_products(query=default_query, num=5)
            except Exception as e:
                logger.warning(f"[Gemini] Fallback search_products failed: {e}")
                products = []
            
            collected_products = products
            func_calls.append({
                "name": "search_products",
                "response": {"name": "search_products", "content": products},
            })

        # Build a follow-up turn including the assistant function_call and our tool responses
        tool_parts = [{
            "function_response": {
                "name": fc["name"],
                "response": {
                    "result": fc["response"]["content"]
                }
            }
        } for fc in func_calls]

        # Continue conversation to get final structured JSON plan
        attempts = 0
        last_error = None
        followup = None
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

        if not followup:
            raise Exception("Failed to get followup response from Gemini after 3 attempts")

        # Extract content text with safety checks
        content_text = ""
        try:
            content_text = getattr(followup, "text", None) or ""
        except Exception:
            pass
            
        if not content_text:
            try:
                # Some SDK versions require extracting from candidates
                if followup.candidates and followup.candidates[0].content and followup.candidates[0].content.parts:
                    content_text = followup.candidates[0].content.parts[0].text
            except Exception as e:
                logger.warning(f"[Gemini] Error extracting text from candidates: {e}")
                content_text = ""

        # Handle empty response
        if not content_text.strip():
            logger.warning("[Gemini] Empty response from model, returning fallback plan")
            fallback_plan = {
                "skin_type": "unknown",
                "diagnosis": "Automated skin analysis completed",
                "concerns": ["See detailed MedGemma analysis"],
                "deficiencies": [],
                "excesses": [],
                "query": "skincare products for acne blackheads dehydration",
                "need_search": True,
                "medgemma_analysis": medgemma_summary[:500] + "..." if len(medgemma_summary) > 500 else medgemma_summary
            }
            logger.info(f"[Gemini] Fallback plan created need_search={fallback_plan.get('need_search')} skin_type={fallback_plan.get('skin_type')}")
            return fallback_plan

        # Parse JSON with error handling
        plan = {}
        try:
            plan = json.loads(content_text)
        except Exception as e:
            logger.warning(f"[Gemini] Plan JSON parse failed: {e}; returning fallback with raw text")
            plan = {
                "skin_type": "unknown",
                "diagnosis": content_text[:200] if content_text else "No detailed diagnosis available",
                "concerns": ["See MedGemma analysis"],
                "deficiencies": [],
                "excesses": [],
                "query": "",
                "need_search": True,
                "medgemma_analysis": medgemma_summary[:300] + "..." if len(medgemma_summary) > 300 else medgemma_summary
            }
        
        # Ensure required fields
        plan.setdefault("skin_type", "unknown")
        plan.setdefault("diagnosis", "Analysis completed")
        plan.setdefault("concerns", [])
        plan.setdefault("deficiencies", [])
        plan.setdefault("excesses", [])
        plan.setdefault("query", "")
        plan.setdefault("need_search", True)
        plan.setdefault("medgemma_analysis", medgemma_summary[:500] + "..." if len(medgemma_summary) > 500 else medgemma_summary)
        
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
        resp = None
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

        if not resp:
            return "{}"

        # Extract content text with safety checks
        content_text = ""
        try:
            content_text = getattr(resp, "text", None) or ""
        except Exception:
            pass
            
        if not content_text:
            try:
                if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                    content_text = resp.candidates[0].content.parts[0].text
            except Exception:
                content_text = ""
        
        return content_text