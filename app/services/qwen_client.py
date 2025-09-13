from __future__ import annotations
from typing import List, Dict, Any
import json
from openai import OpenAI
from app.config import settings
from app.utils.logging import get_logger
from app.tools.search_products import search_products
import time


logger = get_logger("qwen")

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
    "Return only a valid JSON object, no markdown."
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

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_products",
        "description": "Search skincare products on the web via Google Custom Search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
}


class QwenClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
        )
        self.model = settings.qwen_model

    def _chat(self, messages: List[Dict[str, Any]], temperature: float = 0.3,
              tools: list | None = None, tool_choice: str | None = None, max_tokens: int = 1024) -> Any:
        attempts = 0
        last_error: Exception | None = None
        backoffs = [1, 2, 4]  # Exponential backoff in seconds

        while attempts < 3:
            attempts += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60.0,
                    tools=tools,
                    tool_choice=tool_choice if tools else None,
                )
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Qwen chat attempt {attempts} failed: {e}")
                if attempts < 3:
                    delay = backoffs[attempts - 1]
                    try:
                        time.sleep(delay)
                    except Exception:
                        pass
        logger.error("Qwen chat failed after retries")
        raise last_error if last_error else RuntimeError("Qwen chat failed")

    async def plan_with_tool(self, medgemma_summary: str, user_text: str | None) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_PLAN},
            {"role": "user", "content": f"MedGemma Analysis:\n{medgemma_summary}\n\nUser note: {user_text or ''}"},
        ]
        logger.info("Qwen planning started (tool-enabled)")
        # ⬇️ Всегда требуем вызов тулзы
        resp = self._chat(messages, tools=[TOOL_SCHEMA], tool_choice="required", max_tokens=1024)
        choice = resp.choices[0]
        msg = choice.message

        collected_products: List[Dict[str, Any]] = []
        if getattr(msg, "tool_calls", None):
            logger.info(f"Qwen requested tool_calls: {[t.function.name for t in msg.tool_calls]}")
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "search_products":
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                        query = args.get("query")
                        num = int(args.get("num", 10))
                        logger.info(f"Executing tool search_products args={{'query': '{query}', 'num': {num}}}")
                        products = await search_products(query=query, num=num)
                        logger.info(f"Tool search_products returned count={len(products)}")
                        collected_products = products
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "search_products",
                            "content": json.dumps(products, ensure_ascii=False),
                        }
                        messages.append({
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [
                                {"id": tool_call.id, "type": "function", "function": {"name": "search_products", "arguments": tool_call.function.arguments}}
                            ]
                        })
                        messages.append(tool_message)
                    except Exception as e:
                        logger.warning(f"Tool execution failed: {e}")

            messages.insert(0, {"role": "system", "content": "Return only a valid JSON object, no markdown."})
            resp2 = self._chat(messages, temperature=0.2, max_tokens=1024)
            content = resp2.choices[0].message.content or ""
            try:
                plan = json.loads(content)
                plan["tool_products"] = collected_products
                logger.info(f"Qwen plan parsed successfully need_search={plan.get('need_search')} skin_type={plan.get('skin_type')}")
                return plan
            except Exception:
                logger.warning("Qwen plan JSON parse failed; returning fallback")
                return {"skin_type": "unknown", "diagnosis": content[:200], "query": "", "need_search": False, "tool_products": collected_products}

        logger.error("Qwen did not produce any tool_calls despite 'required'")
        return {"skin_type": "unknown", "diagnosis": "No tool call produced", "query": "", "need_search": False, "tool_products": []}

    def finalize_with_products(self, planning_json: str, products_jsonl: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FINAL},
            {"role": "user", "content": f"Plan: {planning_json}\nProducts: {products_jsonl}"},
        ]
        logger.info("Qwen finalizing answer with products")
        resp = self._chat(messages, temperature=0.2, max_tokens=1024)
        content = resp.choices[0].message.content or ""
        logger.info(f"Qwen final output length={len(content)}")
        return content
