from __future__ import annotations
from typing import List, Dict, Any
import json
from openai import OpenAI
from app.config import settings
from app.utils.logging import get_logger
from app.tools.search_products import search_products


logger = get_logger("qwen")

SYSTEM_PROMPT_PLAN = (
	"You are a dermatology assistant. Given a visual analysis summary and optional user note, "
	"infer probable skin type and issues. If needed, call the search_products tool to fetch candidates. "
	"Always produce a compact JSON at the end with keys: skin_type, diagnosis, query, need_search."
)

SYSTEM_PROMPT_FINAL = (
	"You are selecting skin-care products based on user case. Given the prior plan and a list of search results, "
	"pick up to 5 items. Return a JSON with keys: diagnosis, skin_type, explanation, products (list of {name,url,price?,snippet?,image_url?})."
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

	def _chat(self, messages: List[Dict[str, Any]], temperature: float = 0.3, tools: list | None = None) -> Any:
		attempts = 0
		last_error: Exception | None = None
		while attempts < 3:
			attempts += 1
			try:
				response = self.client.chat.completions.create(
					model=self.model,
					messages=messages,
					temperature=temperature,
					max_tokens=768,
					timeout=45.0,
					tools=tools,
				)
				return response
			except Exception as e:
				last_error = e
				logger.warning(f"Qwen chat attempt {attempts} failed: {e}")
		logger.error("Qwen chat failed after retries")
		raise last_error if last_error else RuntimeError("Qwen chat failed")

	async def plan_with_tool(self, medgemma_summary: str, user_text: str | None) -> Dict[str, Any]:
		messages: List[Dict[str, Any]] = [
			{"role": "system", "content": SYSTEM_PROMPT_PLAN},
			{"role": "user", "content": f"Visual analysis: {medgemma_summary}\nUser note: {user_text or ''}"},
		]
		logger.info("Qwen planning started (tool-enabled)")
		resp = self._chat(messages, tools=[TOOL_SCHEMA])
		choice = resp.choices[0]
		msg = choice.message

		# Tool call branch
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

			resp2 = self._chat(messages)
			content = resp2.choices[0].message.content or ""
			try:
				plan = json.loads(content)
				logger.info(f"Qwen plan parsed successfully need_search={plan.get('need_search')} skin_type={plan.get('skin_type')}")
				return plan
			except Exception:
				logger.warning("Qwen plan JSON parse failed; returning fallback")
				return {"skin_type": "unknown", "diagnosis": content[:200], "query": "", "need_search": False}

		# No tool call
		logger.info("Qwen did not request any tool; parsing direct content as plan")
		content = msg.content or ""
		try:
			plan = json.loads(content)
			logger.info(f"Qwen direct plan parsed need_search={plan.get('need_search')} skin_type={plan.get('skin_type')}")
			return plan
		except Exception:
			logger.warning("Qwen direct plan parse failed; returning fallback")
			return {"skin_type": "unknown", "diagnosis": content[:200], "query": "", "need_search": False}

	def finalize_with_products(self, planning_json: str, products_jsonl: str) -> str:
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT_FINAL},
			{"role": "user", "content": f"Plan: {planning_json}\nProducts: {products_jsonl}"},
		]
		logger.info("Qwen finalizing answer with products")
		resp = self._chat(messages, temperature=0.2)
		content = resp.choices[0].message.content or ""
		logger.info(f"Qwen final output length={len(content)}")
		return content
