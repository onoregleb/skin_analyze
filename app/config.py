import os
from dataclasses import dataclass


@dataclass
class Settings:
	api_host: str = os.getenv("API_HOST", "0.0.0.0")
	api_port: int = int(os.getenv("API_PORT", "8000"))

	hf_token: str | None = os.getenv("HF_TOKEN")

	vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
	vllm_api_key: str = os.getenv("VLLM_API_KEY", "dev")
	qwen_model: str = os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

	google_cse_api_key: str | None = os.getenv("GOOGLE_CSE_API_KEY")
	google_cse_cx: str | None = os.getenv("GOOGLE_CSE_CX")

	torch_dtype: str = os.getenv("TORCH_DTYPE", "bf16")
	device_map: str = os.getenv("DEVICE_MAP", "auto")


settings = Settings()
