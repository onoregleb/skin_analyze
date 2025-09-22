import os
from dataclasses import dataclass


@dataclass
class Settings:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8001"))

    google_cse_api_key: str | None = os.getenv("GOOGLE_CSE_API_KEY")
    google_cse_cx: str | None = os.getenv("GOOGLE_CSE_CX")


settings = Settings()
