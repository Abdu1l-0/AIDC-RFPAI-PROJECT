from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict


class Settings(BaseSettings):
    # Model A: self-hosted Qwen via vLLM (serving-stack)
    model_a_name: str = Field(default="qwen-7b")
    model_a_base_url: str = Field(default="http://localhost:8000/v1")
    model_a_api_key: str = Field(default="")

    # Model B: OpenAI
    model_b_name: str = Field(default="gpt-4o-mini")
    model_b_base_url: str = Field(default="https://api.openai.com/v1")
    model_b_api_key: str = Field(default="")

    # Model C: second self-hosted or open-weight model
    model_c_name: str = Field(default="qwen-14b")
    model_c_base_url: str = Field(default="http://localhost:8001/v1")
    model_c_api_key: str = Field(default="")

    # Pricing, per 1M tokens (USD), for cost calculation
    price_per_1m_input: Dict[str, float] = Field(default_factory=lambda: {
        "qwen-7b": 0.0,
        "gpt-4o-mini": 0.15,
        "qwen-14b": 0.0,
    })
    price_per_1m_output: Dict[str, float] = Field(default_factory=lambda: {
        "qwen-7b": 0.0,
        "gpt-4o-mini": 0.60,
        "qwen-14b": 0.0,
    })

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()