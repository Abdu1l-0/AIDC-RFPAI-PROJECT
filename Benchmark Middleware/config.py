import os
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


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

    # Which models are ready to call. Model A and C are the self-hosted vLLM
    # endpoints; turn them on in .env once they are deployed.
    model_a_enabled: bool = Field(default=False)
    model_b_enabled: bool = Field(default=True)
    model_c_enabled: bool = Field(default=False)

    # How each endpoint is told to answer in our JSON schema:
    #   json_schema  OpenAI style response_format (OpenAI enforces this)
    #   guided_json  vLLM's own guided decoding (vLLM enforces this)
    #   json_object  only "must be valid JSON", no shape
    #   none         nothing sent; the prompt has to carry it
    # Tested on our vLLM (2026-09-21):
    #   json_schema  accepted but silently IGNORED
    #   guided_json  works on tiny schemas, but its only grammar backend
    #                (outlines) hangs on our 17-field schema with a real RFP
    # So vLLM models use "none": the prompt spells out every value shape
    # (prompt.py), and validate.py checks the answer. The mode used is saved
    # with every result, so the report can state it.
    model_a_schema_mode: str = Field(default="none")
    model_b_schema_mode: str = Field(default="json_schema")
    model_c_schema_mode: str = Field(default="none")

    # Print every model endpoint's full HTTP response body to the log.
    # The full response is always saved to data/raw/ either way; this only
    # controls the log. Turn off when running big benchmarks (it is long).
    log_full_response: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # .env also holds OPENAI_API_KEY and other keys this model does not
        # declare; ignore them instead of failing to start.
        extra = "ignore"


settings = Settings()

# Convenience: if MODEL_B_API_KEY is not set, fall back to the standard
# OPENAI_API_KEY, so the .env only needs the one key.
if not settings.model_b_api_key:
    settings.model_b_api_key = os.getenv("OPENAI_API_KEY", "")