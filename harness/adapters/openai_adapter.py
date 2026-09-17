import time
import requests
import json
import os
from typing import Dict, Any
from harness.adapters.base import BaseModelAdapter
class OpenAIAdapter(BaseModelAdapter):
    """Adapter for commercial OpenAI API endpoints."""
    def __init__(self, model_key: str = "openai-gpt-4o-mini", api_key: str = None, model_name: str = "gpt-4o-mini", base_url: str = None, **kwargs):
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        super().__init__(model_key, base_url, model_name)
        self.api_key = api_key
        self.endpoint_url = base_url.rstrip("/")
    def invoke(self, messages: list, decoding_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "client_error",
                "response": {
                    "raw_text": "Error: OPENAI_API_KEY environment variable is not set.",
                    "finish_reason": "error",
                    "model_id_returned": self.model_name,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "source": "provider_reported"}
                },
                "timings": {"ms_total": 0.0, "ms_to_first_token": None}
            }
        url = f"{self.endpoint_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": decoding_params.get("temperature", 0.0),
            "max_tokens": decoding_params.get("max_tokens", 4096)
        }
        t0 = time.time()
        status = "ok"
        raw_text = ""
        finish_reason = None
        model_returned = self.model_name
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "source": "provider_reported"}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=decoding_params.get("timeout", 180))
            ms_total = (time.time() - t0) * 1000.0
            
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                raw_text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "stop")
                model_returned = data.get("model", self.model_name)
                if "usage" in data:
                    usage["prompt_tokens"] = data["usage"].get("prompt_tokens", 0)
                    usage["completion_tokens"] = data["usage"].get("completion_tokens", 0)
            elif resp.status_code == 429:
                status = "rate_limited"
                raw_text = f"Rate limited (429): {resp.text}"
            elif resp.status_code >= 500:
                status = "server_error"
                raw_text = f"Server error ({resp.status_code}): {resp.text}"
            else:
                status = "client_error"
                raw_text = f"Client error ({resp.status_code}): {resp.text}"
        except requests.exceptions.Timeout:
            ms_total = (time.time() - t0) * 1000.0
            status = "timeout"
            raw_text = "Request timed out"
        except Exception as e:
            ms_total = (time.time() - t0) * 1000.0
            status = "client_error"
            raw_text = str(e)
        return {
            "status": status,
            "response": {
                "raw_text": raw_text,
                "finish_reason": finish_reason,
                "model_id_returned": model_returned,
                "usage": usage
            },
            "timings": {
                "ms_total": round(ms_total, 2),
                "ms_to_first_token": None
            }
        }