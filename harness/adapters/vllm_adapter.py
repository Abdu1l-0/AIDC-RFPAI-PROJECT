import time
import requests
from typing import Dict, Any
from harness.adapters.base import BaseModelAdapter

class VLLMAdapter(BaseModelAdapter):
    def invoke(self, messages: list, decoding_params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.endpoint_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": decoding_params.get("temperature", 0.0),
            "max_tokens": decoding_params.get("max_tokens", 4096),
            "top_p": decoding_params.get("top_p", 1.0)
        }
        t0 = time.time()
        status = "ok"
        raw_text = ""
        finish_reason = None
        model_returned = self.model_name
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "source": "provider_reported"}
        try:
            resp = requests.post(url, json=payload, timeout=decoding_params.get("timeout", 180))
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
            elif resp.status_code >= 500:
                status = "server_error"
            else:
                status = "client_error"
        except requests.exceptions.Timeout:
            status = "timeout"
            ms_total = (time.time() - t0) * 1000.0
        except Exception:
            status = "client_error"
            ms_total = (time.time() - t0) * 1000.0
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
