import time
import requests
from config import settings
from prompt import build_messages


MODEL_CONFIG = {
    "model_a": {
        "name": settings.model_a_name,
        "base_url": settings.model_a_base_url,
        "api_key": settings.model_a_api_key,
    },
    "model_b": {
        "name": settings.model_b_name,
        "base_url": settings.model_b_base_url,
        "api_key": settings.model_b_api_key,
    },
    "model_c": {
        "name": settings.model_c_name,
        "base_url": settings.model_c_base_url,
        "api_key": settings.model_c_api_key,
    },
}

TIMEOUT_S = 120
MAX_RETRIES = 2


def call_model(model_key: str, text: str) -> dict:
    """Send the document text to one model, return its answer + metadata.

    Returns a dict with: status, raw_text, latency_s, input_tokens,
    output_tokens, error, prompt_sha256.
    """
    if model_key not in MODEL_CONFIG:
        return {
            "status": "error",
            "raw_text": None,
            "latency_s": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": f"Unknown model_key '{model_key}'",
            "prompt_sha256": None,
        }

    cfg = MODEL_CONFIG[model_key]
    prompt_res = build_messages(text)
    messages = prompt_res["messages"]

    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    payload = {
        "model": cfg["name"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT_S)
            latency_s = round(time.time() - t0, 3)

            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                raw_text = choice["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "status": "ok",
                    "raw_text": raw_text,
                    "latency_s": latency_s,
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "error": None,
                    "prompt_sha256": prompt_res["prompt_sha256"],
                }
            elif resp.status_code == 429:
                last_error = f"Rate limited (429): {resp.text[:200]}"
            elif resp.status_code >= 500:
                last_error = f"Server error ({resp.status_code}): {resp.text[:200]}"
            else:
                # client error (bad request, auth, etc.) - don't retry
                return {
                    "status": "error",
                    "raw_text": None,
                    "latency_s": latency_s,
                    "input_tokens": None,
                    "output_tokens": None,
                    "error": f"Client error ({resp.status_code}): {resp.text[:200]}",
                    "prompt_sha256": prompt_res["prompt_sha256"],
                }
        except requests.exceptions.Timeout:
            latency_s = round(time.time() - t0, 3)
            last_error = "Request timed out"
            if attempt == MAX_RETRIES:
                return {
                    "status": "timeout",
                    "raw_text": None,
                    "latency_s": latency_s,
                    "input_tokens": None,
                    "output_tokens": None,
                    "error": last_error,
                    "prompt_sha256": prompt_res["prompt_sha256"],
                }
        except Exception as e:
            latency_s = round(time.time() - t0, 3)
            last_error = str(e)

    return {
        "status": "error",
        "raw_text": None,
        "latency_s": None,
        "input_tokens": None,
        "output_tokens": None,
        "error": last_error,
        "prompt_sha256": prompt_res["prompt_sha256"],
    }