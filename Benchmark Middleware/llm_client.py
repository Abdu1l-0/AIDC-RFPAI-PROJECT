import time
import requests
from config import settings
from fields import JSON_SCHEMA
from prompt import build_messages


MODEL_CONFIG = {
    "model_a": {
        "name": settings.model_a_name,
        "base_url": settings.model_a_base_url,
        "api_key": settings.model_a_api_key,
        "label": f"Model A ({settings.model_a_name}, vLLM)",
        "enabled": settings.model_a_enabled,
    },
    "model_b": {
        "name": settings.model_b_name,
        "base_url": settings.model_b_base_url,
        "api_key": settings.model_b_api_key,
        "label": f"Model B ({settings.model_b_name}, OpenAI)",
        "enabled": settings.model_b_enabled,
    },
    "model_c": {
        "name": settings.model_c_name,
        "base_url": settings.model_c_base_url,
        "api_key": settings.model_c_api_key,
        "label": f"Model C ({settings.model_c_name}, vLLM)",
        "enabled": settings.model_c_enabled,
    },
}


def enabled_models() -> list[str]:
    """The model keys that are switched on in .env."""
    return [key for key, cfg in MODEL_CONFIG.items() if cfg["enabled"]]


def ping_model(model_key: str, timeout_s: float = 6.0) -> str:
    """Cheap check that a model endpoint answers. Returns 'ok', 'off', or an
    error string. Does not spend tokens: it only lists the endpoint's models.
    """
    cfg = MODEL_CONFIG.get(model_key)
    if cfg is None:
        return "unknown model"
    if not cfg["enabled"]:
        return "off"

    url = f"{cfg['base_url'].rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg["api_key"] else {}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
    except requests.exceptions.RequestException as exc:
        return f"unreachable: {type(exc).__name__}"
    if resp.status_code == 200:
        return "ok"
    return f"error {resp.status_code}"

TIMEOUT_S = 90
MAX_RETRIES = 1
MAX_TOKENS = 8000

# Ask the model to answer in exactly our schema. OpenAI and recent vLLM builds
# both accept this; schema_mode in the result records what was actually sent.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "rfp_fields", "strict": True, "schema": JSON_SCHEMA},
}

# 429s that will never succeed on retry (no budget / no quota).
_PERMANENT_429 = ("spend limit", "insufficient_quota", "exceeded your current quota")


def _result(status, *, raw_text=None, latency_s=None, input_tokens=None,
            output_tokens=None, error=None, prompt_sha256=None,
            schema_mode=None, finish_reason=None) -> dict:
    return {
        "status": status,
        "raw_text": raw_text,
        "latency_s": latency_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
        "prompt_sha256": prompt_sha256,
        "schema_mode": schema_mode,
        "finish_reason": finish_reason,
    }


def call_model(model_key: str, text: str) -> dict:
    """Send the document text to one model, return its answer + metadata.

    Returns a dict with: status, raw_text, latency_s, input_tokens,
    output_tokens, error, prompt_sha256, schema_mode, finish_reason.

    status is one of: ok, timeout, rate_limited, error.
    """
    if model_key not in MODEL_CONFIG:
        return _result("error", error=f"Unknown model_key '{model_key}'")

    cfg = MODEL_CONFIG[model_key]
    prompt_res = build_messages(text)
    messages = prompt_res["messages"]

    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    sha = prompt_res["prompt_sha256"]
    payload = {
        "model": cfg["name"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "response_format": RESPONSE_FORMAT,
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
                finish_reason = choice.get("finish_reason")
                usage = data.get("usage", {})
                # "length" means the answer was cut off before it finished.
                status = "error" if finish_reason == "length" else "ok"
                return _result(
                    status,
                    raw_text=raw_text,
                    latency_s=latency_s,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    error=("Answer cut off: hit the max_tokens limit"
                           if status == "error" else None),
                    prompt_sha256=sha,
                    schema_mode="json_schema",
                    finish_reason=finish_reason,
                )

            body = resp.text[:300]

            if resp.status_code == 429:
                # No budget / no quota: retrying will never help.
                if any(s in body.lower() for s in _PERMANENT_429):
                    return _result("rate_limited", latency_s=latency_s,
                                   error=f"Rate limited (429): {body}",
                                   prompt_sha256=sha, schema_mode="json_schema")
                last_error = f"Rate limited (429): {body}"
                if attempt == MAX_RETRIES:
                    return _result("rate_limited", latency_s=latency_s,
                                   error=last_error, prompt_sha256=sha,
                                   schema_mode="json_schema")
                time.sleep(2 ** attempt)  # back off before trying again

            elif resp.status_code >= 500:
                last_error = f"Server error ({resp.status_code}): {body}"
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)

            else:
                # client error (bad request, auth, bad schema) - don't retry
                return _result("error", latency_s=latency_s,
                               error=f"Client error ({resp.status_code}): {body}",
                               prompt_sha256=sha, schema_mode="json_schema")

        except requests.exceptions.Timeout:
            latency_s = round(time.time() - t0, 3)
            last_error = f"Request timed out after {TIMEOUT_S}s"
            if attempt == MAX_RETRIES:
                return _result("timeout", latency_s=latency_s, error=last_error,
                               prompt_sha256=sha, schema_mode="json_schema")
        except Exception as e:
            last_error = str(e)

    return _result("error", error=last_error, prompt_sha256=sha,
                   schema_mode="json_schema")