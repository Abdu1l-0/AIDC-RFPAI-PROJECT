import json
import time
from urllib.parse import urlparse

import requests
from config import settings
from fields import JSON_SCHEMA
from logger import get_logger, preview
from prompt import build_messages

log = get_logger()


MODEL_CONFIG = {
    "model_a": {
        "name": settings.model_a_name,
        "base_url": settings.model_a_base_url,
        "api_key": settings.model_a_api_key,
        "label": f"Model A ({settings.model_a_name}, vLLM)",
        "enabled": settings.model_a_enabled,
        "schema_mode": settings.model_a_schema_mode,
    },
    "model_b": {
        "name": settings.model_b_name,
        "base_url": settings.model_b_base_url,
        "api_key": settings.model_b_api_key,
        "label": f"Model B ({settings.model_b_name}, OpenAI)",
        "enabled": settings.model_b_enabled,
        "schema_mode": settings.model_b_schema_mode,
    },
    "model_c": {
        "name": settings.model_c_name,
        "base_url": settings.model_c_base_url,
        "api_key": settings.model_c_api_key,
        "label": f"Model C ({settings.model_c_name}, vLLM)",
        "enabled": settings.model_c_enabled,
        "schema_mode": settings.model_c_schema_mode,
    },
}


def enabled_models() -> list[str]:
    """The model keys that are switched on in .env."""
    return [key for key, cfg in MODEL_CONFIG.items() if cfg["enabled"]]


def ping_model(model_key: str, timeout_s: float = 6.0) -> str:
    """Cheap check that the endpoint answers AND is serving the model we
    ask for. Returns 'ok', 'off', 'not loaded', or an error string.
    Spends no tokens: it only lists the endpoint's models.
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
    if resp.status_code != 200:
        return f"error {resp.status_code}"

    try:
        served = {m.get("id") for m in resp.json().get("data", [])}
    except ValueError:
        return "ok"
    if not served:
        return "ok"
    return "ok" if cfg["name"] in served else "not loaded"
# Self-hosted models need well over a minute for a full RFP. A timeout is not
# retried: the same slow request would just double the wait.
TIMEOUT_S = 180
MAX_RETRIES = 1

# How many output tokens to ask for. This is NOT a fixed number: a self-hosted
# model has a fixed context window shared between the prompt and the answer, so
# a long document leaves less room for the answer. Arabic makes this sharp -
# the same RFP costs ~1.6x more tokens in Arabic than in English, so a value
# that fits every English document overflows on the long Arabic ones.
# _token_budget() therefore asks the endpoint how long the prompt actually is
# and gives the answer whatever is left.
TOKEN_CEILING = 4000     # never ask for more than this, however much room there is
TOKEN_FLOOR = 1200       # below this an answer cannot be complete; let it fail loudly
CONTEXT_MARGIN = 64       # small slack: the reported prompt count already covers the template

# base_url -> context length, so we ask each endpoint only once.
_context_cache: dict[str, int | None] = {}


def _headers(cfg: dict) -> dict:
    return {"Authorization": f"Bearer {cfg['api_key']}"} if cfg["api_key"] else {}


def _probe_prompt(cfg: dict, messages: list) -> tuple[int | None, int | None]:
    """Ask a vLLM endpoint for the exact prompt length via POST /tokenize.

    vLLM serves /tokenize at the server root, next to /v1, and its reply
    carries both "count" and "max_model_len" - exactly the two numbers the
    budget needs, with no estimating. OpenAI has no such endpoint; it returns
    (None, None) and the caller falls back.
    """
    root = cfg["base_url"].rstrip("/")
    candidates = [root[:-3].rstrip("/") + "/tokenize"] if root.endswith("/v1") else []
    candidates.append(root + "/tokenize")
    for url in candidates:
        try:
            r = requests.post(url, json={"model": cfg["name"], "messages": messages},
                              headers=_headers(cfg), timeout=15)
            if r.status_code == 200:
                d = r.json()
                return d.get("count"), d.get("max_model_len")
        except requests.exceptions.RequestException:
            continue
        except ValueError:
            continue
    return None, None


def _estimate_tokens(messages: list) -> int:
    """Conservative fallback when the endpoint will not tokenize for us.

    Deliberately pessimistic: Arabic script runs about 2.44 characters per
    token on our corpus (measured), Latin about 3.4. Over-estimating the
    prompt only costs a slightly shorter answer; under-estimating costs a
    failed request.
    """
    text = "".join(m["content"] for m in messages)
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06ff")
    ratio = 2.44 if arabic > len(text) * 0.15 else 3.4
    return int(len(text) / ratio) + 1


def _token_budget(cfg: dict, messages: list) -> int:
    """Output tokens to request: whatever the context window has left."""
    count, limit = _probe_prompt(cfg, messages)

    if limit is None:
        limit = _context_cache.get(cfg["base_url"], "miss")
        if limit == "miss":
            limit = None
            try:
                r = requests.get(f"{cfg['base_url'].rstrip('/')}/models",
                                 headers=_headers(cfg), timeout=6)
                if r.status_code == 200:
                    for m in r.json().get("data", []):
                        if m.get("id") == cfg["name"]:
                            limit = m.get("max_model_len")
            except (requests.exceptions.RequestException, ValueError):
                pass
            _context_cache[cfg["base_url"]] = limit

    if not limit:
        # No context length available (OpenAI, whose window dwarfs our prompts).
        return TOKEN_CEILING
    if count is None:
        count = _estimate_tokens(messages)
    return max(TOKEN_FLOOR, min(TOKEN_CEILING, int(limit) - int(count) - CONTEXT_MARGIN))

def apply_schema_mode(payload: dict, schema_mode: str) -> dict:
    """Add the right "answer in this schema" instruction for this endpoint.

    OpenAI enforces response_format json_schema. vLLM accepts that field but
    silently ignores it, so vLLM endpoints must use guided_json instead, or
    their answers are not constrained at all.
    """
    if schema_mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "rfp_fields", "strict": True, "schema": JSON_SCHEMA},
        }
    elif schema_mode == "guided_json":
        payload["guided_json"] = JSON_SCHEMA
    elif schema_mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    # "none": send nothing; the prompt alone has to carry the shape
    return payload

# Modes where the endpoint itself enforces our JSON schema. For every other
# mode, the answer shape has to be written into the prompt instead.
ENFORCING_MODES = {"json_schema", "guided_json"}

# 429s that will never succeed on retry (no budget / no quota).
_PERMANENT_429 = ("spend limit", "insufficient_quota", "exceeded your current quota")


def _result(status, *, raw_text=None, latency_s=None, input_tokens=None,
            output_tokens=None, error=None, prompt_sha256=None,
            schema_mode=None, finish_reason=None, prompt_variant=None) -> dict:
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
        "prompt_variant": prompt_variant,
    }


def call_model(model_key: str, text: str, doc_id: str = "-") -> dict:
    """Send the document text to one model, return its answer + metadata.

    Returns a dict with: status, raw_text, latency_s, input_tokens,
    output_tokens, error, prompt_sha256, schema_mode, finish_reason.

    status is one of: ok, timeout, rate_limited, error.
    doc_id is only used to label the log lines.
    """
    if model_key not in MODEL_CONFIG:
        log.error("model=%s doc=%s unknown model key", model_key, doc_id)
        return _result("error", error=f"Unknown model_key '{model_key}'")

    cfg = MODEL_CONFIG[model_key]
    schema_mode = cfg.get("schema_mode", "json_schema")
    # Shapes go into the prompt only when the endpoint cannot enforce them.
    prompt_res = build_messages(text, shapes_in_prompt=schema_mode not in ENFORCING_MODES)
    messages = prompt_res["messages"]
    variant = prompt_res["prompt_variant"]

    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    sha = prompt_res["prompt_sha256"]
    max_tokens = _token_budget(cfg, messages)
    payload = apply_schema_mode({
        "model": cfg["name"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }, schema_mode)

    # Sizes only - never the text itself, and never the headers (API key).
    prompt_chars = sum(len(m["content"]) for m in messages)
    log.info("-> model=%s doc=%s host=%s served_name=%s schema_mode=%s prompt=%s "
             "prompt_chars=%d body_kb=%.1f max_tokens=%d",
             model_key, doc_id, urlparse(url).netloc, cfg["name"], schema_mode, variant,
             prompt_chars, len(json.dumps(payload)) / 1024, max_tokens)

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
                log.info("<- model=%s doc=%s HTTP 200 status=%s latency=%.1fs "
                         "tokens=%s/%s finish=%s answer_chars=%d",
                         model_key, doc_id, status, latency_s,
                         usage.get("prompt_tokens"), usage.get("completion_tokens"),
                         finish_reason, len(raw_text or ""))
                log.info("   model=%s doc=%s answer: %s", model_key, doc_id, preview(raw_text))
                return _result(
                    status,
                    raw_text=raw_text,
                    latency_s=latency_s,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    error=("Answer cut off: hit the max_tokens limit"
                           if status == "error" else None),
                    prompt_sha256=sha,
                    schema_mode=schema_mode, prompt_variant=variant,
                    finish_reason=finish_reason,
                )

            body = resp.text[:300]
            log.warning("<- model=%s doc=%s HTTP %d after %.1fs (attempt %d/%d): %s",
                        model_key, doc_id, resp.status_code, latency_s,
                        attempt + 1, MAX_RETRIES + 1, preview(body, 200))

            if resp.status_code == 429:
                # No budget / no quota: retrying will never help.
                if any(s in body.lower() for s in _PERMANENT_429):
                    return _result("rate_limited", latency_s=latency_s,
                                   error=f"Rate limited (429): {body}",
                                   prompt_sha256=sha, schema_mode=schema_mode, prompt_variant=variant)
                last_error = f"Rate limited (429): {body}"
                if attempt == MAX_RETRIES:
                    return _result("rate_limited", latency_s=latency_s,
                                   error=last_error, prompt_sha256=sha,
                                   schema_mode=schema_mode, prompt_variant=variant)
                time.sleep(2 ** attempt)  # back off before trying again

            elif resp.status_code >= 500:
                last_error = f"Server error ({resp.status_code}): {body}"
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)

            else:
                # client error (bad request, auth, bad schema) - don't retry
                return _result("error", latency_s=latency_s,
                               error=f"Client error ({resp.status_code}): {body}",
                               prompt_sha256=sha, schema_mode=schema_mode, prompt_variant=variant)

        except requests.exceptions.Timeout:
            latency_s = round(time.time() - t0, 3)
            log.warning("<- model=%s doc=%s TIMEOUT after %.1fs (no answer from the endpoint)",
                        model_key, doc_id, latency_s)
            return _result("timeout", latency_s=latency_s,
                           error=f"Request timed out after {TIMEOUT_S}s",
                           prompt_sha256=sha, schema_mode=schema_mode, prompt_variant=variant)
        except Exception as e:
            last_error = str(e)
            log.warning("<- model=%s doc=%s connection failed after %.1fs: %s: %s",
                        model_key, doc_id, time.time() - t0, type(e).__name__, e)

    return _result("error", error=last_error, prompt_sha256=sha,
                   schema_mode=schema_mode, prompt_variant=variant)