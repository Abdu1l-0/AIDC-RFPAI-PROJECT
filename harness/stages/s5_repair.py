from typing import Dict, Any
def build_repair_prompt(original_messages: list, parser_error: str) -> list:
    repair_msg = (
        f"Your previous output failed strict JSON validation: {parser_error}.\n"
        "Output strictly valid JSON conforming to: {status, value, evidence}."
    )
    return original_messages + [{"role": "user", "content": repair_msg}]
def bounded_reask(adapter, original_messages: list, raw_text: str, parser_error: str, decoding_params: Dict[str, Any]) -> Dict[str, Any]:
    repair_messages = build_repair_prompt(original_messages, parser_error)
    params = dict(decoding_params)
    params["temperature"] = 0.0
    return adapter.invoke(repair_messages, params)
