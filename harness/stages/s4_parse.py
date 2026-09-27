import json
import re
from typing import Dict, Any, Tuple

def parse_and_validate(raw_text: str) -> Tuple[str, Dict[str, Any], str]:
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, dict):
            return "none", data, ""
    except Exception as e0:
        l0_err = str(e0)
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        candidate = text[s:e+1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return "L1", data, ""
        except Exception as e1:
            return "failed", {}, f"L0: {l0_err} | L1: {e1}"
    return "failed", {}, f"L0: {l0_err} | L1: No balanced JSON object found"
