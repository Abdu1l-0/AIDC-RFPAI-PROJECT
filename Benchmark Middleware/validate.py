import json
import re
from schemas import RFPFields, FieldValue
from pydantic import ValidationError


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrapping, if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_outermost_json(text: str) -> str:
    """Pull out the outermost {...} block, ignoring any surrounding prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start:end + 1]


def parse_model_output(raw_text: str) -> dict:
    """Attempt to parse a model's raw text output into RFPFields.

    Returns: {"valid": bool, "fields": RFPFields | None, "error": str | None}
    """
    if not raw_text:
        return {"valid": False, "fields": None, "error": "Empty response"}

    cleaned = _strip_markdown_fences(raw_text)
    cleaned = _extract_outermost_json(cleaned)

    try:
        parsed_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"valid": False, "fields": None, "error": f"JSON parse error: {e}"}

    try:
        fields = RFPFields(**parsed_dict)
    except ValidationError as e:
        return {"valid": False, "fields": None, "error": f"Schema validation error: {e}"}

    return {"valid": True, "fields": fields, "error": None}