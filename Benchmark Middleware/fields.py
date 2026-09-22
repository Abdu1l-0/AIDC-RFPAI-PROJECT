"""The 17 RFP fields, keyed by readable names.

field_set.json in this folder is our own copy of the team's field definitions
(taken from config/field_set@1.0.json, which keys the fields F01..F17). We keep
a copy so this folder does not depend on anything outside it. Here we re-key
the same 17 fields to readable names, and the rest of the middleware (prompt,
schemas, scoring) uses the names only.

Nothing else changes: value shapes, field_class, absent_legal and the
descriptions all come straight from the field set.

If the upstream field set changes, re-copy field_set.json and check NAME_BY_ID
still covers every id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIELD_SET_PATH = Path(__file__).resolve().parent / "field_set.json"

# The one place that knows the old ids. Order matters: it is the field order
# used in the prompt and in the results.
NAME_BY_ID = {
    "F01": "submission_deadline",
    "F02": "questions_deadline",
    "F03": "rfp_contact",
    "F04": "submission_method",
    "F05": "contract_term",
    "F06": "scope_of_deliverables",
    "F07": "mandatory_submission_requirements",
    "F08": "mandatory_technical_requirements",
    "F09": "evaluation_criteria",
    "F10": "minimum_score_threshold",
    "F11": "pricing_structure",
    "F12": "insurance_requirements",
    "F13": "vendor_experience",
    "F14": "references_required",
    "F15": "data_security_requirements",
    "F16": "data_residency",
    "F17": "demo_required",
}

ID_BY_NAME = {name: fid for fid, name in NAME_BY_ID.items()}


def _load() -> dict:
    """Read the field set and re-key it by readable name."""
    with open(FIELD_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)["fields"]

    out = {}
    for fid, fdef in raw.items():
        name = NAME_BY_ID.get(fid)
        if name is None:
            raise KeyError(f"{FIELD_SET_PATH.name} has field '{fid}' with no name in NAME_BY_ID")
        out[name] = {**fdef, "id": fid}
    return out


# name -> field definition (the field set entry, plus its original "id")
FIELDS = _load()

# the 17 names, in field-set order
FIELD_NAMES = list(FIELDS.keys())


def field_class(name: str) -> str:
    """The scoring class for one field (atomic_exact, list_set, free_text, ...)."""
    return FIELDS[name].get("field_class", "atomic_exact")


# ---------------------------------------------------------------------------
# Strict JSON Schema, sent to the models so they cannot invent their own shape
# ---------------------------------------------------------------------------

# Keys the field set uses for documentation but that strict JSON Schema
# does not accept.
_DROP_KEYS = {"format", "cardinality", "absent_legal", "field_class", "name", "id"}


def _nullable(schema: dict) -> dict:
    """Allow null as well as the declared type(s)."""
    out = dict(schema)
    t = out.get("type")
    if isinstance(t, str):
        out["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        out["type"] = [*t, "null"]
    enum = out.get("enum")
    if isinstance(enum, list) and None not in enum:
        out["enum"] = [*enum, None]
    return out


def _strict(schema: dict) -> dict:
    """Rewrite one field-set schema into strict JSON Schema.

    Strict mode requires every object to list all of its properties in
    "required" and to set "additionalProperties": false. Properties the field
    set marked optional stay optional in meaning by allowing null instead.
    """
    out = {k: v for k, v in schema.items() if k not in _DROP_KEYS}

    if "properties" in out:
        original_required = set(out.get("required", []))
        props = {}
        for prop_name, prop_schema in out["properties"].items():
            child = _strict(prop_schema)
            if prop_name not in original_required:
                child = _nullable(child)
            props[prop_name] = child
        out["properties"] = props
        out["required"] = list(out["properties"].keys())
        out["additionalProperties"] = False

    if "items" in out:
        out["items"] = _strict(out["items"])

    return out


_EVIDENCE_SCHEMA = {
    "type": "array",
    "description": "Exact quotes from the document supporting the value. Empty when not_stated.",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["quote", "page"],
        "properties": {
            "quote": {"type": "string", "description": "Verbatim text copied from the document."},
            "page": {"type": ["integer", "null"], "description": "Page number the quote is on."},
        },
    },
}


def build_json_schema() -> dict:
    """The full response schema: 17 named fields, each {status, value, evidence}."""
    properties = {}
    for name, fdef in FIELDS.items():
        value_schema = _nullable(_strict(fdef))
        value_schema["description"] = fdef["description"]
        properties[name] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "value", "evidence"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["found", "not_stated"],
                    "description": "found when the document states it, not_stated otherwise.",
                },
                "value": value_schema,
                "evidence": _EVIDENCE_SCHEMA,
            },
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties.keys()),
        "properties": properties,
    }


JSON_SCHEMA = build_json_schema()


# ---------------------------------------------------------------------------
# Value shapes in plain text, for the prompt
# ---------------------------------------------------------------------------
# Constrained decoding is not available on every endpoint (our vLLM's grammar
# backend hangs on this schema), so the prompt itself must spell out the exact
# shape of every value. Every model gets the same prompt, which also keeps the
# comparison fair.

def _shape(schema: dict) -> Any:
    """A compact example of one value's shape, e.g. {"date": "YYYY-MM-DD"}."""
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    real = [t for t in types if t != "null"]
    nullable = "null" in types

    if "enum" in schema:
        options = [e for e in schema["enum"] if e is not None]
        leaf = " | ".join(str(o) for o in options)
    elif "object" in real and "properties" in schema:
        return {k: _shape(v) for k, v in schema["properties"].items()}
    elif "array" in real:
        return [_shape(schema.get("items", {}))]
    elif "integer" in real:
        leaf = "integer"
    elif "number" in real:
        leaf = "number"
    elif "boolean" in real:
        leaf = "true | false"
    else:
        leaf = schema.get("description") or "string"

    return f"{leaf} | null" if nullable else leaf


def value_shape(name: str) -> str:
    """The shape of one field's value, as one line of JSON."""
    value_schema = JSON_SCHEMA["properties"][name]["properties"]["value"]
    return json.dumps(_shape(value_schema), ensure_ascii=False)
