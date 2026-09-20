"""How two field values are compared, one rule per field_class.

Every comparison returns a similarity from 0.0 to 1.0. scoring.py turns that
into an outcome (correct / partial / extraction_error).

The ladder, cheapest and most predictable first:
  1. normalize  - dates to YYYY-MM-DD, times to HH:MM, numbers to numbers
  2. alias      - SOC2 -> SOC 2 Type II, CGL -> Commercial General Liability
  3. compare    - by shape:
       scalar          exact after normalizing
       object          property by property, over the properties gold fills
       array           best-match pairing between the two lists, then F1
       free text       word overlap (F1)

normalizers.json in this folder is our own copy of config/normalizers.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ALIAS_PATH = Path(__file__).resolve().parent / "normalizers.json"
with open(_ALIAS_PATH, "r", encoding="utf-8") as _f:
    _ALIASES = json.load(_f)

# Which alias table applies to which (field, property).
_ALIAS_FOR = {
    ("data_security_requirements", "standard"): "security_standards_aliases",
    ("insurance_requirements", "coverage_type"): "insurance_type_aliases",
    ("submission_method", "portal_name"): "portal_aliases",
}

# Words that carry no meaning when comparing free text.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "at", "as", "is", "are", "be", "will", "must", "shall", "should", "any",
    "all", "this", "that", "from", "their", "its", "it", "including", "include",
}

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

# How similar two values must be to count as the same answer.
CORRECT_AT = 0.85
PARTIAL_AT = 0.40


# ---------------------------------------------------------------------------
# Normalizing
# ---------------------------------------------------------------------------

def norm_text(value: Any) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    text = str(value).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_alias(field: str, prop: str, value: Any) -> str:
    """Map a known synonym to its canonical name (SOC2 -> SOC 2 Type II)."""
    table_name = _ALIAS_FOR.get((field, prop))
    if not table_name:
        return str(value)
    table = _ALIASES.get(table_name, {})
    return table.get(str(value).strip().lower(), str(value))


def norm_date(value: Any) -> str:
    """Anything date-like to YYYY-MM-DD; unchanged text if it isn't a date."""
    text = str(value).strip()
    iso = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if iso:
        return f"{int(iso.group(1)):04d}-{int(iso.group(2)):02d}-{int(iso.group(3)):02d}"
    named = re.search(r"([a-zA-Z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
    if named and named.group(1).lower() in _MONTHS:
        return f"{int(named.group(3)):04d}-{_MONTHS[named.group(1).lower()]:02d}-{int(named.group(2)):02d}"
    return norm_text(text)


def norm_time(value: Any) -> str:
    """Times to 24-hour HH:MM. Seconds are dropped, so 23:59:59 == 23:59."""
    text = str(value).strip().lower()
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return norm_text(text)
    hour, minute = int(match.group(1)), int(match.group(2))
    if "pm" in text and hour < 12:
        hour += 12
    if "am" in text and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def as_number(value: Any) -> float | None:
    """A number if the value is one (or contains one), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d[\d,]*\.?\d*", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def token_f1(a: Any, b: Any) -> float:
    """Word-overlap F1 between two pieces of text, 0.0 to 1.0."""
    ta = set(norm_text(a).split()) - _STOPWORDS
    tb = set(norm_text(b).split()) - _STOPWORDS
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    shared = len(ta & tb)
    if shared == 0:
        return 0.0
    precision = shared / len(ta)
    recall = shared / len(tb)
    return 2 * precision * recall / (precision + recall)


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def compare_scalar(field: str, prop: str, pred: Any, gold: Any) -> float:
    """One property or one plain value."""
    if _is_empty(pred) and _is_empty(gold):
        return 1.0
    if _is_empty(pred) or _is_empty(gold):
        return 0.0

    if isinstance(gold, bool) or isinstance(pred, bool):
        return 1.0 if bool(pred) == bool(gold) else 0.0

    if prop == "date":
        return 1.0 if norm_date(pred) == norm_date(gold) else 0.0
    if prop == "time":
        return 1.0 if norm_time(pred) == norm_time(gold) else 0.0

    gold_num, pred_num = as_number(gold), as_number(pred)
    if gold_num is not None and pred_num is not None and not isinstance(gold, str):
        return 1.0 if abs(gold_num - pred_num) < 1e-6 else 0.0

    pred_text = apply_alias(field, prop, pred)
    gold_text = apply_alias(field, prop, gold)
    if norm_text(pred_text) == norm_text(gold_text):
        return 1.0
    return token_f1(pred_text, gold_text)


def compare_object(field: str, pred: dict, gold: dict) -> float:
    """Compare only the properties gold actually fills in.

    A property gold leaves null says nothing about the right answer, so extra
    detail from the model is not punished here. scoring.py counts it separately.
    """
    props = [k for k, v in gold.items() if not _is_empty(v)]
    if not props:
        return 1.0
    if not isinstance(pred, dict):
        return 0.0

    scores = []
    for prop in props:
        gold_v, pred_v = gold[prop], pred.get(prop)
        if isinstance(gold_v, list):
            scores.append(compare_list(field, pred_v if isinstance(pred_v, list) else [], gold_v))
        elif isinstance(gold_v, dict):
            scores.append(compare_object(field, pred_v if isinstance(pred_v, dict) else {}, gold_v))
        else:
            scores.append(compare_scalar(field, prop, pred_v, gold_v))
    return sum(scores) / len(scores)


def _item_similarity(field: str, pred_item: Any, gold_item: Any) -> float:
    if isinstance(gold_item, dict) and isinstance(pred_item, dict):
        return compare_object(field, pred_item, gold_item)
    if isinstance(gold_item, dict) or isinstance(pred_item, dict):
        return token_f1(json.dumps(pred_item, sort_keys=True), json.dumps(gold_item, sort_keys=True))
    return compare_scalar(field, "", pred_item, gold_item)


def compare_list(field: str, pred: list, gold: list) -> float:
    """Pair each gold item with its best match in pred, then F1.

    Order does not matter. Missing items lower recall, extra items lower
    precision, so both under- and over-answering are visible.
    """
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    grid = [[_item_similarity(field, p, g) for p in pred] for g in gold]

    # greedy best-first pairing; each item used at most once
    pairs = sorted(
        ((grid[gi][pi], gi, pi) for gi in range(len(gold)) for pi in range(len(pred))),
        reverse=True,
    )
    used_gold, used_pred, total = set(), set(), 0.0
    for score, gi, pi in pairs:
        if score <= 0 or gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        total += score

    recall = total / len(gold)
    precision = total / len(pred)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compare_values(field: str, field_class: str, pred: Any, gold: Any) -> float:
    """How well one predicted value matches the gold value, 0.0 to 1.0."""
    if _is_empty(pred) and _is_empty(gold):
        return 1.0
    if _is_empty(pred) or _is_empty(gold):
        return 0.0

    if isinstance(gold, list):
        return compare_list(field, pred if isinstance(pred, list) else [pred], gold)

    if isinstance(gold, dict):
        if field_class == "free_text":
            # prose: judge the whole thing by word overlap, not property by property
            return token_f1(json.dumps(pred, ensure_ascii=False), json.dumps(gold, ensure_ascii=False))
        return compare_object(field, pred if isinstance(pred, dict) else {}, gold)

    return compare_scalar(field, "", pred, gold)


def count_extra_properties(pred: Any, gold: Any) -> int:
    """Properties the model filled in where gold has nothing.

    Not an error by itself, but worth reporting: it is where over-claiming
    shows up.
    """
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return 0
    return sum(1 for k, v in pred.items() if not _is_empty(v) and _is_empty(gold.get(k)))
