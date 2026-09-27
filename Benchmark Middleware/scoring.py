"""Score one document's extraction against the gold answers.

check_grounding and score_cell are our own copy of harness/stages/s7_score.py,
so this folder does not depend on anything outside it.

NOTE: the copied score_cell only compares properly for `atomic_exact` fields.
Lists and composite objects fall back to a plain string comparison, which is
too strict. Replacing that with one matcher per field_class is a later step.
"""

import json
from pathlib import Path
from typing import Any, Dict

from fields import FIELDS
from matchers import (CORRECT_AT, PARTIAL_AT, _is_empty, compare_values,
                      count_extra_properties, list_precision_recall)

GOLD_DIR = Path(__file__).resolve().parent / "eval_set" / "ground_truth"


def check_grounding(quote: str, source_text: str) -> bool:
    """True when the quote really appears in the document text."""
    if not quote:
        return False
    # Normalize whitespace before comparison
    clean_q = " ".join(quote.split()).lower()
    clean_src = " ".join(source_text.split()).lower()
    return clean_q in clean_src


def score_cell(pred_cell: Dict[str, Any], gold_cell: Dict[str, Any], field_id: str,
               field_class: str, source_text: str) -> Dict[str, Any]:
    """Compare one predicted field with the gold answer.

    Outcomes:
      correct_abstention  gold says nothing, model said nothing
      fabrication         gold says nothing, model answered anyway
      miss                gold has a value, model said nothing
      correct             values match (similarity >= CORRECT_AT)
      partial             values partly match (>= PARTIAL_AT)
      extraction_error    values do not match

    evidence_grounded is reported separately and never changes the outcome:
    a right answer with a bad quote is still a right answer, and grounding is
    its own metric.
    """
    gold_status = gold_cell.get("status", "not_stated")
    pred_status = pred_cell.get("status", "not_stated")
    gold_val = gold_cell.get("value")
    pred_val = pred_cell.get("value")
    evidence = pred_cell.get("evidence", [])

    # Check evidence quote grounding in source document
    quotes = [ev.get("quote", "") for ev in evidence if isinstance(ev, dict) and ev.get("quote")]
    grounded = all(check_grounding(q, source_text) for q in quotes) if quotes else None

    # Gold says nothing: this is the abstention probe.
    if gold_status == "not_stated":
        if pred_status == "not_stated":
            return {"outcome": "correct_abstention", "score": 1.0,
                    "similarity": 1.0, "evidence_grounded": None, "extra_properties": 0}
        return {"outcome": "fabrication", "score": 0.0,
                "similarity": 0.0, "evidence_grounded": grounded, "extra_properties": 0}

    # Gold has a value but the model gave up.
    if pred_status == "not_stated":
        return {"outcome": "miss", "score": 0.0,
                "similarity": 0.0, "evidence_grounded": None, "extra_properties": 0}

    # Both answered: compare by field class.
    similarity = compare_values(field_id, field_class, pred_val, gold_val)
    similarity = round(float(similarity), 3)

    if similarity >= CORRECT_AT:
        outcome, score = "correct", 1.0
    elif similarity >= PARTIAL_AT:
        outcome, score = "partial", similarity
    else:
        outcome, score = "extraction_error", 0.0

    return {
        "outcome": outcome,
        "score": score,
        "similarity": similarity,
        "evidence_grounded": grounded,
        "extra_properties": count_extra_properties(pred_val, gold_val),
    }


_TYPE_CHECKS = {
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
}


def follows_instructions(pred_cell: Dict[str, Any] | None, field_type: str | None) -> bool:
    """Did the model fill this field the way the prompt asks?

      - the field is present in the answer
      - 'found'      -> a value of the right shape and at least one evidence quote
      - 'not_stated' -> no value
    """
    if not pred_cell:
        return False
    value = pred_cell.get("value")
    if pred_cell.get("status") == "not_stated":
        return _is_empty(value)
    if _is_empty(value):
        return False
    shape_ok = _TYPE_CHECKS.get(field_type, lambda v: True)(value)
    has_quote = any(isinstance(ev, dict) and ev.get("quote") for ev in pred_cell.get("evidence") or [])
    return shape_ok and has_quote


def _extra_checks(field_def: dict, field_id: str, pred_cell: dict | None,
                  gold_cell: dict, result: dict) -> dict:
    """Per-field inputs for completeness, relevance, support and instruction
    following. None means the field does not count towards that metric."""
    pred_cell = pred_cell or {}
    answered = pred_cell.get("status") == "found"
    stated = gold_cell.get("status", "not_stated") != "not_stated"
    is_list = field_def.get("type") == "array"

    precision = recall = None
    if is_list:
        pred_val, gold_val = pred_cell.get("value"), gold_cell.get("value")
        pred_list = pred_val if isinstance(pred_val, list) else ([] if _is_empty(pred_val) else [pred_val])
        gold_list = gold_val if isinstance(gold_val, list) else ([] if _is_empty(gold_val) else [gold_val])
        if not stated:
            gold_list = []
        if not answered:
            pred_list = []
        if pred_list or gold_list:
            precision, recall = list_precision_recall(field_id, pred_list, gold_list)
        if not pred_list:
            precision = None          # nothing returned: nothing to judge for relevance
        if not gold_list:
            recall = None             # nothing to find: nothing to judge for completeness

    # completeness: only fields the RFP states. Lists: share of the key covered.
    completeness = None
    if stated:
        completeness = recall if is_list else (1.0 if answered else 0.0)

    # relevance: only answers the model gave. Lists: share of items that match.
    relevance = None
    if answered:
        if is_list and precision is not None:
            relevance = precision
        else:
            relevance = 1.0 if result["similarity"] >= PARTIAL_AT else 0.0

    # supported: an answer with a quote that really is in the document, and not
    # made up where the RFP says nothing.
    supported = None
    if answered:
        supported = result["outcome"] != "fabrication" and result["evidence_grounded"] is True

    return {
        "list_precision": None if precision is None else round(precision, 3),
        "list_recall": None if recall is None else round(recall, 3),
        "completeness": None if completeness is None else round(completeness, 3),
        "relevance": None if relevance is None else round(relevance, 3),
        "supported": supported,
        "follows_instructions": follows_instructions(pred_cell or None, field_def.get("type")),
    }


def load_ground_truth(doc_id: str) -> dict | None:
    """Load the gold answers for one document, or None if there are none yet.

    Looks for eval_set/ground_truth/<doc_id>.json and returns its "fields"
    object.
    """
    gt_path = GOLD_DIR / f"{doc_id}.json"
    if not gt_path.exists():
        return None
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f).get("fields")


def score_document(predicted_fields: dict, gold_fields: dict, source_text: str) -> dict:
    """Score every field of one document's prediction against ground truth.

    Returns: {"per_field": {...}, "macro_accuracy": float, plus the rates the
    benchmark report needs}
    """
    per_field = {}
    scores = []

    for field_name, field_def in FIELDS.items():
        field_class = field_def.get("field_class", "atomic_exact")
        raw_cell = predicted_fields.get(field_name)   # None when the model left it out
        pred_cell = raw_cell or {"status": "not_stated", "value": None}
        gold_cell = gold_fields.get(field_name) or {"status": "not_stated", "value": None}

        result = score_cell(pred_cell, gold_cell, field_name, field_class, source_text)
        result["field_class"] = field_class
        result.update(_extra_checks(field_def, field_name, raw_cell, gold_cell, result))
        per_field[field_name] = result
        scores.append(result["score"])

    cells = list(per_field.values())
    outcomes = [c["outcome"] for c in cells]

    def _rate(count: int, total: int) -> float | None:
        return round(count / total, 3) if total else None

    # Fields where the RFP says nothing: the chance to hallucinate.
    absent = [c for c in cells if c["outcome"] in ("correct_abstention", "fabrication")]
    # Fields where the RFP does say something: the chance to extract.
    present = [c for c in cells if c["outcome"] not in ("correct_abstention", "fabrication")]
    graded = [c for c in cells if c["evidence_grounded"] is not None]

    def _avg(key: str) -> float | None:
        vals = [c[key] for c in cells if c[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    answered = [c for c in cells if c["supported"] is not None]

    return {
        "per_field": per_field,
        "macro_accuracy": round(sum(scores) / len(scores), 3) if scores else None,
        "extraction_accuracy": _rate(sum(c["score"] for c in present), len(present)),
        "abstention_accuracy": _rate(outcomes.count("correct_abstention"), len(absent)),
        "fabrication_rate": _rate(outcomes.count("fabrication"), len(absent)),
        "miss_rate": _rate(outcomes.count("miss"), len(present)),
        "grounding_rate": _rate(sum(1 for c in graded if c["evidence_grounded"]), len(graded)),
        "completeness": _avg("completeness"),
        "relevance": _avg("relevance"),
        "unsupported_rate": _rate(sum(1 for c in answered if not c["supported"]), len(answered)),
        "instruction_following": _rate(sum(1 for c in cells if c["follows_instructions"]), len(cells)),
        "counts": {name: outcomes.count(name) for name in
                   ("correct", "partial", "extraction_error", "miss",
                    "correct_abstention", "fabrication")},
    }
