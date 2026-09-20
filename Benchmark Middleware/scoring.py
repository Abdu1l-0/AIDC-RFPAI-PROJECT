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
from matchers import CORRECT_AT, PARTIAL_AT, compare_values, count_extra_properties

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
        pred_cell = predicted_fields.get(field_name, {"status": "not_stated", "value": None})
        gold_cell = gold_fields.get(field_name, {"status": "not_stated", "value": None})

        result = score_cell(pred_cell, gold_cell, field_name, field_class, source_text)
        result["field_class"] = field_class
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

    return {
        "per_field": per_field,
        "macro_accuracy": round(sum(scores) / len(scores), 3) if scores else None,
        "extraction_accuracy": _rate(sum(c["score"] for c in present), len(present)),
        "abstention_accuracy": _rate(outcomes.count("correct_abstention"), len(absent)),
        "fabrication_rate": _rate(outcomes.count("fabrication"), len(absent)),
        "miss_rate": _rate(outcomes.count("miss"), len(present)),
        "grounding_rate": _rate(sum(1 for c in graded if c["evidence_grounded"]), len(graded)),
        "counts": {name: outcomes.count(name) for name in
                   ("correct", "partial", "extraction_error", "miss",
                    "correct_abstention", "fabrication")},
    }
