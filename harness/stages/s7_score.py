import json
from typing import Dict, Any

def check_grounding(quote: str, source_text: str) -> bool:
    if not quote:
        return False
    # Normalize whitespace before comparison
    clean_q = " ".join(quote.split()).lower()
    clean_src = " ".join(source_text.split()).lower()
    return clean_q in clean_src

def score_cell(pred_cell: Dict[str, Any], gold_cell: Dict[str, Any], field_id: str, field_class: str, source_text: str) -> Dict[str, Any]:
    gold_status = gold_cell.get("status", "not_stated")
    pred_status = pred_cell.get("status", "not_stated")
    gold_val = gold_cell.get("value")
    pred_val = pred_cell.get("value")
    evidence = pred_cell.get("evidence", [])

    # Check evidence quote grounding in source document
    quotes = [ev.get("quote", "") for ev in evidence if isinstance(ev, dict)]
    grounded = all(check_grounding(q, source_text) for q in quotes) if quotes else False

    # Case 1: Gold is not_stated (Abstention probe)
    if gold_status == "not_stated":
        if pred_status == "not_stated":
            return {"outcome": "correct_abstention", "score": 1.0, "evidence_grounded": True}
        else:
            return {"outcome": "fabrication", "score": 0.0, "evidence_grounded": grounded}

    # Case 2: Gold is present, model abstained (Miss)
    if pred_status == "not_stated":
        return {"outcome": "miss", "score": 0.0, "evidence_grounded": None}

    # Case 3: Both present -> Check extraction accuracy by field class
    if field_class == "atomic_exact":
        is_correct = (str(pred_val).strip().lower() == str(gold_val).strip().lower())
        if is_correct:
            return {"outcome": "correct", "score": 1.0, "evidence_grounded": grounded}
        else:
            outcome = "extraction_error" if grounded else "fabrication"
            return {"outcome": outcome, "score": 0.0, "evidence_grounded": grounded}

    # Default fallback for structured/lists (simplified match)
    is_match = (str(pred_val).strip().lower() == str(gold_val).strip().lower())
    if is_match:
        return {"outcome": "correct", "score": 1.0, "evidence_grounded": grounded}
    else:
        outcome = "extraction_error" if grounded else "fabrication"
        return {"outcome": outcome, "score": 0.0, "evidence_grounded": grounded}
