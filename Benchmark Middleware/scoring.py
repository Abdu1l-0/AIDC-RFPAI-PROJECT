import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.stages.s7_score import score_cell

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELD_SET_PATH = REPO_ROOT / "config" / "field_set@1.0.json"
GROUND_TRUTH_DIR = REPO_ROOT / "ground_truth"

with open(FIELD_SET_PATH, "r", encoding="utf-8") as f:
    FIELD_SET = json.load(f)["fields"]


def load_ground_truth(doc_id: str) -> dict | None:
    """Load the frozen gold answers for one document, if they exist yet.
    Looks for ground_truth/adjudicated/<doc_id>.json (adjust path once
    the team's ground truth owner confirms the final frozen location).
    """
    gt_path = GROUND_TRUTH_DIR / "adjudicated" / f"{doc_id}.json"
    if not gt_path.exists():
        return None
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_document(predicted_fields: dict, gold_fields: dict, source_text: str) -> dict:
    """Score every field of one document's prediction against ground truth,
    using the team's outcome taxonomy (correct / correct_abstention / miss /
    extraction_error / fabrication) via harness.stages.s7_score.score_cell.

    Returns: {"per_field": {F01: {outcome, score, evidence_grounded}, ...},
              "macro_accuracy": float}
    """
    per_field = {}
    scores = []

    for field_id, field_def in FIELD_SET.items():
        field_class = field_def.get("field_class", "atomic_exact")
        pred_cell = predicted_fields.get(field_id, {"status": "not_stated", "value": None})
        gold_cell = gold_fields.get(field_id, {"status": "not_stated", "value": None})

        result = score_cell(pred_cell, gold_cell, field_id, field_class, source_text)
        per_field[field_id] = result
        scores.append(result["score"])

    macro_accuracy = round(sum(scores) / len(scores), 3) if scores else None

    return {"per_field": per_field, "macro_accuracy": macro_accuracy}