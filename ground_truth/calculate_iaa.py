"""
Inter-Annotator Agreement (IAA) Quality Gate.

Compares two independent annotators' completed annotation_template.json
files for the same document. Computes:
  - Cohen's Kappa on status labels (found / not_stated) across all 17 fields
  - Class A exact-value-match accuracy for the four atomic_exact fields
    that matter most for downstream scoring (F01-F04)

Quality Gate: kappa >= 0.60 (Substantial Agreement) AND Class A Accuracy >= 0.80

Usage:
    python ground_truth/calculate_iaa.py ann_a_DOC_01.json ann_b_DOC_01.json
"""

import sys
import json

from sklearn.metrics import cohen_kappa_score

CLASS_A_FIELDS = ["F01", "F02", "F03", "F04"]


def values_match(value_a, value_b) -> bool:
    """Structural comparison of two field values (objects, arrays, or
    scalars) rather than naive string comparison. Values are the nested
    dict/list/scalar shapes defined in config/field_set@1.0.json, so a
    plain str() comparison would treat two structurally-identical values
    as different if one annotator omitted a null key the other included.
    """
    if isinstance(value_a, dict) and isinstance(value_b, dict):
        keys = set(value_a.keys()) | set(value_b.keys())
        return all(
            _normalize(value_a.get(k)) == _normalize(value_b.get(k))
            for k in keys
        )
    if isinstance(value_a, list) and isinstance(value_b, list):
        if len(value_a) != len(value_b):
            return False
        return all(values_match(a, b) for a, b in zip(value_a, value_b))
    return _normalize(value_a) == _normalize(value_b)


def _normalize(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v


def calculate_cohen_kappa(annotator_a_file: str, annotator_b_file: str) -> dict:
    with open(annotator_a_file, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(annotator_b_file, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    labels_a, labels_b = [], []
    class_a_matches = 0
    class_a_total = 0
    disagreements = []

    all_field_ids = sorted(set(data_a["fields"].keys()) | set(data_b["fields"].keys()))

    for field_id in all_field_ids:
        content_a = data_a["fields"].get(field_id, {})
        content_b = data_b["fields"].get(field_id, {})

        status_a = content_a.get("status", "not_stated")
        status_b = content_b.get("status", "not_stated")
        labels_a.append(status_a)
        labels_b.append(status_b)

        if status_a != status_b:
            disagreements.append(f"{field_id}: status mismatch ({status_a} vs {status_b})")

        if field_id in CLASS_A_FIELDS:
            class_a_total += 1
            val_a = content_a.get("value")
            val_b = content_b.get("value")
            if status_a == status_b == "not_stated":
                # both correctly abstained -- counts as a match
                class_a_matches += 1
            elif status_a == status_b == "found" and values_match(val_a, val_b):
                class_a_matches += 1
            else:
                disagreements.append(f"{field_id}: value mismatch ({val_a!r} vs {val_b!r})")

    kappa = float(cohen_kappa_score(labels_a, labels_b))
    class_a_acc = (class_a_matches / class_a_total) if class_a_total > 0 else 0.0
    passed = kappa >= 0.60 and class_a_acc >= 0.80

    print(f"Cohen's Kappa: {kappa:.4f} | Class A Match: {class_a_acc:.2%}")
    print(f"IAA Quality Gate Status: {'PASSED (PROCEED)' if passed else 'FAILED (RE-ADJUDICATE)'}")

    if disagreements:
        print(f"\n{len(disagreements)} field-level disagreement(s):")
        for d in disagreements:
            print(f"  - {d}")

    return {
        "kappa": kappa,
        "class_a_accuracy": class_a_acc,
        "passed": passed,
        "disagreements": disagreements,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_iaa.py <annotator_a_file.json> <annotator_b_file.json>")
        sys.exit(1)
    calculate_cohen_kappa(sys.argv[1], sys.argv[2])
