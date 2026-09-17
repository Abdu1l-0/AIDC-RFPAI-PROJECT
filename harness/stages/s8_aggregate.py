import os
import json
import csv
from typing import List, Dict, Any

def compute_aggregates(score_records: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    models = sorted(list(set(r["model_key"] for r in score_records)))
    summary = {}
    for m in models:
        m_recs = [r for r in score_records if r["model_key"] == m]
        total = len(m_recs)
        if total == 0:
            continue
        correct_count = sum(1 for r in m_recs if r["outcome"] in ["correct", "correct_abstention"])
        fabrications = sum(1 for r in m_recs if r["outcome"] == "fabrication")
        macro_acc = correct_count / total
        fab_rate = fabrications / total
        summary[m] = {
            "macro_accuracy": round(macro_acc, 4),
            "fabrication_rate": round(fab_rate, 4),
            "total_cells": total
        }
    # Write summary CSV
    csv_path = os.path.join(out_dir, "per_model.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_key", "macro_accuracy", "fabrication_rate", "total_cells"])
        for m, s in summary.items():
            writer.writerow([m, s["macro_accuracy"], s["fabrication_rate"], s["total_cells"]])
    return summary
