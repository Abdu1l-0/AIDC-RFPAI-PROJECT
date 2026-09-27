"""Re-score a finished run with the current scoring code, without calling any model.

    python rescore.py                 # the newest finished run
    python rescore.py run_20260922_103702

Uses the saved answers in data/results/<doc>__<model>.json. Those hold the
LATEST answer per document and model, so this is exact only for the newest run
that covered them. It warns when a saved answer's score no longer matches what
the run recorded (a sign the answer came from a different run, or that scoring
changed for that document).
"""

from __future__ import annotations

import sys

import storage
from aggregate import build_results, csv_rows
from loader import detect_language, ensure_text
from scoring import load_ground_truth, score_document


def rescore(run_id: str) -> dict:
    manifest = storage.load_run_manifest(run_id)
    old = storage.load_run_results(run_id)
    if not manifest or not old:
        raise SystemExit(f"No finished run '{run_id}'")

    old_score = {(d["doc_id"], d["model"]): d.get("accuracy") for d in old["per_document"]}
    records = []
    for doc in storage.dataset_documents(manifest["dataset"]):
        doc_id = doc["doc_id"]
        text = ensure_text(doc_id, doc["path"])
        gold = load_ground_truth(doc_id)
        for model in manifest["models"]:
            if (doc_id, model) not in old_score:
                continue
            record = storage.load_result(doc_id, model)
            if record is None:
                print(f"  ! no saved answer for {doc_id} / {model}, skipped")
                continue
            if record.get("fields") and gold is not None:
                detail = score_document(record["fields"], gold.get("fields", gold), text)
                record["score_detail"] = detail
                record["score"] = detail["macro_accuracy"]
            record["language"] = detect_language(text)
            records.append(record)

            before = old_score[(doc_id, model)]
            if before is not None and record.get("score") is not None \
                    and abs(before - record["score"]) > 0.001:
                print(f"  ! {doc_id} / {model}: accuracy {before} -> {record['score']}")

    results = build_results(run_id, records)
    results["dataset"] = manifest["dataset"]
    storage.save_run_results(run_id, results)
    storage.export_results_csv(run_id, csv_rows(results))
    return results


def _newest_run() -> str:
    done = [m for m in storage.list_run_manifests() if m.get("status") == "done"]
    if not done:
        raise SystemExit("No finished runs")
    return max(done, key=lambda m: m["run_id"])["run_id"]


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else _newest_run()
    print(f"Re-scoring {run}")
    for row in rescore(run)["summary"]:
        print(f"  {row['model']}: " + ", ".join(
            f"{k}={row.get(k)}" for k in ("accuracy", "completeness", "relevance",
                                          "unsupported_rate", "instruction_following",
                                          "valid_json_rate")))
