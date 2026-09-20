import json
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
RUNS_DIR = DATA_DIR / "runs"

EVAL_DIR = BASE_DIR / "eval_set"
EVAL_DOCS_DIR = EVAL_DIR / "documents"
EVAL_GOLD_DIR = EVAL_DIR / "ground_truth"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# The eval set: PDFs in eval_set/documents, gold answers in eval_set/ground_truth
# ---------------------------------------------------------------------------

def _has_gold(doc_id: str) -> bool:
    """Gold files are named after the PDF. Allow spaces or underscores."""
    for name in (doc_id, doc_id.replace("_", " "), doc_id.replace(" ", "_")):
        if (EVAL_GOLD_DIR / f"{name}.json").exists():
            return True
    return False


def eval_documents() -> list[dict]:
    """Every PDF in the eval set, with whether it has gold answers."""
    if not EVAL_DOCS_DIR.exists():
        return []
    docs = []
    for pdf in sorted(EVAL_DOCS_DIR.glob("*.pdf")):
        doc_id = pdf.stem
        docs.append({"doc_id": doc_id, "path": str(pdf), "scored": _has_gold(doc_id)})
    return docs


def list_datasets() -> list[str]:
    """Dataset names the benchmark can run."""
    return ["eval_scored", "eval_all"]


def dataset_documents(dataset: str) -> list[dict]:
    """The documents in one dataset.

    eval_scored - only documents that have gold answers (these produce scores)
    eval_all    - every PDF in the eval set (the rest run without a score)
    """
    docs = eval_documents()
    if dataset == "eval_scored":
        return [d for d in docs if d["scored"]]
    if dataset == "eval_all":
        return docs
    raise KeyError(f"Unknown dataset '{dataset}'")


def save_result(doc_id: str, model_key: str, result: dict) -> Path:
    """Save one model's result for one document as JSON.
    Path: data/results/<doc_id>__<model_key>.json
    """
    path = RESULTS_DIR / f"{doc_id}__{model_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def load_result(doc_id: str, model_key: str) -> dict | None:
    """Load a previously saved result, or None if it doesn't exist."""
    path = RESULTS_DIR / f"{doc_id}__{model_key}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_run_manifest(run_id: str, manifest: dict) -> Path:
    """Save a benchmark run's manifest/status as JSON.
    Path: data/runs/<run_id>.json
    """
    path = RUNS_DIR / f"{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def load_run_manifest(run_id: str) -> dict | None:
    """Load a run's manifest/status, or None if it doesn't exist."""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_run_manifests() -> list[dict]:
    """Every run, newest first."""
    runs = []
    for path in RUNS_DIR.glob("*.json"):
        if path.name.endswith("_results.json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)


def save_run_results(run_id: str, results: dict) -> Path:
    """Save a finished run's aggregated results."""
    path = RUNS_DIR / f"{run_id}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def load_run_results(run_id: str) -> dict | None:
    path = RUNS_DIR / f"{run_id}_results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def results_csv_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}_results.csv"


def export_results_csv(run_id: str, rows: list[dict]) -> Path:
    """Write a flat list of result rows to CSV for download.
    Path: data/runs/<run_id>_results.csv
    """
    path = RUNS_DIR / f"{run_id}_results.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path