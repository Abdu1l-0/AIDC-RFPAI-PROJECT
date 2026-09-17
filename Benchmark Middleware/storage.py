import json
import csv
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = DATA_DIR / "results"
RUNS_DIR = DATA_DIR / "runs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


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