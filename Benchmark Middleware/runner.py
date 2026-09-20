import asyncio
from datetime import datetime, timezone

import storage
from aggregate import build_results, csv_rows
from display import add_display
from llm_client import call_model, MODEL_CONFIG
from loader import detect_language, ensure_text
from validate import parse_model_output
from storage import save_result
from scoring import load_ground_truth, score_document


async def _run_one_model(doc_id: str, model_key: str, text: str) -> dict:
    """Call one model, validate its output, score it if ground truth
    exists, save the result, return it."""
    raw_result = await asyncio.to_thread(call_model, model_key, text)

    if raw_result["status"] == "ok":
        parsed = parse_model_output(raw_result["raw_text"])
    else:
        parsed = {"valid": False, "fields": None, "error": raw_result["error"]}

    score = None
    score_detail = None
    if parsed["valid"] and parsed["fields"] is not None:
        gold = load_ground_truth(doc_id)
        if gold is not None:
            predicted_dict = parsed["fields"].model_dump()
            gold_fields = gold.get("fields", gold)
            score_detail = score_document(predicted_dict, gold_fields, text)
            score = score_detail["macro_accuracy"]

    result = {
        "doc_id": doc_id,
        "model": model_key,
        "status": raw_result["status"],
        "latency_s": raw_result["latency_s"],
        "input_tokens": raw_result["input_tokens"],
        "output_tokens": raw_result["output_tokens"],
        "valid_json": parsed["valid"],
        # each cell carries value (exact, for scoring) + display (for the UI)
        "fields": add_display(parsed["fields"].model_dump() if parsed["fields"] else None),
        "score": score,
        "score_detail": score_detail,
        "error": raw_result["error"] or parsed.get("error"),
        "schema_mode": raw_result.get("schema_mode"),
        "finish_reason": raw_result.get("finish_reason"),
    }

    save_result(doc_id, model_key, result)
    return result


async def run_document(doc_id: str, text: str, models: list[str] = None) -> dict:
    """Run one document across the chosen models (default: all 3), in parallel.

    Returns: {"doc_id": doc_id, "results": [result, result, result]}
    """
    if models is None:
        models = list(MODEL_CONFIG.keys())

    tasks = [_run_one_model(doc_id, model_key, text) for model_key in models]
    results = await asyncio.gather(*tasks)

    return {"doc_id": doc_id, "results": list(results)}


# ---------------------------------------------------------------------------
# Full benchmark run (background)
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def start_manifest(run_id: str, dataset: str, models: list[str], total: int) -> dict:
    manifest = {
        "run_id": run_id,
        "dataset": dataset,
        "models": models,
        "status": "running",
        "done": 0,
        "total": total,
        "errors": 0,
        "current_doc": None,
        "created_at": _now(),
        "finished_at": None,
    }
    storage.save_run_manifest(run_id, manifest)
    return manifest


async def run_benchmark(run_id: str, dataset: str, models: list[str]) -> None:
    """Run every document in the dataset on every chosen model.

    Documents run one after another; the models for one document run in
    parallel. Progress is written to the run manifest after each document, so
    the frontend can poll it.
    """
    manifest = storage.load_run_manifest(run_id) or {}
    records = []

    try:
        documents = storage.dataset_documents(dataset)
        for doc in documents:
            doc_id = doc["doc_id"]
            manifest["current_doc"] = doc_id
            storage.save_run_manifest(run_id, manifest)

            text = await asyncio.to_thread(ensure_text, doc_id, doc["path"])
            language = detect_language(text)

            outcome = await run_document(doc_id, text, models)
            for result in outcome["results"]:
                result["language"] = language
                records.append(result)
                manifest["done"] += 1
                if result["status"] != "ok":
                    manifest["errors"] += 1

            storage.save_run_manifest(run_id, manifest)

        results = build_results(run_id, records)
        results["dataset"] = dataset
        storage.save_run_results(run_id, results)
        storage.export_results_csv(run_id, csv_rows(results))

        manifest["status"] = "done"
    except Exception as exc:  # a broken run must still report, not hang
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        manifest["current_doc"] = None
        manifest["finished_at"] = _now()
        storage.save_run_manifest(run_id, manifest)