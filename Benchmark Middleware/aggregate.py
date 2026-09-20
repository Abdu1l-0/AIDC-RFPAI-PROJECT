"""Turn per-document results into the tables the frontend shows.

The frontend draws what it gets, so all the adding-up happens here:
  summary       one row per model
  per_field     one row per model and field
  per_document  one row per document and model
"""

from __future__ import annotations

from config import settings
from fields import FIELD_NAMES
from llm_client import MODEL_CONFIG


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def cost_usd(model_key: str, input_tokens: int | None, output_tokens: int | None) -> float:
    """Price one call using the per-1M-token rates in .env / config.py."""
    name = MODEL_CONFIG.get(model_key, {}).get("name", model_key)
    price_in = settings.price_per_1m_input.get(name, 0.0)
    price_out = settings.price_per_1m_output.get(name, 0.0)
    return ((input_tokens or 0) / 1_000_000 * price_in
            + (output_tokens or 0) / 1_000_000 * price_out)


def build_results(run_id: str, records: list[dict]) -> dict:
    """records is one dict per (document, model), as runner saves them."""
    models = sorted({r["model"] for r in records})

    # --- one row per document and model -----------------------------------
    per_document = []
    for r in records:
        detail = r.get("score_detail") or {}
        counts = detail.get("counts") or {}
        per_document.append({
            "doc_id": r["doc_id"],
            "language": r.get("language", "en"),
            "model": r["model"],
            "status": r["status"],
            "valid_json": r.get("valid_json"),
            "accuracy": r.get("score"),
            "hallucinations": counts.get("fabrication"),
            "misses": counts.get("miss"),
            "latency_s": r.get("latency_s"),
            "input_tokens": r.get("input_tokens"),
            "output_tokens": r.get("output_tokens"),
            "cost_usd": round(cost_usd(r["model"], r.get("input_tokens"),
                                       r.get("output_tokens")), 6),
            "error": r.get("error"),
        })

    # --- one row per model and field --------------------------------------
    per_field = []
    for model in models:
        model_records = [r for r in records if r["model"] == model]
        for field in FIELD_NAMES:
            scores = [
                (r.get("score_detail") or {}).get("per_field", {}).get(field, {}).get("score")
                for r in model_records
            ]
            accuracy = _mean(scores)
            if accuracy is not None:
                per_field.append({"model": model, "field": field, "accuracy": accuracy})

    # --- one row per model -------------------------------------------------
    summary = []
    for model in models:
        model_records = [r for r in records if r["model"] == model]
        scored = [r for r in model_records if r.get("score_detail")]
        details = [r["score_detail"] for r in scored]

        summary.append({
            "model": model,
            "documents": len(model_records),
            "accuracy": _mean([d.get("macro_accuracy") for d in details]),
            "extraction_accuracy": _mean([d.get("extraction_accuracy") for d in details]),
            "hallucination_rate": _mean([d.get("fabrication_rate") for d in details]),
            "miss_rate": _mean([d.get("miss_rate") for d in details]),
            "grounding_rate": _mean([d.get("grounding_rate") for d in details]),
            "valid_json_rate": _mean([1.0 if r.get("valid_json") else 0.0 for r in model_records]),
            "avg_latency_s": _mean([r.get("latency_s") for r in model_records]),
            "avg_input_tokens": _mean([r.get("input_tokens") for r in model_records]),
            "avg_output_tokens": _mean([r.get("output_tokens") for r in model_records]),
            "total_cost_usd": round(sum(
                cost_usd(model, r.get("input_tokens"), r.get("output_tokens"))
                for r in model_records), 4),
            "errors": sum(1 for r in model_records if r["status"] != "ok"),
        })

    return {
        "run_id": run_id,
        "summary": summary,
        "per_field": per_field,
        "per_document": per_document,
    }


def csv_rows(results: dict) -> list[dict]:
    """The rows written to results.csv (the per-document table)."""
    return [{"run_id": results["run_id"], **row} for row in results["per_document"]]
