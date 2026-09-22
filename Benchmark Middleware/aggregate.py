"""Turn per-document results into the tables the frontend shows.

The frontend draws what it gets, so all the adding-up happens here:
  summary       one row per model
  per_field     one row per model and field
  per_document  one row per document and model
"""

from __future__ import annotations

import storage
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

    # A call that failed or could not be read, on a document that HAS an answer
    # key, counts as 0 (every field wrong). Leaving it out would make a model
    # that fails on hard documents look better. Documents without an answer key
    # have nothing to be scored against and stay out.
    def is_scored(r: dict) -> bool:
        return bool(r.get("score_detail")) or storage.has_gold(r["doc_id"])

    def field_score(r: dict, field: str) -> float:
        detail = r.get("score_detail")
        return detail["per_field"][field]["score"] if detail else 0.0

    # --- one row per model and field --------------------------------------
    per_field = []
    for model in models:
        model_records = [r for r in records if r["model"] == model and is_scored(r)]
        for field in FIELD_NAMES:
            accuracy = _mean([field_score(r, field) for r in model_records])
            if accuracy is not None:
                per_field.append({"model": model, "field": field, "accuracy": accuracy})

    # --- one row per model -------------------------------------------------
    summary = []
    for model in models:
        model_records = [r for r in records if r["model"] == model]
        scored = [r for r in model_records if is_scored(r)]
        details = [r["score_detail"] for r in scored if r.get("score_detail")]

        # how the 17 x documents field answers ended up, for the outcome chart
        outcome_counts = {}
        for d in details:
            for name, count in (d.get("counts") or {}).items():
                outcome_counts[name] = outcome_counts.get(name, 0) + count
        failed = sum(1 for r in scored if not r.get("score_detail"))
        if failed:
            outcome_counts["failed_call"] = failed * len(FIELD_NAMES)

        summary.append({
            "model": model,
            # how the answer format reached this model - must be stated in the report
            "schema_mode": next((r.get("schema_mode") for r in model_records if r.get("schema_mode")), None),
            "prompt_variant": next((r.get("prompt_variant") for r in model_records if r.get("prompt_variant")), None),
            "documents": len(model_records),
            "accuracy": _mean([(r["score_detail"]["macro_accuracy"] if r.get("score_detail") else 0.0)
                               for r in scored]),
            "extraction_accuracy": _mean([(r["score_detail"].get("extraction_accuracy") if r.get("score_detail") else 0.0)
                                          for r in scored]),
            # a failed call produced nothing, so it is 0 for completeness and
            # instruction following, like for accuracy
            "completeness": _mean([(r["score_detail"].get("completeness") if r.get("score_detail") else 0.0)
                                   for r in scored]),
            "relevance": _mean([d.get("relevance") for d in details]),
            "unsupported_rate": _mean([d.get("unsupported_rate") for d in details]),
            "instruction_following": _mean([(r["score_detail"].get("instruction_following")
                                             if r.get("score_detail") else 0.0) for r in scored]),
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
            "languages": sorted({r.get("language", "en") for r in model_records}),
            "outcome_counts": outcome_counts,
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
