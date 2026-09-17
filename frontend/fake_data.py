"""Fake responses used when USE_FAKE_DATA=true.

Every function here returns data in the SAME shape as the real middleware
(see the API contract in CLAUDE.md), so the pages work the same whether the
data is fake or real.

Special behaviour asked for in CLAUDE.md:
  - Model C fails with a "timeout" on every 2nd extract call, so we can
    show error handling in the UI.
  - Benchmark runs advance a little on every status poll, then finish.
"""

from __future__ import annotations

import time

from fields import FIELD_KEYS, MODEL_IDS

# ---------------------------------------------------------------------------
# Small internal state (module-level). Fine for a fake/demo backend.
# ---------------------------------------------------------------------------

_extract_calls = 0                 # counts extract() calls (for Model C timeout)
_run_progress: dict[str, int] = {}  # run_id -> how many docs are "done"

_DATASET_SIZES = {"eval_5_docs": 5, "eval_full": 50}


# ---------------------------------------------------------------------------
# Fake extracted field values
# ---------------------------------------------------------------------------

def _fields_for(model: str) -> dict:
    """Return a full set of 17 fields for one model.

    The three models give slightly different answers on purpose, so the
    side-by-side view has something to compare. Some values are None to
    show the "Not found" case.
    """
    base = {
        "submission_deadline": "2026-10-30 14:00 ET",
        "questions_deadline": "2026-10-10 17:00 ET",
        "rfp_contact": "Jane Doe, procurement@example.gov",
        "submission_method": "Upload via the BeamData vendor portal",
        "contract_term": "3 years, with two 1-year renewal options",
        "scope_of_deliverables": (
            "Design, build, and support a document extraction platform, "
            "including training and 12 months of maintenance."
        ),
        "mandatory_submission_requirements": [
            "Signed cover letter",
            "Completed pricing sheet (Appendix B)",
            "Proof of insurance",
        ],
        "mandatory_technical_requirements": [
            "SOC 2 Type II certification",
            "Data encrypted at rest and in transit",
            "99.9% uptime SLA",
        ],
        "evaluation_criteria": [
            {"category": "Technical approach", "points": 40},
            {"category": "Price", "points": 30},
            {"category": "Experience", "points": 20},
            {"category": "References", "points": 10},
        ],
        "minimum_score_threshold": "70 out of 100 to advance",
        "pricing_structure": "Fixed price, itemized by phase",
        "insurance_requirements": "General liability of at least $2,000,000",
        "vendor_experience": "At least 5 years and 3 similar projects",
        "references_required": "3 references",
        "data_security_requirements": "Comply with GDPR and local privacy law",
        "data_residency": "Data must stay within the EU",
        "demo_required": "Live demo required for shortlisted vendors",
    }

    if model == "model_a":
        # Model A misses a couple of fields.
        base["data_residency"] = None
        base["references_required"] = None
    elif model == "model_b":
        # Model B is the most complete but words things a bit differently.
        base["submission_deadline"] = "October 30, 2026 at 2:00 PM ET"
        base["references_required"] = "Three (3) client references"
    elif model == "model_c":
        # Model C is shorter and misses more.
        base["scope_of_deliverables"] = "Build a document extraction platform"
        base["insurance_requirements"] = None
        base["data_security_requirements"] = None

    # Make sure every key exists (any missing key -> None).
    return {key: base.get(key) for key in FIELD_KEYS}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def health() -> dict:
    """GET /health"""
    return {
        "status": "ok",
        "models": {"model_a": "ok", "model_b": "ok", "model_c": "ok"},
    }


def upload_document(filename: str) -> dict:
    """POST /documents"""
    return {
        "doc_id": "rfp_017",
        "pages": 12,
        "chars": 28450,
        "language": "en",
        "warnings": [],
    }


def get_document_text(doc_id: str) -> dict:
    """GET /documents/{doc_id}/text"""
    text = (
        "REQUEST FOR PROPOSAL\n\n"
        "Section 1 — Overview\n"
        "The Agency invites proposals for a document extraction platform.\n\n"
        "Section 2 — Key dates\n"
        "Questions are due 2026-10-10 17:00 ET.\n"
        "Proposals are due 2026-10-30 14:00 ET.\n\n"
        "Section 3 — Submission\n"
        "Submit via the BeamData vendor portal. Late proposals are rejected.\n\n"
        "(...fake extracted text for the demo...)"
    )
    return {"doc_id": doc_id, "text": text}


def extract(doc_id: str, models: list[str]) -> dict:
    """POST /extract

    Model C fails with a timeout on every 2nd call, on purpose.
    """
    global _extract_calls
    _extract_calls += 1
    c_should_timeout = (_extract_calls % 2 == 0)

    results = []
    for model in models:
        if model == "model_c" and c_should_timeout:
            results.append({
                "model": model,
                "status": "timeout",
                "latency_s": 120.0,
                "input_tokens": None,
                "output_tokens": None,
                "valid_json": False,
                "fields": None,
                "score": None,
                "error": "Model timed out after 120 s",
            })
            continue

        latency = {"model_a": 2.8, "model_b": 3.6, "model_c": 4.1}.get(model, 3.0)
        results.append({
            "model": model,
            "status": "ok",
            "latency_s": latency,
            "input_tokens": 2610,
            "output_tokens": 340,
            "valid_json": True,
            "fields": _fields_for(model),
            "score": None,  # only set for eval-dataset docs
            "error": None,
        })

    return {"doc_id": doc_id, "results": results}


def list_datasets() -> list[str]:
    """GET /datasets"""
    return ["eval_5_docs", "eval_full"]


def list_runs() -> list[dict]:
    """GET /benchmark/runs"""
    return [
        {"run_id": "run_003", "dataset": "eval_5_docs", "status": "done",
         "created_at": "2026-09-16 10:12"},
        {"run_id": "run_002", "dataset": "eval_5_docs", "status": "done",
         "created_at": "2026-09-15 14:40"},
        {"run_id": "run_001", "dataset": "eval_full", "status": "done",
         "created_at": "2026-09-14 09:05"},
    ]


def start_run(dataset: str, models: list[str]) -> dict:
    """POST /benchmark/runs"""
    run_id = f"run_{int(time.time()) % 1000:03d}"
    _run_progress[run_id] = 0
    return {"run_id": run_id}


def get_run(run_id: str) -> dict:
    """GET /benchmark/runs/{run_id}

    Advances by a few docs on every call so the polling UI shows movement,
    then reports "done". Old/unknown runs are treated as already finished.
    """
    total = _DATASET_SIZES["eval_5_docs"]
    done = _run_progress.get(run_id)

    if done is None:
        # A run we did not start this session (e.g. from list_runs) -> finished.
        return {"run_id": run_id, "status": "done", "done": total,
                "total": total, "errors": 1, "current_doc": None}

    done = min(done + 2, total)
    _run_progress[run_id] = done

    if done >= total:
        return {"run_id": run_id, "status": "done", "done": total,
                "total": total, "errors": 1, "current_doc": None}

    return {"run_id": run_id, "status": "running", "done": done,
            "total": total, "errors": 0, "current_doc": f"rfp_{done + 1:03d}"}


def get_results(run_id: str) -> dict:
    """GET /benchmark/runs/{run_id}/results"""
    summary = [
        {"model": "model_a", "accuracy": 0.78, "hallucination_rate": 0.06,
         "avg_latency_s": 2.8, "avg_input_tokens": 2610, "avg_output_tokens": 330,
         "total_cost_usd": 0.00, "errors": 0},
        {"model": "model_b", "accuracy": 0.88, "hallucination_rate": 0.03,
         "avg_latency_s": 3.6, "avg_input_tokens": 2610, "avg_output_tokens": 355,
         "total_cost_usd": 0.42, "errors": 0},
        {"model": "model_c", "accuracy": 0.71, "hallucination_rate": 0.09,
         "avg_latency_s": 4.1, "avg_input_tokens": 2610, "avg_output_tokens": 300,
         "total_cost_usd": 0.00, "errors": 1},
    ]

    # A small accuracy number per (model, field).
    per_field = []
    accs = {"model_a": 0.78, "model_b": 0.88, "model_c": 0.71}
    for i, key in enumerate(FIELD_KEYS):
        for model in MODEL_IDS:
            # vary a little per field so the pivot is not flat
            acc = round(min(1.0, max(0.0, accs[model] + ((i % 3) - 1) * 0.05)), 2)
            per_field.append({"model": model, "field": key, "accuracy": acc})

    # Per document: 5 docs x 3 models.
    per_document = []
    docs = [
        ("rfp_001", "en"), ("rfp_002", "en"), ("rfp_003", "ar"),
        ("rfp_004", "en"), ("rfp_005", "ar"),
    ]
    for doc_id, lang in docs:
        for model in MODEL_IDS:
            status = "ok"
            acc = accs[model]
            halluc = 1 if model == "model_c" else 0
            lat = {"model_a": 2.8, "model_b": 3.6, "model_c": 4.1}[model]
            if model == "model_c" and doc_id == "rfp_003":
                status = "timeout"
                acc = 0.0
                halluc = 0
                lat = 120.0
            per_document.append({
                "doc_id": doc_id, "language": lang, "model": model,
                "status": status, "accuracy": acc,
                "hallucinations": halluc, "latency_s": lat,
            })

    return {
        "run_id": run_id,
        "summary": summary,
        "per_field": per_field,
        "per_document": per_document,
    }
