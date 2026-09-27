import asyncio
import re
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

import storage
from fields import FIELDS
from method import build_method
from loader import load_pdf
from runner import new_run_id, run_benchmark, run_document, start_manifest
from llm_client import MODEL_CONFIG, enabled_models, ping_model
from schemas import BenchmarkRequest, ExtractRequest
from storage import DATA_DIR

app = FastAPI(title="RFP Benchmark Middleware")

UPLOAD_DIR = DATA_DIR / "uploads"
TEXT_DIR = DATA_DIR / "text"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

_ARABIC = re.compile(r"[؀-ۿ]")


@app.get("/health")
def health():
    """App is up, plus one line per model endpoint.

    models is {model_key: "ok" | "off" | "<error>"}, which is what the
    frontend shows as green/red.
    """
    models = {key: ping_model(key) for key in MODEL_CONFIG}
    reachable = [k for k, v in models.items() if v == "ok"]
    return {
        "status": "ok" if reachable else "degraded",
        "models": models,
        "enabled": enabled_models(),
    }


@app.get("/models")
def list_models():
    """The 3 models and whether they are ready to call."""
    return [
        {
            "id": key,
            "label": cfg["label"],
            "name": cfg["name"],
            "base_url": cfg["base_url"],
            "enabled": cfg["enabled"],
        }
        for key, cfg in MODEL_CONFIG.items()
    ]


@app.get("/fields")
def list_fields():
    """The 17 fields, in order, so the frontend does not hard-code them."""
    return [
        {
            "key": name,
            "label": fdef["name"],
            "description": fdef["description"],
            "field_class": fdef.get("field_class"),
        }
        for name, fdef in FIELDS.items()
    ]


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = Path(file.filename).stem.replace(" ", "_")
    pdf_path = UPLOAD_DIR / (doc_id + ".pdf")

    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    result = load_pdf(str(pdf_path), doc_id)

    # Rough language flag: Arabic if the text has Arabic letters, else English.
    text_path = TEXT_DIR / (doc_id + ".txt")
    sample = text_path.read_text(encoding="utf-8")[:20000] if text_path.exists() else ""
    result["language"] = "ar" if _ARABIC.search(sample) else "en"

    return result


@app.get("/documents/{doc_id}/text")
def get_document_text(doc_id: str):
    text_path = TEXT_DIR / (doc_id + ".txt")
    if not text_path.exists():
        msg = "No text found for doc_id: " + doc_id
        raise HTTPException(status_code=404, detail=msg)
    with open(text_path, "r", encoding="utf-8") as f:
        return {"doc_id": doc_id, "text": f.read()}


@app.post("/extract")
async def extract(request: ExtractRequest):
    """Run one document on the chosen models. Body: {doc_id, models}."""
    text_path = TEXT_DIR / (request.doc_id + ".txt")
    if not text_path.exists():
        msg = "No prepared text for doc_id: " + request.doc_id + ". Upload it first."
        raise HTTPException(status_code=404, detail=msg)

    available = enabled_models()
    models = request.models or available

    unknown = [m for m in models if m not in MODEL_CONFIG]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown models: {unknown}")

    wanted = [m for m in models if m in available]
    skipped = [m for m in models if m not in available]

    with open(text_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    result = {"doc_id": request.doc_id, "results": []}
    if wanted:
        result = await run_document(request.doc_id, doc_text, wanted)

    _add_disabled_rows(result, request.doc_id, skipped)
    return result


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

@app.get("/datasets")
def list_datasets():
    """Dataset names that can be benchmarked."""
    return storage.list_datasets()


@app.get("/datasets/{dataset}")
def describe_dataset(dataset: str):
    """Which documents are in a dataset, and which of them have gold answers."""
    try:
        return storage.dataset_documents(dataset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/benchmark/runs")
async def start_run(request: BenchmarkRequest):
    """Start a full run in the background. Returns the run_id to poll."""
    try:
        documents = storage.dataset_documents(request.dataset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not documents:
        raise HTTPException(status_code=400, detail=f"Dataset '{request.dataset}' has no documents")

    available = enabled_models()
    models = [m for m in (request.models or available) if m in available]
    if not models:
        raise HTTPException(status_code=400, detail="No enabled models were selected")

    run_id = new_run_id()
    start_manifest(run_id, request.dataset, models, total=len(documents) * len(models))
    asyncio.create_task(run_benchmark(run_id, request.dataset, models))
    return {"run_id": run_id}


@app.get("/benchmark/method")
def benchmark_method():
    """How every benchmark number is calculated, in plain words.

    Built from the live code values (thresholds, prices, each model's setup),
    so the explanation always matches what the benchmark actually computes.
    """
    return build_method()


@app.get("/benchmark/runs")
def list_runs():
    """Every run, newest first."""
    return storage.list_run_manifests()


@app.get("/benchmark/runs/{run_id}")
def get_run(run_id: str):
    """Progress of one run. status is running | done | failed."""
    manifest = storage.load_run_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"No run '{run_id}'")
    return manifest


@app.get("/benchmark/runs/{run_id}/results")
def get_run_results(run_id: str):
    """Summary, per-field and per-document tables for a finished run."""
    results = storage.load_run_results(run_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f"No results for run '{run_id}' yet")
    return results


@app.get("/benchmark/runs/{run_id}/download")
def download_run_results(run_id: str):
    """results.csv for a finished run."""
    path = storage.results_csv_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No results.csv for run '{run_id}' yet")
    return FileResponse(path, media_type="text/csv", filename=f"{run_id}_results.csv")


def _add_disabled_rows(result: dict, doc_id: str, skipped: list[str]) -> None:
    """Models that are switched off still get a row, so the UI can show why."""
    for model_key in skipped:
        result["results"].append({
            "doc_id": doc_id,
            "model": model_key,
            "status": "error",
            "latency_s": None,
            "input_tokens": None,
            "output_tokens": None,
            "valid_json": False,
            "fields": None,
            "score": None,
            "error": "Model is not enabled yet (set it in .env when the endpoint is ready)",
        })
