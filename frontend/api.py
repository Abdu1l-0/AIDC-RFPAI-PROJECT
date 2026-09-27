"""All calls to the middleware live here.

The pages NEVER call the middleware or the models directly. They call the
functions in this file. This is the only place that knows about httpx and
the API contract.

Set USE_FAKE_DATA=true in .env to use fake data (see fake_data.py) and never
touch the network. Set it to false when the middleware is ready. Nothing else
in the app needs to change.
"""

from __future__ import annotations

import io
import os

import httpx
import pandas as pd
from dotenv import load_dotenv

import fake_data

load_dotenv()

MIDDLEWARE_URL = os.getenv("MIDDLEWARE_URL", "http://localhost:8000")
USE_FAKE_DATA = os.getenv("USE_FAKE_DATA", "true").lower() == "true"

# One shared timeout. Extract calls run several models, each up to ~90 s with
# one retry, so this has to be longer than the middleware's worst case.
_TIMEOUT = httpx.Timeout(300.0)


class ApiError(Exception):
    """Raised on any network or HTTP error. Pages show it with st.error."""


def _get(path: str, **kwargs):
    url = f"{MIDDLEWARE_URL}{path}"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise ApiError(f"Server returned {exc.response.status_code} for {path}") from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"Could not reach the middleware at {url}: {exc}") from exc


def _post(path: str, *, json=None, files=None):
    url = f"{MIDDLEWARE_URL}{path}"
    try:
        resp = httpx.post(url, json=json, files=files, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        raise ApiError(f"Server returned {exc.response.status_code} for {path}") from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"Could not reach the middleware at {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Health / models
# ---------------------------------------------------------------------------

def health() -> dict:
    if USE_FAKE_DATA:
        return fake_data.health()
    return _get("/health")


def list_models() -> list[dict]:
    """The 3 models with their labels and whether they are ready to call."""
    if USE_FAKE_DATA:
        return fake_data.list_models()
    return _get("/models")


def list_fields() -> list[dict]:
    """The 17 fields, in order, with their labels."""
    if USE_FAKE_DATA:
        return fake_data.list_fields()
    return _get("/fields")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def upload_document(uploaded_file) -> dict:
    """Send a PDF to the middleware. `uploaded_file` is a Streamlit file."""
    if USE_FAKE_DATA:
        return fake_data.upload_document(uploaded_file.name)
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    return _post("/documents", files=files)


def get_document_text(doc_id: str) -> dict:
    if USE_FAKE_DATA:
        return fake_data.get_document_text(doc_id)
    return _get(f"/documents/{doc_id}/text")


# ---------------------------------------------------------------------------
# Extract (single document)
# ---------------------------------------------------------------------------

def extract(doc_id: str, models: list[str]) -> dict:
    if USE_FAKE_DATA:
        return fake_data.extract(doc_id, models)
    return _post("/extract", json={"doc_id": doc_id, "models": models})


# ---------------------------------------------------------------------------
# Datasets / benchmark runs
# ---------------------------------------------------------------------------

def list_datasets() -> list[str]:
    if USE_FAKE_DATA:
        return fake_data.list_datasets()
    return _get("/datasets")


def list_runs() -> list[dict]:
    if USE_FAKE_DATA:
        return fake_data.list_runs()
    return _get("/benchmark/runs")


def start_run(dataset: str, models: list[str]) -> dict:
    if USE_FAKE_DATA:
        return fake_data.start_run(dataset, models)
    return _post("/benchmark/runs", json={"dataset": dataset, "models": models})


def get_method() -> dict | None:
    """How every benchmark number is calculated (GET /benchmark/method).

    Built by the middleware from its live code values. Not available on fake
    data: returns None, and the page shows the numbers without explanations.
    """
    if USE_FAKE_DATA:
        return None
    return _get("/benchmark/method")


def get_run(run_id: str) -> dict:
    if USE_FAKE_DATA:
        return fake_data.get_run(run_id)
    return _get(f"/benchmark/runs/{run_id}")


def get_results(run_id: str) -> dict:
    if USE_FAKE_DATA:
        return fake_data.get_results(run_id)
    return _get(f"/benchmark/runs/{run_id}/results")


def download_results_csv(run_id: str) -> bytes:
    """Return the results CSV as bytes, ready for st.download_button."""
    if USE_FAKE_DATA:
        results = fake_data.get_results(run_id)
        df = pd.DataFrame(results["per_document"])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    url = f"{MIDDLEWARE_URL}/benchmark/runs/{run_id}/download"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPError as exc:
        raise ApiError(f"Could not download results from {url}: {exc}") from exc
