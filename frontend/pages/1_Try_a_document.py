"""Page 1 — Try a document.

Upload one RFP PDF, pick models, and compare their answers field by field.
"""

from __future__ import annotations

import streamlit as st

import api
import ui
from fields import MODELS

st.set_page_config(page_title="Try a document", page_icon="📄", layout="wide")
st.title("Try a document")


@st.cache_data(ttl=60)
def _catalog():
    """Models and field labels, straight from the middleware.

    Falls back to the built-in list if the middleware cannot be reached, so
    the page still renders instead of crashing.
    """
    try:
        models = api.list_models()
    except api.ApiError:
        models = [{"id": m["id"], "label": m["label"], "enabled": True} for m in MODELS]
    try:
        field_list = api.list_fields()
    except api.ApiError:
        field_list = None
    return models, field_list


models_info, field_list = _catalog()
model_labels = {m["id"]: m["label"] for m in models_info}

# ---------------------------------------------------------------------------
# 1. Upload a PDF (uploaded only once per file)
# ---------------------------------------------------------------------------

uploaded = st.file_uploader("Upload an RFP PDF", type=["pdf"])

doc_info = None
if uploaded is not None:
    # Each upload has a unique file_id. Cache the upload result on it so we
    # do not send the same file to the middleware again on every rerun.
    cache = st.session_state.setdefault("_uploads", {})
    if uploaded.file_id not in cache:
        try:
            with st.spinner("Sending PDF to the middleware..."):
                cache[uploaded.file_id] = api.upload_document(uploaded)
        except api.ApiError as exc:
            st.error(str(exc))
    doc_info = cache.get(uploaded.file_id)

if doc_info:
    cols = st.columns(4)
    cols[0].metric("Doc ID", doc_info.get("doc_id", "—"))
    cols[1].metric("Pages", doc_info.get("pages", "—"))
    cols[2].metric("Characters", doc_info.get("chars", "—"))
    cols[3].metric("Language", doc_info.get("language", "—"))

    for warning in doc_info.get("warnings", []):
        st.warning(warning)

    with st.expander("Show extracted text"):
        try:
            text = api.get_document_text(doc_info["doc_id"]).get("text", "")
            st.text(text)
        except api.ApiError as exc:
            st.error(str(exc))

# ---------------------------------------------------------------------------
# 2. Pick models and run extraction
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Models")

chosen = []
cols = st.columns(len(models_info))
for col, model in zip(cols, models_info):
    ready = model.get("enabled", True)
    label = model["label"] if ready else f"{model['label']} — not ready"
    if col.checkbox(label, value=ready, disabled=not ready, key=f"pick_{model['id']}"):
        chosen.append(model["id"])

run = st.button("Extract", type="primary", disabled=(doc_info is None or not chosen))

if run and doc_info:
    try:
        with st.spinner("Running the models..."):
            data = api.extract(doc_info["doc_id"], chosen)
        st.session_state["_extract_result"] = data
    except api.ApiError as exc:
        st.error(str(exc))

# ---------------------------------------------------------------------------
# 3. Show results
# ---------------------------------------------------------------------------

result = st.session_state.get("_extract_result")
if result and doc_info and result.get("doc_id") == doc_info["doc_id"]:
    st.divider()
    st.subheader("Model status")
    status_cols = st.columns(len(result["results"]))
    for col, item in zip(status_cols, result["results"]):
        with col:
            ui.render_model_status(item, model_labels)

    st.divider()
    st.subheader("Extracted fields")
    ui.render_field_grid(result["results"], field_list, model_labels)
