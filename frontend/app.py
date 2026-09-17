"""Home page: short intro + middleware health check.

Run from the frontend/ folder:
    uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import api
from fields import MODELS

st.set_page_config(page_title="AIDC RFP Benchmark", page_icon="📄", layout="wide")

st.title("📄 AIDC RFP Benchmark")
st.write(
    "Test how well three language models pull key fields out of RFP "
    "(Request for Proposal) PDFs."
)

if api.USE_FAKE_DATA:
    st.info("Running on **fake data**. Set `USE_FAKE_DATA=false` in `.env` to use the real middleware.")

st.subheader("Pages")
st.markdown(
    "- **Try a document** — upload one PDF and compare the models side by side.\n"
    "- **Run benchmark** — run the whole test dataset and track progress.\n"
    "- **Results** — see accuracy, hallucination, latency, tokens, and cost."
)

st.divider()
st.subheader("Middleware health")

if st.button("Check now", type="primary"):
    st.session_state["_ran_health"] = True

if st.session_state.get("_ran_health"):
    try:
        data = api.health()
    except api.ApiError as exc:
        st.error(str(exc))
    else:
        overall = data.get("status", "unknown")
        if overall == "ok":
            st.success(f"Middleware status: {overall}")
        else:
            st.warning(f"Middleware status: {overall}")

        model_health = data.get("models", {})
        cols = st.columns(len(MODELS))
        for col, model in zip(cols, MODELS):
            state = model_health.get(model["id"], "unknown")
            mark = "✅" if state == "ok" else "❌"
            col.metric(model["label"], f"{mark} {state}")
else:
    st.caption("Click **Check now** to ping the middleware and each model.")
