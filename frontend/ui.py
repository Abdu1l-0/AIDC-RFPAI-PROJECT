"""Shared display helpers used by more than one page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from fields import FIELD_LABELS, FIELDS, MODEL_LABELS

# status -> (emoji, short text)
_STATUS_LOOK = {
    "ok": ("✅", "OK"),
    "invalid_json": ("⚠️", "Bad JSON"),
    "timeout": ("⏱️", "Timed out"),
    "error": ("❌", "Error"),
}


def status_badge(status: str) -> str:
    emoji, text = _STATUS_LOOK.get(status, ("❓", status))
    return f"{emoji} {text}"


def value_to_markdown(shape: str, value) -> str:
    """Turn one field value into markdown text, based on its shape."""
    if value is None or value == "" or value == []:
        return "_Not found_"

    if shape == "str":
        return str(value)

    if shape == "list":
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        return str(value)

    if shape == "criteria":
        # list of {category, points} -> a small markdown table
        if isinstance(value, list) and value:
            rows = ["| Category | Points |", "| --- | --- |"]
            for item in value:
                if isinstance(item, dict):
                    cat = item.get("category", "")
                    pts = item.get("points", "")
                    rows.append(f"| {cat} | {pts} |")
                else:
                    rows.append(f"| {item} |  |")
            return "\n".join(rows)
        return str(value)

    return str(value)


def render_model_status(result: dict) -> None:
    """One compact status block for a single model result."""
    st.markdown(f"**{MODEL_LABELS.get(result['model'], result['model'])}**")
    st.write(status_badge(result.get("status", "")))

    cols = st.columns(3)
    latency = result.get("latency_s")
    cols[0].metric("Latency (s)", f"{latency:.1f}" if latency is not None else "—")
    cols[1].metric("In tokens", result.get("input_tokens") or "—")
    cols[2].metric("Out tokens", result.get("output_tokens") or "—")

    if result.get("error"):
        st.error(result["error"])


def render_field_grid(results: list[dict]) -> None:
    """Show one row per field, one column per model, so answers line up.

    `results` is the list from POST /extract (one item per model).
    """
    # Map model -> fields dict (may be None on failure).
    fields_by_model = {r["model"]: (r.get("fields") or {}) for r in results}
    models = [r["model"] for r in results]

    # Header row: blank cell + one model label per column.
    weights = [2] + [3] * len(models)
    header = st.columns(weights)
    header[0].markdown("**Field**")
    for i, model in enumerate(models):
        header[i + 1].markdown(f"**{MODEL_LABELS.get(model, model)}**")

    st.divider()

    for key, label, shape in FIELDS:
        row = st.columns(weights)
        row[0].markdown(f"**{label}**")
        for i, model in enumerate(models):
            value = fields_by_model.get(model, {}).get(key)
            row[i + 1].markdown(value_to_markdown(shape, value))
        st.divider()


def _model_name(mid: str) -> str:
    return MODEL_LABELS.get(mid, mid)


def render_results(results: dict, key_prefix: str = "") -> None:
    """Render a full benchmark result: summary, chart, per-field, per-doc.

    `results` is the dict from GET /benchmark/runs/{run_id}/results.
    `key_prefix` keeps widget keys unique when shown more than once.
    """
    # --- Summary table ---
    st.markdown("**Summary**")
    summary_df = pd.DataFrame(results.get("summary", []))
    if not summary_df.empty:
        summary_df["model"] = summary_df["model"].map(_model_name)
    st.dataframe(summary_df, width="stretch", hide_index=True)

    # --- Bar chart (pick a metric) ---
    if not summary_df.empty:
        st.markdown("**Compare models**")
        metric_options = [c for c in summary_df.columns if c != "model"]
        metric = st.selectbox("Metric", metric_options, index=0, key=f"{key_prefix}metric")
        fig = px.bar(summary_df, x="model", y=metric, color="model", text_auto=True)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=metric)
        st.plotly_chart(fig, width="stretch")

    # --- Accuracy per field (pivot: field x model) ---
    per_field_df = pd.DataFrame(results.get("per_field", []))
    if not per_field_df.empty:
        st.markdown("**Accuracy per field**")
        pivot = per_field_df.pivot(index="field", columns="model", values="accuracy")
        pivot.index = [FIELD_LABELS.get(k, k) for k in pivot.index]
        pivot.columns = [_model_name(c) for c in pivot.columns]
        st.dataframe(pivot, width="stretch")

    # --- Per-document table ---
    per_doc_df = pd.DataFrame(results.get("per_document", []))
    if not per_doc_df.empty:
        st.markdown("**Per document**")
        view = per_doc_df.copy()
        view["model"] = view["model"].map(_model_name)
        st.dataframe(view, width="stretch", hide_index=True)
