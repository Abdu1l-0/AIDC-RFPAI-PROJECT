"""Shared display helpers used by more than one page."""

from __future__ import annotations

import json

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


def cell_to_markdown(cell) -> str:
    """Turn one field cell into markdown.

    The middleware already formats the value into a "display" string, so the
    frontend never has to know a field's internal shape.
    """
    if not isinstance(cell, dict):
        return "_Not found_"

    if cell.get("status") != "found":
        return "_Not found_"

    text = cell.get("display")
    if not text:
        # middleware did not send a display string: fall back to the raw value
        value = cell.get("value")
        if value is None or value == [] or value == {}:
            return "_Not found_"
        text = str(value)

    return str(text)


def render_model_status(result: dict, model_labels: dict | None = None) -> None:
    """One compact status block for a single model result."""
    labels = {**MODEL_LABELS, **(model_labels or {})}
    st.markdown(f"**{labels.get(result['model'], result['model'])}**")
    st.write(status_badge(result.get("status", "")))

    cols = st.columns(3)
    latency = result.get("latency_s")
    cols[0].metric("Latency (s)", f"{latency:.1f}" if latency is not None else "—")
    cols[1].metric("In tokens", result.get("input_tokens") or "—")
    cols[2].metric("Out tokens", result.get("output_tokens") or "—")

    if result.get("error"):
        st.error(result["error"])

    # How the call was made and how it ended - useful when an answer looks wrong.
    details = []
    if result.get("schema_mode"):
        details.append(f"schema mode: `{result['schema_mode']}`")
    if result.get("finish_reason"):
        details.append(f"finish: `{result['finish_reason']}`")
    if result.get("valid_json") is not None:
        details.append(f"valid JSON: {'yes' if result['valid_json'] else 'no'}")
    if details:
        st.caption(" · ".join(details))

    # The model's answer exactly as it came back, before any parsing.
    raw = result.get("raw_text")
    if raw:
        with st.expander(f"Raw model response ({len(raw):,} chars)"):
            try:
                st.code(json.dumps(json.loads(raw), indent=2, ensure_ascii=False), language="json")
            except (ValueError, TypeError):
                st.code(raw, language="text")  # not clean JSON: show it as is


def render_field_grid(results: list[dict], field_list: list[dict] | None = None,
                      model_labels: dict | None = None, show_evidence: bool = True) -> None:
    """Show one row per field, one column per model, so answers line up.

    `results`    is the list from POST /extract (one item per model).
    `field_list` is [{key, label}, ...] from GET /fields; falls back to the
                 built-in list when the middleware cannot be reached.
    """
    if field_list:
        rows = [(f["key"], f.get("label", f["key"])) for f in field_list]
    else:
        rows = [(key, label) for key, label, _shape in FIELDS]

    labels = {**MODEL_LABELS, **(model_labels or {})}

    # Map model -> fields dict (may be None on failure).
    fields_by_model = {r["model"]: (r.get("fields") or {}) for r in results}
    models = [r["model"] for r in results]

    # Header row: blank cell + one model label per column.
    weights = [2] + [3] * len(models)
    header = st.columns(weights)
    header[0].markdown("**Field**")
    for i, model in enumerate(models):
        header[i + 1].markdown(f"**{labels.get(model, model)}**")

    st.divider()

    for key, label in rows:
        row = st.columns(weights)
        row[0].markdown(f"**{label}**")
        for i, model in enumerate(models):
            cell = fields_by_model.get(model, {}).get(key)
            with row[i + 1]:
                st.markdown(cell_to_markdown(cell))
                evidence = (cell or {}).get("evidence") if isinstance(cell, dict) else None
                if show_evidence and evidence:
                    with st.expander(f"Evidence ({len(evidence)})"):
                        for ev in evidence:
                            page = ev.get("page")
                            page_text = f" _(page {page})_" if page else ""
                            st.markdown(f"> {ev.get('quote', '')}{page_text}")
        st.divider()


def _model_name(mid: str, labels: dict | None = None) -> str:
    return (labels or {}).get(mid) or MODEL_LABELS.get(mid, mid)


# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------

# summary key -> (column label, kind). kind picks how the column is shown.
_SUMMARY_COLUMNS = {
    "accuracy": ("Accuracy", "score"),
    "extraction_accuracy": ("Extraction accuracy", "score"),
    "completeness": ("Completeness", "score"),
    "relevance": ("Relevance", "score"),
    "unsupported_rate": ("Unsupported answers", "score"),
    "instruction_following": ("Instruction following", "score"),
    "hallucination_rate": ("Hallucination rate", "score"),
    "miss_rate": ("Miss rate", "score"),
    "grounding_rate": ("Grounding rate", "score"),
    "valid_json_rate": ("Valid JSON", "score"),
    "avg_latency_s": ("Avg latency", "seconds"),
    "avg_input_tokens": ("Avg input tokens", "int"),
    "avg_output_tokens": ("Avg output tokens", "int"),
    "total_cost_usd": ("Total cost", "usd"),
    "errors": ("Errors", "int"),
}

# per-document key -> (column label, kind)
_DOC_COLUMNS = {
    "doc_id": ("Document", "text"),
    "model": ("Model", "text"),
    "status": ("Status", "text"),
    "accuracy": ("Accuracy", "score"),
    "hallucinations": ("Made up", "int"),
    "misses": ("Missed", "int"),
    "latency_s": ("Latency", "seconds"),
    "input_tokens": ("Input tokens", "int"),
    "output_tokens": ("Output tokens", "int"),
    "cost_usd": ("Cost", "usd"),
}

# outcome -> (label, colour). Dict order = stacking order in the chart.
_OUTCOMES = {
    "correct": ("Correct", "#2e7d32"),
    "partial": ("Partly correct", "#9ccc65"),
    "correct_abstention": ("Correctly 'not stated'", "#42a5f5"),
    "extraction_error": ("Wrong value", "#ffa726"),
    "miss": ("Missed", "#bdbdbd"),
    "fabrication": ("Made up (hallucination)", "#e53935"),
    "failed_call": ("Call failed", "#6d4c41"),
}


def _metric_help(method: dict | None) -> dict:
    """summary key -> how it is calculated, as sent by the middleware."""
    if not method:
        return {}
    return {m["key"]: m["how"] + (" Higher is better." if m["better"] == "higher" else " Lower is better.")
            for m in method.get("metrics", [])}


def _column(label: str, kind: str, help_text: str | None):
    if kind == "score":
        return st.column_config.ProgressColumn(label, help=help_text, min_value=0.0,
                                               max_value=1.0, format="%.3f")
    if kind == "seconds":
        return st.column_config.NumberColumn(label, help=help_text, format="%.1f s")
    if kind == "usd":
        return st.column_config.NumberColumn(label, help=help_text, format="$%.4f")
    if kind == "int":
        return st.column_config.NumberColumn(label, help=help_text, format="%d")
    return st.column_config.TextColumn(label, help=help_text)


def _render_setup(summary: list[dict], labels: dict) -> None:
    """How the answer format reached each model - part of a fair reading."""
    for s in summary:
        variant = s.get("prompt_variant")
        if variant == "schema_enforced":
            how = "answer format **enforced** by the endpoint's JSON schema"
        elif variant == "shapes_in_prompt":
            how = "answer format **written into the prompt** (this endpoint cannot enforce it)"
        else:
            how = "answer format delivery was not recorded for this run"
        st.caption(f"**{_model_name(s['model'], labels)}**: {how}")


def _fmt(value, kind: str = "num") -> str:
    if value is None:
        return "—"
    if kind == "usd":
        return f"${value:.4f}"
    if kind == "seconds":
        return f"{value:.1f} s"
    if kind == "int":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _render_quick_read(summary: list[dict], labels: dict) -> None:
    """Raw numbers, one column per model: what was run and where the answers ended up."""
    def fmt_delivery(v):
        return {"schema_enforced": "enforced by endpoint",
                "shapes_in_prompt": "written into prompt"}.get(v, "—")

    rows = [
        ("Documents run", lambda s: _fmt(s.get("documents"), "int")),
        ("Failed calls", lambda s: _fmt(s.get("errors"), "int")),
        ("Field answers scored", lambda s: _fmt(sum((s.get("outcome_counts") or {}).values()) or None, "int")),
        *[(f"  {label}", (lambda key: lambda s: _fmt((s.get("outcome_counts") or {}).get(key, 0), "int"))(key))
          for key, (label, _c) in _OUTCOMES.items() if key != "failed_call"],
        ("Avg latency per document", lambda s: _fmt(s.get("avg_latency_s"), "seconds")),
        ("Avg input tokens", lambda s: _fmt(s.get("avg_input_tokens"), "int")),
        ("Avg output tokens", lambda s: _fmt(s.get("avg_output_tokens"), "int")),
        ("Total cost", lambda s: _fmt(s.get("total_cost_usd"), "usd")),
        ("Answer format", lambda s: fmt_delivery(s.get("prompt_variant"))),
        ("Document languages", lambda s: ", ".join(s.get("languages") or []) or "—"),
    ]
    table = pd.DataFrame({"": [name for name, _ in rows],
                          **{_model_name(s["model"], labels): [fn(s) for _, fn in rows] for s in summary}})
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(rows) + 1) + 3)


_STATUS = {"measured": "✅ measured", "proxy": "🟡 proxy", "not_measured": "⚪ not measured"}


def _render_scorecard(summary: list[dict], method: dict | None, labels: dict) -> None:
    """The 9 model-quality dimensions from the RFP, one row each, one column per model."""
    dims = (method or {}).get("dimensions")
    if not dims:
        st.caption("The scorecard needs the method description from the middleware.")
        return
    metric_info = {m["key"]: m for m in method.get("metrics", [])}

    rows = []
    for d in dims:
        info = metric_info.get(d["metric"]) if d["metric"] else None
        icon, _, status = _STATUS.get(d["status"], d["status"]).partition(" ")
        row = {"Dimension": f"{icon} **{d['name']}**"}
        for s in summary:
            row[_model_name(s["model"], labels)] = _fmt(s.get(d["metric"])) if d["metric"] else "—"
        how = (f"**{info['name']}** ({'↑ higher' if info['better'] == 'higher' else '↓ lower'} is better). "
               if info else f"*{status}.* ")
        if d["status"] == "proxy":
            how = f"*Proxy:* {how}"
        row["How it is measured"] = how + d["note"]
        rows.append(row)

    # a markdown table, so the text wraps instead of being cut off
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    st.markdown("\n".join(lines))
    st.caption("✅ measured · 🟡 proxy (closest available signal) · ⚪ not measured in this benchmark")
    if any(d["metric"] and d["metric"] not in summary[0] for d in dims):
        st.caption("Some values are missing because this run was scored before those metrics existed. "
                   "Run `python rescore.py <run_id>` in the middleware folder to fill them in.")


def _render_outcomes(summary: list[dict], labels: dict) -> None:
    rows = []
    for s in summary:
        counts = s.get("outcome_counts") or {}
        for key, (label, _colour) in _OUTCOMES.items():
            if counts.get(key):
                rows.append({"model": _model_name(s["model"], labels), "outcome": label,
                             "field answers": counts[key]})
    if not rows:
        st.caption("Outcome breakdown is not available for runs made before this view existed.")
        return
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="field answers", y="model", color="outcome", orientation="h",
                 text="field answers",
                 color_discrete_map={label: colour for label, colour in _OUTCOMES.values()},
                 category_orders={"outcome": [label for label, _ in _OUTCOMES.values()]})
    fig.update_layout(barmode="stack", height=140 + 55 * len(summary), yaxis_title="",
                      legend_title_text="", legend={"orientation": "h", "y": -0.35},
                      margin={"t": 10, "b": 10})
    st.plotly_chart(fig, width="stretch")


def _render_field_heatmap(per_field: list[dict], method: dict | None, labels: dict) -> None:
    df = pd.DataFrame(per_field)
    if df.empty:
        return
    pivot = df.pivot(index="field", columns="model", values="accuracy")
    field_info = {f["key"]: f for f in (method or {}).get("fields", [])}
    order = [f["key"] for f in (method or {}).get("fields", [])] or [k for k, _l, _s in FIELDS]
    pivot = pivot.reindex([k for k in order if k in pivot.index])

    def row_label(key: str) -> str:
        info = field_info.get(key)
        name = info["label"] if info else FIELD_LABELS.get(key, key)
        return f"{name}  ({info['kind']})" if info else name

    fig = px.imshow(pivot.values, x=[_model_name(c, labels) for c in pivot.columns],
                    y=[row_label(k) for k in pivot.index], zmin=0, zmax=1,
                    color_continuous_scale="RdYlGn", text_auto=".2f", aspect="auto")
    fig.update_layout(height=60 + 34 * len(pivot.index), margin={"t": 10, "b": 10},
                      coloraxis_colorbar={"title": "score"})
    fig.update_xaxes(side="top", title="")
    st.plotly_chart(fig, width="stretch")


def render_results(results: dict, key_prefix: str = "", method: dict | None = None,
                   model_labels: dict | None = None) -> None:
    """Render a full benchmark result: headline, setup, outcomes, summary,
    comparison chart, per-field heatmap and per-document table.

    `method` (GET /benchmark/method) supplies the 'how it is calculated' text
    shown when hovering a metric. Without it the numbers still render.
    """
    labels = model_labels or {}
    help_map = _metric_help(method)
    summary = results.get("summary", [])
    if not summary:
        st.info("This run has no results.")
        return

    st.markdown("#### Quick read")
    _render_quick_read(summary, labels)

    st.markdown("#### Scorecard — the 9 quality dimensions")
    _render_scorecard(summary, method, labels)
    _render_setup(summary, labels)
    st.caption("Scores run from 0 to 1. With this few documents, differences below about 0.05 between "
               "models are within run-to-run noise. How every number is calculated is in the "
               "**How we measure** tab.")

    st.markdown("#### Where every field answer ended up")
    st.caption("Each model answered 17 fields per document. Every answer ends in exactly one outcome.")
    _render_outcomes(summary, labels)

    st.markdown("#### All metrics")
    summary_df = pd.DataFrame(summary)
    summary_df["model"] = summary_df["model"].map(lambda m: _model_name(m, labels))
    shown = ["model"] + [k for k in _SUMMARY_COLUMNS if k in summary_df.columns]
    config = {"model": st.column_config.TextColumn("Model")}
    config.update({k: _column(label, kind, help_map.get(k)) for k, (label, kind) in _SUMMARY_COLUMNS.items()})
    st.dataframe(summary_df[shown], column_config=config, hide_index=True, width="stretch")

    st.markdown("#### Compare one metric")
    options = [k for k in _SUMMARY_COLUMNS if k in summary_df.columns]
    metric = st.selectbox("Metric", options, index=0, key=f"{key_prefix}metric",
                          format_func=lambda k: _SUMMARY_COLUMNS[k][0])
    fig = px.bar(summary_df, x="model", y=metric, color="model", text_auto=True)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=_SUMMARY_COLUMNS[metric][0],
                      margin={"t": 10})
    st.plotly_chart(fig, width="stretch")
    if help_map.get(metric):
        st.info(f"**How it is calculated:** {help_map[metric]}")

    if results.get("per_field"):
        st.markdown("#### Accuracy per field")
        st.caption("Average score of each field over all documents: green = right, red = wrong or missing. "
                   "The comparison rule for each kind of field is in **How we measure**.")
        _render_field_heatmap(results["per_field"], method, labels)

    per_doc = pd.DataFrame(results.get("per_document", []))
    if not per_doc.empty:
        st.markdown("#### Per document")
        per_doc["model"] = per_doc["model"].map(lambda m: _model_name(m, labels))
        doc_help = (method or {}).get("per_document", {})
        shown = [k for k in _DOC_COLUMNS if k in per_doc.columns]
        config = {k: _column(label, kind, doc_help.get(k)) for k, (label, kind) in _DOC_COLUMNS.items()}
        st.dataframe(per_doc[shown], column_config=config, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# How we measure
# ---------------------------------------------------------------------------

def render_method(method: dict | None) -> None:
    """The measurement method, as described by the middleware from its live code."""
    if not method:
        st.info("The method description comes from the middleware. It is not available on fake data.")
        return

    st.markdown(method["summary"])

    st.markdown("#### 1. What happens in a run")
    st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(method["steps"], start=1)))

    st.markdown("#### 2. How each model is called")
    s = method["settings"]
    st.markdown(
        f"Temperature **{s['temperature']}** (no randomness), at most **{s['max_output_tokens']:,}** "
        f"output tokens, timeout **{s['timeout_s']} s**, **{s['retries']}** retry on server or "
        f"connection errors (timeouts are not retried)."
    )
    models = [m for m in method["models"] if m["enabled"]]
    st.dataframe(pd.DataFrame([{
        "Model": m["label"],
        "How the answer format is delivered": m["format_delivery"],
        "Price per 1M tokens (in / out)": f"${m['price_in_per_1m']:.2f} / ${m['price_out_per_1m']:.2f}",
    } for m in models]), hide_index=True, width="stretch")
    st.caption("Every model gets the same instructions, field descriptions and document text. The only "
               "difference is how the answer format reaches it, because not every endpoint can enforce "
               "a JSON schema.")

    st.markdown("#### 3. How each field answer is judged")
    st.dataframe(pd.DataFrame([{
        "Answer key says": o["answer_key"], "Model says": o["model"],
        "Outcome": o["outcome"], "Field score": o["score"],
    } for o in method["outcomes"]]), hide_index=True, width="stretch")
    t = method["thresholds"]
    st.caption(f"Similarity is a number from 0 to 1. At {t['correct_at']:.2f} or above the answer counts "
               f"as correct; from {t['partial_at']:.2f} it earns partial credit equal to the similarity.")

    st.markdown("#### 4. How two values are compared")
    by_kind: dict[str, dict] = {}
    for f in method["fields"]:
        entry = by_kind.setdefault(f["kind"], {"rule": f["rule"], "fields": []})
        entry["fields"].append(f["label"])
    for kind, entry in by_kind.items():
        with st.container(border=True):
            st.markdown(f"**{kind.capitalize()} fields** — {entry['rule']}")
            st.caption(", ".join(entry["fields"]))
    st.markdown("\n".join(f"- {rule}" for rule in method["text_rules"]))

    st.markdown("#### 5. What each number means")
    st.dataframe(pd.DataFrame([{
        "Metric": m["name"], "How it is calculated": m["how"],
        "Better when": "higher ↑" if m["better"] == "higher" else "lower ↓",
    } for m in method["metrics"]]), hide_index=True, width="stretch",
        column_config={"How it is calculated": st.column_config.TextColumn(width="large")})

    if method.get("dimensions"):
        st.markdown("#### 6. The 9 quality dimensions and what measures them")
        st.dataframe(pd.DataFrame([{
            "Dimension": d["name"], "The question": d["asks"],
            "Metric": next((m["name"] for m in method["metrics"] if m["key"] == d["metric"]), "—"),
            "Status": _STATUS.get(d["status"], d["status"]), "Notes": d["note"],
        } for d in method["dimensions"]]), hide_index=True, width="stretch",
            column_config={"Notes": st.column_config.TextColumn(width="large")})

    st.markdown(f"#### {7 if method.get('dimensions') else 6}. Limits of this benchmark")
    st.warning("\n".join(f"- {limit}" for limit in method["limits"]))
