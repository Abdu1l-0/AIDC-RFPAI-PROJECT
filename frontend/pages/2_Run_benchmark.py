"""Page 2 — Run benchmark (with results and the measurement method).

Tab 1: start a new run (collapsible) and its progress at the top, then the
results of a finished run (quick read, scorecard, details).
Tab 2: how every number is calculated, as described by the middleware.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

import api
import ui
from fields import MODELS

st.set_page_config(page_title="Run benchmark", page_icon="📊", layout="wide")
st.title("RFP extraction benchmark")
st.markdown(
    "Several language models read the **same RFP documents** and fill in the **same 17 fields** "
    "(deadlines, contacts, evaluation criteria, insurance, …). Each answer is compared field by "
    "field with an **answer key** written for that RFP, so we can see which model extracts most "
    "accurately, how often it makes things up, how fast it is and what it costs."
)


@st.cache_data(ttl=60, show_spinner=False)
def _catalog():
    """Models (with live labels) and the measurement method, from the middleware."""
    try:
        models = api.list_models()
    except api.ApiError:
        models = [{"id": m["id"], "label": m["label"], "enabled": True} for m in MODELS]
    try:
        method = api.get_method()
    except api.ApiError:
        method = None
    return models, method


def _elapsed(start: str | None, end: str | None = None) -> str:
    """'2m 05s' between two UTC timestamps written by the middleware."""
    if not start:
        return "—"
    fmt = "%Y-%m-%d %H:%M:%S"
    t0 = datetime.strptime(start, fmt).replace(tzinfo=UTC)
    t1 = datetime.strptime(end, fmt).replace(tzinfo=UTC) if end else datetime.now(UTC)
    seconds = max(0, int((t1 - t0).total_seconds()))
    return f"{seconds // 60}m {seconds % 60:02d}s"


models_info, method = _catalog()
model_labels = {m["id"]: m["label"] for m in models_info}

tab_run, tab_method = st.tabs(["▶  Run & results", "📐  How we measure"])

# ===========================================================================
# Tab 2 - How we measure (drawn first, so it shows even if tab 1 has errors)
# ===========================================================================

with tab_method:
    ui.render_method(method)

# ===========================================================================
# Tab 1 - Run & results
# ===========================================================================

with tab_run:
    try:
        runs = api.list_runs()
    except api.ApiError as exc:
        st.error(str(exc))
        runs = []

    # --- start a new run ----------------------------------------------------
    with st.expander("▶  Start a new benchmark run", expanded=not any(r.get("status") == "done" for r in runs)):
        # pick dataset + models and start ------------------------------------
        try:
            datasets = api.list_datasets()
        except api.ApiError as exc:
            st.error(str(exc))
            datasets = []

        left, right = st.columns([1, 2])
        with left:
            dataset = st.selectbox("Dataset", datasets,
                                   help="eval_scored: RFPs that have an answer key, so they get a score. "
                                        "eval_all: every RFP; those without an answer key are run but not scored.")
        with right:
            st.write("Models")
            chosen = []
            cols = st.columns(len(models_info))
            for col, model in zip(cols, models_info):
                ready = model.get("enabled", True)
                label = model["label"] if ready else f"{model['label']} — not ready"
                if col.checkbox(label, value=ready, disabled=not ready, key=f"bench_{model['id']}"):
                    chosen.append(model["id"])

        if st.button("Start benchmark", type="primary", disabled=not (chosen and dataset)):
            try:
                data = api.start_run(dataset, chosen)
                st.session_state["bench_run_id"] = data["run_id"]
                st.session_state["bench_running"] = True
                st.session_state.pop("bench_failed", None)
            except api.ApiError as exc:
                st.error(str(exc))

    # --- progress (polls only while running) ---------------------------------
    run_id = st.session_state.get("bench_run_id")
    running = st.session_state.get("bench_running", False)
    if st.session_state.get("bench_failed"):
        st.error(st.session_state["bench_failed"])
    if run_id and running:
        st.subheader(f"Run {run_id}")

        @st.fragment(run_every=2 if running else None)
        def show_progress() -> None:
            try:
                status = api.get_run(run_id)
            except api.ApiError as exc:
                st.error(str(exc))
                return

            done = status.get("done", 0)
            total = status.get("total", 0) or 1
            st.progress(done / total, text=f"{done} / {total} model calls finished")

            cols = st.columns(3)
            cols[0].metric("Status", status.get("status", "—"))
            cols[1].metric("Errors", status.get("errors", 0))
            cols[2].metric("Elapsed", _elapsed(status.get("created_at"), status.get("finished_at")))
            if status.get("current_doc"):
                st.caption(f"Now reading: **{status['current_doc']}** (all selected models in parallel)")

            if status.get("status") != "running":
                # Finished: stop polling by doing a full rerun with running=False.
                if status.get("status") != "done":
                    st.session_state["bench_failed"] = (f"Run {run_id} ended with status: "
                                                        f"{status.get('status')} {status.get('error', '')}")
                st.session_state["bench_running"] = False
                st.rerun()

        show_progress()

    # --- results (this run, or any earlier finished run) --------------------
    st.divider()
    st.subheader("Results")

    # label -> run_id, newest / current run first.
    options: dict[str, str] = {}
    active = st.session_state.get("bench_run_id")
    if active and not st.session_state.get("bench_running", False):
        options[f"{active} (this run)"] = active
    for r in runs:
        if r.get("status") == "done" and r["run_id"] not in options.values():
            models_txt = ", ".join(model_labels.get(m, m) for m in r.get("models", []))
            options[f"{r['run_id']} — {r['dataset']} — {models_txt} ({r['created_at']} UTC)"] = r["run_id"]

    if not options:
        st.info("No finished runs yet. Start one above.")
    else:
        picked_label = st.selectbox("Show results of", list(options.keys()))
        chosen_run = options[picked_label]

        try:
            results = api.get_results(chosen_run)
        except api.ApiError as exc:
            st.error(str(exc))
        else:
            ui.render_results(results, key_prefix=f"{chosen_run}_", method=method,
                              model_labels=model_labels)
            try:
                st.download_button(
                    "Download results CSV",
                    data=api.download_results_csv(chosen_run),
                    file_name=f"{chosen_run}_results.csv",
                    mime="text/csv",
                )
            except api.ApiError as exc:
                st.error(str(exc))
