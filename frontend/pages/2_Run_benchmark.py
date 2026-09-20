"""Page 2 — Run benchmark (with results).

Pick a dataset and models, start a full run, and watch the progress. When the
run finishes, its results (summary, chart, per-field, per-document, CSV) show
on the same page. You can also pick an earlier finished run to view.
"""

from __future__ import annotations

import streamlit as st

import api
import ui
from fields import MODELS

st.set_page_config(page_title="Run benchmark", page_icon="📊", layout="wide")
st.title("Run benchmark")

# ---------------------------------------------------------------------------
# Pick dataset + models and start
# ---------------------------------------------------------------------------

try:
    datasets = api.list_datasets()
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

dataset = st.selectbox("Dataset", datasets)

try:
    models_info = api.list_models()
except api.ApiError:
    models_info = [{"id": m["id"], "label": m["label"], "enabled": True} for m in MODELS]

st.write("Models")
chosen = []
cols = st.columns(len(models_info))
for col, model in zip(cols, models_info):
    ready = model.get("enabled", True)
    label = model["label"] if ready else f"{model['label']} — not ready"
    if col.checkbox(label, value=ready, disabled=not ready, key=f"bench_{model['id']}"):
        chosen.append(model["id"])

if st.button("Start", type="primary", disabled=not chosen):
    try:
        data = api.start_run(dataset, chosen)
        st.session_state["bench_run_id"] = data["run_id"]
        st.session_state["bench_running"] = True
    except api.ApiError as exc:
        st.error(str(exc))

# ---------------------------------------------------------------------------
# Progress (polls only while running)
# ---------------------------------------------------------------------------

run_id = st.session_state.get("bench_run_id")
if run_id:
    st.divider()
    st.subheader(f"Run {run_id}")

    running = st.session_state.get("bench_running", False)

    @st.fragment(run_every=2 if running else None)
    def show_progress() -> None:
        try:
            status = api.get_run(run_id)
        except api.ApiError as exc:
            st.error(str(exc))
            return

        done = status.get("done", 0)
        total = status.get("total", 0) or 1
        st.progress(done / total, text=f"{done} / {total} documents")

        cols = st.columns(3)
        cols[0].metric("Status", status.get("status", "—"))
        cols[1].metric("Errors", status.get("errors", 0))
        cols[2].metric("Current doc", status.get("current_doc") or "—")

        if status.get("status") != "running":
            # Finished: stop polling by doing a full rerun with running=False.
            if st.session_state.get("bench_running"):
                st.session_state["bench_running"] = False
                st.rerun()
            if status.get("status") != "done":
                st.error(f"Run ended with status: {status.get('status')}")

    show_progress()

# ---------------------------------------------------------------------------
# Results (this run, or any earlier finished run)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Results")

try:
    runs = api.list_runs()
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

# label -> run_id, newest / current run first.
options: dict[str, str] = {}

active = st.session_state.get("bench_run_id")
active_done = active and not st.session_state.get("bench_running", False)
if active_done:
    options[f"{active} (this run)"] = active

for r in runs:
    if r.get("status") == "done" and r["run_id"] not in options.values():
        options[f"{r['run_id']} — {r['dataset']} ({r['created_at']})"] = r["run_id"]

if not options:
    st.info("No finished runs yet. Start one above.")
else:
    picked_label = st.selectbox("Run", list(options.keys()))
    chosen_run = options[picked_label]

    try:
        results = api.get_results(chosen_run)
    except api.ApiError as exc:
        st.error(str(exc))
    else:
        ui.render_results(results, key_prefix=f"{chosen_run}_")

        try:
            csv_bytes = api.download_results_csv(chosen_run)
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name=f"{chosen_run}_results.csv",
                mime="text/csv",
            )
        except api.ApiError as exc:
            st.error(str(exc))
