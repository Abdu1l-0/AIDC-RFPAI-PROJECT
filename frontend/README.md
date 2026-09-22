# Frontend (Streamlit)

The user-facing app. It only uploads PDFs and shows results. All logic lives in
the middleware. Every call to the middleware goes through `api.py`.

## Run

```powershell
uv run streamlit run app.py
```

## Settings (`.env`)

- `MIDDLEWARE_URL` — where the middleware runs (default `http://localhost:8000`).
- `USE_FAKE_DATA` — `true` to use fake data and never call the middleware.
  Set to `false` when the middleware is ready. Nothing else needs to change.

## Files

| File | Job |
|---|---|
| `app.py` | Home page + middleware health check |
| `api.py` | All middleware calls; fake-data switch; raises `ApiError` |
| `fake_data.py` | Fake responses that match the API contract |
| `fields.py` | The 17 field keys + labels, and the 3 model names |
| `ui.py` | Shared display helpers |
| `pages/1_Try_a_document.py` | Upload one PDF, compare models side by side |
| `pages/2_Run_benchmark.py` | Run the dataset, poll progress |
| `pages/3_Results.py` | Summary, charts, per-field/per-doc tables, CSV |
