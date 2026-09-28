# RFP AI — Model Benchmark

**SDA AI Data Center Bootcamp — Team 7, Use Case 2**

Beam Data needs to know which language model best extracts structured information
from RFP (Request for Proposal) documents, so it can advise clients on model choice.
This project deploys open-weight models on Kubernetes, runs them against a fixed set
of RFPs alongside a commercial API, and measures how accurately each one extracts a
defined list of fields.

The deliverable is the **benchmark**, not a product. The system exists to produce
trustworthy numbers.

---

## Results

Three models, ten documents (five RFPs in English plus Arabic translations of the
same five), seventeen fields per document.

| Model | Serving | Lang | Accuracy | Fabrications | Misses | Latency | Cost (5 docs) |
|---|---|---|---|---|---|---|---|
| **gpt-4o-mini** | OpenAI API | en | **0.550** | 0 | 15 | 19.6 s | $0.020 |
| **gpt-4o-mini** | OpenAI API | ar | **0.477** | 0 | 11 | 23.7 s | $0.024 |
| **Llama-3.1-8B-Instruct** | self-hosted, AWQ 4-bit | en | 0.492 | 4 | 1 | 26.3 s | — |
| **Llama-3.1-8B-Instruct** | self-hosted, AWQ 4-bit | ar | 0.295 | 2 | 6 | 32.2 s | — |
| **Qwen2.5-7B-Instruct** | self-hosted, AWQ 4-bit | en | 0.459 | 4 | 7 | 41.5 s | — |
| **Qwen2.5-7B-Instruct** | self-hosted, AWQ 4-bit | ar | 0.169 | 4 | 10 | 39.8 s | — |

Runs `20260928_121819` (Llama + OpenAI) and `20260928_123403` (Qwen + OpenAI).

### What the numbers say

**On English the three are closer than the table suggests.** gpt-4o-mini's 0.058
advantage over Llama sits close to the measured run-to-run variance (below), so
English performance is better described as comparable than as a clear win.

**The two open-weight models are indistinguishable from each other.** Llama 0.492
against Qwen 0.459 is a 0.033 gap — the same size as the noise. Neither can be
recommended over the other on this evidence.

**Arabic is where they separate.** Every model degrades, but by very different
amounts. Percentages are each model's loss relative to its own English score:

```
gpt-4o-mini   0.550 -> 0.477    -13%
Llama         0.492 -> 0.295    -40%
Qwen          0.459 -> 0.169    -63%
```

**The most important finding is how the models fail, not how often.** Over the ten
documents, gpt-4o-mini produced **zero fabrications** — and none in its second run
either, so none across twenty document-runs. Llama produced 6 and Qwen 8. The
pattern inverts on misses: gpt-4o-mini 26, Llama 7, Qwen 17. The
commercial model answers "not stated" when it doesn't know; the self-hosted models
invent a value. For a client checking an extracted RFP, a fabricated submission
deadline is more damaging than a blank field they can look up.

Both open-weight models also failed documents outright by falling into repetition
loops until they exhausted the output budget — Qwen twice, Llama once. gpt-4o-mini
never did.

### Measurement noise

gpt-4o-mini was run twice over the same ten documents:

| | |
|---|---|
| Mean absolute difference per document | **0.032** |
| Largest single-document difference | **0.132** |
| Run means | 0.514 vs 0.508 |

Aggregate figures are stable; individual document scores are not. Differences
smaller than about 0.05 between models should not be treated as real.

---

## How it works

```
PDF  ─►  text extraction  ─►  prompt assembly  ─►  model call  ─►  parse & validate
                                                                        │
                                       score against answer key  ◄──────┘
                                                   │
                                            results + CSV export
```

A **FastAPI middleware** owns the whole pipeline. A **Streamlit frontend** talks only
to the middleware and never to a model directly, so the extraction logic has one home.

### Middleware (`Benchmark Middleware/`)

| Module | Job |
|---|---|
| `config.py` | Model names, endpoints, keys, prices, per-model schema mode — all from `.env` |
| `loader.py` | PDF → text; flags pages with no extractable text |
| `prompt.py` | Builds the messages; injects answer shapes and the language instruction |
| `llm_client.py` | One call path for every model; retries, timeouts, per-request token budget |
| `validate.py` | Parses the answer into the 17-field schema; marks valid or invalid |
| `matchers.py` | Field-level comparison rules per field class |
| `scoring.py` | Outcome per field: correct / partial / miss / extraction error / fabrication |
| `aggregate.py` | Run-level summaries and CSV export |
| `rescore.py` | Re-scores a finished run from stored outputs — no model calls |
| `runner.py` | One document across models, or a whole dataset |
| `storage.py` | Documents, raw answers, results, run manifests |
| `method.py` | Serves the "How we measure" panel from the live configuration |
| `main.py` | FastAPI routes |

Every model's raw answer is written to `data/raw/` **before** parsing, so a malformed
response can be examined later without re-calling the model.

### Frontend (`frontend/`)

Three pages: try a single document across models, run the benchmark over the dataset,
and browse results. See `frontend/README.md`.

---

## Infrastructure

Both open-weight models run as vLLM containers on the team Kubernetes cluster
(namespace `team`), exposed through the `team-serving` NodePort.

**One GPU.** Kubernetes assigns a whole GPU to one pod, so only one self-hosted model
can be resident at a time. The two are therefore benchmarked in **separate passes**,
with the deployment swapped between them, while the OpenAI leg runs alongside either.
Results are stored per model and compared afterwards. This does not weaken the
comparison — and it removes GPU contention as a confound in the latency figures.

Swapping models:

```bash
kubectl -n team scale deploy/llama --replicas=0
kubectl -n team wait --for=delete pod -l app=llama --timeout=120s
kubectl -n team scale deploy/rfp-vllm --replicas=1
kubectl -n team patch svc team-serving -p '{"spec":{"selector":{"app":"rfp-vllm"}}}'
kubectl -n team rollout status deploy/rfp-vllm --timeout=600s
```

Manifests are in `k8s/`. Both models are pinned to matched serving settings —
4-bit AWQ, `--max-model-len=32768`, `--gpu-memory-utilization=0.90`, identical
batching limits — so that differences between them reflect the models, not the
configuration.

**Measured footprint** (from each engine's startup log):

| | Qwen2.5-7B-AWQ (vLLM 0.6.3) | Llama-3.1-8B-AWQ (vLLM 0.27.1) |
|---|---|---|
| Model weights | 5.20 GB | 5.39 GiB |
| Peak activation | not reported by this version | 3.37 GiB |
| CUDA graph memory | not reported | 0.13 GiB |
| **Model footprint** | ~5.2 GB + activation | **~8.9 GiB** |
| KV cache capacity | ~612,000 tokens (38,261 blocks) | 277,536 tokens |

The cluster GPU has **47.5 GiB**, so at `--gpu-memory-utilization=0.90` vLLM allocates
around 33.9 GiB to the KV cache. That allocation is a property of the card, not of the
model: Llama's own footprint — weights, peak activation and CUDA graphs — is about
8.9 GiB, comfortably inside the 16 GB budget the capstone guidance sets. On a 16 GB
card the same model would load with a correspondingly smaller cache.

Llama's KV cache holds under half as many tokens as Qwen's for the same memory, which
follows from the architecture: 8 key-value heads across 32 layers against Qwen's 4
across 28, roughly twice the cost per token. A single Arabic RFP (~30,000 tokens) is
therefore a much larger share of Llama's cache than of Qwen's, which limits how many
documents either model can serve concurrently.

### Monitoring

Prometheus and Grafana run in the same namespace. Prometheus scrapes vLLM's
`/metrics` endpoint on both serving Services (`k8s/prometheus-scrape.yaml`, job
`serving`, 15-second interval); Grafana is exposed on NodePort 30300.

The **Team service** dashboard (`k8s/grafana/team-service-dashboard.json`, importable
into Grafana) watches the self-hosted models during benchmark and load-test runs:

| Panel | Query | What it shows |
|---|---|---|
| Traffic Over Time | `sum(rate(vllm:request_success_total[5m])) * 60` | Successful completions per minute, 5-minute rolling rate |
| Completed Requests per Minute | same, as a single stat | Current engine throughput |
| Time to First Token | `histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))` | p95 prefill latency — how long before a caller sees the first token |
| p95 TTFT gauge | same, scoped to `job="serving"`, scaled 0–1 s | The same figure against the **1.0 s p95 SLO** |

Time to first token is the metric that matters for this workload: the prompts are
whole RFPs, so prefill dominates, and the gauge's 0–1 second range is the service
objective the deployment is held to. The dashboard refreshes every 10 seconds over a
24-hour window.


### Load testing

`locustfile.py` drives concurrent requests at the serving endpoint; results are in
`reports/`.

> **Caveat:** the `gpt4o_mini-*` report recorded 107 requests and 107 failures — all
> HTTP 429 from a project spend limit — so it contains no usable latency data. The
> load-test payload is also a single short prompt, not a full RFP, so those figures
> describe endpoint behaviour under concurrency rather than real extraction throughput.

---

## Running it

### Middleware

```powershell
cd "Benchmark Middleware"
python -m venv ../venv
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then fill in endpoints and keys
uvicorn main:app --reload
```

`http://localhost:8000/docs` for the API, `/health` for a quick check.

### Frontend

```powershell
cd frontend
copy .env.example .env      # set USE_FAKE_DATA=false
uv sync
uv run streamlit run app.py
```

### The evaluation set

`Benchmark Middleware/eval_set/` holds `documents/` (the RFP PDFs, **not committed**)
and `ground_truth/` (the answer keys, committed). A document is scored when a JSON
file in `ground_truth/` matches its filename. Arabic documents carry an `_AR` suffix.

The RFP PDFs stay on the team drive. Copy them into `eval_set/documents/` before
running a benchmark.

---

## Method

**Field set.** Seventeen fields from Appendix A of the capstone guidance, defined in
`field_set.json` with a type, cardinality, and whether absence is legal.

**Prompting.** Every model receives the same base instructions, field list and
document text. The only difference is how the answer *shape* reaches it: OpenAI
enforces a JSON schema at the endpoint, while the vLLM endpoints cannot, so their
prompts carry the shape explicitly. The mode used is recorded with every result.

**Arabic.** The prompt detects Arabic in the document and instructs the model to
answer, and to quote, in the document's own script. Without this the models answered
English and quoted invented English sentences — and, in one measured case, returned a
fabricated date. Requiring source-language quotes corrected the extraction as well as
the language.

**Token budget.** A self-hosted model's context window is shared between prompt and
answer. Arabic costs roughly 1.6× more tokens than English for the same RFP
(measured: ~2.45 characters per token against ~4.0), so a fixed output allowance that
fits every English document overflows on the long Arabic ones. Each request therefore
asks the endpoint how long the prompt is and requests whatever the window has left,
capped at 4000 tokens and floored at 1200.

**Scoring.** Each field gets an outcome: `correct`, `partial`, `miss` (the key has a
value, the model abstained), `correct_abstention`, `extraction_error` (wrong value,
but the quote is real) or `fabrication` (wrong value with no grounding). Separating
the last two matters: a misread is a different failure from an invention.

---

## Limitations

**Small sample.** Five source RFPs and their Arabic translations, one benchmark pass
per self-hosted model (gpt-4o-mini ran in both passes). With measured variance of
0.032, differences below ~0.05 are not meaningful.

**The answer keys were drafted with a model and reviewed by a person.** They are not
double-annotated, and inter-annotator agreement has not been computed.

**One field measured nothing.** `pricing_structure` scored 0.00 for all three models
on all ten documents. Its key value describes the *submission format* in a slot named
`pricing_model`, which every model fills with a pricing type. The field is
mis-specified; its scores should be disregarded.

**Scoring is strict on multi-part fields.** Free-text and list fields are compared by
word overlap, and answers below a similarity threshold receive zero rather than
partial credit. Recomputing over the stored per-field records with both relaxations
raises gpt-4o-mini from 0.522 to 0.597, Llama from 0.405 to 0.489 and Qwen from 0.314
to 0.410 — **leaving the ranking and the gaps essentially unchanged.** (Those baselines
are per-field means over `data/results/`, so they differ slightly from the run-level
figures in the table above.) The conclusions do not depend on this choice.

**Arabic documents are machine-translated**, not native Arabic RFPs, and PDF text
extraction degrades Arabic — dates and numbers get joined to adjacent words and split
across lines. Both affect all models equally, so the comparison holds, but the
absolute Arabic scores are not a measure of native Arabic performance.

**Engine versions differ.** The two self-hosted models run on different vLLM releases
(v0.6.3 and v0.27.1), because the older release predates Llama 3.1 AWQ support. Part
of any latency difference between them is runtime, not model.

**Latency across serving types is not comparable.** The self-hosted models share one
GPU on the team cluster; gpt-4o-mini runs on OpenAI's infrastructure.

---

## Repository layout

| Path | Contents |
|---|---|
| `Benchmark Middleware/` | FastAPI service: extraction, scoring, benchmark runner |
| `frontend/` | Streamlit app |
| `k8s/` | Deployment manifests for both models, Prometheus, and the Grafana dashboard |
| `ground_truth/` | Annotation guideline, template, and inter-annotator agreement script |
| `reports/` | Load-test output |
| `locustfile.py` | Load-test definition |
| `harness/`, `runs/`, `config/` | Earlier CLI pipeline, superseded by the middleware; kept for reference |

> **Note:** `config/field_set@1.0.json` and `Benchmark Middleware/field_set.json` are
> separate copies of the field specification. The middleware uses its own.
