"""How the benchmark measures - in plain words, built from the live code values.

GET /benchmark/method returns this, and the frontend shows it next to the
numbers. Thresholds, prices and each model's setup are read from the code that
actually uses them, so the explanation cannot drift from what is computed.
"""

from __future__ import annotations

from config import settings
from fields import FIELDS
from llm_client import ENFORCING_MODES, MAX_RETRIES, MAX_TOKENS, MODEL_CONFIG, TIMEOUT_S
from matchers import CORRECT_AT, PARTIAL_AT
from prompt import VARIANT_ENFORCED, VARIANT_SHAPES
import storage


def _pct(x: float) -> str:
    return f"{x:.2f}"


def _comparison_rule(fdef: dict) -> tuple[str, str]:
    """(kind, rule) for one field, matching what matchers.compare_values does."""
    ftype = fdef.get("type")
    if ftype == "array":
        return ("list", "Each answer-key item is paired with the best-matching model item "
                        "(order does not matter). Missing items lower recall, extra items lower "
                        "precision; the field score is their F1.")
    if ftype == "object" and fdef.get("field_class") == "free_text":
        return ("free text", "Word overlap (F1) between the model's text and the answer key, "
                             "after lowercasing, removing punctuation and common words.")
    if ftype == "object":
        return ("structured", "Compared property by property, only over the properties the "
                              "answer key fills in; the field score is their average. Dates become "
                              "YYYY-MM-DD, times HH:MM (seconds dropped), numbers are compared as numbers.")
    return ("number", "Exact match after turning both sides into numbers.")


# (summary key, display name, how it is calculated, better when)
METRICS = [
    ("accuracy", "Accuracy",
     "For each document: the average score of its 17 fields. Then the average over documents. "
     "A field scores 1 when correct (or correctly left empty), its similarity when partly right, "
     "and 0 when wrong, missed or made up. A call that failed or could not be read counts as 0.", "higher"),
    ("extraction_accuracy", "Extraction accuracy",
     "Like accuracy, but only over fields the RFP really states (the answer key has a value). "
     "Shows how well the model pulls out information that is there.", "higher"),
    ("completeness", "Completeness",
     "Of the fields the RFP states, how much the model delivered. A list field counts the share of "
     "answer-key items the model's list covers (matched items, weighted by similarity); any other "
     "field counts 1 when answered and 0 when the model said 'not stated'. Averaged per document, "
     "then over documents. A failed call counts as 0.", "higher"),
    ("relevance", "Relevance",
     "Of the answers the model gave, how much is on target. A list field counts the share of the "
     "model's items that match an answer-key item (extra, off-topic items lower it); any other field "
     f"counts 1 when it at least partly matches the answer key (similarity >= {PARTIAL_AT:.2f}), "
     "else 0. A value where the RFP says nothing counts 0. Only readable answers count.", "higher"),
    ("unsupported_rate", "Unsupported answers",
     "Of the answers the model gave, the share not backed by the document: made up where the RFP "
     "says nothing, no evidence quote, or a quote that does not appear in the document. "
     "Wider than the hallucination rate, which only sees the few fields the RFP leaves empty. "
     "Only readable answers count.", "lower"),
    ("instruction_following", "Instruction following",
     "Share of the 17 fields filled the way the prompt asks: the field is present; 'found' comes "
     "with a value of the right shape (list, object, number) and at least one evidence quote; "
     "'not stated' comes with no value. A failed or unreadable call counts as 0.", "higher"),
    ("hallucination_rate", "Hallucination rate",
     "Of the fields the RFP does NOT state (the answer key says 'not stated'), the share where the "
     "model still gave a value. Averaged over documents that have such fields; only readable answers count.", "lower"),
    ("miss_rate", "Miss rate",
     "Of the fields the RFP does state, the share where the model said 'not stated'. "
     "Only readable answers count.", "lower"),
    ("grounding_rate", "Grounding rate",
     "Of the answers that come with evidence quotes, the share where every quote appears word for "
     "word in the document (spacing and capital letters ignored). Low grounding means reworded or "
     "invented quotes.", "higher"),
    ("valid_json_rate", "Valid JSON rate",
     "Share of calls whose answer could be read as the 17-field format. A call that cannot be "
     "read counts as 0 in accuracy.", "higher"),
    ("avg_latency_s", "Avg latency (s)",
     "Average seconds from sending the request to receiving the complete answer.", "lower"),
    ("avg_input_tokens", "Avg input tokens",
     "Average prompt size in tokens, as reported by the model endpoint.", "lower"),
    ("avg_output_tokens", "Avg output tokens",
     "Average answer size in tokens, as reported by the model endpoint.", "lower"),
    ("total_cost_usd", "Total cost (USD)",
     "Sum over all calls of input tokens x input price + output tokens x output price, per 1M "
     "tokens. Self-hosted models count as $0 (GPU cost is not included).", "lower"),
    ("errors", "Errors",
     "Calls that failed: timeout, HTTP error, or an answer cut off at the token limit.", "lower"),
]

# The 9 model-quality dimensions the RFP asks for (section 5), and which
# summary metric answers each. status: measured / proxy / not_measured.
DIMENSIONS = [
    {"name": "Correctness", "asks": "Is the answer/task result correct?",
     "metric": "accuracy", "status": "measured",
     "note": "Every field scored against the answer key; partial credit for near matches."},
    {"name": "Completeness", "asks": "Did the model produce all required information?",
     "metric": "completeness", "status": "measured",
     "note": "Stated fields answered; for lists, share of answer-key items covered."},
    {"name": "Relevance", "asks": "Is the response relevant to the task?",
     "metric": "relevance", "status": "measured",
     "note": "Share of returned answers / list items that match the answer key."},
    {"name": "Hallucination", "asks": "Does the output contain unsupported information?",
     "metric": "unsupported_rate", "status": "measured",
     "note": "Made-up values, missing quotes or quotes not in the document."},
    {"name": "Instruction following", "asks": "Does the model follow the requested format and instructions?",
     "metric": "instruction_following", "status": "measured",
     "note": "Every field present, right value shape, evidence quote given when 'found'."},
    {"name": "Arabic quality", "asks": "How well does the model perform on Arabic tasks?",
     "metric": None, "status": "not_measured",
     "note": "The evaluation set has no Arabic RFP with an answer key yet. The pipeline "
             "detects the language per document, so an Arabic RFP plus answer key is all it needs."},
    {"name": "Structured output", "asks": "Does the model reliably return the required JSON/structure?",
     "metric": "valid_json_rate", "status": "measured",
     "note": "Answer parses as JSON with exactly the 17 fields in {status, value, evidence} form."},
    {"name": "Retrieval quality", "asks": "For RAG tasks, does the system retrieve useful supporting information?",
     "metric": "grounding_rate", "status": "proxy",
     "note": "No RAG step: the whole RFP is in the prompt. Closest signal: the model's evidence "
             "quotes really appear in the document."},
    {"name": "Human preference", "asks": "How do reviewers compare outputs?",
     "metric": None, "status": "not_measured",
     "note": "No reviewer comparison was run. The per-document results can be compared side "
             "by side on the Try a document page."},
]

PER_DOCUMENT = {
    "accuracy": "The average score of this document's 17 fields.",
    "hallucinations": "Number of fields the RFP does not state but the model filled in.",
    "misses": "Number of fields the RFP states but the model said 'not stated'.",
    "latency_s": "Seconds for this call.",
    "cost_usd": "Cost of this call in USD.",
}


def build_method() -> dict:
    fields = []
    for name, fdef in FIELDS.items():
        kind, rule = _comparison_rule(fdef)
        fields.append({"key": name, "label": fdef["name"], "kind": kind, "rule": rule})

    models = []
    for key, cfg in MODEL_CONFIG.items():
        mode = cfg.get("schema_mode")
        enforced = mode in ENFORCING_MODES
        models.append({
            "id": key,
            "label": cfg["label"],
            "enabled": cfg["enabled"],
            "schema_mode": mode,
            "prompt_variant": VARIANT_ENFORCED if enforced else VARIANT_SHAPES,
            "format_delivery": (
                "Answer format enforced by the endpoint (JSON schema); short prompt."
                if enforced else
                "Endpoint cannot enforce the format, so the exact shape of every field is "
                "written into the prompt."),
            "price_in_per_1m": settings.price_per_1m_input.get(cfg["name"], 0.0),
            "price_out_per_1m": settings.price_per_1m_output.get(cfg["name"], 0.0),
        })

    scored_docs = storage.dataset_documents("eval_scored")

    return {
        "summary": (
            "Every model reads the same RFP text and fills the same 17 fields. Each answer is "
            "compared field by field with an answer key (ground truth) written for that RFP."
        ),
        "steps": [
            "PDF to text: each page's text is extracted once and reused, so every model reads identical text.",
            "Prompt: the same instructions and the same 17 field descriptions for every model, temperature 0.",
            "Model call: one request per document per model; latency and tokens are recorded.",
            "Validation: the answer must be JSON with exactly the 17 fields, each {status, value, evidence}.",
            "Scoring: each field is compared with the answer key and gets an outcome and a score (0 to 1).",
            "Aggregation: field scores are averaged per document, then per model.",
        ],
        "outcomes": [
            {"answer_key": "not stated", "model": "not stated", "outcome": "correct abstention", "score": "1"},
            {"answer_key": "not stated", "model": "a value", "outcome": "fabrication (hallucination)", "score": "0"},
            {"answer_key": "a value", "model": "not stated", "outcome": "miss", "score": "0"},
            {"answer_key": "a value", "model": "a value, similarity >= " + _pct(CORRECT_AT),
             "outcome": "correct", "score": "1"},
            {"answer_key": "a value", "model": f"a value, similarity {_pct(PARTIAL_AT)} - {_pct(CORRECT_AT)}",
             "outcome": "partial", "score": "the similarity"},
            {"answer_key": "a value", "model": "a value, similarity < " + _pct(PARTIAL_AT),
             "outcome": "extraction error", "score": "0"},
        ],
        "thresholds": {"correct_at": CORRECT_AT, "partial_at": PARTIAL_AT},
        "text_rules": [
            "Text is compared after lowercasing and removing punctuation; an exact match scores 1, "
            "otherwise the word overlap (F1).",
            "Known synonyms count as equal, e.g. 'SOC2' = 'SOC 2', 'SOC 2 Type 2' = 'SOC 2 Type II', "
            "'CGL' = 'Commercial General Liability', portal names. Different standards stay different: "
            "'SOC 2' and 'SOC 2 Type II' are not the same.",
            "In structured fields, details the model adds where the answer key is empty are not counted "
            "against it. In lists, extra items do lower the score (precision).",
        ],
        "fields": fields,
        "metrics": [{"key": k, "name": n, "how": h, "better": b} for k, n, h, b in METRICS],
        "dimensions": DIMENSIONS,
        "per_document": PER_DOCUMENT,
        "models": models,
        "settings": {
            "temperature": 0.0,
            "max_output_tokens": MAX_TOKENS,
            "timeout_s": TIMEOUT_S,
            "retries": MAX_RETRIES,
            "scored_documents": len(scored_docs),
        },
        "limits": [
            f"Small evaluation set: {len(scored_docs)} RFPs, one run each. Repeated runs of the same "
            "model vary by about 0.03-0.05, so smaller differences between models are not meaningful.",
            "The hallucination rate can only be measured on fields the answer key marks 'not stated', "
            "and there are few of those.",
            "Free-text fields (scope, pricing) are scored by word overlap, which is strict with "
            "answers that say the same thing in different words.",
            "The answer keys were drafted with a model and reviewed by a person; they are not yet "
            "double-annotated.",
        ],
    }
