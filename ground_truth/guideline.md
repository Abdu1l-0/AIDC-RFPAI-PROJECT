# Ground Truth Annotation Guideline

**Owner:** Mohannad (Ground Truth & Evaluation Lead)
**Applies to:** the 5 documents in `corpus_source/` (or `RFP Tender Documents/`)
**Schema of record:** `config/field_set@1.0.json` — this guideline follows that file exactly. If the two ever disagree, `field_set@1.0.json` wins; update this guideline to match, not the other way around.
**Consumed by:** `ground_truth/calculate_iaa.py`, `harness/stages/s7_score.py` (`score_cell`), `scoring/`

---

## 1. Purpose

This guideline defines how a human annotator fills out `ground_truth/annotation_template.json` for one RFP document. The result is a **gold-standard answer key**: every field's true value, structured exactly as `field_set@1.0.json` requires, with page-level evidence. Every model's extraction gets scored against this — a wrong or malformed annotation produces a wrong answer key, and every model gets scored against the wrong thing.

Each document is annotated **independently by two annotators** (double-blind — no comparing notes while annotating). Their outputs are reconciled into one adjudicated `gold_standard.json` via `calculate_iaa.py`'s Cohen's Kappa gate.

---

## 2. The 17-Field Taxonomy (F01–F17)

**This table is a summary for quick reference only. The authoritative definition — including the exact JSON Schema for each field's `value` object — lives in `config/field_set@1.0.json`. Always check that file directly before annotating a field you're unsure about.**

| Key | Name | `type` | `field_class` | `absent_legal` |
|---|---|---|---|---|
| F01 | Submission Deadline | object | atomic_exact | false — should basically always be present; search hard before marking absent |
| F02 | Deadline for Questions / Inquiries | object | atomic_exact | true — genuinely often absent |
| F03 | RFP Contact | array | list_set | false |
| F04 | Submission Method / Portal | array | short_string | false |
| F05 | Contract Term / Duration | object | composite | false |
| F06 | Scope of Deliverables / Services Requested | object | free_text | false |
| F07 | Mandatory Submission Requirements | array | list_set | false |
| F08 | Mandatory Technical Requirements | array | list_set | true |
| F09 | Evaluation Criteria & Weighting | array | list_set | true |
| F10 | Minimum Score Threshold to Advance | object | atomic_exact | true |
| F11 | Pricing Structure / Cost Submission Requirements | object | free_text | false |
| F12 | Minimum Insurance Coverage Requirements | array | composite | true |
| F13 | Required Vendor Experience / Qualifications | array | composite | true |
| F14 | Number of References Required | integer | atomic_exact | true |
| F15 | Data Security / Privacy Compliance Requirements | array | list_set | true |
| F16 | Data Hosting / Residency Requirements | object | short_string | true |
| F17 | Vendor Demonstration Requirement | object | composite | true |

**`absent_legal` is a real signal, not decoration.** It tells you how surprising it would be for this field to be missing from a properly-drafted RFP:
- **`absent_legal: false`** (F01, F03, F04, F05, F06, F07, F11) — a normal RFP should state this. If you can't find it, search *harder* than the 3-term minimum before concluding it's absent — this is more likely an oversight in your search than a genuine gap in the document.
- **`absent_legal: true`** (F02, F08, F09, F10, F12, F13, F14, F15, F16, F17) — perfectly normal for this to be unstated in many RFPs. The standard 3-search minimum (Section 4) is sufficient.

---

## 3. Value Structure — Per-Field Shape

**Do not fill `value` as a flat string.** Every field's `value` must match the object/array shape defined in `config/field_set@1.0.json`. Get the exact shape from that file, not from memory or a prior document's annotation — fields can have optional sub-properties that aren't always populated.

Quick shape reference (see `field_set@1.0.json` for full property lists, enums, and `required` sub-fields):

- **F01, F02** (`atomic_exact`, object): `{"date": "YYYY-MM-DD", "time": "HH:MM" or null, "tz_raw": "<as written>" or null}`. `date` is required; `time`/`tz_raw` are not.
- **F03** (`list_set`, array): list of `{"name": ..., "email": ..., "title": ..., "phone": ...}` — one object per contact.
- **F04** (`short_string`, array): list of `{"method": "portal"|"email"|"physical"|"other", "portal_name": ..., "url": ...}`. `method` is required.
- **F05** (`composite`, object): `{"base_length": <number>, "base_unit": "days"|"months"|"years", "renewals": [{"count": ..., "length": ..., "unit": ...}], "total_max_years": <number> or null}`.
- **F06** (`free_text`, object): `{"summary": "<required, 1-3 sentences>", "key_deliverables": ["...", "..."]}`.
- **F07, F08** (`list_set`, array): plain list of strings.
- **F09** (`list_set`, array): list of `{"criterion": ..., "weight": <number>, "unit": "points"|"percent"}`. All three required per item.
- **F10** (`atomic_exact`, object): `{"value": <number> or null, "unit": "points"|"percent"|null, "applies_to": "<what stage/component this threshold gates>" or null}`.
- **F11** (`free_text`, object): `{"pricing_model": "<e.g. fixed-fee, T&M, milestone>", "submission_requirements": ["...", "..."]}`.
- **F12** (`composite`, array): list of `{"coverage_type": ..., "amount": <number>, "currency": ..., "basis": "per_occurrence"|"aggregate"|"unspecified"}`. `coverage_type` and `amount` required.
- **F13** (`composite`, array): list of `{"requirement_type": ..., "years": <number> or null, "detail": "<required>"}`.
- **F14** (`atomic_exact`, integer): a plain integer. If the RFP gives a range (e.g. "2 to 5"), see Section 6 for how to record it — do not silently pick one end.
- **F15** (`list_set`, array): list of `{"standard": "<e.g. SOC2, ISO27001>", "detail": ... or null}`. `standard` required.
- **F16** (`short_string`, object): `{"required": <boolean, required>, "location": "<e.g. Canada>" or null, "detail": ... or null}`.
- **F17** (`composite`, object): `{"required": <boolean, required>, "stage": "<which evaluation stage>" or null, "detail": ... or null}`.

If a field is `found` but the document only gives partial information for a multi-property object (e.g. F01 gives a date but no timezone), fill the properties you have and leave the rest `null` — do not invent a value for a property the document doesn't specify.

---

## 4. The Mandatory Negative Search Rule

**An annotator is strictly forbidden from recording `status: "not_stated"` without documenting at least three distinct keyword searches in that field's `search_record` array.** For fields where `absent_legal: false` (see Section 2), search more thoroughly than the 3-term minimum — treat the minimum as a floor, not a target, for those fields specifically.

**How to do it:**
1. Before marking a field `not_stated`, search the raw document text for at least 3 distinct, plausible keyword variants that would surface the field's value if present.
2. Record all search terms in `search_record`, including ones that returned nothing.
3. Only after all searches come back empty may `status` be `"not_stated"`.

**Example** (F16 — data residency):
```json
"search_record": ["data residency", "canadian soil", "sovereignty", "hosted in canada", "jurisdiction"]
```

Pick terms reflecting how the *concept* might be phrased, not just the field's own name — RFPs use inconsistent terminology across issuers.

If a field is `found`, `search_record` is `[]` — the rule only applies to absence claims.

---

## 5. Evidence Requirements

Every field marked `status: "found"` must include at least one entry in its `evidence` array:

```json
"evidence": [
  {"quote": "<exact verbatim text from the document>", "page": <page number>}
]
```

The `quote` must be an **exact substring** of the raw document text — this is checked programmatically by `harness/stages/s7_score.py`'s `check_grounding` function, which normalizes whitespace and case but otherwise requires the quote to literally appear in the source. Do not paraphrase or lightly edit — copy exactly. A quote that fails this check is treated as ungrounded, which for *gold-standard* ground truth means that entry can never legitimately be marked "correct" by the automated grounding check.

For multi-property fields (F01, F05, F09, F12, etc.), if different sub-properties come from different parts of the document, include one evidence entry per distinct quoted passage.

For list fields (F03, F07, F08, F09, F12, F13, F15), each list item does not need its own separate evidence entry unless it comes from a genuinely distinct part of the document — one evidence entry covering the relevant passage/table is sufficient if that passage contains multiple items.

---

## 6. Status Values

Only two `status` values are valid: **`"found"`** and **`"not_stated"`**.

- **`"found"`** — requires `value` populated per Section 3's shape, and at least one `evidence` entry (Section 5).
- **`"not_stated"`** — requires the Negative Search Rule (Section 4) satisfied. `value` is `null`, `evidence` is `[]`.

**Ranges (e.g. F14 "2 to 5 references"):** F14's schema is a plain integer, but a range is not a single integer. Record the field as `found`, and in `value`, use the maximum stated value (the more conservative/inclusive number for a "how many do we need" question), and note the full range in the evidence quote itself so the range is never lost — the quote will show "2 to 5" even though `value` records `5`. This is a documented, deliberate choice; do not silently average or pick the low end.

---

## 7. Normalization Rules

- **Dates:** `date` sub-property is ISO 8601 (`YYYY-MM-DD`). `time` is 24-hour `HH:MM`. `tz_raw` is copied as written in the document (do not convert timezones yourself).
- **Currency (F12):** always populate `currency` (e.g. `"CAD"`), not folded into `amount`.
- **Booleans (F16 `required`, F17 `required`):** use `true` only when the document is unambiguous. A discretionary or "may be requested" demo (see F17) is `false`, not `true` — read intent, not just keyword presence.
- **Lists:** each item in F03/F07/F08/F09/F12/F13/F15 arrays is a distinct, atomic entry per that field's item schema — do not merge multiple requirements into one string.

---

## 8. Why This Matters

Per the project's risk register: *"Low Annotator Agreement (κ < 0.60) — Ground truth is unreliable; evaluation results cannot be published or defended."* Every rule above exists to reduce disagreement between independent annotators before it reaches the Cohen's Kappa gate. The most common sources of disagreement are: skipping the negative search rule, filling `value` as a flat string instead of the schema's object shape, paraphrasing quotes instead of copying them exactly, and silently resolving ranges/ambiguity instead of following Section 6's rule. Get these right and IAA holds; get them wrong and the whole benchmark's evidentiary basis is compromised.
