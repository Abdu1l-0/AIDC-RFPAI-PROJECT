"""The 17 RFP fields and the 3 model names.

This is the single source of truth for the frontend. Display code and
fake data both read from here. The keys MUST match the middleware
`RFPFields` Pydantic model.

Each field has:
  key   -> the JSON key used by the API
  label -> the human-friendly name shown in the UI
  shape -> how to display the value:
             "str"       plain text
             "list"      list of strings -> bullets
             "criteria"  list of {category, points} dicts -> table
"""

# (key, label, shape)
FIELDS = [
    ("submission_deadline", "Submission deadline", "str"),
    ("questions_deadline", "Deadline for questions", "str"),
    ("rfp_contact", "RFP contact (name / email)", "str"),
    ("submission_method", "Submission method / portal", "str"),
    ("contract_term", "Contract term / duration", "str"),
    ("scope_of_deliverables", "Scope of deliverables / services", "str"),
    ("mandatory_submission_requirements", "Mandatory submission requirements", "list"),
    ("mandatory_technical_requirements", "Mandatory technical requirements", "list"),
    ("evaluation_criteria", "Evaluation criteria & weighting", "criteria"),
    ("minimum_score_threshold", "Minimum score to advance", "str"),
    ("pricing_structure", "Pricing structure / cost submission", "str"),
    ("insurance_requirements", "Minimum insurance coverage", "str"),
    ("vendor_experience", "Required vendor experience", "str"),
    ("references_required", "Number of references required", "str"),
    ("data_security_requirements", "Data security / privacy", "str"),
    ("data_residency", "Data hosting / residency", "str"),
    ("demo_required", "Vendor demonstration requirement", "str"),
]

# Ordered list of just the keys, handy for loops.
FIELD_KEYS = [key for key, _label, _shape in FIELDS]

# key -> label
FIELD_LABELS = {key: label for key, label, _shape in FIELDS}

# key -> shape
FIELD_SHAPES = {key: shape for key, _label, shape in FIELDS}


# The 3 models we benchmark.
# id     -> the value sent to / returned by the API
# label  -> shown in the UI
# kind   -> short note about what it is
MODELS = [
    {"id": "model_a", "label": "Model A (open-weight, vLLM)", "kind": "open-weight"},
    {"id": "model_b", "label": "Model B (OpenAI API)", "kind": "commercial"},
    {"id": "model_c", "label": "Model C (open-weight, vLLM)", "kind": "open-weight"},
]

MODEL_IDS = [m["id"] for m in MODELS]

MODEL_LABELS = {m["id"]: m["label"] for m in MODELS}
