from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any, List, Dict, Literal
from datetime import datetime


class Evidence(BaseModel):
    """One supporting quote for an extracted field value."""
    model_config = ConfigDict(extra="forbid")

    quote: str
    page: Optional[int] = None


class FieldValue(BaseModel):
    """The uniform shape every one of the 17 extracted fields must follow.
    Matches harness/schemas/field_value.schema.json.
    """
    model_config = ConfigDict(extra="forbid")

    status: Literal["found", "not_stated"]
    value: Optional[Any] = None
    evidence: List[Evidence] = Field(default_factory=list)


class RFPFields(BaseModel):
    """All 17 fields extracted from one RFP document.

    Keys are the readable names in fields.py. They are the same 17 fields as
    F01..F17 in config/field_set@1.0.json, only re-keyed.

    extra="forbid": an answer using different keys is invalid JSON, not an
    answer with 17 empty fields.
    """
    model_config = ConfigDict(extra="forbid")

    submission_deadline: Optional[FieldValue] = None
    questions_deadline: Optional[FieldValue] = None
    rfp_contact: Optional[FieldValue] = None
    submission_method: Optional[FieldValue] = None
    contract_term: Optional[FieldValue] = None
    scope_of_deliverables: Optional[FieldValue] = None
    mandatory_submission_requirements: Optional[FieldValue] = None
    mandatory_technical_requirements: Optional[FieldValue] = None
    evaluation_criteria: Optional[FieldValue] = None
    minimum_score_threshold: Optional[FieldValue] = None
    pricing_structure: Optional[FieldValue] = None
    insurance_requirements: Optional[FieldValue] = None
    vendor_experience: Optional[FieldValue] = None
    references_required: Optional[FieldValue] = None
    data_security_requirements: Optional[FieldValue] = None
    data_residency: Optional[FieldValue] = None
    demo_required: Optional[FieldValue] = None


class DocumentInfo(BaseModel):
    """Returned after a PDF is uploaded and text-extracted."""
    doc_id: str
    pages: int
    chars: int
    language: str = "en"
    warnings: List[str] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    """Body of POST /extract."""
    doc_id: str
    models: Optional[List[str]] = None  # None means every enabled model


class BenchmarkRequest(BaseModel):
    """Body of POST /benchmark/runs."""
    dataset: str = "eval_scored"
    models: Optional[List[str]] = None  # None means every enabled model


class ModelResult(BaseModel):
    """The outcome of running one model against one document."""
    doc_id: str
    model: str
    status: Literal["ok", "invalid_json", "timeout", "error", "rate_limited"]
    latency_s: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    valid_json: bool = False
    fields: Optional[RFPFields] = None
    score: Optional[float] = None
    error: Optional[str] = None


class BenchmarkJob(BaseModel):
    """Tracks the progress of a full benchmark run across documents/models."""
    run_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    total: int = 0
    done: int = 0
    errors: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None