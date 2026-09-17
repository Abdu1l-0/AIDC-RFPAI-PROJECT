from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict, Literal
from datetime import datetime


class Evidence(BaseModel):
    """One supporting quote for an extracted field value."""
    quote: str
    page: Optional[int] = None


class FieldValue(BaseModel):
    """The uniform shape every one of the 17 extracted fields must follow.
    Matches harness/schemas/field_value.schema.json.
    """
    status: Literal["found", "not_stated"]
    value: Optional[Any] = None
    evidence: List[Evidence] = Field(default_factory=list)


class RFPFields(BaseModel):
    """All 17 fields extracted from one RFP document, keyed F01..F17."""
    F01: Optional[FieldValue] = None
    F02: Optional[FieldValue] = None
    F03: Optional[FieldValue] = None
    F04: Optional[FieldValue] = None
    F05: Optional[FieldValue] = None
    F06: Optional[FieldValue] = None
    F07: Optional[FieldValue] = None
    F08: Optional[FieldValue] = None
    F09: Optional[FieldValue] = None
    F10: Optional[FieldValue] = None
    F11: Optional[FieldValue] = None
    F12: Optional[FieldValue] = None
    F13: Optional[FieldValue] = None
    F14: Optional[FieldValue] = None
    F15: Optional[FieldValue] = None
    F16: Optional[FieldValue] = None
    F17: Optional[FieldValue] = None


class DocumentInfo(BaseModel):
    """Returned after a PDF is uploaded and text-extracted."""
    doc_id: str
    pages: int
    chars: int
    warnings: List[str] = Field(default_factory=list)


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