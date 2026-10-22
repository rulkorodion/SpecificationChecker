"""Canonical internal contracts independent from the concrete 1C API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.enums import CheckStatus, FinalDecision
from app.persistence.models import JobStatus
from app.schemas import ReferenceData


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CanonicalDocument(CanonicalModel):
    """Trusted server-side document representation."""

    path: Path
    filename: str = Field(min_length=1)
    media_type: Literal[
        "application/pdf", "image/jpeg", "image/png"
    ]
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CanonicalCheckInput(CanonicalModel):
    """Canonical request after any external adapter has normalized its input."""

    document: CanonicalDocument
    external_id: str | None = Field(default=None, max_length=128)
    reference: ReferenceData

    @field_validator("external_id", mode="before")
    @classmethod
    def normalize_external_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value


class CanonicalCheckSummary(CanonicalModel):
    code: str
    title: str | None = None
    status: CheckStatus
    actual: Any = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    message: str | None = None


class CanonicalCheckResult(CanonicalModel):
    """Stable worker-to-integration result; not the unknown wire contract."""

    job_id: UUID
    external_id: str | None = None
    status: JobStatus
    decision: FinalDecision | None = None
    comment_1c: str | None = None
    reasons: list[str] = Field(default_factory=list)
    checks: list[CanonicalCheckSummary] = Field(default_factory=list)
    extracted_values: dict[str, Any] = Field(default_factory=dict)
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    completed_at: datetime | None = None
    processing_time_seconds: float | None = Field(default=None, ge=0.0)


class DeliveryReceipt(CanonicalModel):
    adapter: str
    accepted: bool
    sent_at: datetime
    external_message_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
