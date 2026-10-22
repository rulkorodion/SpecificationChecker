"""Pydantic contract of the versioned HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.enums import FinalDecision
from app.persistence.models import JobStatus
from app.schemas import CheckResult


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckRequestForm(APIModel):
    """Non-file fields of the multipart request."""

    reference: str = Field(
        min_length=2,
        description="Reference JSON as a JSON object encoded in a form field.",
    )
    external_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional immutable document identifier from 1C.",
    )

    @field_validator("external_id", mode="before")
    @classmethod
    def normalize_external_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @classmethod
    def as_form(
        cls,
        reference: Annotated[
            str,
            Form(
                description=(
                    "Reference JSON, for example: "
                    '{"specification_id":"1",...}'
                )
            ),
        ],
        external_id: Annotated[
            str | None,
            Form(description="Optional document ID from 1C."),
        ] = None,
    ) -> "CheckRequestForm":
        return cls(reference=reference, external_id=external_id)


class Base64Document(APIModel):
    """One source document embedded into the JSON transport used by 1C."""

    filename: str = Field(
        min_length=1,
        max_length=255,
        description="Original filename or path supplied by 1C.",
    )
    content_type: str | None = Field(
        default=None,
        max_length=128,
        description="Optional advisory MIME type; the API detects the real format from bytes.",
    )
    content_base64: str = Field(
        min_length=1,
        description="Raw Base64 payload (a data: URI is accepted as well).",
    )

    @field_validator("filename", "content_type", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value


class CheckRequestJSON(APIModel):
    """JSON/Base64 transport accepted alongside the legacy multipart form."""

    documents: list[Base64Document] = Field(min_length=1)
    reference: dict[str, Any]
    external_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("external_id", mode="before")
    @classmethod
    def normalize_external_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class ConfidencePayload(APIModel):
    overall: float = Field(ge=0.0, le=1.0)
    fields: dict[str, float | None]


class SpecificationCheckResult(APIModel):
    """Stable JSON payload returned by SpecificationCheckerService."""

    decision: FinalDecision
    extracted_values: dict[str, Any]
    field_details: dict[str, Any]
    checks: list[CheckResult]
    reasons: list[str]
    comment_1c: str
    confidence: ConfidencePayload
    metadata: dict[str, Any]


class ErrorDetail(APIModel):
    code: str
    message: str
    job_id: str | None = None


class CheckJobResponse(APIModel):
    job_id: str = Field(description="Server-generated UUID of this check.")
    external_id: str | None = None
    status: JobStatus
    created_at: datetime
    started_at: datetime
    completed_at: datetime | None = None
    processing_time_seconds: float | None = Field(default=None, ge=0.0)
    result: SpecificationCheckResult | None = None
    error: ErrorDetail | None = None


class HealthResponse(APIModel):
    status: str
    service: str
    api_version: str
    checker_initialized: bool
    ocr_backend: str | None = None
    database_available: bool
    database_error: str | None = None
    redis_available: bool
    redis_error: str | None = None
    worker_registered: bool
    worker_id: str | None = None
    worker_registration_ttl_seconds: int | None = None
    worker_count: int = 0
    worker_ids: list[str] = Field(default_factory=list)


class ErrorResponse(APIModel):
    detail: ErrorDetail
