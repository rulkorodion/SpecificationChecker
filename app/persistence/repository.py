"""Repository containing all SQLAlchemy operations for check jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.database import Database
from app.persistence.models import CheckJob, JobStatus, OneCDeliveryStatus


class JobNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class CheckJobRecord:
    job_id: str
    external_id: str | None
    status: JobStatus
    decision: str | None
    comment_1c: str | None
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    error_details: dict[str, Any] | None
    document_path: str | None
    document_sha256: str | None
    created_at: datetime
    started_at: datetime
    completed_at: datetime | None
    processing_time_seconds: float | None
    onec_delivery_status: OneCDeliveryStatus | None
    onec_delivery_attempts: int
    onec_payload: dict[str, Any] | None
    onec_receipt: dict[str, Any] | None
    onec_last_error: dict[str, Any] | None
    onec_delivery_updated_at: datetime | None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_optional(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None


def _record(model: CheckJob) -> CheckJobRecord:
    return CheckJobRecord(
        job_id=model.job_id,
        external_id=model.external_id,
        status=model.status,
        decision=model.decision,
        comment_1c=model.comment_1c,
        result=model.result,
        error_code=model.error_code,
        error_message=model.error_message,
        error_details=model.error_details,
        document_path=model.document_path,
        document_sha256=model.document_sha256,
        created_at=_utc(model.created_at),
        started_at=_utc(model.started_at),
        completed_at=_utc_optional(model.completed_at),
        processing_time_seconds=model.processing_time_seconds,
        onec_delivery_status=model.onec_delivery_status,
        onec_delivery_attempts=model.onec_delivery_attempts,
        onec_payload=model.onec_payload,
        onec_receipt=model.onec_receipt,
        onec_last_error=model.onec_last_error,
        onec_delivery_updated_at=_utc_optional(model.onec_delivery_updated_at),
    )


def _locked_job(session: Session, job_id: str) -> CheckJob:
    model = session.get(CheckJob, job_id, with_for_update=True)
    if model is None:
        raise JobNotFoundError(job_id)
    return model


class CheckJobRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create_processing(
        self,
        *,
        job_id: str,
        external_id: str | None,
        created_at: datetime,
        started_at: datetime,
        document_path: str | None,
        document_sha256: str | None = None,
    ) -> CheckJobRecord:
        with self.database.session() as session:
            model = CheckJob(
                job_id=job_id,
                external_id=external_id,
                status=JobStatus.PROCESSING,
                created_at=created_at,
                started_at=started_at,
                document_path=document_path,
                document_sha256=document_sha256,
            )
            session.add(model)
            session.flush()
            return _record(model)

    def set_document(
        self, job_id: str, *, document_path: str, document_sha256: str
    ) -> CheckJobRecord:
        with self.database.session() as session:
            model = _locked_job(session, job_id)
            model.document_path = document_path
            model.document_sha256 = document_sha256
            session.flush()
            return _record(model)

    def complete(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        decision: str,
        comment_1c: str,
        completed_at: datetime,
        processing_time_seconds: float,
    ) -> CheckJobRecord:
        with self.database.session() as session:
            model = _locked_job(session, job_id)
            model.status = JobStatus.COMPLETED
            model.result = result
            model.decision = decision
            model.comment_1c = comment_1c
            model.completed_at = completed_at
            model.processing_time_seconds = processing_time_seconds
            model.error_code = None
            model.error_message = None
            model.error_details = None
            session.flush()
            return _record(model)

    def fail(
        self,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        error_details: dict[str, Any],
        completed_at: datetime,
        processing_time_seconds: float,
    ) -> CheckJobRecord:
        with self.database.session() as session:
            model = _locked_job(session, job_id)
            model.status = JobStatus.FAILED
            model.error_code = error_code
            model.error_message = error_message
            model.error_details = error_details
            model.completed_at = completed_at
            model.processing_time_seconds = processing_time_seconds
            session.flush()
            return _record(model)

    def get(self, job_id: str) -> CheckJobRecord | None:
        with self.database.session() as session:
            model = session.get(CheckJob, job_id)
            return _record(model) if model is not None else None

    def set_onec_delivery(
        self,
        job_id: str,
        *,
        status: OneCDeliveryStatus,
        attempts: int,
        payload: dict[str, Any],
        receipt: dict[str, Any] | None,
        last_error: dict[str, Any] | None,
        updated_at: datetime,
    ) -> CheckJobRecord:
        with self.database.session() as session:
            model = _locked_job(session, job_id)
            model.onec_delivery_status = status
            model.onec_delivery_attempts = attempts
            model.onec_payload = payload
            model.onec_receipt = receipt
            model.onec_last_error = last_error
            model.onec_delivery_updated_at = updated_at
            session.flush()
            return _record(model)

    def list_expired_job_files(
        self, *, cutoff: datetime, limit: int
    ) -> list[CheckJobRecord]:
        with self.database.session() as session:
            statement = (
                select(CheckJob)
                .where(
                    CheckJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
                    CheckJob.completed_at.is_not(None),
                    CheckJob.completed_at < cutoff,
                    CheckJob.document_path.is_not(None),
                )
                .order_by(CheckJob.completed_at.asc())
                .limit(limit)
            )
            return [_record(model) for model in session.scalars(statement)]

    def mark_job_files_removed(self, job_id: str) -> CheckJobRecord:
        with self.database.session() as session:
            model = _locked_job(session, job_id)
            model.document_path = None
            session.flush()
            return _record(model)
