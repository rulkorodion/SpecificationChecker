"""Application-facing persistence service; API code never issues SQL."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, TypeVar

from sqlalchemy.exc import SQLAlchemyError

from app.persistence.database import Database
from app.persistence.repository import CheckJobRecord, CheckJobRepository
from app.persistence.models import OneCDeliveryStatus

T = TypeVar("T")


class JobPersistenceError(RuntimeError):
    """A database operation failed without leaking credentials to the API."""


class JobPersistenceService:
    def __init__(
        self, database: Database, repository: CheckJobRepository | None = None
    ) -> None:
        self.database = database
        self.repository = repository or CheckJobRepository(database)

    @staticmethod
    def _call(operation: Callable[..., T], *args, **kwargs) -> T:
        try:
            return operation(*args, **kwargs)
        except SQLAlchemyError as exc:
            raise JobPersistenceError(
                "Операция постоянного хранения задания не выполнена"
            ) from exc

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
        return self._call(
            self.repository.create_processing,
            job_id=job_id,
            external_id=external_id,
            created_at=created_at,
            started_at=started_at,
            document_path=document_path,
            document_sha256=document_sha256,
        )

    def set_document(
        self, job_id: str, *, document_path: str, document_sha256: str
    ) -> CheckJobRecord:
        return self._call(
            self.repository.set_document,
            job_id,
            document_path=document_path,
            document_sha256=document_sha256,
        )

    def complete(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        completed_at: datetime,
        processing_time_seconds: float,
    ) -> CheckJobRecord:
        return self._call(
            self.repository.complete,
            job_id,
            result=result,
            decision=str(result["decision"]),
            comment_1c=str(result["comment_1c"]),
            completed_at=completed_at,
            processing_time_seconds=processing_time_seconds,
        )

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
        return self._call(
            self.repository.fail,
            job_id,
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            completed_at=completed_at,
            processing_time_seconds=processing_time_seconds,
        )

    def get(self, job_id: str) -> CheckJobRecord | None:
        return self._call(self.repository.get, job_id)

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
        return self._call(
            self.repository.set_onec_delivery,
            job_id,
            status=status,
            attempts=attempts,
            payload=payload,
            receipt=receipt,
            last_error=last_error,
            updated_at=updated_at,
        )

    def list_expired_job_files(
        self, *, cutoff: datetime, limit: int
    ) -> list[CheckJobRecord]:
        return self._call(
            self.repository.list_expired_job_files,
            cutoff=cutoff,
            limit=limit,
        )

    def mark_job_files_removed(self, job_id: str) -> CheckJobRecord:
        return self._call(self.repository.mark_job_files_removed, job_id)

    def health_check(self) -> None:
        self._call(self.database.ping)

    def close(self) -> None:
        self.database.dispose()
