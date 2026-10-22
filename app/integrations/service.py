"""Durable result delivery isolated from OCR and the rule engine."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.enums import FinalDecision
from app.integrations.base import OneCAdapter, OneCIntegrationError
from app.integrations.contracts import CanonicalCheckResult, CanonicalCheckSummary
from app.integrations.settings import OneCIntegrationSettings
from app.persistence.models import JobStatus, OneCDeliveryStatus
from app.persistence.repository import CheckJobRecord
from app.persistence.service import JobPersistenceService
from app.reporting.console import format_console_report
from app.schemas import CheckResult, ReferenceData

LOGGER = logging.getLogger(__name__)


DeliveryStatus = OneCDeliveryStatus


class DeliveryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: DeliveryStatus
    attempts: int = Field(ge=0)
    payload: dict[str, Any]
    receipt: dict[str, Any] | None = None
    last_error: dict[str, Any] | None = None
    updated_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_reference(record: CheckJobRecord) -> ReferenceData:
    if not record.document_path:
        raise OneCIntegrationError(
            "Невозможно сформировать отчёт для 1С: путь документа отсутствует"
        )
    reference_path = Path(record.document_path).parent / "reference.json"
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
        return ReferenceData.model_validate(payload)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise OneCIntegrationError(
            "Невозможно сформировать отчёт для 1С: reference.json недоступен или повреждён"
        ) from exc


def _format_onec_comment(
    record: CheckJobRecord,
    result: dict[str, Any],
    decision: FinalDecision,
    reasons: list[str],
) -> str:
    raw_checks = result.get("checks", [])
    if not isinstance(raw_checks, list):
        raise OneCIntegrationError(
            "Невозможно сформировать отчёт для 1С: checks должен быть массивом"
        )
    try:
        checks = [CheckResult.model_validate(item) for item in raw_checks]
    except ValidationError as exc:
        raise OneCIntegrationError(
            "Невозможно сформировать отчёт для 1С: сохранённые проверки повреждены"
        ) from exc

    reference = _load_reference(record)
    return format_console_report(
        SimpleNamespace(
            checks=checks,
            final_decision=decision,
            decision_reasons=reasons,
        ),
        application_number=reference.application_number,
        job_id=record.job_id,
    )


class OneCIntegrationService:
    """Build, persist and deliver callbacks without touching checker state."""

    def __init__(
        self,
        *,
        settings: OneCIntegrationSettings,
        adapter: OneCAdapter,
        persistence: JobPersistenceService,
    ) -> None:
        self.settings = settings
        self.adapter = adapter
        self.persistence = persistence

    @staticmethod
    def _state_from_record(record: CheckJobRecord) -> DeliveryState | None:
        if record.onec_delivery_status is None:
            return None
        return DeliveryState(
            job_id=record.job_id,
            status=record.onec_delivery_status,
            attempts=record.onec_delivery_attempts,
            payload=record.onec_payload or {},
            receipt=record.onec_receipt,
            last_error=record.onec_last_error,
            updated_at=record.onec_delivery_updated_at or _utc_now(),
        )

    @staticmethod
    def canonical_result(record: CheckJobRecord) -> CanonicalCheckResult:
        result = record.result or {}
        checks: list[CanonicalCheckSummary] = []
        for raw in result.get("checks", []):
            if not isinstance(raw, dict) or not raw.get("code") or not raw.get("status"):
                continue
            checks.append(
                CanonicalCheckSummary(
                    code=str(raw["code"]),
                    title=str(raw["title"]) if raw.get("title") is not None else None,
                    status=raw["status"],
                    actual=raw.get("actual"),
                    confidence=raw.get("confidence"),
                    message=str(raw["message"]) if raw.get("message") is not None else None,
                )
            )
        reasons = [str(item) for item in result.get("reasons", [])]
        if record.status is JobStatus.FAILED and record.error_message:
            reasons = reasons or [record.error_message]
        decision_value = record.decision or result.get("decision")
        decision = FinalDecision(decision_value) if decision_value else None
        confidence = result.get("confidence", {})
        overall_confidence = (
            confidence.get("overall") if isinstance(confidence, dict) else None
        )
        comment_1c = record.comment_1c or result.get("comment_1c")
        if record.status is JobStatus.COMPLETED and decision is not None:
            comment_1c = _format_onec_comment(
                record,
                result,
                decision,
                reasons,
            )
        return CanonicalCheckResult(
            job_id=record.job_id,
            external_id=record.external_id,
            status=record.status,
            decision=decision,
            comment_1c=comment_1c,
            reasons=reasons,
            checks=checks,
            extracted_values=(
                dict(result.get("extracted_values", {}))
                if isinstance(result.get("extracted_values", {}), dict)
                else {}
            ),
            overall_confidence=overall_confidence,
            completed_at=record.completed_at,
            processing_time_seconds=record.processing_time_seconds,
        )

    def deliver(
        self, record: CheckJobRecord, *, force: bool = False
    ) -> DeliveryState | None:
        """Attempt delivery once; failures are durable and never raised."""

        if not self.settings.enabled:
            return None
        if record.status not in {JobStatus.COMPLETED, JobStatus.FAILED}:
            raise ValueError("Результат можно отправлять только для финального задания")
        previous = self._state_from_record(record)
        if previous is not None and previous.status is DeliveryStatus.SENT:
            return previous
        if (
            previous is not None
            and previous.attempts >= self.settings.max_delivery_attempts
            and not force
        ):
            return previous

        attempts = (previous.attempts if previous else 0) + 1
        payload = previous.payload if previous else {}
        try:
            if not payload:
                canonical = self.canonical_result(record)
                payload = self.adapter.build_result_payload(canonical)
            receipt = self.adapter.send_result(payload)
            state = DeliveryState(
                job_id=record.job_id,
                status=DeliveryStatus.SENT,
                attempts=attempts,
                payload=payload,
                receipt=receipt.model_dump(mode="json"),
                last_error=None,
                updated_at=_utc_now(),
            )
        except Exception as exc:
            state = DeliveryState(
                job_id=record.job_id,
                status=DeliveryStatus.PENDING,
                attempts=attempts,
                payload=payload,
                receipt=None,
                last_error={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "occurred_at": _utc_now().isoformat(),
                },
                updated_at=_utc_now(),
            )
            LOGGER.warning(
                "1C result delivery failed for job %s (attempt %d/%d)",
                record.job_id,
                attempts,
                self.settings.max_delivery_attempts,
                exc_info=True,
            )
        self.persistence.set_onec_delivery(
            record.job_id,
            status=state.status,
            attempts=state.attempts,
            payload=state.payload,
            receipt=state.receipt,
            last_error=state.last_error,
            updated_at=state.updated_at,
        )
        return state

    def retry(self, record: CheckJobRecord) -> DeliveryState | None:
        """Explicit operator retry; reuses the persisted payload, never OCR."""

        return self.deliver(record, force=True)

    def close(self) -> None:
        self.adapter.close()
