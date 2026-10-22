"""Separate Redis worker for specification checks."""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from app.api.storage import APISettings
from app.config import load_config
from app.job_storage import JobStorage, JobStorageError
from app.job_cleanup import JobFileCleanupService, JobRetentionSettings
from app.exceptions import PageSelectionError
from app.integrations.factory import build_onec_adapter
from app.integrations.service import OneCIntegrationService
from app.integrations.settings import OneCIntegrationSettings
from app.persistence import JobPersistenceError, JobPersistenceService, JobStatus
from app.persistence.database import Database
from app.persistence.repository import CheckJobRecord
from app.queueing import JobQueue, QueueSettings, QueueUnavailableError, RedisJobQueue
from app.service import SpecificationCheckerService

LOGGER = logging.getLogger(__name__)
FINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED}


class Checker(Protocol):
    def check_specification(
        self,
        document_path: str | Path,
        reference: Mapping[str, Any],
        runtime_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _elapsed(started_at: datetime, completed_at: datetime) -> float:
    return max(0.0, (completed_at - started_at).total_seconds())


def _safe_error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, PageSelectionError):
        return {
            "code": "page_selection_failed",
            "message": "Не удалось однозначно выбрать страницы нужного приложения.",
            "diagnostic_code": exc.diagnostic_code,
            "exception_type": type(exc).__name__,
            "technical_message": str(exc),
            "page_selection_artifact": (
                str(exc.artifact_path) if exc.artifact_path is not None else None
            ),
        }
    return {
        "code": "worker_check_failed",
        "message": "Фоновая проверка завершилась технической ошибкой.",
        "exception_type": type(exc).__name__,
        "technical_message": str(exc),
    }


class SpecificationWorker:
    """Sequential worker; one instance owns and reuses one checker/model."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        job_service: JobPersistenceService,
        checker: Checker,
        storage: JobStorage,
        integration_service: OneCIntegrationService | None = None,
        cleanup_service: JobFileCleanupService | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.queue = queue
        self.job_service = job_service
        self.checker = checker
        self.storage = storage
        self.integration_service = integration_service
        self.cleanup_service = cleanup_service
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4()}"

    def _persist_outcome(
        self, record: CheckJobRecord, outcome: dict[str, Any]
    ) -> CheckJobRecord:
        completed_at = datetime.fromisoformat(str(outcome["completed_at"]))
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        completed_at = completed_at.astimezone(timezone.utc)
        duration = float(outcome["processing_time_seconds"])
        if outcome["kind"] == "completed":
            return self.job_service.complete(
                record.job_id,
                result=outcome["result"],
                completed_at=completed_at,
                processing_time_seconds=duration,
            )
        error = outcome["error"]
        return self.job_service.fail(
            record.job_id,
            error_code=str(error["code"]),
            error_message=str(error["message"]),
            error_details=dict(error),
            completed_at=completed_at,
            processing_time_seconds=duration,
        )

    def _finish_delivery(self, job_id: str) -> None:
        self.queue.acknowledge(job_id, self.worker_id)
        try:
            self.storage.delete_outcome(job_id)
        except OSError:
            LOGGER.warning("Could not remove worker outcome for job %s", job_id, exc_info=True)

    def _write_outcome(self, job_id: str, outcome: dict[str, Any]) -> None:
        try:
            self.storage.write_outcome(job_id, outcome)
        except OSError:
            # PostgreSQL can still receive the terminal state. If PostgreSQL is
            # down too, startup recovery may repeat the checker; the error is
            # explicit rather than terminating the whole worker silently.
            LOGGER.error("Could not write durable worker outcome for job %s", job_id, exc_info=True)

    def _notify_integration(self, record: CheckJobRecord) -> None:
        if self.integration_service is None:
            return
        try:
            self.integration_service.deliver(record)
        except Exception:
            # Integration is intentionally downstream from the persisted OCR /
            # Rule Engine result and can never change that terminal status.
            LOGGER.exception("Unexpected 1C integration error for job %s", record.job_id)

    def process_one(self, *, timeout_seconds: int | None = None) -> str:
        """Process at most one delivery and return a diagnostic outcome label."""

        job_id = self.queue.dequeue(self.worker_id, timeout_seconds)
        if job_id is None:
            return "empty"
        try:
            record = self.job_service.get(job_id)
        except JobPersistenceError:
            self.queue.requeue(job_id, self.worker_id)
            raise
        if record is None:
            LOGGER.warning("Ignoring queue item for absent job_id=%s", job_id)
            self.queue.acknowledge(job_id, self.worker_id)
            return "missing"
        if record.status in FINAL_STATUSES:
            self._notify_integration(record)
            self._finish_delivery(job_id)
            return "already_final"

        try:
            saved_outcome = self.storage.read_outcome(job_id)
            if saved_outcome is not None:
                final_record = self._persist_outcome(record, saved_outcome)
                self._notify_integration(final_record)
                self._finish_delivery(job_id)
                return "recovered_outcome"

            job_input = self.storage.load_input(record)
            try:
                result = self.checker.check_specification(
                    job_input.document_path,
                    job_input.reference,
                    {
                        "output_dir": job_input.output_dir,
                        "debug_level": "none",
                        "reuse_ocr_cache": True,
                        "force_ocr": False,
                        "emit_console": False,
                        "job_id": job_id,
                    },
                )
                completed_at = _utc_now()
                duration = _elapsed(record.started_at, completed_at)
                if duration > self.queue.settings.job_timeout_seconds:
                    LOGGER.warning(
                        "Job %s exceeded configured soft timeout: %.3fs > %ss",
                        job_id,
                        duration,
                        self.queue.settings.job_timeout_seconds,
                    )
                outcome = {
                    "kind": "completed",
                    "completed_at": completed_at.isoformat(),
                    "processing_time_seconds": duration,
                    "result": result,
                }
            except Exception as exc:
                LOGGER.exception("Checker failed for job %s", job_id)
                completed_at = _utc_now()
                outcome = {
                    "kind": "failed",
                    "completed_at": completed_at.isoformat(),
                    "processing_time_seconds": _elapsed(
                        record.started_at, completed_at
                    ),
                    "error": _safe_error(exc),
                }
            self._write_outcome(job_id, outcome)
            final_record = self._persist_outcome(record, outcome)
            self._notify_integration(final_record)
            self._finish_delivery(job_id)
            return str(outcome["kind"])
        except JobPersistenceError:
            # The durable sidecar prevents a second expensive checker call after
            # PostgreSQL recovers. Return the delivery to pending with a bounded
            # reconnect delay controlled by the outer loop.
            self.queue.requeue(job_id, self.worker_id)
            raise
        except JobStorageError as exc:
            completed_at = _utc_now()
            error = _safe_error(exc)
            error["code"] = "job_input_unavailable"
            outcome = {
                "kind": "failed",
                "completed_at": completed_at.isoformat(),
                "processing_time_seconds": _elapsed(record.started_at, completed_at),
                "error": error,
            }
            self._write_outcome(job_id, outcome)
            final_record = self._persist_outcome(record, outcome)
            self._notify_integration(final_record)
            self._finish_delivery(job_id)
            return "failed"

    def run_forever(self, stop_event: threading.Event) -> None:
        if not self.queue.acquire_worker(self.worker_id):
            raise RuntimeError(
                f"Worker с идентификатором {self.worker_id!r} уже зарегистрирован"
            )
        heartbeat_stop = threading.Event()
        registration_lost = threading.Event()
        heartbeat_interval = max(
            5.0,
            min(30.0, self.queue.settings.worker_lock_ttl_seconds / 3.0),
        )

        def heartbeat() -> None:
            while not heartbeat_stop.wait(heartbeat_interval):
                try:
                    if not self.queue.refresh_worker(self.worker_id):
                        LOGGER.error(
                            "Worker registration was lost: %s", self.worker_id
                        )
                        registration_lost.set()
                        return
                except QueueUnavailableError:
                    LOGGER.warning(
                        "Could not refresh worker registration: %s",
                        self.worker_id,
                        exc_info=True,
                    )

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"worker-heartbeat-{self.worker_id}",
            daemon=True,
        )
        try:
            recovered = self.queue.recover_inflight(
                self.worker_id, include_current=True
            )
            if recovered:
                LOGGER.warning("Recovered %d in-flight queue items", recovered)
            heartbeat_thread.start()
            while not stop_event.is_set():
                try:
                    if registration_lost.is_set():
                        raise RuntimeError(
                            f"Worker registration was lost: {self.worker_id}"
                        )
                    recovered = self.queue.recover_inflight(self.worker_id)
                    if recovered:
                        LOGGER.warning(
                            "Recovered %d deliveries from stopped workers",
                            recovered,
                        )
                    if self.cleanup_service is not None:
                        cleanup_report = self.cleanup_service.run_if_due()
                        if cleanup_report and cleanup_report.removed:
                            LOGGER.info(
                                "Job retention removed %d directories (%d bytes)",
                                cleanup_report.removed,
                                cleanup_report.freed_bytes,
                            )
                    self.process_one()
                except (QueueUnavailableError, JobPersistenceError):
                    LOGGER.exception("Worker infrastructure operation failed")
                    stop_event.wait(self.queue.settings.reconnect_delay_seconds)
                except Exception:
                    LOGGER.exception("Unexpected job error; worker remains alive")
                    try:
                        self.queue.recover_inflight(self.worker_id)
                    except QueueUnavailableError:
                        LOGGER.warning("Could not recover in-flight job", exc_info=True)
                    if registration_lost.is_set():
                        raise
                    stop_event.wait(self.queue.settings.reconnect_delay_seconds)
        finally:
            heartbeat_stop.set()
            if heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=max(1.0, heartbeat_interval + 1.0))
            try:
                self.queue.release_worker(self.worker_id)
            except QueueUnavailableError:
                LOGGER.warning("Could not release Redis worker registration", exc_info=True)

    def close(self) -> None:
        if self.integration_service is not None:
            self.integration_service.close()
        self.job_service.close()
        self.queue.close()
        close = getattr(self.checker, "close", None)
        if callable(close):
            close()


def build_worker() -> SpecificationWorker:
    queue_settings = QueueSettings.from_environment()
    queue = RedisJobQueue(queue_settings)
    service = JobPersistenceService(Database.from_environment())
    config_path_value = os.getenv("SPECIFICATION_CHECKER_CONFIG", "").strip()
    options: dict[str, Any] = {
        "reuse_ocr_cache": True,
        "debug_level": "none",
        "emit_console": False,
    }
    if config_path_value:
        options["config_path"] = Path(config_path_value)
    checker = SpecificationCheckerService(options)
    ocr_info = checker.prepare()
    LOGGER.info(
        "Worker OCR initialized once: profile=%s dtype=%s attention=%s "
        "use_cache=%s GPU=%s VRAM=%s MiB",
        ocr_info.get("inference_profile"),
        ocr_info.get("effective_dtype"),
        ocr_info.get("effective_attention_backend"),
        ocr_info.get("transformers_use_kv_cache"),
        ocr_info.get("gpu_model") or "n/a",
        ocr_info.get("gpu_memory_mib") or "n/a",
    )
    config = load_config(Path(config_path_value) if config_path_value else None)
    storage_root = APISettings.from_config(config).jobs_dir
    storage = JobStorage(storage_root)
    integration_settings = OneCIntegrationSettings.from_environment(config)
    integration_service = None
    if integration_settings.enabled:
        integration_service = OneCIntegrationService(
            settings=integration_settings,
            adapter=build_onec_adapter(integration_settings),
            persistence=service,
        )
    cleanup_service = JobFileCleanupService(
        settings=JobRetentionSettings.from_environment(config),
        persistence=service,
        storage=storage,
    )
    return SpecificationWorker(
        queue=queue,
        job_service=service,
        checker=checker,
        storage=storage,
        integration_service=integration_service,
        cleanup_service=cleanup_service,
        worker_id=os.getenv("WORKER_ID", "").strip() or None,
    )


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stop_event = threading.Event()

    def stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    worker = build_worker()
    try:
        worker.run_forever(stop_event)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
