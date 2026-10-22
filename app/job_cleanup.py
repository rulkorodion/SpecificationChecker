"""Retention cleanup for job files; PostgreSQL records are never deleted."""

from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from app.job_storage import JobStorage, JobStorageError
from app.persistence.repository import CheckJobRecord
from app.persistence.service import JobPersistenceService

LOGGER = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} должен быть true/false")


class JobRetentionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    cleanup_interval_seconds: int = Field(default=3600, ge=60)
    batch_size: int = Field(default=100, ge=1, le=10000)

    @classmethod
    def from_environment(
        cls, config: Mapping[str, Any] | None = None
    ) -> "JobRetentionSettings":
        section = dict((config or {}).get("job_retention", {}))
        return cls(
            enabled=_env_bool(
                "JOB_CLEANUP_ENABLED", bool(section.get("enabled", True))
            ),
            retention_days=int(
                os.getenv("JOB_RETENTION_DAYS", section.get("retention_days", 30))
            ),
            cleanup_interval_seconds=int(
                os.getenv(
                    "JOB_CLEANUP_INTERVAL_SECONDS",
                    section.get("cleanup_interval_seconds", 3600),
                )
            ),
            batch_size=int(
                os.getenv("JOB_CLEANUP_BATCH_SIZE", section.get("batch_size", 100))
            ),
        )


class JobCleanupReport(BaseModel):
    scanned: int = 0
    removed: int = 0
    missing: int = 0
    failed: int = 0
    freed_bytes: int = 0
    dry_run: bool = False


class JobFileCleanupService:
    def __init__(
        self,
        *,
        settings: JobRetentionSettings,
        persistence: JobPersistenceService,
        storage: JobStorage,
    ) -> None:
        self.settings = settings
        self.persistence = persistence
        self.storage = storage
        self._next_cleanup_at = 0.0

    @staticmethod
    def _directory_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _trusted_job_dir(self, record: CheckJobRecord) -> Path:
        if not record.document_path:
            raise JobStorageError("У задания уже нет сохранённого пути")
        job_dir = self.storage.job_dir(record.job_id).resolve()
        root = self.storage.root.resolve()
        if job_dir.parent != root:
            raise JobStorageError("Каталог задания находится вне JOB_STORAGE_ROOT")
        document = Path(record.document_path).resolve()
        try:
            document.relative_to(job_dir)
        except ValueError as exc:
            raise JobStorageError("Путь документа находится вне каталога задания") from exc
        return job_dir

    def run(self, *, now: datetime | None = None, dry_run: bool = False) -> JobCleanupReport:
        report = JobCleanupReport(dry_run=dry_run)
        if not self.settings.enabled:
            return report
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        cutoff = current - timedelta(days=self.settings.retention_days)
        records = self.persistence.list_expired_job_files(
            cutoff=cutoff, limit=self.settings.batch_size
        )
        report.scanned = len(records)
        for record in records:
            try:
                job_dir = self._trusted_job_dir(record)
                if not job_dir.exists():
                    report.missing += 1
                    if not dry_run:
                        self.persistence.mark_job_files_removed(record.job_id)
                    continue
                size = self._directory_size(job_dir)
                if not dry_run:
                    shutil.rmtree(job_dir)
                    self.persistence.mark_job_files_removed(record.job_id)
                report.removed += 1
                report.freed_bytes += size
            except (OSError, JobStorageError):
                report.failed += 1
                LOGGER.warning("Could not cleanup job files for %s", record.job_id, exc_info=True)
        return report

    def run_if_due(self) -> JobCleanupReport | None:
        if not self.settings.enabled:
            return None
        now = time.monotonic()
        if now < self._next_cleanup_at:
            return None
        self._next_cleanup_at = now + self.settings.cleanup_interval_seconds
        return self.run()
