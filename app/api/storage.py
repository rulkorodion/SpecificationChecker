"""Small, durable per-request workspace used before a database is added."""

from __future__ import annotations

import json
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APISettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    jobs_dir: Path
    max_upload_size_bytes: int = Field(gt=0)
    max_reference_size_bytes: int = Field(gt=0)
    upload_chunk_size_bytes: int = Field(gt=0)
    max_documents_per_job: int = Field(default=10, gt=0, le=100)
    docs_enabled: bool = True
    rejected_requests_enabled: bool = True
    rejected_request_retention_hours: int = Field(default=24, ge=1, le=720)
    rejected_request_max_entries: int = Field(default=200, ge=1, le=10_000)
    rejected_request_max_total_size_bytes: int = Field(
        default=1024 * 1024 * 1024,
        gt=0,
    )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "APISettings":
        api = config.get("api", {})
        return cls(
            jobs_dir=Path(
                os.getenv("JOB_STORAGE_ROOT", str(api.get("jobs_dir", "runtime/jobs")))
            ),
            max_upload_size_bytes=int(
                float(api.get("max_upload_size_mb", 25)) * 1024 * 1024
            ),
            max_reference_size_bytes=int(
                float(api.get("max_reference_size_kb", 256)) * 1024
            ),
            upload_chunk_size_bytes=int(
                float(api.get("upload_chunk_size_kb", 1024)) * 1024
            ),
            max_documents_per_job=int(
                os.getenv(
                    "API_MAX_DOCUMENTS_PER_JOB",
                    api.get("max_documents_per_job", 10),
                )
            ),
            docs_enabled=os.getenv(
                "API_DOCS_ENABLED", str(api.get("docs_enabled", True))
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            rejected_requests_enabled=os.getenv(
                "API_REJECTED_REQUESTS_ENABLED",
                str(api.get("rejected_requests_enabled", True)),
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            rejected_request_retention_hours=int(
                os.getenv(
                    "API_REJECTED_REQUEST_RETENTION_HOURS",
                    api.get("rejected_request_retention_hours", 24),
                )
            ),
            rejected_request_max_entries=int(
                os.getenv(
                    "API_REJECTED_REQUEST_MAX_ENTRIES",
                    api.get("rejected_request_max_entries", 200),
                )
            ),
            rejected_request_max_total_size_bytes=int(
                float(
                    os.getenv(
                        "API_REJECTED_REQUEST_MAX_TOTAL_SIZE_MB",
                        api.get("rejected_request_max_total_size_mb", 1024),
                    )
                )
                * 1024
                * 1024
            ),
        )


class JobWorkspace:
    """Own files belonging to exactly one API request."""

    def __init__(self, root: Path, job_id: str) -> None:
        self.path = root / job_id
        self.path.mkdir(parents=True, exist_ok=False)

    def write_json(self, filename: str, value: Any) -> Path:
        destination = self.path / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def update_manifest(
        self,
        *,
        status: str,
        created_at: datetime,
        external_id: str | None,
        original_filename: str | None,
        error: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "job_id": self.path.name,
            "status": status,
            "created_at": created_at.isoformat(),
            "updated_at": datetime.now(created_at.tzinfo).isoformat(),
            "external_id": external_id,
            "original_filename": original_filename,
        }
        if error is not None:
            payload["error"] = error
        self.write_json("job.json", payload)
