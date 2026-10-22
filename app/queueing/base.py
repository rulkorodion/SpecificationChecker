"""Queue contract independent from redis-py and FastAPI."""

from __future__ import annotations

import os
import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueueUnavailableError(RuntimeError):
    """Redis operation failed or queue configuration is missing."""


class QueueSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    redis_url: str = Field(min_length=1)
    queue_name: str = Field(default="specification_checks", min_length=1)
    block_timeout_seconds: int = Field(default=5, ge=1, le=60)
    socket_timeout_seconds: int = Field(default=10, ge=1, le=120)
    reconnect_delay_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    job_timeout_seconds: int = Field(default=3600, ge=60)
    worker_lock_ttl_seconds: int = Field(default=120, ge=60)

    @field_validator("queue_name")
    @classmethod
    def safe_queue_name(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", normalized):
            raise ValueError("REDIS_QUEUE_NAME содержит недопустимые символы")
        return normalized

    @classmethod
    def from_environment(cls) -> "QueueSettings":
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            raise QueueUnavailableError("REDIS_URL не задан")
        return cls(
            redis_url=redis_url,
            queue_name=os.getenv(
                "REDIS_QUEUE_NAME", "specification_checks"
            ),
            block_timeout_seconds=int(
                os.getenv("REDIS_BLOCK_TIMEOUT_SECONDS", "5")
            ),
            socket_timeout_seconds=int(
                os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "10")
            ),
            reconnect_delay_seconds=float(
                os.getenv("WORKER_RECONNECT_DELAY_SECONDS", "2")
            ),
            job_timeout_seconds=int(os.getenv("WORKER_JOB_TIMEOUT_SECONDS", "3600")),
            worker_lock_ttl_seconds=int(
                os.getenv("WORKER_LOCK_TTL_SECONDS", "120")
            ),
        )


class JobQueue(Protocol):
    settings: QueueSettings

    def ping(self) -> None: ...

    def enqueue(self, job_id: str) -> None: ...

    def dequeue(
        self, worker_id: str, timeout_seconds: int | None = None
    ) -> str | None: ...

    def acknowledge(self, job_id: str, worker_id: str) -> None: ...

    def requeue(self, job_id: str, worker_id: str) -> None: ...

    def recover_inflight(
        self, worker_id: str, *, include_current: bool = False
    ) -> int: ...

    def acquire_worker(self, worker_id: str) -> bool: ...

    def refresh_worker(self, worker_id: str) -> bool: ...

    def release_worker(self, worker_id: str) -> None: ...

    def worker_registration(self) -> tuple[str | None, int | None]: ...

    def worker_registrations(self) -> list[tuple[str, int]]: ...

    def close(self) -> None: ...
