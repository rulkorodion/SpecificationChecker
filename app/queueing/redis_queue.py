"""Reliable Redis list queue containing job IDs only."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from typing import Any, TypeVar

import redis
from redis.exceptions import RedisError

from app.queueing.base import QueueSettings, QueueUnavailableError

T = TypeVar("T")

_COMPARE_EXPIRE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""

_COMPARE_DELETE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


class RedisJobQueue:
    def __init__(
        self,
        settings: QueueSettings,
        *,
        client: redis.Redis | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            # RESP2 keeps the client compatible with Redis 5+; none of the
            # queue operations requires RESP3.
            protocol=2,
            socket_connect_timeout=settings.socket_timeout_seconds,
            socket_timeout=max(
                settings.socket_timeout_seconds,
                settings.block_timeout_seconds + 2,
            ),
            health_check_interval=30,
        )
        prefix = settings.queue_name
        self.pending_key = f"{prefix}:pending"
        self.legacy_processing_key = f"{prefix}:processing"
        self.worker_key = f"{prefix}:worker"
        self.worker_prefix = f"{prefix}:workers:"
        self.processing_prefix = f"{prefix}:processing:"

    @staticmethod
    def _worker_token(worker_id: str) -> str:
        return hashlib.sha256(worker_id.encode("utf-8")).hexdigest()[:24]

    def _registration_key(self, worker_id: str) -> str:
        return f"{self.worker_prefix}{self._worker_token(worker_id)}"

    def _processing_key(self, worker_id: str) -> str:
        return f"{self.processing_prefix}{self._worker_token(worker_id)}"

    @staticmethod
    def _translate(operation: Callable[..., T], *args, **kwargs) -> T:
        try:
            return operation(*args, **kwargs)
        except RedisError as exc:
            raise QueueUnavailableError("Операция Redis не выполнена") from exc

    def ping(self) -> None:
        self._translate(self.client.ping)

    def enqueue(self, job_id: str) -> None:
        self._translate(self.client.lpush, self.pending_key, job_id)

    def dequeue(
        self, worker_id: str, timeout_seconds: int | None = None
    ) -> str | None:
        timeout = timeout_seconds or self.settings.block_timeout_seconds
        return self._translate(
            self.client.brpoplpush,
            self.pending_key,
            self._processing_key(worker_id),
            timeout=timeout,
        )

    def acknowledge(self, job_id: str, worker_id: str) -> None:
        self._translate(
            self.client.lrem, self._processing_key(worker_id), 1, job_id
        )

    def requeue(self, job_id: str, worker_id: str) -> None:
        def operation() -> Any:
            with self.client.pipeline(transaction=True) as pipeline:
                pipeline.lrem(self._processing_key(worker_id), 1, job_id)
                pipeline.lpush(self.pending_key, job_id)
                return pipeline.execute()

        self._translate(operation)

    def _recover_list(self, processing_key: str) -> int:
        recovered = 0
        while self._translate(
            self.client.rpoplpush, processing_key, self.pending_key
        ) is not None:
            recovered += 1
        return recovered

    def recover_inflight(
        self, worker_id: str, *, include_current: bool = False
    ) -> int:
        """Recover only deliveries whose owning worker is no longer registered."""

        current_token = self._worker_token(worker_id)
        recovered = 0
        registrations = self.worker_registrations()
        # The old single-worker build used one shared processing list. It is
        # safe to migrate it only while this is the sole active worker.
        if len(registrations) <= 1:
            recovered += self._recover_list(self.legacy_processing_key)
        pattern = f"{self.processing_prefix}*"
        processing_keys = self._translate(
            lambda: list(self.client.scan_iter(match=pattern, count=100))
        )
        for processing_key in processing_keys:
            token = str(processing_key)[len(self.processing_prefix) :]
            if token == current_token and not include_current:
                continue
            registration_key = f"{self.worker_prefix}{token}"
            if token != current_token and self._translate(
                self.client.exists, registration_key
            ):
                continue
            recovered += self._recover_list(str(processing_key))
        return recovered

    def acquire_worker(self, worker_id: str) -> bool:
        result = self._translate(
            self.client.set,
            self._registration_key(worker_id),
            worker_id,
            nx=True,
            ex=self.settings.worker_lock_ttl_seconds,
        )
        if result:
            # Compatibility heartbeat for API images from the single-worker
            # release. It is informational and no longer acts as a lock.
            self._translate(
                self.client.set,
                self.worker_key,
                worker_id,
                ex=self.settings.worker_lock_ttl_seconds,
            )
        return bool(result)

    def refresh_worker(self, worker_id: str) -> bool:
        result = self._translate(
            self.client.eval,
            _COMPARE_EXPIRE,
            1,
            self._registration_key(worker_id),
            worker_id,
            self.settings.worker_lock_ttl_seconds,
        )
        if result:
            self._translate(
                self.client.set,
                self.worker_key,
                worker_id,
                ex=self.settings.worker_lock_ttl_seconds,
            )
        return bool(result)

    def release_worker(self, worker_id: str) -> None:
        self._translate(
            self.client.eval,
            _COMPARE_DELETE,
            1,
            self._registration_key(worker_id),
            worker_id,
        )
        current = self._translate(self.client.get, self.worker_key)
        if current == worker_id:
            registrations = self.worker_registrations()
            if registrations:
                replacement, ttl = registrations[0]
                self._translate(
                    self.client.set,
                    self.worker_key,
                    replacement,
                    ex=max(1, ttl),
                )
            else:
                self._translate(self.client.delete, self.worker_key)

    def worker_registrations(self) -> list[tuple[str, int]]:
        pattern = f"{self.worker_prefix}*"

        def operation() -> list[tuple[str, int]]:
            result: list[tuple[str, int]] = []
            for key in self.client.scan_iter(match=pattern, count=100):
                with self.client.pipeline(transaction=False) as pipeline:
                    pipeline.get(key)
                    pipeline.ttl(key)
                    worker_id, ttl = pipeline.execute()
                if worker_id and int(ttl) > 0:
                    result.append((str(worker_id), int(ttl)))
            return sorted(result, key=lambda item: item[0])

        return self._translate(operation)

    def worker_registration(self) -> tuple[str | None, int | None]:
        registrations = self.worker_registrations()
        if registrations:
            return registrations[0]
        # Backward-compatible read during a rolling update from the old image.
        def legacy_operation() -> tuple[str | None, int | None]:
            with self.client.pipeline(transaction=False) as pipeline:
                pipeline.get(self.worker_key)
                pipeline.ttl(self.worker_key)
                worker_id, ttl = pipeline.execute()
            return worker_id, int(ttl) if worker_id is not None else None

        return self._translate(legacy_operation)

    def close(self) -> None:
        self.client.close()
