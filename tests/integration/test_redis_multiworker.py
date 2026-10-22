from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.queueing import QueueSettings, RedisJobQueue


@pytest.mark.integration
def test_two_workers_have_independent_registrations_and_inflight_lists() -> None:
    redis_url = os.getenv("TEST_REDIS_URL", "").strip()
    if not redis_url:
        pytest.skip("TEST_REDIS_URL is not configured")
    queue_name = f"specification_checks_test_{uuid4().hex}"
    queue = RedisJobQueue(
        QueueSettings(
            redis_url=redis_url,
            queue_name=queue_name,
            block_timeout_seconds=1,
            worker_lock_ttl_seconds=60,
        )
    )
    worker_0 = "worker-gpu-0"
    worker_1 = "worker-gpu-1"
    try:
        assert queue.acquire_worker(worker_0) is True
        assert queue.acquire_worker(worker_1) is True
        assert queue.acquire_worker(worker_0) is False
        assert [item[0] for item in queue.worker_registrations()] == [
            worker_0,
            worker_1,
        ]

        queue.enqueue("job-0")
        queue.enqueue("job-1")
        assert queue.dequeue(worker_0, 1) == "job-0"
        assert queue.dequeue(worker_1, 1) == "job-1"

        queue.acknowledge("job-0", worker_0)
        assert queue.client.llen(queue._processing_key(worker_0)) == 0
        assert queue.client.lrange(queue._processing_key(worker_1), 0, -1) == [
            "job-1"
        ]

        queue.release_worker(worker_1)
        assert queue.recover_inflight(worker_0) == 1
        assert queue.dequeue(worker_0, 1) == "job-1"
        queue.acknowledge("job-1", worker_0)
        assert queue.client.llen(queue.pending_key) == 0
        assert queue.client.llen(queue._processing_key(worker_0)) == 0
    finally:
        for key in queue.client.scan_iter(match=f"{queue_name}:*"):
            queue.client.delete(key)
        queue.close()
