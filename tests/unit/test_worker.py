from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from app.job_storage import JobStorage
from app.persistence import Base, Database, JobPersistenceService, JobStatus
from app.queueing import QueueSettings
from app.worker import SpecificationWorker


def result():
    return {
        "decision": "APPROVE", "extracted_values": {}, "field_details": {},
        "checks": [], "reasons": [], "comment_1c": "Согласовано.",
        "confidence": {"overall": 0.99, "fields": {}}, "metadata": {},
    }


class Queue:
    def __init__(self):
        self.settings = QueueSettings(redis_url="redis://test", block_timeout_seconds=1)
        self.pending = []
        self.processing_by_worker = {}
        self.workers = {}
    @property
    def processing(self):
        return [item for values in self.processing_by_worker.values() for item in values]
    def ping(self): pass
    def enqueue(self, item): self.pending.append(item)
    def dequeue(self, worker_id, timeout_seconds=None):
        if not self.pending: return None
        item = self.pending.pop(0)
        self.processing_by_worker.setdefault(worker_id, []).append(item)
        return item
    def acknowledge(self, item, worker_id):
        processing = self.processing_by_worker.setdefault(worker_id, [])
        if item in processing: processing.remove(item)
    def requeue(self, item, worker_id):
        self.acknowledge(item, worker_id); self.pending.append(item)
    def recover_inflight(self, worker_id, include_current=False): return 0
    def acquire_worker(self, worker_id):
        if worker_id in self.workers: return False
        self.workers[worker_id] = 60; return True
    def refresh_worker(self, worker_id): return worker_id in self.workers
    def release_worker(self, worker_id): self.workers.pop(worker_id, None)
    def worker_registration(self):
        return next(iter(self.workers.items()), (None, None))
    def worker_registrations(self): return sorted(self.workers.items())
    def close(self): pass


class Checker:
    def __init__(self): self.calls = 0; self.error = None; self.runtime_configs = []
    def check_specification(self, document, reference, runtime_config=None):
        self.calls += 1
        self.runtime_configs.append(dict(runtime_config or {}))
        if self.error: raise self.error
        return result()


@pytest.fixture
def worker_setup(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    Base.metadata.create_all(database.engine)
    service = JobPersistenceService(database)
    queue = Queue(); checker = Checker(); storage = JobStorage(tmp_path / "jobs")
    worker = SpecificationWorker(queue=queue, job_service=service, checker=checker, storage=storage, worker_id="test-worker")
    return worker, queue, checker, service, storage


def add_job(service, storage, queue, *, present=True):
    job_id = str(uuid4()); folder = storage.job_dir(job_id); folder.mkdir(parents=True)
    document = folder / "document.pdf"
    content = b"%PDF-1.4"
    if present: document.write_bytes(content)
    (folder / "reference.json").write_text(json.dumps({"specification_id": "x"}), "utf-8")
    now = datetime.now(timezone.utc)
    service.create_processing(job_id=job_id, external_id=None, created_at=now, started_at=now, document_path=str(document), document_sha256=hashlib.sha256(content).hexdigest())
    queue.enqueue(job_id)
    return job_id


def test_worker_completes_and_persists_result(worker_setup):
    worker, queue, checker, service, storage = worker_setup
    job_id = add_job(service, storage, queue)
    assert worker.process_one(timeout_seconds=1) == "completed"
    record = service.get(job_id)
    assert record.status is JobStatus.COMPLETED
    assert record.decision == "APPROVE" and record.result == result()
    assert record.completed_at.utcoffset().total_seconds() == 0
    assert record.processing_time_seconds >= 0
    assert checker.calls == 1 and queue.processing == []
    assert checker.runtime_configs[0]["job_id"] == job_id


def test_worker_checker_error_becomes_failed(worker_setup):
    worker, queue, checker, service, storage = worker_setup
    checker.error = ValueError("simulated")
    job_id = add_job(service, storage, queue)
    assert worker.process_one(timeout_seconds=1) == "failed"
    record = service.get(job_id)
    assert record.status is JobStatus.FAILED
    assert record.error_details["exception_type"] == "ValueError"
    assert record.error_details["technical_message"] == "simulated"


def test_completed_job_is_not_run_twice(worker_setup):
    worker, queue, checker, service, storage = worker_setup
    job_id = add_job(service, storage, queue)
    worker.process_one(timeout_seconds=1)
    queue.enqueue(job_id)
    assert worker.process_one(timeout_seconds=1) == "already_final"
    assert checker.calls == 1


def test_missing_job_does_not_stop_worker(worker_setup):
    worker, queue, checker, _, _ = worker_setup
    queue.enqueue(str(uuid4()))
    assert worker.process_one(timeout_seconds=1) == "missing"
    assert checker.calls == 0 and queue.processing == []


def test_missing_document_is_failed_without_checker(worker_setup):
    worker, queue, checker, service, storage = worker_setup
    job_id = add_job(service, storage, queue, present=False)
    assert worker.process_one(timeout_seconds=1) == "failed"
    record = service.get(job_id)
    assert record.status is JobStatus.FAILED
    assert record.error_code == "job_input_unavailable"
    assert checker.calls == 0


def test_saved_outcome_is_reused_without_checker(worker_setup):
    worker, queue, checker, service, storage = worker_setup
    job_id = add_job(service, storage, queue)
    now = datetime.now(timezone.utc)
    storage.write_outcome(job_id, {"kind": "completed", "completed_at": now.isoformat(), "processing_time_seconds": 0.2, "result": result()})
    assert worker.process_one(timeout_seconds=1) == "recovered_outcome"
    assert service.get(job_id).status is JobStatus.COMPLETED
    assert checker.calls == 0


def test_two_workers_claim_different_jobs_from_the_same_queue(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'parallel.sqlite'}")
    Base.metadata.create_all(database.engine)
    service = JobPersistenceService(database)
    queue = Queue()
    storage = JobStorage(tmp_path / "jobs")
    checker_0 = Checker()
    checker_1 = Checker()
    worker_0 = SpecificationWorker(
        queue=queue,
        job_service=service,
        checker=checker_0,
        storage=storage,
        worker_id="worker-gpu-0",
    )
    worker_1 = SpecificationWorker(
        queue=queue,
        job_service=service,
        checker=checker_1,
        storage=storage,
        worker_id="worker-gpu-1",
    )
    job_0 = add_job(service, storage, queue)
    job_1 = add_job(service, storage, queue)

    assert worker_0.process_one(timeout_seconds=1) == "completed"
    assert worker_1.process_one(timeout_seconds=1) == "completed"

    assert service.get(job_0).status is JobStatus.COMPLETED
    assert service.get(job_1).status is JobStatus.COMPLETED
    assert checker_0.calls == checker_1.calls == 1
    assert queue.processing == []
