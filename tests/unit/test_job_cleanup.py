from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.job_cleanup import JobFileCleanupService, JobRetentionSettings
from app.job_storage import JobStorage
from app.persistence import Base, Database, JobPersistenceService, JobStatus


def make_job(service, storage, *, completed_at=None):
    job_id = str(uuid4())
    folder = storage.job_dir(job_id); folder.mkdir(parents=True)
    document = folder / "document.pdf"; document.write_bytes(b"%PDF-1.4")
    (folder / "reference.json").write_text("{}", "utf-8")
    started = (completed_at or datetime.now(timezone.utc)) - timedelta(seconds=1)
    sha = hashlib.sha256(document.read_bytes()).hexdigest()
    service.create_processing(
        job_id=job_id, external_id=None, created_at=started, started_at=started,
        document_path=str(document), document_sha256=sha,
    )
    if completed_at is not None:
        service.complete(
            job_id,
            result={"decision": "APPROVE", "comment_1c": "ok"},
            completed_at=completed_at,
            processing_time_seconds=1.0,
        )
    return job_id, folder, sha


def test_cleanup_removes_only_expired_final_job_files_and_keeps_database(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    Base.metadata.create_all(database.engine)
    persistence = JobPersistenceService(database)
    storage = JobStorage(tmp_path / "jobs")
    now = datetime.now(timezone.utc)
    old_id, old_folder, old_sha = make_job(
        persistence, storage, completed_at=now - timedelta(days=31)
    )
    recent_id, recent_folder, _ = make_job(
        persistence, storage, completed_at=now - timedelta(days=2)
    )
    active_id, active_folder, _ = make_job(persistence, storage)
    cleanup = JobFileCleanupService(
        settings=JobRetentionSettings(retention_days=30, batch_size=10),
        persistence=persistence,
        storage=storage,
    )

    preview = cleanup.run(now=now, dry_run=True)
    assert preview.scanned == 1 and preview.removed == 1
    assert old_folder.exists()
    assert persistence.get(old_id).document_path is not None

    report = cleanup.run(now=now)
    old_record = persistence.get(old_id)
    assert report.removed == 1 and report.failed == 0
    assert not old_folder.exists()
    assert old_record is not None and old_record.status is JobStatus.COMPLETED
    assert old_record.result == {"decision": "APPROVE", "comment_1c": "ok"}
    assert old_record.document_path is None
    assert old_record.document_sha256 == old_sha
    assert recent_folder.exists() and persistence.get(recent_id).document_path
    assert active_folder.exists() and persistence.get(active_id).status is JobStatus.PROCESSING


def test_cleanup_marks_missing_expired_directory_without_deleting_record(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    Base.metadata.create_all(database.engine)
    persistence = JobPersistenceService(database)
    storage = JobStorage(tmp_path / "jobs")
    now = datetime.now(timezone.utc)
    job_id, folder, _ = make_job(
        persistence, storage, completed_at=now - timedelta(days=90)
    )
    for item in folder.iterdir():
        item.unlink()
    folder.rmdir()
    cleanup = JobFileCleanupService(
        settings=JobRetentionSettings(retention_days=30),
        persistence=persistence,
        storage=storage,
    )
    report = cleanup.run(now=now)
    assert report.missing == 1
    assert persistence.get(job_id) is not None
    assert persistence.get(job_id).document_path is None
