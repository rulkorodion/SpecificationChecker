from __future__ import annotations

import os
import time

from app.api.rejections import cleanup_rejected_requests
from app.api.storage import APISettings


def test_rejected_request_cleanup_removes_expired_and_oldest_entries(tmp_path):
    settings = APISettings(
        jobs_dir=tmp_path / "jobs",
        max_upload_size_bytes=1024,
        max_reference_size_bytes=1024,
        upload_chunk_size_bytes=16,
        rejected_request_retention_hours=1,
        rejected_request_max_entries=1,
        rejected_request_max_total_size_bytes=1024,
    )
    root = settings.jobs_dir / "_rejected"
    expired = root / "expired"
    older = root / "older"
    newest = root / "newest"
    for path in (expired, older, newest):
        path.mkdir(parents=True)
        (path / "rejection.json").write_text("{}", encoding="utf-8")

    now = time.time()
    os.utime(expired, (now - 7200, now - 7200))
    os.utime(older, (now - 120, now - 120))
    os.utime(newest, (now - 60, now - 60))

    result = cleanup_rejected_requests(settings, preserve_request_id="newest")

    assert result["removed"] == 2
    assert not expired.exists()
    assert not older.exists()
    assert newest.is_dir()
