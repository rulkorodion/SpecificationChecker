"""Deterministic checker process used only by the real infrastructure smoke-test."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.storage import APISettings  # noqa: E402
from app.config import load_config  # noqa: E402
from app.job_storage import JobStorage  # noqa: E402
from app.persistence import Database, JobPersistenceService  # noqa: E402
from app.queueing import QueueSettings, RedisJobQueue  # noqa: E402
from app.worker import SpecificationWorker  # noqa: E402


class DeterministicChecker:
    def __init__(self) -> None:
        self.calls_path = Path(os.environ["SMOKE_CHECKER_CALLS_PATH"])

    def check_specification(self, document_path, reference, runtime_config=None):
        with self.calls_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"document": str(document_path), "reference": reference}) + "\n")
        time.sleep(0.35)
        if reference.get("simulate_failure"):
            raise RuntimeError("deterministic smoke failure")
        return {
            "decision": "APPROVE",
            "extracted_values": {"application_number": "SMOKE-1"},
            "field_details": {},
            "checks": [],
            "reasons": ["Детерминированная smoke-проверка завершена"],
            "comment_1c": "Спецификация согласована.",
            "confidence": {"overall": 1.0, "fields": {}},
            "metadata": {"checker": "deterministic-smoke"},
        }


def main() -> None:
    config = load_config()
    settings = APISettings.from_config(config)
    worker = SpecificationWorker(
        queue=RedisJobQueue(QueueSettings.from_environment()),
        job_service=JobPersistenceService(Database.from_environment()),
        checker=DeterministicChecker(),
        storage=JobStorage(settings.jobs_dir),
    )
    import threading
    import signal

    stop = threading.Event()
    stop_file_value = os.getenv("SMOKE_WORKER_STOP_FILE")
    if stop_file_value:
        stop_file = Path(stop_file_value)

        def watch_stop_file() -> None:
            while not stop.wait(0.1):
                if stop_file.exists():
                    stop.set()

        threading.Thread(target=watch_stop_file, daemon=True).start()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        worker.run_forever(stop)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
