"""Docker health probe for the worker infrastructure and registration."""

import os

from app.persistence import Database
from app.queueing import QueueSettings, RedisJobQueue


def main() -> None:
    database = Database.from_environment()
    queue = RedisJobQueue(QueueSettings.from_environment())
    try:
        database.ping()
        queue.ping()
        registrations = queue.worker_registrations()
        expected_worker_id = os.getenv("WORKER_ID", "").strip()
        if not registrations:
            raise SystemExit("worker registration is absent")
        if expected_worker_id and expected_worker_id not in {
            worker_id for worker_id, _ttl in registrations
        }:
            raise SystemExit(
                f"worker registration is absent: {expected_worker_id}"
            )
    finally:
        queue.close()
        database.dispose()


if __name__ == "__main__":
    main()
