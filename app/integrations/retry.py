"""Manual resend command for a final job; never invokes OCR."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.config import load_config
from app.integrations.factory import build_onec_adapter
from app.integrations.service import OneCIntegrationService
from app.integrations.settings import OneCIntegrationSettings
from app.persistence import Database, JobPersistenceService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Повторно отправить сохранённый результат в адаптер 1С без OCR"
    )
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    config_value = os.getenv("SPECIFICATION_CHECKER_CONFIG", "").strip()
    config = load_config(Path(config_value) if config_value else None)
    settings = OneCIntegrationSettings.from_environment(config)
    adapter = build_onec_adapter(settings)
    persistence = JobPersistenceService(Database.from_environment())
    integration = OneCIntegrationService(
        settings=settings,
        adapter=adapter,
        persistence=persistence,
    )
    try:
        record = persistence.get(args.job_id)
        if record is None:
            parser.error("job_id не найден в PostgreSQL")
        state = integration.retry(record)
        if state is None:
            parser.error("ONEC_INTEGRATION_ENABLED=false")
        print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        integration.close()
        persistence.close()


if __name__ == "__main__":
    main()
