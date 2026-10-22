"""Local JSON/file adapter used until the real 1C contract is known."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.integrations.base import OneCAdapter, OneCIntegrationError
from app.integrations.contracts import (
    CanonicalCheckResult,
    CanonicalDocument,
    DeliveryReceipt,
)
from app.integrations.settings import OneCIntegrationSettings
from app.schemas import ReferenceData

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


class MockOneCAdapter(OneCAdapter):
    name = "mock"

    def __init__(self, settings: OneCIntegrationSettings) -> None:
        self.settings = settings

    def reference_to_internal(
        self, source: Mapping[str, Any] | str | Path
    ) -> ReferenceData:
        try:
            if isinstance(source, Mapping):
                payload = dict(source)
            else:
                path = Path(source)
                payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise OneCIntegrationError("Reference JSON должен быть объектом")
            return ReferenceData.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise OneCIntegrationError("Не удалось преобразовать reference в ReferenceData") from exc

    def get_document(self, source: Any) -> CanonicalDocument:
        try:
            path = Path(source).resolve(strict=True)
        except (TypeError, OSError) as exc:
            raise OneCIntegrationError("Локальный документ недоступен") from exc
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in MEDIA_TYPES:
            raise OneCIntegrationError("Mock-адаптер поддерживает PDF/JPG/JPEG/PNG")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return CanonicalDocument(
            path=path,
            filename=path.name,
            media_type=MEDIA_TYPES[suffix],
            size_bytes=path.stat().st_size,
            sha256=digest.hexdigest(),
        )

    def build_result_payload(
        self, result: CanonicalCheckResult
    ) -> dict[str, Any]:
        # This is deliberately the canonical JSON itself. A future adapter may
        # rename/nest fields only here after the official 1C contract arrives.
        return result.model_dump(mode="json")

    def send_result(self, payload: Mapping[str, Any]) -> DeliveryReceipt:
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            raise OneCIntegrationError("В результате отсутствует job_id")
        destination = self.settings.mock_outbox_dir / f"{job_id}.json"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(dict(payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except OSError as exc:
            raise OneCIntegrationError("MockOneCAdapter не смог сохранить ответ") from exc
        return DeliveryReceipt(
            adapter=self.name,
            accepted=True,
            sent_at=datetime.now(timezone.utc),
            external_message_id=job_id,
            details={"path": str(destination.resolve())},
        )
