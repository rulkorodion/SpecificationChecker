"""Trusted shared filesystem storage used by API and worker processes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from app.persistence.repository import CheckJobRecord


class JobStorageError(RuntimeError):
    """A persisted job does not point to a valid, trusted input."""


@dataclass(frozen=True)
class JobInput:
    document_path: Path
    reference: dict[str, Any]
    output_dir: Path


class JobStorage:
    ALLOWED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    @staticmethod
    def _validated_job_id(job_id: str) -> str:
        try:
            normalized = str(UUID(job_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise JobStorageError("Некорректный job_id") from exc
        if normalized != job_id.lower():
            raise JobStorageError("job_id должен быть каноническим UUID")
        return normalized

    def job_dir(self, job_id: str) -> Path:
        return self.root / self._validated_job_id(job_id)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_input(self, record: CheckJobRecord) -> JobInput:
        if not record.document_path or not record.document_sha256:
            raise JobStorageError("В задании отсутствуют путь или SHA-256 документа")
        job_dir = self.job_dir(record.job_id).resolve()
        document = Path(record.document_path).resolve()
        try:
            document.relative_to(job_dir)
        except ValueError as exc:
            raise JobStorageError("Путь документа находится вне каталога задания") from exc
        if document.parent != job_dir or document.stem != "document":
            raise JobStorageError("Недопустимое имя документа задания")
        if document.suffix.lower() not in self.ALLOWED_SUFFIXES:
            raise JobStorageError("Недопустимый формат документа задания")
        if not document.is_file():
            raise JobStorageError("Файл документа задания отсутствует")
        actual_hash = self._sha256(document)
        if actual_hash != record.document_sha256.lower():
            raise JobStorageError("SHA-256 документа не совпадает с сохранённым")
        reference_path = job_dir / "reference.json"
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JobStorageError("Reference задания отсутствует или повреждён") from exc
        if not isinstance(reference, dict):
            raise JobStorageError("Reference задания должен быть JSON-объектом")
        return JobInput(document, reference, job_dir / "result")

    def outcome_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "worker_outcome.json"

    def write_outcome(self, job_id: str, payload: dict[str, Any]) -> None:
        destination = self.outcome_path(job_id)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
        )
        temporary.replace(destination)

    def read_outcome(self, job_id: str) -> dict[str, Any] | None:
        path = self.outcome_path(job_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JobStorageError("Сохранённый результат worker повреждён") from exc
        if not isinstance(value, dict):
            raise JobStorageError("Сохранённый результат worker имеет неверный формат")
        return value

    def delete_outcome(self, job_id: str) -> None:
        self.outcome_path(job_id).unlink(missing_ok=True)
