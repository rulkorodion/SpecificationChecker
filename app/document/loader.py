"""Валидация документа и вычисление SHA-256."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from app.exceptions import InputValidationError


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    path: Path
    extension: str
    mime_type: str
    sha256: str
    size_bytes: int


class DocumentLoader:
    """Проверить входной путь без изменения исходного файла."""

    def __init__(self, supported_extensions: list[str]) -> None:
        self.supported = {item.lower() for item in supported_extensions}

    def load(self, path: Path) -> LoadedDocument:
        if not path.exists():
            raise InputValidationError(f"Документ не найден: {path}")
        if not path.is_file():
            raise InputValidationError(f"Путь не является файлом: {path}")
        extension = path.suffix.lower()
        if extension not in self.supported:
            raise InputValidationError(
                f"Формат {extension or '<без расширения>'} не поддерживается"
            )
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            size = path.stat().st_size
        except OSError as exc:
            raise InputValidationError(f"Не удалось прочитать документ: {exc}") from exc
        if size == 0:
            raise InputValidationError("Документ пуст")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return LoadedDocument(path, extension, mime, digest.hexdigest(), size)

