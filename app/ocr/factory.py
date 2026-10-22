"""OCR backend factory."""

from __future__ import annotations

from typing import Any

from app.exceptions import OCREngineError
from app.ocr.base import OCREngine
from app.ocr.monkey_engine import MonkeyOCRParsingEngine


def build_ocr_engine(config: dict[str, Any], language: str | None = None) -> OCREngine:
    ocr = config["ocr"]
    backend = str(ocr.get("backend", "monkeyocr_v2")).lower().replace("-", "_")
    if backend == "monkeyocr_v2":
        monkey = dict(ocr.get("monkeyocr_v2", {}))
        monkey.update(dict(monkey.get("effective", {})))
        monkey["device"] = ocr.get("device", "auto")
        return MonkeyOCRParsingEngine(monkey)
    raise OCREngineError(f"Неизвестный OCR backend: {backend}")
