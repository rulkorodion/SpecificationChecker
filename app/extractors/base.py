"""Общие безопасные фабрики ExtractedField."""

from __future__ import annotations

from typing import Any

from app.schemas import BoundingBox, ExtractedField


def extracted_field(
    code: str,
    raw: Any,
    normalized: Any,
    confidence: float,
    method: str,
    page: int | None = None,
    box: BoundingBox | None = None,
    evidence: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ExtractedField:
    return ExtractedField(
        code=code,
        raw_value=raw,
        normalized_value=normalized,
        confidence=max(0.0, min(1.0, confidence)),
        page_number=page,
        bounding_box=box,
        extraction_method=method,
        evidence=evidence or [],
        warnings=warnings or [],
    )


def uncertain_field(code: str, reason: str, method: str) -> ExtractedField:
    return extracted_field(
        code,
        raw=None,
        normalized=None,
        confidence=0.0,
        method=method,
        warnings=[reason],
    )

