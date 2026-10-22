"""Извлечение номера и даты приложения из OCR, без эталонных подсказок."""

from __future__ import annotations

import re

from app.extractors.base import extracted_field, uncertain_field
from app.extractors.utils import blocks_containing, union_box
from app.normalization.dates import normalize_date
from app.normalization.text import normalize_application_number
from app.ocr.models import OCRDocumentResult
from app.schemas import ExtractedField

DATE_PATTERN = (
    r"(?:\d{1,2}[./]\d{1,2}[./]\d{4}|"
    r"\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)\s+\d{4})"
)


def extract_application(
    document: OCRDocumentResult,
) -> tuple[ExtractedField, ExtractedField]:
    text = document.normalized_full_text
    match = re.search(
        rf"приложени[ея]\s*(?:№|no|nº|n)?\s*([0-9a-zа-я][0-9a-zа-я./-]*)\s*от\s*({DATE_PATTERN})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        missing = "Не найден уверенный заголовок «Приложение № ... от ...»"
        return (
            uncertain_field("APPLICATION_NUMBER", missing, "application_regex"),
            uncertain_field("APPLICATION_DATE", missing, "application_regex"),
        )
    raw_number, raw_date = match.group(1), match.group(2)
    blocks = blocks_containing(document.pages, ["приложение", raw_number])
    header_blocks = [
        block
        for block in blocks
        if block.bounding_box.y_min / block.bounding_box.page_height < 0.35
    ]
    evidence = [match.group(0)]
    confidence = min((x.effective_confidence for x in header_blocks), default=0.9)
    page = header_blocks[0].page_number if header_blocks else 1
    box = union_box(header_blocks)
    number = extracted_field(
        "APPLICATION_NUMBER",
        raw_number,
        normalize_application_number(raw_number),
        confidence,
        "application_anchor_regex",
        page,
        box,
        evidence,
    )
    try:
        normalized_date = normalize_date(raw_date)
        date_field = extracted_field(
            "APPLICATION_DATE",
            raw_date,
            normalized_date,
            confidence,
            "application_anchor_regex",
            page,
            box,
            evidence,
        )
    except ValueError as exc:
        date_field = uncertain_field("APPLICATION_DATE", str(exc), "application_anchor_regex")
    return number, date_field
