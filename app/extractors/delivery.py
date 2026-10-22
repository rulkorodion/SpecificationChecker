"""Извлечение срока и вида поставки."""

from __future__ import annotations

import re

from app.extractors.base import extracted_field, uncertain_field
from app.extractors.utils import (
    blocks_containing,
    categorical_confidence,
    context_window,
    label_value_context,
    union_box,
)
from app.normalization.conditions import classify_delivery
from app.normalization.dates import normalize_date
from app.ocr.models import OCRDocumentResult, OCRTextBlock
from app.schemas import ExtractedField

DATE_RE = re.compile(
    r"\b(?:[\dОоOo]{1,2}[./][\dОоOo]{1,2}[./][\dОоOo]{4}|"
    r"\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)\s+\d{4})\b"
)


def _repair_numeric_date_ocr(value: str) -> str:
    """Исправить только однозначную OCR-подмену нуля внутри числовой даты."""

    if not re.fullmatch(r"[\dОоOo]{1,2}[./][\dОоOo]{1,2}[./][\dОоOo]{4}", value):
        return value
    return value.translate(str.maketrans({"О": "0", "о": "0", "O": "0", "o": "0"}))


def _date_block_near_anchor(
    document: OCRDocumentResult,
    anchor: OCRTextBlock | None,
) -> tuple[re.Match[str] | None, OCRTextBlock | None]:
    """Prefer a date geometrically tied to the label over unrelated text above it."""

    if anchor is None:
        return None, None
    page = next(
        (item for item in document.pages if item.page_number == anchor.page_number),
        None,
    )
    if page is None:
        return None, None
    anchor_box = anchor.bounding_box
    anchor_y = (anchor_box.y_min + anchor_box.y_max) / 2
    tolerance = max(
        anchor_box.y_max - anchor_box.y_min,
        page.height * 0.014,
    )
    ranked: list[tuple[float, re.Match[str], OCRTextBlock]] = []
    for block in page.blocks:
        match = DATE_RE.search(block.text)
        if match is None:
            continue
        box = block.bounding_box
        center_y = (box.y_min + box.y_max) / 2
        delta_y = abs(center_y - anchor_y)
        if delta_y > tolerance:
            continue
        if block is not anchor and box.x_max < anchor_box.x_min:
            continue
        horizontal_gap = max(0.0, box.x_min - anchor_box.x_max) / page.width
        same_or_right = box.x_min >= anchor_box.x_min - page.width * 0.02
        score = (
            delta_y / page.height
            + horizontal_gap * 0.35
            + (0.0 if same_or_right else 0.25)
        )
        ranked.append((score, match, block))
    if not ranked:
        return None, None
    _, match, block = min(ranked, key=lambda item: item[0])
    return match, block


def extract_delivery(
    document: OCRDocumentResult, config: dict
) -> tuple[ExtractedField, ExtractedField]:
    deadline_link = label_value_context(
        document.pages,
        "срок поставки",
        competing_anchors=["товар поставляется на условиях", "условия оплаты"],
    )
    deadline_context = deadline_link.text or context_window(
        document.normalized_full_text, "срок поставки", 260
    )
    date_match, date_block = _date_block_near_anchor(
        document, deadline_link.anchor
    )
    if date_match is not None and date_block is not None:
        deadline_context = " ".join(
            item
            for item in (
                deadline_link.anchor.text if deadline_link.anchor else "",
                date_block.text,
            )
            if item
        )
        deadline_blocks = list(
            {
                id(block): block
                for block in (deadline_link.anchor, date_block)
                if block is not None
            }.values()
        )
    else:
        date_match = DATE_RE.search(deadline_context)
        deadline_blocks = deadline_link.blocks or blocks_containing(
            document.pages, ["срок поставки", "не позднее"]
        )
    if not date_match:
        deadline = uncertain_field(
            "DELIVERY_DEADLINE",
            "После якоря «Срок поставки» не найдена однозначная дата",
            "delivery_context",
        )
    else:
        try:
            repaired_date = _repair_numeric_date_ocr(date_match.group(0))
            normalized = normalize_date(repaired_date)
            ocr_quality = (
                sum(x.effective_confidence for x in deadline_blocks) / len(deadline_blocks)
                if deadline_blocks
                else 0.0
            )
            confidence = min(
                0.99,
                0.70 * ocr_quality
                + 0.30
                * (
                    deadline_link.geometry_score
                    if deadline_link.anchor is not None
                    else 0.5
                ),
            )
            deadline = extracted_field(
                "DELIVERY_DEADLINE",
                date_match.group(0),
                normalized,
                confidence,
                "delivery_context",
                deadline_blocks[0].page_number if deadline_blocks else None,
                union_box(deadline_blocks),
                [
                    deadline_context,
                    f"label_geometry={deadline_link.geometry_score:.4f}",
                ],
            )
        except ValueError as exc:
            deadline = uncertain_field("DELIVERY_DEADLINE", str(exc), "delivery_context")

    terms_link = label_value_context(
        document.pages,
        "товар поставляется на условиях",
        competing_anchors=["срок поставки", "условия оплаты"],
    )
    terms = terms_link.text or context_window(
        document.normalized_full_text, "товар поставляется на условиях", 350
    )
    delivery_type = classify_delivery(terms, config) if terms else None
    term_blocks = terms_link.blocks or blocks_containing(
        document.pages, ["товар поставляется на условиях", "доставка", "самовывоз"]
    )
    if delivery_type is None:
        type_field = uncertain_field(
            "DELIVERY_TYPE",
            "В контексте условий поставки нет однозначной классификации",
            "delivery_terms_keywords",
        )
    else:
        confidence = categorical_confidence(
            term_blocks,
            geometry_score=terms_link.geometry_score if terms_link.anchor else 0.5,
            dictionary_match=True,
            no_conflict=True,
        )
        type_field = extracted_field(
            "DELIVERY_TYPE",
            terms,
            delivery_type,
            confidence,
            "delivery_terms_keywords",
            term_blocks[0].page_number if term_blocks else None,
            union_box(term_blocks),
            [
                terms,
                f"categorical_confidence={confidence:.4f}",
                f"geometry_score={terms_link.geometry_score:.4f}",
            ],
        )
    return deadline, type_field
