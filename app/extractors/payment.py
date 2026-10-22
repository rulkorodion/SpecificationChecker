"""Извлечение вида оплаты и дней отсрочки."""

from __future__ import annotations

import re

from app.enums import PaymentType
from app.extractors.base import extracted_field, uncertain_field
from app.extractors.utils import (
    blocks_containing,
    categorical_confidence,
    context_window,
    label_value_context,
    union_box,
)
from app.normalization.conditions import (
    classify_payment,
    normalize_payment_numeric_tokens,
)
from app.ocr.models import OCRDocumentResult
from app.schemas import ExtractedField


def extract_payment(
    document: OCRDocumentResult, config: dict
) -> tuple[ExtractedField, ExtractedField]:
    linked = label_value_context(
        document.pages,
        "условия оплаты",
        competing_anchors=["срок поставки", "товар поставляется на условиях"],
    )
    context = linked.text
    normalized_context, numeric_repairs = normalize_payment_numeric_tokens(context)
    payment_type = (
        classify_payment(normalized_context, config) if normalized_context else None
    )
    # Parser models sometimes put a label and its value into consecutive
    # blocks instead of one row. If the geometrically linked block contains
    # only the label, inspect the bounded payment context before failing.
    if payment_type is None:
        fallback_context = context_window(
            document.normalized_full_text, "условия оплаты", 500
        )
        fallback_normalized, fallback_repairs = normalize_payment_numeric_tokens(
            fallback_context
        )
        fallback_type = (
            classify_payment(fallback_normalized, config)
            if fallback_normalized
            else None
        )
        if fallback_type is not None:
            context = fallback_context
            normalized_context = fallback_normalized
            numeric_repairs.extend(fallback_repairs)
            payment_type = fallback_type
    blocks = linked.blocks or blocks_containing(
        document.pages,
        [
            "условия оплаты",
            "предоплат",
            "предполат",
            "отсроч",
            "календарных дней",
            "по факту готовности",
        ],
    )
    confidence = categorical_confidence(
        blocks,
        geometry_score=linked.geometry_score if linked.anchor else 0.5,
        dictionary_match=payment_type is not None,
        no_conflict=payment_type is not None,
    )
    if payment_type is None:
        payment = uncertain_field(
            "PAYMENT_TYPE",
            "В пункте условий оплаты нет однозначной классификации",
            "payment_terms_keywords",
        )
        days = uncertain_field(
            "DEFERMENT_DAYS", "Тип оплаты не определён", "payment_terms_regex"
        )
        return payment, days
    payment = extracted_field(
        "PAYMENT_TYPE",
        context,
        payment_type,
        confidence,
        "payment_terms_keywords",
        blocks[0].page_number if blocks else None,
        union_box(blocks),
        [
            normalized_context,
            f"categorical_confidence={confidence:.4f}",
            f"geometry_score={linked.geometry_score:.4f}",
            *numeric_repairs,
        ],
    )
    if payment_type is PaymentType.PREPAYMENT:
        days = extracted_field(
            "DEFERMENT_DAYS",
            None,
            None,
            1.0,
            "not_applicable_prepayment",
            evidence=["Неприменимо при предоплате"],
        )
        return payment, days
    match = re.search(
        r"(?:в течение|отсроч\w*)\s*(\d{1,4})\s*(?:календарн|банковск)?",
        normalized_context,
    )
    if not match:
        match = re.search(
            r"(\d{1,4})\s+(?:календарных|банковских)\s+дн",
            normalized_context,
        )
    if not match:
        days = uncertain_field(
            "DEFERMENT_DAYS",
            "Отсрочка найдена, но число дней не извлечено",
            "payment_terms_regex",
        )
    else:
        value = int(match.group(1))
        days = extracted_field(
            "DEFERMENT_DAYS",
            match.group(1),
            value if value > 0 else None,
            confidence if value > 0 else 0.0,
            "payment_terms_regex",
            blocks[0].page_number if blocks else None,
            union_box(blocks),
            [normalized_context, *numeric_repairs],
        )
    return payment, days
