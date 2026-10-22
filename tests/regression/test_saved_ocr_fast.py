"""Быстрая регрессия extractors на сохранённом OCR без запуска модели."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import cv2

from app.config import load_config
from app.enums import PresenceStatus
from app.extractors.buyer_block import _buyer_label_anchors
from app.extractors.delivery import extract_delivery
from app.extractors.payment import extract_payment
from app.extractors.totals import extract_totals
from app.ocr.models import OCRDocumentResult, OCRPageResult
from app.vision.seal_detector import OpenCVSealDetector, SealDetectionContext
from app.vision.signature_detector import (
    OpenCVSignatureDetector,
    SignatureDetectionContext,
)

OUTPUT = Path("output")


def _saved_ocr(number: int) -> OCRDocumentResult:
    directory = OUTPUT / f"specification_{number}" / "ocr"
    paths = sorted(directory.glob("page_*.json"))
    if not paths:
        pytest.skip(f"Нет локального OCR fixture: {directory}")
    pages = [
        OCRPageResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    return OCRDocumentResult(
        pages=pages,
        normalized_full_text=(directory / "full_text.txt").read_text(encoding="utf-8"),
        engine_name="saved-regression-ocr",
    )


@pytest.mark.parametrize(
    ("number", "quantity", "total"),
    [
        (25, Decimal("5.040"), Decimal("309960")),
        (27, Decimal("7.326"), Decimal("492840")),
        (28, Decimal("3.000"), Decimal("274500")),
        (29, Decimal("12.177"), Decimal("775944.27")),
    ],
)
def test_saved_ocr_totals(number: int, quantity: Decimal, total: Decimal):
    actual_quantity, actual_total, _ = extract_totals(_saved_ocr(number))
    assert actual_quantity.normalized_value == quantity
    assert actual_total.normalized_value == total


def test_specification_27_contextual_deferment_days():
    payment, days = extract_payment(_saved_ocr(27), load_config())
    assert payment.normalized_value.value == "DEFERMENT"
    assert days.normalized_value == 30


def test_specification_26_fuzzy_buyer_anchor_and_saved_visual_roi():
    document = _saved_ocr(26)
    anchors = _buyer_label_anchors(document.pages[0])
    assert anchors, "OCR-вариант «Мокупателв» должен геометрически связать buyer ROI"

    page_path = OUTPUT / "specification_26" / "pages" / "page_001_original.png"
    if not page_path.is_file():
        pytest.skip(f"Нет локального visual fixture: {page_path}")
    page = cv2.imread(str(page_path))
    roi = page[2598:3508, 1299:2456]
    anchor = anchors[-1]
    relative_anchor = (
        int(anchor.x_min) - 1299,
        int(anchor.y_min) - 2598,
        int(anchor.x_max) - 1299,
        int(anchor.y_max) - 2598,
    )
    seal = OpenCVSealDetector().detect(
        roi,
        SealDetectionContext("anchor_right_extended", relative_anchor, True),
    )
    signature = OpenCVSignatureDetector().detect(
        roi,
        SignatureDetectionContext("anchor_right_extended", relative_anchor, True),
    )
    assert seal.status is PresenceStatus.PRESENT
    assert signature.status is PresenceStatus.PRESENT


def test_specification_29_label_value_geometry():
    document = _saved_ocr(29)
    deadline, delivery = extract_delivery(document, load_config())
    payment, _ = extract_payment(document, load_config())
    assert deadline.normalized_value == date(2026, 8, 7)
    assert delivery.normalized_value.value == "PICKUP"
    assert payment.normalized_value.value == "PREPAYMENT"


def test_specification_31_dictionary_geometry_confidence():
    _, delivery = extract_delivery(_saved_ocr(31), load_config())
    assert delivery.normalized_value.value == "PICKUP"
    assert delivery.confidence >= load_config()["extraction"]["field_confidence_threshold"]
