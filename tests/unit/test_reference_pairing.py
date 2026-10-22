from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.enums import DeliveryType, PaymentType
from app.extractors.base import extracted_field, uncertain_field
from app.ocr.models import OCRDocumentResult
from app.pipeline import possible_reference_pairing_warning
from app.schemas import ExtractedSpecification, ReferenceData


def _extracted(number: str, confidence: float) -> ExtractedSpecification:
    number_field = extracted_field(
        "APPLICATION_NUMBER",
        number,
        number,
        confidence,
        "test",
    )
    unknown = uncertain_field("UNKNOWN", "test", "test")
    return ExtractedSpecification(
        application_number=number_field,
        application_date=unknown.model_copy(update={"code": "APPLICATION_DATE"}),
        total_quantity=unknown.model_copy(update={"code": "TOTAL_QUANTITY"}),
        total_with_vat=unknown.model_copy(update={"code": "TOTAL_WITH_VAT"}),
        transportation_costs_included=unknown.model_copy(
            update={"code": "TRANSPORTATION_COSTS"}
        ),
        delivery_deadline=unknown.model_copy(update={"code": "DELIVERY_DEADLINE"}),
        delivery_type=unknown.model_copy(update={"code": "DELIVERY_TYPE"}),
        payment_type=unknown.model_copy(update={"code": "PAYMENT_TYPE"}),
        deferment_days=unknown.model_copy(update={"code": "DEFERMENT_DAYS"}),
        buyer_signature_present=unknown.model_copy(
            update={"code": "BUYER_SIGNATURE"}
        ),
        buyer_seal_present=unknown.model_copy(update={"code": "BUYER_SEAL"}),
    )


def _reference(number: str) -> ReferenceData:
    return ReferenceData(
        specification_id="test",
        check_date=date(2026, 7, 30),
        application_number=number,
        application_date=date(2026, 7, 29),
        total_quantity=Decimal("1"),
        total_with_vat=Decimal("1"),
        transportation_costs_included=False,
        delivery_deadline=date(2026, 7, 31),
        delivery_type=DeliveryType.PICKUP,
        payment_type=PaymentType.PREPAYMENT,
        deferment_days=None,
    )


def test_repeated_document_number_flags_probable_reference_mixup():
    document = OCRDocumentResult(
        pages=[],
        normalized_full_text=(
            "приложение no 208353 от 29 июля 2026 "
            "приложение no 208353 стр 2 из 2"
        ),
        engine_name="fake",
    )
    warning = possible_reference_pairing_warning(
        _extracted("208353", 0.97),
        _reference("208371"),
        document,
    )
    assert warning is not None
    assert "PDF/JSON" in warning


def test_low_confidence_or_expected_number_in_text_does_not_flag_mixup():
    document = OCRDocumentResult(
        pages=[],
        normalized_full_text="208353 208353 208371",
        engine_name="fake",
    )
    assert (
        possible_reference_pairing_warning(
            _extracted("208353", 0.97),
            _reference("208371"),
            document,
        )
        is None
    )
    assert (
        possible_reference_pairing_warning(
            _extracted("208353", 0.70),
            _reference("208371"),
            document.model_copy(
                update={"normalized_full_text": "208353 208353"}
            ),
        )
        is None
    )
