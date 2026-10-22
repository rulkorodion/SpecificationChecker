from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.config import load_config
from app.enums import DeliveryType, PaymentType, PresenceStatus
from app.schemas import (
    ExtractedField,
    ExtractedSpecification,
    ReferenceData,
)


def field(code: str, value, confidence: float = 0.99) -> ExtractedField:
    return ExtractedField(
        code=code,
        raw_value=str(value) if value is not None else None,
        normalized_value=value,
        confidence=confidence,
        page_number=1,
        extraction_method="test",
        evidence=[],
    )


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def reference() -> ReferenceData:
    return ReferenceData(
        specification_id="test-001",
        check_date=date(2026, 4, 1),
        application_number="76930",
        application_date=date(2026, 4, 1),
        total_quantity=Decimal("69.284"),
        total_with_vat=Decimal("5272993.70"),
        transportation_costs_included=True,
        delivery_deadline=date(2026, 4, 24),
        delivery_type=DeliveryType.DELIVERY,
        payment_type=PaymentType.PREPAYMENT,
        deferment_days=None,
    )


@pytest.fixture
def extracted() -> ExtractedSpecification:
    return ExtractedSpecification(
        application_number=field("APPLICATION_NUMBER", "76930"),
        application_date=field("APPLICATION_DATE", date(2026, 4, 1)),
        total_quantity=field("TOTAL_QUANTITY", Decimal("69.284")),
        total_with_vat=field("TOTAL_WITH_VAT", Decimal("5272993.70")),
        transportation_costs_included=field("TRANSPORTATION_COSTS", True),
        delivery_deadline=field("DELIVERY_DEADLINE", date(2026, 4, 24)),
        delivery_type=field("DELIVERY_TYPE", DeliveryType.DELIVERY),
        payment_type=field("PAYMENT_TYPE", PaymentType.PREPAYMENT),
        deferment_days=field("DEFERMENT_DAYS", None),
        buyer_signature_present=field("BUYER_SIGNATURE", PresenceStatus.PRESENT),
        buyer_seal_present=field("BUYER_SEAL", PresenceStatus.PRESENT),
    )
