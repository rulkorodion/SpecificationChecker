"""Загрузка mock extraction через те же модели и нормализаторы."""

from __future__ import annotations

import json
from pathlib import Path

from app.enums import DeliveryType, PaymentType, PresenceStatus
from app.extractors.base import extracted_field
from app.normalization.dates import normalize_date
from app.normalization.numbers import normalize_decimal
from app.schemas import ExtractedSpecification

FIELD_CODES = {
    "application_number": "APPLICATION_NUMBER",
    "application_date": "APPLICATION_DATE",
    "total_quantity": "TOTAL_QUANTITY",
    "total_with_vat": "TOTAL_WITH_VAT",
    "transportation_costs_included": "TRANSPORTATION_COSTS",
    "delivery_deadline": "DELIVERY_DEADLINE",
    "delivery_type": "DELIVERY_TYPE",
    "payment_type": "PAYMENT_TYPE",
    "deferment_days": "DEFERMENT_DAYS",
    "buyer_signature_present": "BUYER_SIGNATURE",
    "buyer_seal_present": "BUYER_SEAL",
}


def _normalize(name: str, value):
    if value is None:
        return None
    if name in {"application_date", "delivery_deadline"}:
        return normalize_date(value)
    if name in {"total_quantity", "total_with_vat"}:
        return normalize_decimal(value)
    if name == "delivery_type":
        return DeliveryType(value)
    if name == "payment_type":
        return PaymentType(value)
    if name in {"buyer_signature_present", "buyer_seal_present"}:
        return PresenceStatus(value)
    return value


def load_mock_extracted(path: Path) -> ExtractedSpecification:
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = {}
    for name, code in FIELD_CODES.items():
        item = data[name]
        value = _normalize(name, item.get("value"))
        fields[name] = extracted_field(
            code,
            item.get("value"),
            value,
            float(item.get("confidence", 0.99)),
            "mock_input",
            1,
            evidence=["Доверенный mock extraction"],
        )
    return ExtractedSpecification(**fields)
