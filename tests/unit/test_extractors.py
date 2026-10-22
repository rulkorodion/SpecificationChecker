from datetime import date

import pytest

from app.extractors.application import extract_application
from app.extractors.delivery import extract_delivery
from app.extractors.payment import extract_payment
from app.normalization.conditions import classify_delivery
from app.normalization.conditions import normalize_payment_numeric_tokens
from app.ocr.models import OCRDocumentResult, OCRPageResult, OCRTextBlock
from app.schemas import BoundingBox


def block(text, x, y, confidence=0.98):
    return OCRTextBlock(
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + 300,
            y_max=y + 30,
            page_width=1000,
            page_height=1400,
        ),
        page_number=1,
        preprocessing_profile="original",
        engine_name="fake",
    )


def document(text, blocks=None):
    page = OCRPageResult(
        page_number=1,
        width=1000,
        height=1400,
        blocks=blocks or [block(text, 10, 10)],
        preprocessing_profile="original",
        engine_name="fake",
    )
    return OCRDocumentResult(pages=[page], normalized_full_text=text, engine_name="fake")


def test_application_handles_ocr_no_without_space():
    doc = document(
        "приложение no 76930от 1 апреля 2026 г.",
        [block("Приложение No 76930 от 1 апреля 2026 г.", 200, 100)],
    )
    number, application_date = extract_application(doc)
    assert number.normalized_value == "76930"
    assert application_date.normalized_value == date(2026, 4, 1)


def test_payment_extracts_deferment_days(config):
    text = "условия оплаты: оплата не позднее 30 календарных дней с момента отгрузки"
    payment, days = extract_payment(document(text), config)
    assert payment.normalized_value.value == "DEFERMENT"
    assert days.normalized_value == 30


def test_payment_repairs_local_prepayment_typo(config):
    blocks = [
        block("10. Условия оплаты:", 100, 620),
        block("Предполата 100%, не позднее 30.07.2026", 100, 650),
    ]
    text = " ".join(item.text for item in blocks)
    payment, days = extract_payment(document(text, blocks), config)
    assert payment.normalized_value.value == "PREPAYMENT"
    assert days.normalized_value is None


def test_payment_on_product_readiness_is_prepayment_without_deferment(config):
    text = "10. Условия оплаты: Оплата по факту готовности трубной продукции"
    payment, days = extract_payment(document(text), config)
    assert payment.normalized_value.value == "PREPAYMENT"
    assert payment.confidence >= 0.85
    assert days.normalized_value is None


def test_same_block_label_value_has_strong_geometry(config):
    blocks = [
        block(
            "8. Срок поставки: Не позднее 04.08.2026",
            100,
            500,
            confidence=0.86,
        ),
        block("9. Товар поставляется на условиях: самовывоз", 100, 560, 0.86),
    ]
    doc = document(" ".join(item.text for item in blocks), blocks)
    deadline, delivery_type = extract_delivery(doc, config)
    assert deadline.confidence >= 0.85
    assert delivery_type.confidence >= 0.85


@pytest.mark.parametrize(
    "token",
    ["30", "з0", "ЗО", "3О", "I0", "l0", "I4", "l4"],
)
def test_payment_repairs_only_numeric_like_tokens(token, config):
    expected = 14 if token.endswith("4") else (10 if token[0] in "Il" else 30)
    text = f"Условия оплаты: Оплата не позднее {token} календарных дней"
    payment, days = extract_payment(document(text), config)
    assert payment.normalized_value.value == "DEFERMENT"
    assert days.normalized_value == expected


def test_payment_does_not_replace_ordinary_letter():
    text = "Поставщик сообщает: здание готово к отгрузке"
    normalized, replacements = normalize_payment_numeric_tokens(text)
    assert normalized == text
    assert replacements == []


def test_contradictory_delivery_is_uncertain(config):
    assert classify_delivery("доставка, далее самовывоз", config) is None


def test_delivery_date_prefers_same_row_over_unrelated_date_above(config):
    blocks = [
        block("8. Срок поставки:", 100, 500),
        block("р/с открыт 25.04.2022 в банке", 450, 470),
        block(
            "Не позднее 08.08.2026 г. Поставщик вправе осуществить досрочную поставку",
            450,
            505,
        ),
        block("9. Товар поставляется на условиях: самовывоза", 100, 560),
        block("10. Условия оплаты: Предоплата 100%, не позднее 29.07.2026", 100, 620),
    ]
    doc = document(" ".join(item.text for item in blocks), blocks)
    deadline, delivery_type = extract_delivery(doc, config)
    assert deadline.normalized_value == date(2026, 8, 8)
    assert delivery_type.normalized_value.value == "PICKUP"
