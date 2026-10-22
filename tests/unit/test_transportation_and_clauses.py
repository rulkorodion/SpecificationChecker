from __future__ import annotations

from app.extractors.transportation import extract_transportation
from app.ocr.models import OCRDocumentResult, OCRPageResult, OCRTextBlock
from app.schemas import BoundingBox


def _document(lines: list[str], confidence: float = 0.97) -> OCRDocumentResult:
    blocks = [
        OCRTextBlock(
            text=text,
            confidence=confidence,
            bounding_box=BoundingBox(
                x_min=40,
                y_min=50 + index * 35,
                x_max=1500,
                y_max=75 + index * 35,
                page_width=1600,
                page_height=2200,
            ),
            page_number=1,
            preprocessing_profile="original",
            engine_name="fake",
        )
        for index, text in enumerate(lines)
    ]
    page = OCRPageResult(
        page_number=1,
        width=1600,
        height=2200,
        blocks=blocks,
        preprocessing_profile="original",
        engine_name="fake",
    )
    return OCRDocumentResult(
        pages=[page],
        normalized_full_text=" ".join(lines),
        engine_name="fake",
    )


def test_transport_positive_phrase(config):
    result = extract_transportation(
        _document(["Включены расходы по оплате транспортных услуг"]), config
    )
    assert result.normalized_value is True


def test_transport_positive_phrase_split_across_lines(config):
    result = extract_transportation(
        _document(
            [
                "В общую сумму Приложения включены расходы",
                "по оплате транспортных услуг",
            ]
        ),
        config,
    )
    assert result.normalized_value is True


def test_transport_negative_phrase(config):
    result = extract_transportation(
        _document(["Транспортные расходы не включены"]), config
    )
    assert result.normalized_value is False


def test_transport_paid_separately(config):
    result = extract_transportation(
        _document(["Расходы оплачиваются отдельно"]), config
    )
    assert result.normalized_value is False


def test_transport_negative_split_across_lines(config):
    result = extract_transportation(
        _document(
            [
                "В общую сумму Приложения не включены расходы",
                "по оплате транспортных услуг",
                "Данные расходы оплачиваются отдельно",
            ]
        ),
        config,
    )
    assert result.normalized_value is False


def test_pickup_default_is_uncertain(config):
    result = extract_transportation(
        _document(["Товар поставляется на условиях самовывоза"]), config
    )
    assert result.normalized_value is None
    assert result.raw_value["pickup_inference"]["rule_enabled"] is False


def test_pickup_configured_rule_is_false(config):
    config["transportation"]["pickup_implies_not_included"] = True
    result = extract_transportation(
        _document(["Товар поставляется на условиях самовывоза"]), config
    )
    assert result.normalized_value is False
    assert result.raw_value["pickup_inference"]["used_as_negative_evidence"] is True


def test_transport_conflict_is_uncertain(config):
    result = extract_transportation(
        _document(
            [
                "В общую сумму Приложения включены расходы по оплате транспортных услуг",
                "Транспортные расходы не включены и оплачиваются отдельно",
            ]
        ),
        config,
    )
    assert result.normalized_value is None
    assert result.raw_value["conflicts"]


def test_transport_without_evidence_is_uncertain(config):
    assert extract_transportation(_document(["Нет условий доставки"]), config).normalized_value is None
