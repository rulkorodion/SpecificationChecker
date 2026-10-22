from __future__ import annotations

from decimal import Decimal

from app.enums import TableType
from app.extractors.totals import extract_totals
from app.extractors.transportation import extract_transportation
from app.ocr.models import OCRDocumentResult, OCRPageResult, OCRTextBlock
from app.schemas import BoundingBox


def _block(
    text: str, x: int, y: int, confidence: float | None = 0.97
) -> OCRTextBlock:
    return OCRTextBlock(
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + max(70, len(text) * 7),
            y_max=y + 24,
            page_width=1600,
            page_height=2200,
        ),
        page_number=1,
        preprocessing_profile="original",
        engine_name="fake",
    )


def _document(lines: list[list[str]]) -> OCRDocumentResult:
    blocks = [
        _block(text, 40 + column * 250, 80 + row * 45)
        for row, line in enumerate(lines)
        for column, text in enumerate(line)
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
        normalized_full_text=" ".join(block.text for block in blocks),
        engine_name="fake",
    )


def _goods(quantity: str, total: str) -> list[list[str]]:
    return [
        ["Наименование товара", "Ед. изм.", "Кол-во", "Цена за ед.", "Всего с НДС"],
        ["Труба стальная", "т", quantity, "100,00", total],
        ["Итого:", quantity, total],
    ]


def _services(quantity: str, total: str) -> list[list[str]]:
    return [
        ["Услуга", "Расшифровка услуги", "Количество", "Всего с НДС"],
        ["Доставка", "Транспортировка товара", quantity, total],
        ["Итого:", quantity, total],
    ]


def test_one_goods_table():
    quantity, total, analysis = extract_totals(_document(_goods("10,5", "1000,00")))
    assert quantity.normalized_value == Decimal("10.5")
    assert total.normalized_value == Decimal("1000.00")
    assert analysis.tables[0].table_type is TableType.GOODS


def test_unscored_vlm_table_uses_structural_confidence_without_crashing():
    document = _document(_goods("10,5", "1000,00"))
    for block in document.pages[0].blocks:
        block.confidence = None
    quantity, total, analysis = extract_totals(document)
    assert quantity.normalized_value == Decimal("10.5")
    assert total.normalized_value == Decimal("1000.00")
    assert analysis.tables[0].confidence == 0.95


def test_two_goods_tables_are_aggregated():
    quantity, total, analysis = extract_totals(
        _document([*_goods("10", "1000"), *_goods("5", "700")])
    )
    assert len(analysis.tables) == 2
    assert quantity.normalized_value == Decimal("15")
    assert total.normalized_value == Decimal("1700")


def test_goods_and_services_keep_service_quantity_out():
    quantity, total, analysis = extract_totals(
        _document([*_goods("75,146", "4672610"), *_services("4", "60000")])
    )
    assert [item.table_type for item in analysis.tables] == [
        TableType.GOODS,
        TableType.SERVICES,
    ]
    assert quantity.normalized_value == Decimal("75.146")
    assert total.normalized_value == Decimal("4732610")


def test_document_grand_total_is_not_double_counted():
    quantity, total, analysis = extract_totals(
        _document(
            [
                *_goods("10", "1000"),
                *_services("1", "600"),
                ["Общий итог:", "1600"],
            ]
        )
    )
    assert quantity.normalized_value == Decimal("10")
    assert total.normalized_value == Decimal("1600")
    assert analysis.document_grand_total == Decimal("1600")
    assert analysis.extraction_method == "DOCUMENT_GRAND_TOTAL_VALIDATED"


def test_conflicting_document_grand_total_is_uncertain():
    _, total, analysis = extract_totals(
        _document(
            [
                *_goods("10", "1000"),
                *_services("1", "600"),
                ["Общий итог:", "1700"],
            ]
        )
    )
    assert total.normalized_value is None
    assert analysis.conflict is True


def test_unknown_table_is_excluded():
    quantity, total, analysis = extract_totals(
        _document([["Произвольный блок"], ["Итого:", "3", "999"]])
    )
    assert quantity.normalized_value is None
    assert total.normalized_value is None
    assert analysis.tables[0].table_type is TableType.UNKNOWN
    assert analysis.excluded_candidates


def test_delivery_service_table_is_transport_evidence(config):
    _, _, analysis = extract_totals(
        _document([*_goods("10", "1000"), *_services("1", "600")])
    )
    result = extract_transportation(_document([]), config, analysis)
    assert result.normalized_value is True
    assert result.raw_value["service_table_evidence"]


def test_concatenated_goods_header_and_corrupt_total_are_validated_from_rows():
    blocks = [
        _block(
            "Наименованиетовара Ед.изм. Кол-во Цена за ед. (с НДС 22%) "
            "Сумма НДС 22% Сумма без НДС Всего с НДС",
            40,
            100,
        )
    ]
    for row, quantity in enumerate(("1,000", "1,000", "1,000"), start=1):
        y = 100 + row * 45
        blocks.extend(
            [
                _block(f"Труба {row} 80x80", 40, y),
                _block("т", 600, y),
                _block(quantity, 780, y),
                _block("100,00", 950, y),
                _block("18,00", 1120, y),
                _block("82,00", 1300, y),
                _block("100,00", 1480, y),
            ]
        )
    blocks.extend(
        [
            _block("Итого:", 40, 300),
            _block("3,000", 780, 300),
            _block("54,00", 1120, 300),
            _block("246,00", 1300, 300),
            _block("30,00", 1480, 300),
        ]
    )
    page = OCRPageResult(
        page_number=1,
        width=1600,
        height=2200,
        blocks=blocks,
        preprocessing_profile="original",
        engine_name="fake",
    )
    doc = OCRDocumentResult(
        pages=[page],
        normalized_full_text=" ".join(item.text for item in blocks),
        engine_name="fake",
    )
    quantity, total, analysis = extract_totals(doc)
    assert analysis.tables[0].table_type is TableType.GOODS
    assert quantity.normalized_value == Decimal("3.000")
    assert total.normalized_value == Decimal("300.00")
    assert any("количество × цена" in item for item in total.warnings)
