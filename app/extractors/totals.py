"""Поиск, классификация и агрегация нескольких табличных частей."""

from __future__ import annotations

import re
from decimal import Decimal
from difflib import SequenceMatcher
from statistics import median

from app.enums import TableType
from app.extractors.base import extracted_field
from app.extractors.utils import union_box
from app.layout.reading_order import page_lines
from app.normalization.numbers import normalize_decimal
from app.normalization.text import normalize_text
from app.ocr.models import OCRDocumentResult, OCRPageResult, OCRTextBlock
from app.schemas import ExtractedField, TableAggregation, TableResult

NUMBER_RE = re.compile(r"[-+]?\d[\d\s\u00a0\u202f.,]*\d|[-+]?\d")
SERVICE_KEYWORDS = (
    "доставка",
    "транспортировка",
    "перевозка",
    "транспортные услуги",
    "автотранспорт",
)


def _is_total_anchor(text: str) -> bool:
    normalized = normalize_text(text)
    for token in re.findall(r"[a-zа-я]+", normalized):
        if token == "итог":
            return True
        if len(token) != 5:
            continue
        repaired = token.replace("m", "т")
        if repaired.startswith("нтого"):
            repaired = "и" + repaired[1:]
        if repaired == "итого" or SequenceMatcher(None, repaired, "итого").ratio() >= 0.78:
            return True
    return False


def _line_text(line: list[OCRTextBlock]) -> str:
    return " ".join(block.text for block in line).strip()


def _numeric_blocks(line: list[OCRTextBlock]) -> list[tuple[OCRTextBlock, Decimal, str]]:
    result: list[tuple[OCRTextBlock, Decimal, str]] = []
    for block in line:
        for raw in NUMBER_RE.findall(block.text):
            try:
                result.append((block, normalize_decimal(raw), raw))
            except ValueError:
                continue
    return result


def _line_y(line: list[OCRTextBlock]) -> tuple[float, float]:
    return (
        min(block.bounding_box.y_min for block in line),
        max(block.bounding_box.y_max for block in line),
    )


def _find_numeric_total_line(
    lines: list[list[OCRTextBlock]],
    index: int,
    page_height: int,
    min_numeric: int = 2,
) -> tuple[list[OCRTextBlock], list[tuple[OCRTextBlock, Decimal, str]]] | None:
    """Связать «Итого» с числами в той же или соседней OCR-строке."""

    candidates: list[tuple[int, list[OCRTextBlock], list[tuple[OCRTextBlock, Decimal, str]]]] = []
    anchor_top, anchor_bottom = _line_y(lines[index])
    same_line_numeric = _numeric_blocks(lines[index])
    if len(same_line_numeric) >= min_numeric:
        return lines[index], same_line_numeric
    for candidate_index in (index, index - 1, index + 1):
        if candidate_index < 0 or candidate_index >= len(lines):
            continue
        line = lines[candidate_index]
        numeric = _numeric_blocks(line)
        if len(numeric) < min_numeric:
            continue
        top, bottom = _line_y(line)
        gap = max(0.0, top - anchor_bottom, anchor_top - bottom)
        if candidate_index != index and gap > page_height * 0.025:
            continue
        preference = len(numeric)
        candidates.append((preference, line, numeric))
    if not candidates:
        return None
    _, line, numeric = max(candidates, key=lambda item: item[0])
    return line, numeric


def _table_region_start(
    lines: list[list[OCRTextBlock]],
    total_index: int,
    previous_total_index: int,
    page_height: int,
) -> int:
    lower_bound = max(previous_total_index + 1, 0)
    total_top, _ = _line_y(lines[total_index])
    fallback = max(lower_bound, total_index - 10)
    for index in range(total_index - 1, lower_bound - 1, -1):
        top, _ = _line_y(lines[index])
        if total_top - top > page_height * 0.22:
            break
        text = normalize_text(_line_text(lines[index]))
        compact = text.replace(" ", "")
        if (
            "наименование товара" in text
            or "наименованиетовара" in compact
            or "расшифровка услуг" in text
            or ("услуг" in text and "количеств" in text)
        ):
            # Заголовок часто разбит на две соседние строки.
            return max(lower_bound, index - 1)
        fallback = index
    return fallback


def _classify_table(
    region_lines: list[list[OCRTextBlock]],
) -> tuple[TableType, float, list[str]]:
    text = normalize_text(" ".join(_line_text(line) for line in region_lines))
    compact = text.replace(" ", "")
    service_hits = [
        keyword
        for keyword in ("услуга", "расшифровка услуги", *SERVICE_KEYWORDS)
        if normalize_text(keyword) in text
    ]
    goods_hits = [
        keyword
        for keyword in ("наименование товара", "ед. изм", "кол-во", "цена за ед", "товар")
        if normalize_text(keyword) in text
    ]
    if "услуг" in text and (
        "расшифров" in text or any(item in text for item in SERVICE_KEYWORDS)
    ):
        return TableType.SERVICES, 0.96, service_hits
    if "наименование товара" in text or "наименованиетовара" in compact or (
        "ед. изм" in text and "кол" in text and ("цена" in text or "сумма" in text)
    ):
        return TableType.GOODS, 0.95, goods_hits
    return TableType.UNKNOWN, 0.58, [*goods_hits, *service_hits]


def _headers_and_rows(
    region_lines: list[list[OCRTextBlock]],
    numeric_line: list[OCRTextBlock],
) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    rows: list[list[str]] = []
    numeric_y = _line_y(numeric_line)[0]
    for line in region_lines:
        text = _line_text(line)
        normalized = normalize_text(text)
        compact = normalized.replace(" ", "")
        if line is numeric_line:
            continue
        if any(
            marker in normalized
            for marker in (
                "наименование товара",
                "ед. изм",
                "кол-во",
                "расшифровка услуг",
                "всего с ндс",
            )
        ) or "наименованиетовара" in compact:
            headers.append(text)
        elif _line_y(line)[0] < numeric_y and text:
            rows.append([block.text for block in line])
    return headers, rows


def _coordinate_total_values(
    page: OCRPageResult,
    region_lines: list[list[OCRTextBlock]],
    anchor_line: list[OCRTextBlock],
) -> tuple[
    tuple[OCRTextBlock, Decimal, str] | None,
    tuple[OCRTextBlock, Decimal, str] | None,
]:
    """Найти quantity/total по X колонок, независимо от OCR reading order."""

    blocks = [block for line in region_lines for block in line]
    anchor_top, anchor_bottom = _line_y(anchor_line)
    header_search = [
        block
        for block in page.blocks
        if block.bounding_box.y_max <= anchor_bottom
        and anchor_top - block.bounding_box.y_max <= page.height * 0.12
    ]
    quantity_headers = [
        block
        for block in header_search
        if "кол" in normalize_text(block.text)
        and (
            "кол-во" in normalize_text(block.text)
            or "количество" in normalize_text(block.text)
            or "объем" in normalize_text(block.text)
        )
    ]
    total_headers = [
        block
        for block in header_search
        if "всего с ндс" in normalize_text(block.text)
        or ("сумма" in normalize_text(block.text) and "ндс" in normalize_text(block.text))
    ]
    if not quantity_headers or not total_headers:
        return None, None
    quantity_header = max(quantity_headers, key=lambda item: item.bounding_box.x_min)
    total_header = max(total_headers, key=lambda item: item.bounding_box.x_min)
    quantity_x = (
        quantity_header.bounding_box.x_min + quantity_header.bounding_box.x_max
    ) / 2
    total_x = (total_header.bounding_box.x_min + total_header.bounding_box.x_max) / 2
    candidates: list[tuple[OCRTextBlock, Decimal, str]] = []
    for block in blocks:
        if block is quantity_header or block is total_header:
            continue
        box = block.bounding_box
        if box.y_min > anchor_bottom + page.height * 0.01:
            continue
        if anchor_top - box.y_max > page.height * 0.055:
            continue
        candidates.extend(_numeric_blocks([block]))

    def select(
        target_x: float,
        *,
        excluded: OCRTextBlock | None = None,
    ) -> tuple[OCRTextBlock, Decimal, str] | None:
        ranked = []
        for item in candidates:
            block, value, raw = item
            if block is excluded:
                continue
            center_x = (block.bounding_box.x_min + block.bounding_box.x_max) / 2
            x_distance = abs(center_x - target_x) / page.width
            if x_distance > 0.14:
                continue
            vertical_gap = max(0.0, anchor_top - block.bounding_box.y_max) / page.height
            # Геометрия колонки важнее небольшого перспективного сдвига по Y.
            score = x_distance * 2.4 + vertical_gap
            ranked.append((score, item))
        return min(ranked, key=lambda item: item[0])[1] if ranked else None

    quantity = select(quantity_x)
    total = select(total_x, excluded=quantity[0] if quantity else None)
    return quantity, total


def _validated_row_sums(
    page: OCRPageResult,
    region_lines: list[list[OCRTextBlock]],
    total_line: list[OCRTextBlock],
) -> tuple[Decimal, Decimal, int, float] | None:
    """Recalculate totals only when quantity×price and VAT+net independently agree."""

    quantity_sum = Decimal(0)
    total_sum = Decimal(0)
    candidate_rows = 0
    valid_rows = 0
    confidences: list[float] = []
    for line in region_lines:
        if line is total_line or _is_total_anchor(_line_text(line)):
            continue
        normalized_line = normalize_text(_line_text(line))
        if any(
            marker in normalized_line
            for marker in (
                "цена за ед",
                "сумма ндс",
                "всего с ндс",
                "% от объема",
            )
        ):
            continue
        numeric = sorted(
            (
                item
                for item in _numeric_blocks(line)
                if (
                    item[0].bounding_box.x_min
                    + item[0].bounding_box.x_max
                )
                / 2
                >= page.width * 0.45
            ),
            key=lambda item: (
                item[0].bounding_box.x_min,
                item[0].text.find(item[2]),
            ),
        )
        # Standard goods/services rows: quantity, gross unit price, VAT, net, gross.
        if len(numeric) < 5:
            continue
        candidate_rows += 1
        quantity = numeric[0][1]
        unit_price = numeric[1][1]
        vat = numeric[-3][1]
        net = numeric[-2][1]
        if quantity <= 0 or unit_price <= 0 or vat < 0 or net <= 0:
            continue
        calculated = (quantity * unit_price).quantize(Decimal("0.01"))
        tax_components = (vat + net).quantize(Decimal("0.01"))
        tolerance = max(Decimal("0.05"), abs(calculated) * Decimal("0.002"))
        if abs(calculated - tax_components) > tolerance:
            continue
        valid_rows += 1
        quantity_sum += quantity
        total_sum += calculated
        confidences.extend(
            item[0].effective_confidence
            for item in (numeric[0], numeric[1], numeric[-3], numeric[-2])
        )
    if candidate_rows < 2 or valid_rows != candidate_rows:
        return None
    return (
        quantity_sum,
        total_sum,
        valid_rows,
        float(median(confidences)),
    )


def _table_result(
    page: OCRPageResult,
    lines: list[list[OCRTextBlock]],
    total_index: int,
    previous_total_index: int,
) -> TableResult | None:
    numeric_match = _find_numeric_total_line(lines, total_index, page.height)
    if numeric_match is None:
        return None
    numeric_line, numeric = numeric_match
    start = _table_region_start(lines, total_index, previous_total_index, page.height)
    end = max(total_index, lines.index(numeric_line))
    region_lines = lines[start : end + 1]
    blocks = [block for line in region_lines for block in line]
    table_type, classification_confidence, keywords = _classify_table(region_lines)
    headers, rows = _headers_and_rows(region_lines, numeric_line)
    numeric = sorted(
        numeric,
        key=lambda item: (
            item[0].bounding_box.x_min,
            item[0].text.find(item[2]),
        ),
    )
    coordinate_quantity, coordinate_total = _coordinate_total_values(
        page, region_lines, lines[total_index]
    )
    coordinate_used = (
        table_type is TableType.GOODS
        and numeric_line is not lines[total_index]
        and coordinate_quantity is not None
        and coordinate_total is not None
    )
    if coordinate_used:
        numeric = [coordinate_quantity, coordinate_total]
        numeric_line = list({id(item[0]): item[0] for item in numeric}.values())
    total_quantity = numeric[0][1] if table_type is TableType.GOODS else None
    total_with_vat = numeric[-1][1]
    ocr_confidence = float(median(item[0].effective_confidence for item in numeric))
    confidence = min(classification_confidence, ocr_confidence)
    warnings: list[str] = []
    consistency_evidence: list[str] = []
    row_sums = (
        _validated_row_sums(page, region_lines, numeric_line)
        if table_type is TableType.GOODS
        else None
    )
    if row_sums is not None:
        row_quantity, row_total, row_count, row_confidence = row_sums
        quantity_tolerance = max(
            Decimal("0.001"), abs(row_quantity) * Decimal("0.002")
        )
        total_tolerance = max(
            Decimal("0.05"), abs(row_total) * Decimal("0.002")
        )
        if (
            total_quantity is not None
            and abs(total_quantity - row_quantity) > quantity_tolerance
        ):
            warnings.append(
                "Итоговое количество восстановлено как сумма геометрически связанных строк"
            )
            total_quantity = row_quantity
        if abs(total_with_vat - row_total) > total_tolerance:
            warnings.append(
                "Итоговая сумма восстановлена по строкам: количество × цена "
                "подтверждено суммой НДС и стоимости без НДС"
            )
            total_with_vat = row_total
        confidence = min(classification_confidence, row_confidence, 0.95)
        consistency_evidence.append(
            f"validated_row_sums=count:{row_count},"
            f"quantity:{row_quantity},total:{row_total}"
        )
    if table_type is TableType.UNKNOWN:
        warnings.append("Тип таблицы не подтверждён заголовками")
    if numeric_line is not lines[total_index]:
        warnings.append("Числовой итог связан с соседней OCR-строкой")
    if coordinate_used:
        warnings.append("Итоговые значения привязаны к X-координатам колонок")
    return TableResult(
        page_number=page.page_number,
        bounding_box=union_box(blocks),  # type: ignore[arg-type]
        table_type=table_type,
        headers=headers,
        rows=rows,
        totals_row=[block.text for block in numeric_line],
        total_quantity=total_quantity,
        total_with_vat=total_with_vat,
        confidence=confidence,
        evidence=[
            f"anchor={_line_text(lines[total_index])}",
            f"numeric_row={_line_text(numeric_line)}",
            f"classification={table_type.value}",
            f"coordinate_columns={coordinate_used}",
            *consistency_evidence,
        ],
        warnings=warnings,
        total_row_bounding_box=union_box(numeric_line),
        service_keywords_found=[
            keyword for keyword in SERVICE_KEYWORDS if keyword in normalize_text(
                " ".join(_line_text(line) for line in region_lines)
            )
        ],
    )


def analyze_tables(document: OCRDocumentResult) -> TableAggregation:
    tables: list[TableResult] = []
    grand_total: Decimal | None = None
    grand_total_evidence: list[str] = []
    rejected: list[dict[str, object]] = []
    for page in document.pages:
        lines = page_lines(page)
        total_indexes = [
            index
            for index, line in enumerate(lines)
            if _is_total_anchor(_line_text(line))
        ]
        previous_total = -1
        for index in total_indexes:
            anchor_text = normalize_text(_line_text(lines[index]))
            if any(marker in anchor_text for marker in ("общий итог", "всего по документ")):
                numeric_match = _find_numeric_total_line(
                    lines, index, page.height, min_numeric=1
                )
                if numeric_match:
                    grand_total = _numeric_blocks(numeric_match[0])[-1][1]
                    grand_total_evidence.append(_line_text(numeric_match[0]))
                previous_total = index
                continue
            result = _table_result(
                page,
                lines,
                index,
                previous_total,
            )
            previous_total = index
            if result is None:
                rejected.append(
                    {
                        "page_number": page.page_number,
                        "anchor": _line_text(lines[index]),
                        "reason": "Не найдена числовая строка рядом с якорем",
                    }
                )
                continue
            tables.append(result)

    included = [
        index
        for index, table in enumerate(tables)
        if table.table_type in {TableType.GOODS, TableType.SERVICES}
        and table.total_with_vat is not None
    ]
    quantity_indexes = [
        index
        for index, table in enumerate(tables)
        if table.table_type is TableType.GOODS and table.total_quantity is not None
    ]
    total_indexes = list(included)
    for index, table in enumerate(tables):
        if index not in included:
            rejected.append(
                {
                    "table_index": index,
                    "page_number": table.page_number,
                    "table_type": table.table_type.value,
                    "reason": "UNKNOWN или нет подтверждённого итога с НДС",
                }
            )

    quantity = (
        sum((tables[index].total_quantity or Decimal(0) for index in quantity_indexes), Decimal(0))
        if quantity_indexes
        else None
    )
    component_total = (
        sum((tables[index].total_with_vat or Decimal(0) for index in total_indexes), Decimal(0))
        if total_indexes
        else None
    )
    conflict = False
    warnings: list[str] = [
        warning
        for index in included
        for warning in tables[index].warnings
    ]
    method = "AGGREGATED_TABLE_TOTALS"
    total = component_total
    if grand_total is not None:
        method = "DOCUMENT_GRAND_TOTAL_VALIDATED"
        if component_total is None or abs(grand_total - component_total) > Decimal("0.01"):
            conflict = True
            total = None
            warnings.append(
                "Общий итог документа противоречит сумме итогов отдельных таблиц"
            )
        else:
            total = grand_total

    confidences = [tables[index].confidence for index in included]
    confidence = min(confidences) if confidences else 0.0
    evidence = [
        f"table_{index + 1}:{tables[index].table_type.value}:"
        f"quantity={tables[index].total_quantity}:total={tables[index].total_with_vat}"
        for index in included
    ]
    evidence.extend(f"grand_total={item}" for item in grand_total_evidence)
    return TableAggregation(
        tables=tables,
        included_table_indexes=included,
        excluded_candidates=rejected,
        quantity_table_indexes=quantity_indexes,
        total_table_indexes=total_indexes,
        aggregated_quantity=quantity,
        aggregated_total_with_vat=total,
        document_grand_total=grand_total,
        extraction_method=method,
        confidence=confidence,
        conflict=conflict,
        evidence=evidence,
        warnings=warnings,
    )


def extract_totals(
    document: OCRDocumentResult,
) -> tuple[ExtractedField, ExtractedField, TableAggregation]:
    analysis = analyze_tables(document)
    raw = analysis.model_dump(mode="json")
    goods = [
        analysis.tables[index]
        for index in analysis.quantity_table_indexes
    ]
    totals = [
        analysis.tables[index]
        for index in analysis.total_table_indexes
    ]
    if analysis.aggregated_quantity is None:
        quantity = extracted_field(
            "TOTAL_QUANTITY",
            raw,
            None,
            0.0,
            "multi_table_aggregation",
            evidence=analysis.evidence,
            warnings=[
                "Не найден уверенный итог количества в таблице GOODS",
                *analysis.warnings,
            ],
        )
    else:
        quantity = extracted_field(
            "TOTAL_QUANTITY",
            raw,
            analysis.aggregated_quantity,
            min(table.confidence for table in goods),
            "AGGREGATED_GOODS_QUANTITY",
            goods[0].page_number,
            goods[0].total_row_bounding_box,
            analysis.evidence,
            analysis.warnings,
        )
    if analysis.aggregated_total_with_vat is None:
        total = extracted_field(
            "TOTAL_WITH_VAT",
            raw,
            None,
            0.0,
            analysis.extraction_method,
            evidence=analysis.evidence,
            warnings=[
                "Не найден непротиворечивый итог с НДС",
                *analysis.warnings,
            ],
        )
    else:
        total = extracted_field(
            "TOTAL_WITH_VAT",
            raw,
            analysis.aggregated_total_with_vat,
            min(table.confidence for table in totals),
            analysis.extraction_method,
            totals[0].page_number,
            totals[0].total_row_bounding_box,
            analysis.evidence,
            analysis.warnings,
        )
    return quantity, total, analysis
