"""Объяснимое определение включения транспортных расходов."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from app.enums import TableType
from app.extractors.base import extracted_field
from app.extractors.utils import union_box
from app.layout.reading_order import page_lines
from app.normalization.text import normalize_text
from app.ocr.models import OCRDocumentResult, OCRTextBlock
from app.schemas import ExtractedField, TableAggregation


@dataclass(slots=True)
class _TextWindow:
    page_number: int
    start_line: int
    end_line: int
    text: str
    normalized: str
    blocks: list[OCRTextBlock]
    confidence: float


def _token_similarity(phrase: str, text: str) -> float:
    wanted = normalize_text(phrase).split()
    actual = normalize_text(text).split()
    if not wanted or not actual:
        return 0.0
    compact_phrase = " ".join(wanted)
    compact_text = " ".join(actual)
    if compact_phrase in compact_text:
        return 1.0
    token_scores: list[float] = []
    positions: list[int] = []
    cursor = 0
    for token in wanted:
        best_score = 0.0
        best_index = cursor
        for index in range(cursor, min(len(actual), cursor + 16)):
            score = SequenceMatcher(None, token, actual[index]).ratio()
            if score > best_score:
                best_score = score
                best_index = index
        token_scores.append(best_score)
        positions.append(best_index)
        cursor = min(len(actual), best_index + 1)
    ordered = all(left <= right for left, right in zip(positions, positions[1:]))
    return min(1.0, sum(token_scores) / len(token_scores) * 0.9 + (0.1 if ordered else 0.0))


def _sliding_windows(document: OCRDocumentResult, max_size: int = 4) -> list[_TextWindow]:
    windows: list[_TextWindow] = []
    for page in document.pages:
        lines = page_lines(page)
        for start in range(len(lines)):
            for size in range(1, max_size + 1):
                selected = lines[start : start + size]
                if len(selected) != size:
                    break
                blocks = [block for line in selected for block in line]
                text = " ".join(block.text for block in blocks)
                windows.append(
                    _TextWindow(
                        page_number=page.page_number,
                        start_line=start,
                        end_line=start + size - 1,
                        text=text,
                        normalized=normalize_text(text),
                        blocks=blocks,
                        confidence=float(median(block.effective_confidence for block in blocks)),
                    )
                )
    return windows


def _best_phrase_matches(
    windows: list[_TextWindow],
    phrases: list[str],
    *,
    positive: bool,
) -> list[dict]:
    matches: list[dict] = []
    for phrase in phrases:
        best: tuple[float, _TextWindow] | None = None
        for window in windows:
            if positive and (
                "не включ" in window.normalized
                or "оплачива" in window.normalized and "отдельно" in window.normalized
            ):
                continue
            normalized_phrase = normalize_text(phrase)
            if not positive:
                if "не включ" in normalized_phrase and "не включ" not in window.normalized:
                    continue
                if "отдельно" in normalized_phrase and "отдельно" not in window.normalized:
                    continue
                if "не входят" in normalized_phrase and "не вход" not in window.normalized:
                    continue
            score = _token_similarity(phrase, window.normalized)
            if score >= 0.82 and (best is None or score > best[0]):
                best = (score, window)
        if best is None:
            continue
        score, window = best
        matches.append(
            {
                "phrase": phrase,
                "found_text": window.text,
                "similarity": round(score, 6),
                "ocr_confidence": round(window.confidence, 6),
                "page_number": window.page_number,
                "start_line": window.start_line,
                "end_line": window.end_line,
                "bounding_box": (
                    union_box(window.blocks).model_dump(mode="json")
                    if union_box(window.blocks)
                    else None
                ),
                "_window": window,
            }
        )
    return matches


def extract_transportation(
    document: OCRDocumentResult,
    config: dict,
    tables: TableAggregation | None = None,
) -> ExtractedField:
    section = config["transportation"]
    included_phrases = section.get(
        "included_phrases", section.get("included_keywords", [])
    )
    excluded_phrases = section.get(
        "excluded_phrases", section.get("excluded_keywords", [])
    )
    service_keywords = [
        normalize_text(item)
        for item in section.get(
            "service_keywords",
            ["доставка", "транспортировка", "перевозка", "транспортные услуги"],
        )
    ]
    windows = _sliding_windows(document)
    positive_matches = _best_phrase_matches(
        windows, included_phrases, positive=True
    )
    negative_matches = _best_phrase_matches(
        windows, excluded_phrases, positive=False
    )
    service_evidence: list[dict] = []
    if tables is not None:
        included_indexes = set(tables.included_table_indexes)
        for index, table in enumerate(tables.tables):
            found = [
                keyword
                for keyword in service_keywords
                if keyword
                in normalize_text(
                    " ".join(
                        [
                            *table.headers,
                            *(cell for row in table.rows for cell in row),
                            *table.totals_row,
                        ]
                    )
                )
            ]
            if (
                index in included_indexes
                and table.table_type is TableType.SERVICES
                and table.total_with_vat is not None
                and table.total_with_vat > 0
                and found
            ):
                service_evidence.append(
                    {
                        "table_index": index,
                        "page_number": table.page_number,
                        "keywords": found,
                        "total_with_vat": str(table.total_with_vat),
                        "confidence": table.confidence,
                        "included_in_document_total": True,
                    }
                )

    normalized_text = normalize_text(document.normalized_full_text)
    pickup_found = any(
        marker in normalized_text
        for marker in ("самовывоз", "самовывоза", "выборка покупателем")
    )
    pickup_rule = bool(section.get("pickup_implies_not_included", False))
    pickup_inference = {
        "pickup_found": pickup_found,
        "rule_enabled": pickup_rule,
        "used_as_negative_evidence": pickup_found and pickup_rule,
    }

    positive = bool(positive_matches or service_evidence)
    negative = bool(negative_matches or pickup_inference["used_as_negative_evidence"])
    conflicts: list[str] = []
    if positive and negative:
        conflicts.append(
            "Одновременно найдены положительные и отрицательные доказательства"
        )
    raw = {
        "positive_evidence": [
            {key: value for key, value in item.items() if key != "_window"}
            for item in positive_matches
        ],
        "negative_evidence": [
            {key: value for key, value in item.items() if key != "_window"}
            for item in negative_matches
        ],
        "service_table_evidence": service_evidence,
        "pickup_inference": pickup_inference,
        "conflicts": conflicts,
    }
    evidence = [
        *(f"positive: {item['found_text']}" for item in positive_matches),
        *(f"negative: {item['found_text']}" for item in negative_matches),
        *(
            f"service table {item['table_index'] + 1}: "
            f"{', '.join(item['keywords'])}, total={item['total_with_vat']}"
            for item in service_evidence
        ),
    ]
    if pickup_inference["used_as_negative_evidence"]:
        evidence.append(
            "negative: применено конфигурационное правило "
            "pickup_implies_not_included=true"
        )

    text_matches = positive_matches or negative_matches
    first_window = text_matches[0]["_window"] if text_matches else None
    first_table = (
        tables.tables[service_evidence[0]["table_index"]]
        if tables is not None and service_evidence
        else None
    )
    page = first_window.page_number if first_window else (
        first_table.page_number if first_table else None
    )
    box = union_box(first_window.blocks) if first_window else (
        first_table.bounding_box if first_table else None
    )
    confidence_values = [
        item["ocr_confidence"] * item["similarity"]
        for item in [*positive_matches, *negative_matches]
    ]
    confidence_values.extend(item["confidence"] for item in service_evidence)
    confidence = min(0.99, max(confidence_values, default=0.0))

    if positive == negative:
        warnings = conflicts or [
            "Нет явной непротиворечивой формулировки или услуги транспорта"
        ]
        if pickup_found and not pickup_rule:
            warnings.append(
                "Самовывоз найден, но pickup_implies_not_included=false"
            )
        return extracted_field(
            "TRANSPORTATION_COSTS",
            raw,
            None,
            min(confidence, 0.84),
            "transportation_evidence_ensemble",
            page,
            box,
            evidence,
            warnings,
        )
    return extracted_field(
        "TRANSPORTATION_COSTS",
        raw,
        positive,
        max(0.86, confidence),
        "transportation_evidence_ensemble",
        page,
        box,
        evidence,
    )
