"""Координатные и текстовые вспомогательные функции extractors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.normalization.text import normalize_text
from app.ocr.models import OCRPageResult, OCRTextBlock
from app.schemas import BoundingBox


@dataclass(frozen=True, slots=True)
class LabelValueContext:
    text: str
    blocks: list[OCRTextBlock]
    anchor: OCRTextBlock | None
    geometry_score: float


def union_box(blocks: list[OCRTextBlock]) -> BoundingBox | None:
    if not blocks:
        return None
    return BoundingBox(
        x_min=min(x.bounding_box.x_min for x in blocks),
        y_min=min(x.bounding_box.y_min for x in blocks),
        x_max=max(x.bounding_box.x_max for x in blocks),
        y_max=max(x.bounding_box.y_max for x in blocks),
        page_width=blocks[0].bounding_box.page_width,
        page_height=blocks[0].bounding_box.page_height,
    )


def context_window(text: str, anchor: str, chars: int = 350) -> str:
    normalized = normalize_text(text)
    index = normalized.find(normalize_text(anchor))
    if index < 0:
        return ""
    return normalized[index : index + chars]


def blocks_containing(
    pages: list[OCRPageResult], fragments: list[str]
) -> list[OCRTextBlock]:
    wanted = [normalize_text(item) for item in fragments]
    return [
        block
        for page in pages
        for block in page.blocks
        if any(item in normalize_text(block.text) for item in wanted)
    ]


def _anchor_similarity(text: str, anchor: str) -> float:
    normalized = normalize_text(text)
    wanted = normalize_text(anchor)
    if wanted in normalized:
        return 1.0
    wanted_words = [word for word in wanted.split() if len(word) >= 4]
    word_coverage = (
        sum(word in normalized for word in wanted_words) / len(wanted_words)
        if wanted_words
        else 0.0
    )
    sequence = SequenceMatcher(None, normalized[: len(wanted) + 12], wanted).ratio()
    return max(sequence, word_coverage)


def label_value_context(
    pages: list[OCRPageResult],
    anchor_text: str,
    *,
    competing_anchors: list[str] | None = None,
) -> LabelValueContext:
    """Связать label со значением по строке, устойчиво к небольшому наклону."""

    competing = competing_anchors or []
    anchors = [
        block
        for page in pages
        for block in page.blocks
        if _anchor_similarity(block.text, anchor_text) >= 0.72
    ]
    best: LabelValueContext | None = None
    for anchor in anchors:
        page = next(page for page in pages if page.page_number == anchor.page_number)
        box = anchor.bounding_box
        anchor_y = (box.y_min + box.y_max) / 2
        competing_centers = [
            (block.bounding_box.y_min + block.bounding_box.y_max) / 2
            for block in page.blocks
            if any(
                _anchor_similarity(block.text, other) >= 0.76
                for other in competing
            )
        ]
        previous_labels = [center for center in competing_centers if center < anchor_y]
        next_labels = [center for center in competing_centers if center > anchor_y]
        row_min = (
            (max(previous_labels) + anchor_y) / 2 if previous_labels else float("-inf")
        )
        row_max = (
            (min(next_labels) + anchor_y) / 2 if next_labels else float("inf")
        )
        initial: list[tuple[float, OCRTextBlock]] = []
        for block in page.blocks:
            candidate = block.bounding_box
            center_y = (candidate.y_min + candidate.y_max) / 2
            delta_y = abs(center_y - anchor_y)
            same_block = block is anchor
            if not same_block and not (row_min <= center_y <= row_max):
                continue
            right_linked = candidate.x_min >= box.x_max - page.width * 0.035
            vertically_close = delta_y <= max(
                box.y_max - box.y_min, page.height * 0.032
            )
            if not same_block and not (right_linked and vertically_close):
                continue
            normalized = normalize_text(block.text)
            if (
                not same_block
                and any(_anchor_similarity(normalized, other) >= 0.76 for other in competing)
            ):
                continue
            horizontal_gap = max(0.0, candidate.x_min - box.x_max) / page.width
            # На перспективных сканах правое значение часто оказывается немного
            # выше своего label. Следующая строка, напротив, обычно уже ниже label.
            same_or_above = center_y <= anchor_y + max(
                (box.y_max - box.y_min) * 0.35, page.height * 0.003
            )
            score = (
                (1.0 if same_block else 0.0)
                + max(0.0, 1.0 - delta_y / (page.height * 0.04))
                + max(0.0, 0.6 - horizontal_gap)
                + (0.35 if same_or_above else 0.0)
            )
            initial.append((score, block))
        if not initial:
            continue
        non_anchor = [(score, block) for score, block in initial if block is not anchor]
        selected = [anchor]
        anchor_normalized = normalize_text(anchor.text)
        wanted_normalized = normalize_text(anchor_text)
        anchor_suffix = (
            anchor_normalized.split(wanted_normalized, 1)[1].strip()
            if wanted_normalized in anchor_normalized
            else ""
        )
        anchor_value = re.sub(r"[^\w]+", "", anchor_suffix, flags=re.UNICODE)
        # Label and value in the same OCR block is the strongest possible
        # geometric link, even when no separate block exists to its right.
        geometry = 1.0 if anchor_value else 0.72
        if non_anchor:
            _, closest = max(non_anchor, key=lambda item: item[0])
            closest_y = (
                closest.bounding_box.y_min + closest.bounding_box.y_max
            ) / 2
            row_tolerance = max(page.height * 0.012, box.y_max - box.y_min)
            selected.extend(
                block
                for _, block in non_anchor
                if abs(
                    (block.bounding_box.y_min + block.bounding_box.y_max) / 2
                    - closest_y
                )
                <= row_tolerance
            )
            geometry = max(
                0.72,
                1.0 - abs(closest_y - anchor_y) / (page.height * 0.045),
            )
        selected = list({id(block): block for block in selected}.values())
        selected.sort(key=lambda item: item.bounding_box.x_min)
        value = LabelValueContext(
            text=" ".join(normalize_text(block.text) for block in selected),
            blocks=selected,
            anchor=anchor,
            geometry_score=min(1.0, geometry),
        )
        if best is None or value.geometry_score > best.geometry_score:
            best = value
    return best or LabelValueContext("", [], None, 0.0)


def categorical_confidence(
    blocks: list[OCRTextBlock],
    *,
    geometry_score: float,
    dictionary_match: bool,
    no_conflict: bool,
) -> float:
    """Объяснимый ансамбль OCR + словарь + связь с label + непротиворечивость."""

    ocr_quality = (
        sum(block.effective_confidence for block in blocks) / len(blocks) if blocks else 0.0
    )
    score = (
        0.45 * ocr_quality
        + 0.30 * float(dictionary_match)
        + 0.15 * geometry_score
        + 0.10 * float(no_conflict)
    )
    return min(0.99, max(0.0, score))
