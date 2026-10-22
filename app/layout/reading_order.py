"""Восстановление естественного порядка OCR-блоков."""

from __future__ import annotations

from app.normalization.text import normalize_text
from app.ocr.models import OCRPageResult, OCRTextBlock


def ordered_blocks(page: OCRPageResult) -> list[OCRTextBlock]:
    """Сгруппировать блоки по строкам и отсортировать слева направо."""

    if not page.blocks:
        return []
    tolerance = max(8.0, page.height * 0.004)
    source = sorted(
        page.blocks,
        key=lambda item: (
            (item.bounding_box.y_min + item.bounding_box.y_max) / 2,
            item.bounding_box.x_min,
        ),
    )
    lines: list[list[OCRTextBlock]] = []
    centers: list[float] = []
    for block in source:
        center = (block.bounding_box.y_min + block.bounding_box.y_max) / 2
        if not lines or abs(center - centers[-1]) > tolerance:
            lines.append([block])
            centers.append(center)
        else:
            lines[-1].append(block)
            centers[-1] = sum(
                (x.bounding_box.y_min + x.bounding_box.y_max) / 2 for x in lines[-1]
            ) / len(lines[-1])
    return [block for line in lines for block in sorted(line, key=lambda x: x.bounding_box.x_min)]


def page_lines(page: OCRPageResult) -> list[list[OCRTextBlock]]:
    blocks = ordered_blocks(page)
    if not blocks:
        return []
    tolerance = max(8.0, page.height * 0.004)
    lines: list[list[OCRTextBlock]] = []
    for block in blocks:
        center = (block.bounding_box.y_min + block.bounding_box.y_max) / 2
        if not lines:
            lines.append([block])
            continue
        previous = sum(
            (x.bounding_box.y_min + x.bounding_box.y_max) / 2 for x in lines[-1]
        ) / len(lines[-1])
        if abs(center - previous) <= tolerance:
            lines[-1].append(block)
        else:
            lines.append([block])
    return lines


def normalized_document_text(pages: list[OCRPageResult]) -> str:
    text = "\n".join(
        " ".join(block.text for block in line)
        for page in sorted(pages, key=lambda item: item.page_number)
        for line in page_lines(page)
    )
    return normalize_text(text)

