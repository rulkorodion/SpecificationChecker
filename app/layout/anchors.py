"""Поиск текстовых якорей без использования reference."""

from __future__ import annotations

from difflib import SequenceMatcher

from app.normalization.text import normalize_text
from app.ocr.models import OCRPageResult, OCRTextBlock


def find_anchor(
    pages: list[OCRPageResult], variants: list[str], threshold: float = 0.8
) -> tuple[OCRTextBlock | None, float]:
    best: OCRTextBlock | None = None
    best_score = 0.0
    normalized_variants = [normalize_text(value) for value in variants]
    for page in pages:
        for block in page.blocks:
            text = normalize_text(block.text)
            for variant in normalized_variants:
                score = 1.0 if variant in text else SequenceMatcher(None, text, variant).ratio()
                score *= block.effective_confidence
                if score > best_score:
                    best, best_score = block, score
    return (best, best_score) if best_score >= threshold else (None, best_score)
