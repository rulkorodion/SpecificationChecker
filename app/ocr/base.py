"""Интерфейс OCR и тестовый double."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.ocr.models import OCRPageResult


class OCREngine(ABC):
    @abstractmethod
    def recognize_page(
        self,
        image: np.ndarray,
        page_number: int,
        preprocessing_profile: str,
    ) -> OCRPageResult:
        """Распознать одну страницу с координатами и confidence."""


class FakeOCREngine(OCREngine):
    """Детерминированный OCR для unit/integration tests."""

    def __init__(self, pages: dict[int, OCRPageResult]) -> None:
        self.pages = pages

    def recognize_page(
        self, image: np.ndarray, page_number: int, preprocessing_profile: str
    ) -> OCRPageResult:
        return self.pages[page_number]

