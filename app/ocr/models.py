"""Neutral OCR structures consumed by layout-aware parsing and extractors."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas import BoundingBox, StrictModel


class OCRTextBlock(StrictModel):
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox
    page_number: int
    preprocessing_profile: str
    engine_name: str
    block_type: str = "Text"
    source: str = "ocr"
    reading_order: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_confidence(self) -> float:
        """Compatibility score for geometry helpers, not a fabricated model score."""

        return self.confidence if self.confidence is not None else 1.0


class OCRLayoutElement(StrictModel):
    block_type: str
    page_number: int = Field(ge=1)
    reading_order: int = Field(ge=0)
    bbox_source_width: int = Field(gt=0)
    bbox_source_height: int = Field(gt=0)
    original_page_width: int = Field(gt=0)
    original_page_height: int = Field(gt=0)
    normalized_bbox: list[float] = Field(min_length=4, max_length=4)
    original_pixel_bbox: list[int] = Field(min_length=4, max_length=4)
    text: str = ""
    raw_response: str = ""
    native_content: str = ""
    crop_sha256: str | None = None
    model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRPageResult(StrictModel):
    page_number: int
    width: int
    height: int
    blocks: list[OCRTextBlock]
    preprocessing_profile: str
    engine_name: str
    warnings: list[str] = Field(default_factory=list)
    native_markdown: str = ""
    native_tables: list[dict[str, Any]] = Field(default_factory=list)
    layout_elements: list[OCRLayoutElement] = Field(default_factory=list)
    processing_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def mean_confidence(self) -> float:
        scored = [block.confidence for block in self.blocks if block.confidence is not None]
        # An unscored VLM page is not low-quality merely because the model does
        # not publish token/block probabilities.
        return sum(scored) / len(scored) if scored else (1.0 if self.blocks else 0.0)


class OCRDocumentResult(StrictModel):
    pages: list[OCRPageResult]
    normalized_full_text: str
    engine_name: str
    warnings: list[str] = Field(default_factory=list)
    full_text: str = ""
    native_markdown: str = ""
    native_tables: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str | None = None
    model_version: str | None = None
    pipeline_version: str | None = None
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
