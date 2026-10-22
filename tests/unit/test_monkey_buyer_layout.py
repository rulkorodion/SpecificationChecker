from __future__ import annotations

import numpy as np

from app.document.renderer import PageImage
from app.extractors.buyer_block import _layout_picture_roi
from app.ocr.models import OCRLayoutElement, OCRPageResult


def _element(kind: str, order: int, bbox: list[int], text: str = "") -> OCRLayoutElement:
    return OCRLayoutElement(
        block_type=kind,
        page_number=1,
        reading_order=order,
        bbox_source_width=1000,
        bbox_source_height=1000,
        original_page_width=1000,
        original_page_height=1000,
        normalized_bbox=[float(value) for value in bbox],
        original_pixel_bbox=bbox,
        text=text,
    )


def _page(elements: list[OCRLayoutElement]) -> tuple[PageImage, OCRPageResult]:
    rendered = PageImage(1, np.full((1000, 1000, 3), 255, np.uint8), 1000, 1000)
    ocr = OCRPageResult(
        page_number=1,
        width=1000,
        height=1000,
        blocks=[],
        preprocessing_profile="original",
        engine_name="fake",
        layout_elements=elements,
    )
    return rendered, ocr


def test_layout_locator_selects_buyer_picture_not_supplier_or_logo():
    rendered, ocr = _page(
        [
            _element("Picture", 0, [50, 30, 180, 140]),
            _element("Text", 1, [80, 700, 190, 730], "Поставщик:"),
            _element("Picture", 2, [80, 650, 390, 900]),
            _element("Text", 3, [580, 700, 700, 730], "Покупатель:"),
            _element("Picture", 4, [520, 640, 920, 920]),
        ]
    )
    selected, diagnostics = _layout_picture_roi(rendered, ocr)
    assert selected is not None
    assert selected[0] == "layout_picture"
    assert diagnostics["selected_picture_bbox"] == [520, 640, 920, 920]
    rejected = diagnostics["rejection_reasons"]
    assert any("upper_page_logo_or_qr" in item["reasons"] for item in rejected)
    assert any("closer_to_supplier_anchor" in item["reasons"] for item in rejected)


def test_layout_locator_without_picture_falls_back_to_anchor_geometry():
    rendered, ocr = _page(
        [_element("Text", 0, [580, 700, 700, 730], "Покупатель:")]
    )
    selected, diagnostics = _layout_picture_roi(rendered, ocr)
    assert selected is None
    assert diagnostics["locator_confidence"] == 0.0


def test_layout_locator_without_buyer_anchor_rejects_all_pictures():
    rendered, ocr = _page([_element("Picture", 0, [520, 640, 920, 920])])
    selected, _ = _layout_picture_roi(rendered, ocr)
    assert selected is None
