from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.config import load_config
from app.ocr.base import OCREngine
from app.ocr.batch import run_ocr_batch
from app.ocr.cache import OCRCache
from app.ocr.models import OCRPageResult


class CountingOCREngine(OCREngine):
    def __init__(self) -> None:
        self.documents_started = 0
        self.pages_recognized = 0

    def start_document(self) -> None:
        self.documents_started += 1

    def recognize_page(
        self,
        image: np.ndarray,
        page_number: int,
        preprocessing_profile: str,
    ) -> OCRPageResult:
        self.pages_recognized += 1
        height, width = image.shape[:2]
        return OCRPageResult(
            page_number=page_number,
            width=width,
            height=height,
            blocks=[],
            preprocessing_profile=preprocessing_profile,
            engine_name="counting-fake",
        )


def test_ocr_batch_warms_and_reuses_compact_cache(tmp_path: Path):
    document = tmp_path / "scan.jpg"
    Image.new("RGB", (1200, 1600), "white").save(document, quality=85)
    output = tmp_path / "reports" / "summary.json"
    cache = OCRCache(
        tmp_path / "cache",
        max_size_mb=10,
        retention_days=30,
        cleanup_on_start=False,
    )
    config = load_config()
    config["ocr"]["retry_with_preprocessing"] = False

    first_engine = CountingOCREngine()
    first = run_ocr_batch(
        [document],
        output,
        config,
        first_engine,
        cache,
        reuse_cache=True,
    )
    assert first_engine.documents_started == 1
    assert first_engine.pages_recognized == 1
    assert first["documents"][0]["cache"]["created"] is True
    assert first["documents"][0]["cache"]["size_bytes"] < 4096

    second_engine = CountingOCREngine()
    second = run_ocr_batch(
        [document],
        output,
        config,
        second_engine,
        cache,
        reuse_cache=True,
    )
    assert second_engine.documents_started == 0
    assert second_engine.pages_recognized == 0
    assert second["documents"][0]["cache"]["hit"] is True
