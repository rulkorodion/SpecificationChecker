from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from app.config import load_config
from app.cli import _parse_id_filter
from app.document.loader import LoadedDocument
from app.ocr.cache import OCRCache, cache_fingerprint, cache_key
from app.ocr.models import OCRDocumentResult, OCRPageResult


def _loaded(digest: str = "a" * 64) -> LoadedDocument:
    return LoadedDocument(Path("document.pdf"), ".pdf", "application/pdf", digest, 100)


def _ocr() -> OCRDocumentResult:
    page = OCRPageResult(
        page_number=1,
        width=100,
        height=200,
        blocks=[],
        preprocessing_profile="original",
        engine_name="fake",
    )
    return OCRDocumentResult(pages=[page], normalized_full_text="", engine_name="fake")


def test_cache_key_tracks_document_render_preprocessing_and_model():
    config = load_config()
    original = cache_key(cache_fingerprint(_loaded(), config))
    changed_hash = cache_key(cache_fingerprint(_loaded("b" * 64), config))
    render = deepcopy(config)
    render["document"]["render_dpi"] = 200
    changed_render = cache_key(cache_fingerprint(_loaded(), render))
    model = deepcopy(config)
    model["ocr"]["monkeyocr_v2"]["model_id"] = "other-model"
    changed_model = cache_key(cache_fingerprint(_loaded(), model))
    assert len({original, changed_hash, changed_render, changed_model}) == 4


def test_cache_key_tracks_monkey_adapter_and_crop_settings():
    original = load_config()
    changed_adapter = deepcopy(original)
    changed_adapter["ocr"]["monkeyocr_v2"]["adapter_version"] += 1
    changed_crop = deepcopy(original)
    changed_crop["ocr"]["monkeyocr_v2"]["crop_padding_ratio"] = 0.01
    keys = {
        cache_key(cache_fingerprint(_loaded(), original)),
        cache_key(cache_fingerprint(_loaded(), changed_adapter)),
        cache_key(cache_fingerprint(_loaded(), changed_crop)),
    }
    assert len(keys) == 3


def test_gzip_cache_roundtrip_and_dry_cleanup(tmp_path):
    cache = OCRCache(
        tmp_path / "ocr",
        max_size_mb=10,
        retention_days=30,
        cleanup_on_start=False,
    )
    fingerprint = cache_fingerprint(_loaded(), load_config())
    stored = cache.put(fingerprint, _ocr())
    restored, metadata = cache.get(fingerprint)
    assert stored["size_bytes"] < 4096
    assert metadata["hit"] is True
    assert restored is not None
    assert restored.pages[0].width == 100
    dry = OCRCache(
        tmp_path / "ocr",
        max_size_mb=10,
        retention_days=0,
        cleanup_on_start=False,
    ).cleanup(dry_run=True)
    assert dry["removed_files"] == 0


def test_batch_id_filter_supports_ranges():
    assert _parse_id_filter("25-29,31") == {"25", "26", "27", "28", "29", "31"}
