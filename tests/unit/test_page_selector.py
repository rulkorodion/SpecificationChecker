from __future__ import annotations

import hashlib
import os
from pathlib import Path

import fitz
import pytest

from app.document.page_selector import (
    PageMarker,
    prepare_document_for_ocr,
    select_application_pages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PDF = Path(os.getenv("PAGE_SELECTOR_REAL_PDF", r"D:\onec-test-document.pdf"))
LOCAL_TESSDATA = PROJECT_ROOT / "tmp" / "tessdata"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_digest(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False)
    return hashlib.sha256(pixmap.samples).hexdigest()


def _make_pdf(path: Path, page_count: int) -> None:
    document = fitz.open()
    try:
        for number in range(1, page_count + 1):
            page = document.new_page()
            page.insert_text((72, 72), f"page {number}")
        document.save(path)
    finally:
        document.close()


def test_one_and_two_page_documents_bypass_selector(tmp_path):
    for page_count in (1, 2):
        source = tmp_path / f"short-{page_count}.pdf"
        _make_pdf(source, page_count)
        original_hash = _sha256(source)

        result = prepare_document_for_ocr(
            source,
            reference_source={"application_number": "243740"},
            output_dir=tmp_path / f"output-{page_count}",
            page_limit=2,
            settings={"command": "definitely-not-installed-tesseract"},
        )

        assert result.document_path == source
        assert result.metadata is None
        assert _sha256(source) == original_hash
        assert not (tmp_path / f"output-{page_count}" / "selected_document.pdf").exists()
        assert not (tmp_path / f"output-{page_count}" / "page_selection.json").exists()


@pytest.mark.skipif(
    not REAL_PDF.is_file() or not (LOCAL_TESSDATA / "rus.traineddata").is_file(),
    reason="real 1C PDF or local Russian Tesseract data is unavailable",
)
def test_real_six_page_pdf_selects_243740_in_logical_order(tmp_path):
    original_hash = _sha256(REAL_PDF)
    result = select_application_pages(
        REAL_PDF,
        target_application_number="243740",
        output_dir=tmp_path,
        page_limit=2,
        settings={
            "command": "tesseract",
            "language": "rus",
            "tessdata_dir": str(LOCAL_TESSDATA),
            "dpi": 300,
            "top_crop_ratio": 0.24,
            "bottom_crop_ratio": 0.18,
            "timeout_seconds": 20,
        },
    )

    assert _sha256(REAL_PDF) == original_hash
    assert result.metadata is not None
    assert result.metadata["status"] == "COMPLETED"
    assert [
        item["source_page_number"] for item in result.metadata["selected_pages"]
    ] == [4, 3]
    assert [
        item["logical_page_number"] for item in result.metadata["selected_pages"]
    ] == [1, 2]
    assert (tmp_path / "page_selection.json").is_file()
    with fitz.open(REAL_PDF) as source, fitz.open(result.document_path) as selected:
        assert selected.page_count == 2
        assert _render_digest(selected[0]) == _render_digest(source[3])
        assert _render_digest(selected[1]) == _render_digest(source[2])


def test_ambiguous_selection_fails_closed_and_writes_diagnostic(tmp_path, monkeypatch):
    source = tmp_path / "compound.pdf"
    _make_pdf(source, 3)
    markers = {
        1: [PageMarker("243740", 1, 2, "bottom")],
        2: [PageMarker("243740", 1, 2, "bottom")],
        3: [PageMarker("243740", 2, 2, "bottom")],
    }
    monkeypatch.setattr(
        "app.document.page_selector._resolve_tesseract", lambda _command: "tesseract"
    )
    monkeypatch.setattr(
        "app.document.page_selector._page_markers",
        lambda page, **_kwargs: markers[page.number + 1],
    )

    from app.exceptions import PageSelectionError

    with pytest.raises(PageSelectionError) as raised:
        select_application_pages(
            source,
            target_application_number="243740",
            output_dir=tmp_path / "output",
            page_limit=2,
            settings={},
        )
    assert raised.value.diagnostic_code == "ambiguous_page_set"
    diagnostic = (tmp_path / "output" / "page_selection.json").read_text(
        encoding="utf-8"
    )
    assert '"status": "FAILED"' in diagnostic
    assert '"code": "ambiguous_page_set"' in diagnostic
