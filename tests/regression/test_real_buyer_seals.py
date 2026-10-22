"""Regression on real rendered pages without reference or ground-truth input."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from app.document.renderer import PageImage
from app.document.loader import DocumentLoader
from app.document.renderer import DocumentRenderer
from app.enums import PresenceStatus
from app.extractors.buyer_block import extract_buyer_visuals
from app.ocr.models import OCRDocumentResult, OCRPageResult
from app.vision.seal_detector import OpenCVSealDetector
from app.vision.signature_detector import OpenCVSignatureDetector


DATASET_OUTPUT = Path("output/dataset_evaluation")


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("testdata/specifications/documents/Спецификация 35.jpg").is_file(),
    reason="private document corpus is not available",
)
def test_specification_35_lower_right_buyer_row_without_ocr_anchor():
    image_path = Path("testdata/specifications/documents/Спецификация 35.jpg")
    image = cv2.imdecode(
        np.fromfile(str(image_path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert image is not None
    height, width = image.shape[:2]
    page = PageImage(1, image, width, height)
    ocr_page = OCRPageResult(
        page_number=1,
        width=width,
        height=height,
        blocks=[],
        preprocessing_profile="saved",
        engine_name="no-anchor-regression",
    )
    document = OCRDocumentResult(
        pages=[ocr_page],
        normalized_full_text="",
        engine_name="no-anchor-regression",
    )
    config = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))

    signature, seal, _ = extract_buyer_visuals(
        [page],
        document,
        config,
        OpenCVSealDetector(
            float(config["buyer_visual_checks"]["min_seal_confidence"])
        ),
        OpenCVSignatureDetector(
            float(config["buyer_visual_checks"]["min_signature_confidence"])
        ),
    )

    assert signature.normalized_value is PresenceStatus.PRESENT
    assert seal.normalized_value is PresenceStatus.PRESENT
    assert signature.confidence >= 0.85
    assert seal.confidence >= 0.90
    assert signature.raw_value["selected_roi"] == "fallback_lower_right_signature_row"
    assert seal.raw_value["selected_roi"] == "fallback_lower_right_signature_row"


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename",
    [
        "manual_review_260934.pdf",
        "manual_review_264878.pdf",
    ],
)
def test_manual_review_false_negatives_have_buyer_signature_and_seal(filename):
    path = Path("testdata/specifications/documents") / filename
    if not path.is_file():
        pytest.skip("private document corpus is not available")
    loaded = DocumentLoader([".pdf"]).load(path)
    pages = DocumentRenderer(dpi=300).render(loaded)
    ocr_pages = [
        OCRPageResult(
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            blocks=[],
            preprocessing_profile="saved",
            engine_name="no-anchor-regression",
        )
        for page in pages
    ]
    document = OCRDocumentResult(
        pages=ocr_pages,
        normalized_full_text="",
        engine_name="no-anchor-regression",
    )
    config = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))

    signature, seal, _ = extract_buyer_visuals(
        pages,
        document,
        config,
        OpenCVSealDetector(
            float(config["buyer_visual_checks"]["min_seal_confidence"])
        ),
        OpenCVSignatureDetector(
            float(config["buyer_visual_checks"]["min_signature_confidence"])
        ),
    )

    assert signature.normalized_value is PresenceStatus.PRESENT
    assert seal.normalized_value is PresenceStatus.PRESENT
    assert signature.confidence >= 0.85
    assert seal.confidence >= 0.85


def _dataset_runs() -> list[Path]:
    return sorted(
        {
            info.parent
            for info in DATASET_OUTPUT.glob("*/document_info.json")
            if (info.parent / "ocr").is_dir() and (info.parent / "pages").is_dir()
        }
    )


@pytest.mark.integration
@pytest.mark.slow
def test_all_saved_real_documents_run_true_buyer_extraction():
    runs = _dataset_runs()
    if len(runs) < 4:
        pytest.skip("Run evaluate-dataset first to create four diagnostic datasets")
    config = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    results: list[
        tuple[Path, PresenceStatus, PresenceStatus, int | None]
    ] = []

    for run in runs:
        pages: list[PageImage] = []
        ocr_pages: list[OCRPageResult] = []
        for image_path in sorted((run / "pages").glob("page_*_original.png")):
            page_number = int(image_path.stem.split("_")[1])
            image = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert image is not None
            pages.append(
                PageImage(
                    page_number=page_number,
                    image=image,
                    width=image.shape[1],
                    height=image.shape[0],
                )
            )
            ocr_pages.append(
                OCRPageResult.model_validate_json(
                    (run / "ocr" / f"page_{page_number:03d}.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        document = OCRDocumentResult(
            pages=ocr_pages,
            normalized_full_text="",
            engine_name="saved-real-ocr",
        )
        signature, seal, _ = extract_buyer_visuals(
            pages,
            document,
            config,
            OpenCVSealDetector(
                float(config["buyer_visual_checks"]["min_seal_confidence"])
            ),
            OpenCVSignatureDetector(
                float(config["buyer_visual_checks"]["min_signature_confidence"])
            ),
        )
        results.append(
            (
                run,
                signature.normalized_value,
                seal.normalized_value,
                seal.page_number,
            )
        )

    assert len(results) == len(runs)
    assert all(
        signature is PresenceStatus.PRESENT
        for _, signature, _, _ in results
    )
    assert all(seal is PresenceStatus.PRESENT for _, _, seal, _ in results)
