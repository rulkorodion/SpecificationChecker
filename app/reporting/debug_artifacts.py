"""Сохранение OCR JSON, страниц, boxes и релевантных crops."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.document.preprocessing import preprocess
from app.document.renderer import PageImage, save_page
from app.extractors.buyer_block import BuyerROIArtifact, BuyerVisualArtifacts
from app.ocr.models import OCRDocumentResult, OCRPageResult
from app.schemas import CheckResult, ExtractedSpecification, TableAggregation


def save_ocr_artifacts(document: OCRDocumentResult, output_dir: Path) -> None:
    ocr_dir = output_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    for page in document.pages:
        (ocr_dir / f"page_{page.page_number:03d}.json").write_text(
            page.model_dump_json(indent=2), encoding="utf-8"
        )
    (ocr_dir / "full_text.txt").write_text(
        document.normalized_full_text, encoding="utf-8"
    )
    if document.native_markdown:
        (ocr_dir / "native_markdown.md").write_text(
            document.native_markdown, encoding="utf-8"
        )
    if document.native_tables:
        (ocr_dir / "native_tables.json").write_text(
            json.dumps(document.native_tables, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def save_monkey_artifacts(
    pages: list[PageImage], document: OCRDocumentResult, output_dir: Path
) -> None:
    """Persist the complete two-stage trace only for debug-level=full."""

    if document.pipeline_version is None:
        return
    root = output_dir / "monkey"
    crops_dir = root / "crops"
    raw_dir = root / "recognition_raw"
    root.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    page_map = {page.page_number: page for page in pages}
    timings: dict[str, object] = {"pages": []}
    for ocr_page in document.pages:
        page = page_map.get(ocr_page.page_number)
        if page is None:
            continue
        save_page(root / f"page_{ocr_page.page_number:03d}_original.png", page.image)
        preview = page.image.copy()
        for element in ocr_page.layout_elements:
            x1, y1, x2, y2 = element.original_pixel_bbox
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 220), 3)
            cv2.putText(
                preview,
                f"{element.reading_order}:{element.block_type}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 210),
                1,
                cv2.LINE_AA,
            )
            crop = page.image[y1:y2, x1:x2]
            stem = (
                f"page_{ocr_page.page_number}_element_"
                f"{element.reading_order + 1:03d}_{_safe_name(element.block_type.lower())}"
            )
            if crop.size:
                save_page(crops_dir / f"{stem}.png", crop)
            suffix = ".html" if element.block_type == "Table" else ".txt"
            (raw_dir / f"{stem}{suffix}").write_text(
                element.raw_response, encoding="utf-8"
            )
        save_page(
            root / f"page_{ocr_page.page_number:03d}_layout_preview.png",
            preview,
        )
        (root / f"page_{ocr_page.page_number:03d}_layout_raw.json").write_text(
            json.dumps(
                {
                    "raw": ocr_page.processing_metadata.get("layout_raw", ""),
                    "input_size": ocr_page.processing_metadata.get("layout_input_size"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / f"page_{ocr_page.page_number:03d}_layout_normalized.json").write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in ocr_page.layout_elements],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        timings["pages"].append(ocr_page.processing_metadata.get("timings", {}))
    (root / "native_markdown.md").write_text(document.native_markdown, encoding="utf-8")
    (root / "native_tables.json").write_text(
        json.dumps(document.native_tables, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "ocr_document.json").write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    (root / "timings.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_error_artifacts(
    output_dir: Path,
    checks: list[CheckResult],
    buyer: BuyerVisualArtifacts | None,
) -> None:
    """Сохранить только минимальные доказательства проблемных проверок."""

    problematic = [item for item in checks if item.status.value != "PASS"]
    if not problematic:
        return
    error_dir = output_dir / "debug_errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    (error_dir / "checks.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in problematic],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if buyer is None or not any(item.code.startswith("BUYER_") for item in problematic):
        return
    selected = next(
        (
            item
            for item in buyer.roi_candidates
            if item.page_number == buyer.page_number
            and item.name == buyer.selected_roi_name
        ),
        None,
    )
    if selected is None:
        return
    image = selected.roi
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side > 1200:
        scale = 1200 / max_side
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(
        str(error_dir / "buyer_selected_roi.jpg"),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 78],
    )


def _draw_boxes(page: PageImage, ocr: OCRPageResult) -> np.ndarray:
    image = page.image.copy()
    for block in ocr.blocks:
        box = block.bounding_box
        p1 = (int(box.x_min), int(box.y_min))
        p2 = (int(box.x_max), int(box.y_max))
        cv2.rectangle(image, p1, p2, (0, 180, 0), 2)
        score = "n/a" if block.confidence is None else f"{block.confidence:.2f}"
        label = f"{block.text[:24]} {score}"
        cv2.putText(
            image,
            label,
            (p1[0], max(15, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 200),
            1,
            cv2.LINE_AA,
        )
    return image


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )


def _draw_candidate(artifact: BuyerROIArtifact) -> np.ndarray:
    image = artifact.roi.copy()
    if artifact.seal.candidate_bbox is not None:
        x1, y1, x2, y2 = artifact.seal.candidate_bbox
        color = (
            (0, 150, 0)
            if artifact.seal.status.value == "PRESENT"
            else (0, 170, 255)
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    label = (
        f"{artifact.name}: {artifact.seal.status.value} "
        f"{artifact.seal.confidence:.3f}"
    )
    cv2.putText(
        image,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 210),
        2,
        cv2.LINE_AA,
    )
    return image


def _draw_signature_candidate(artifact: BuyerROIArtifact) -> np.ndarray:
    image = artifact.roi.copy()
    if artifact.signature.candidate_bbox is not None:
        x1, y1, x2, y2 = artifact.signature.candidate_bbox
        color = (
            (0, 150, 0)
            if artifact.signature.status.value == "PRESENT"
            else (0, 170, 255)
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    cv2.putText(
        image,
        f"{artifact.name}: {artifact.signature.status.value} "
        f"{artifact.signature.confidence:.3f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 210),
        2,
        cv2.LINE_AA,
    )
    return image


def _draw_buyer_rois(page: PageImage, candidates: list[BuyerROIArtifact]) -> np.ndarray:
    image = page.image.copy()
    palette = [
        (30, 180, 30),
        (230, 140, 20),
        (190, 40, 190),
        (30, 170, 230),
    ]
    for index, artifact in enumerate(candidates):
        x1, y1, x2, y2 = artifact.bounds
        color = palette[index % len(palette)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            image,
            f"{artifact.name} {artifact.seal.status.value} "
            f"{artifact.seal.confidence:.3f}",
            (x1 + 6, max(24, y1 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        if artifact.anchor_bbox is not None:
            ax1, ay1, ax2, ay2 = artifact.anchor_bbox
            cv2.rectangle(image, (ax1, ay1), (ax2, ay2), (0, 0, 230), 3)
        if artifact.seal.candidate_bbox is not None:
            cx1, cy1, cx2, cy2 = artifact.seal.candidate_bbox
            cv2.rectangle(
                image,
                (x1 + cx1, y1 + cy1),
                (x1 + cx2, y1 + cy2),
                (255, 40, 20),
                4,
            )
    return image


def save_debug_artifacts(
    pages: list[PageImage],
    ocr: OCRDocumentResult,
    output_dir: Path,
    buyer: BuyerVisualArtifacts | None = None,
    extracted: ExtractedSpecification | None = None,
    tables: TableAggregation | None = None,
) -> None:
    save_monkey_artifacts(pages, ocr, output_dir)
    pages_dir = output_dir / "pages"
    for page, ocr_page in zip(pages, ocr.pages, strict=False):
        save_page(pages_dir / f"page_{page.page_number:03d}_original.png", page.image)
        save_page(
            pages_dir / f"page_{page.page_number:03d}_enhanced.png",
            preprocess(page.image, "enhanced_gray"),
        )
        save_page(
            pages_dir / f"page_{page.page_number:03d}_geometry_corrected.png",
            preprocess(page.image, "geometry_corrected"),
        )
        save_page(
            pages_dir / f"page_{page.page_number:03d}_thresholded.png",
            preprocess(page.image, "thresholded"),
        )
        save_page(
            pages_dir / f"page_{page.page_number:03d}_ocr_boxes.png",
            _draw_boxes(page, ocr_page),
        )
    if buyer is not None:
        save_page(
            output_dir / "crops" / f"buyer_block_page_{buyer.page_number:03d}.png",
            buyer.roi,
        )
        selected_written = False
        for page in pages:
            page_candidates = [
                candidate
                for candidate in buyer.roi_candidates
                if candidate.page_number == page.page_number
            ]
            if page_candidates:
                save_page(
                    pages_dir
                    / f"page_{page.page_number:03d}_buyer_rois.png",
                    _draw_buyer_rois(page, page_candidates),
                )
            for candidate in page_candidates:
                name = _safe_name(candidate.name)
                save_page(
                    output_dir
                    / "crops"
                    / f"buyer_roi_page_{candidate.page_number:03d}_{name}.png",
                    candidate.roi,
                )
                save_page(
                    output_dir
                    / "buyer_regions"
                    / f"page_{candidate.page_number:03d}_{name}.png",
                    candidate.roi,
                )
                candidate_image = _draw_candidate(candidate)
                save_page(
                    output_dir
                    / "seal_candidates"
                    / f"page_{candidate.page_number:03d}_{name}.png",
                    candidate_image,
                )
                save_page(
                    output_dir
                    / "seal"
                    / f"page_{candidate.page_number:03d}_{name}_candidate.png",
                    candidate_image,
                )
                signature_image = _draw_signature_candidate(candidate)
                save_page(
                    output_dir
                    / "signature"
                    / f"page_{candidate.page_number:03d}_{name}_candidate.png",
                    signature_image,
                )
                if (
                    not selected_written
                    and candidate.page_number == buyer.page_number
                    and candidate.name == buyer.selected_roi_name
                ):
                    save_page(
                        output_dir / "seal_candidates" / "seal_candidate_selected.png",
                        candidate_image,
                    )
                    selected_written = True
                for profile_name, profile in candidate.preprocessing_profiles.items():
                    save_page(
                        output_dir
                        / "seal_profiles"
                        / f"page_{candidate.page_number:03d}_{name}"
                        / f"{_safe_name(profile_name)}.png",
                        profile,
                    )
                    save_page(
                        output_dir
                        / "seal"
                        / f"page_{candidate.page_number:03d}_{name}"
                        / f"{_safe_name(profile_name)}.png",
                        profile,
                    )
                for profile_name, profile in candidate.signature_profiles.items():
                    save_page(
                        output_dir
                        / "signature"
                        / f"page_{candidate.page_number:03d}_{name}"
                        / f"{_safe_name(profile_name)}.png",
                        profile,
                    )

    if tables is not None:
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        page_map = {page.page_number: page for page in pages}
        for index, table in enumerate(tables.tables, start=1):
            (tables_dir / f"table_{index:03d}.json").write_text(
                table.model_dump_json(indent=2), encoding="utf-8"
            )
            page = page_map.get(table.page_number)
            if page is not None:
                image = page.image.copy()
                box = table.bounding_box
                cv2.rectangle(
                    image,
                    (int(box.x_min), int(box.y_min)),
                    (int(box.x_max), int(box.y_max)),
                    (30, 170, 30),
                    5,
                )
                cv2.putText(
                    image,
                    f"table_{index:03d} {table.table_type.value} "
                    f"q={table.total_quantity} total={table.total_with_vat}",
                    (int(box.x_min), max(30, int(box.y_min) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (20, 20, 210),
                    2,
                    cv2.LINE_AA,
                )
                save_page(
                    tables_dir / f"table_{index:03d}_detected.png",
                    image,
                )
        (tables_dir / "aggregation.json").write_text(
            tables.model_dump_json(indent=2), encoding="utf-8"
        )

    if extracted is not None:
        transportation_dir = output_dir / "transportation"
        transportation_dir.mkdir(parents=True, exist_ok=True)
        (transportation_dir / "evidence.json").write_text(
            json.dumps(
                extracted.transportation_costs_included.raw_value,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def save_document_info(data: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "document_info.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
