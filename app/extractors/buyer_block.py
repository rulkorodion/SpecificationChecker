"""Anchor-based buyer regions and fail-closed page/ROI aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from app.document.renderer import PageImage
from app.enums import PresenceStatus
from app.extractors.base import extracted_field
from app.normalization.text import normalize_text
from app.ocr.models import OCRDocumentResult, OCRPageResult
from app.schemas import BoundingBox, ExtractedField
from app.vision.models import VisualDetection
from app.vision.seal_detector import SealDetectionContext, SealDetector
from app.vision.signature_detector import SignatureDetectionContext, SignatureDetector


@dataclass(slots=True)
class BuyerROIArtifact:
    page_number: int
    name: str
    bounds: tuple[int, int, int, int]
    anchor_bbox: tuple[int, int, int, int] | None
    roi: np.ndarray
    seal: VisualDetection
    signature: VisualDetection
    preprocessing_profiles: dict[str, np.ndarray] = field(default_factory=dict)
    signature_profiles: dict[str, np.ndarray] = field(default_factory=dict)
    locator_strategy: str = "geometric"
    locator_confidence: float = 0.0
    selected_picture_bbox: tuple[int, int, int, int] | None = None
    distance_to_buyer_anchor: float | None = None
    distance_to_supplier_anchor: float | None = None
    rejection_reasons: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class BuyerVisualArtifacts:
    page_number: int
    roi: np.ndarray
    seal_status: PresenceStatus
    signature_status: PresenceStatus
    selected_roi_name: str
    roi_candidates: list[BuyerROIArtifact] = field(default_factory=list)


def _clip_bounds(
    page: PageImage,
    values: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x1 = max(0, min(page.width - 1, int(round(values[0]))))
    y1 = max(0, min(page.height - 1, int(round(values[1]))))
    x2 = max(x1 + 1, min(page.width, int(round(values[2]))))
    y2 = max(y1 + 1, min(page.height, int(round(values[3]))))
    return x1, y1, x2, y2


def _crop(page: PageImage, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    return page.image[y1:y2, x1:x2].copy()


def _buyer_label_anchors(ocr_page: OCRPageResult | None) -> list[BoundingBox]:
    if ocr_page is None:
        return []
    anchors: list[BoundingBox] = []
    for block in ocr_page.blocks:
        text = normalize_text(block.text)
        compact = "".join(character for character in text if character.isalpha())
        buyer_similarity = SequenceMatcher(
            None, compact[: len("покупатель") + 2], "покупатель"
        ).ratio()
        if (
            len(text) <= 40
            and (
                compact.startswith("покупател")
                or buyer_similarity >= 0.72
            )
            and block.effective_confidence >= 0.70
        ):
            anchors.append(block.bounding_box)
    return anchors


def _layout_picture_roi(
    page: PageImage,
    ocr_page: OCRPageResult | None,
) -> tuple[
    tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]] | None,
    dict,
]:
    """Select a buyer Picture using MonkeyOCR layout, without treating it as proof."""

    diagnostics: dict = {
        "locator_strategy": "layout_picture",
        "selected_picture_bbox": None,
        "distance_to_buyer_anchor": None,
        "distance_to_supplier_anchor": None,
        "rejection_reasons": [],
        "locator_confidence": 0.0,
    }
    if ocr_page is None or not ocr_page.layout_elements:
        return None, diagnostics

    def label_matches(value: str, target: str) -> bool:
        normalized = normalize_text(value)
        compact = "".join(character for character in normalized if character.isalpha())
        return target in compact and len(normalized) <= 50

    buyer_elements = [
        item
        for item in ocr_page.layout_elements
        if item.block_type != "Picture" and label_matches(item.text, "покупател")
    ]
    if not buyer_elements:
        return None, diagnostics
    buyer = max(buyer_elements, key=lambda item: item.original_pixel_bbox[1])
    supplier_elements = [
        item
        for item in ocr_page.layout_elements
        if item.block_type != "Picture" and label_matches(item.text, "поставщик")
    ]
    supplier = (
        min(
            supplier_elements,
            key=lambda item: abs(
                item.original_pixel_bbox[1] - buyer.original_pixel_bbox[1]
            ),
        )
        if supplier_elements
        else None
    )
    buyer_bbox = tuple(int(value) for value in buyer.original_pixel_bbox)
    bx = (buyer_bbox[0] + buyer_bbox[2]) / 2
    by = (buyer_bbox[1] + buyer_bbox[3]) / 2

    ranked: list[tuple[float, tuple[int, int, int, int], float, float | None]] = []
    for item in ocr_page.layout_elements:
        if item.block_type != "Picture":
            continue
        picture = tuple(int(value) for value in item.original_pixel_bbox)
        px = (picture[0] + picture[2]) / 2
        py = (picture[1] + picture[3]) / 2
        width = picture[2] - picture[0]
        height = picture[3] - picture[1]
        area_ratio = width * height / max(1, page.width * page.height)
        buyer_distance = (
            ((px - bx) / page.width) ** 2 + ((py - by) / page.height) ** 2
        ) ** 0.5
        supplier_distance = None
        if supplier is not None:
            supplier_bbox = supplier.original_pixel_bbox
            sx = (supplier_bbox[0] + supplier_bbox[2]) / 2
            sy = (supplier_bbox[1] + supplier_bbox[3]) / 2
            supplier_distance = (
                ((px - sx) / page.width) ** 2
                + ((py - sy) / page.height) ** 2
            ) ** 0.5
        reasons: list[str] = []
        if py < page.height * 0.35 and by > page.height * 0.50:
            reasons.append("upper_page_logo_or_qr")
        if abs(py - by) > page.height * 0.25:
            reasons.append("outside_buyer_signature_band")
        if buyer_distance > 0.48:
            reasons.append("too_far_from_buyer_anchor")
        if supplier_distance is not None and supplier_distance + 0.02 < buyer_distance:
            reasons.append("closer_to_supplier_anchor")
        if area_ratio > 0.35:
            reasons.append("oversized_watermark_or_page_picture")
        if reasons:
            diagnostics["rejection_reasons"].append(
                {"bbox": list(picture), "reasons": reasons}
            )
            continue
        score = buyer_distance + (0.08 if area_ratio < 0.0008 else 0.0)
        ranked.append((score, picture, buyer_distance, supplier_distance))

    if not ranked:
        return None, diagnostics
    _, picture, buyer_distance, supplier_distance = min(ranked, key=lambda item: item[0])
    padding = max(page.width, page.height) * 0.012
    padded = _clip_bounds(
        page,
        (
            picture[0] - padding,
            picture[1] - padding,
            picture[2] + padding,
            picture[3] + padding,
        ),
    )
    diagnostics.update(
        {
            "selected_picture_bbox": list(picture),
            "distance_to_buyer_anchor": round(buyer_distance, 6),
            "distance_to_supplier_anchor": (
                round(supplier_distance, 6) if supplier_distance is not None else None
            ),
            "locator_confidence": max(0.70, min(0.98, 0.98 - buyer_distance)),
        }
    )
    return ("layout_picture", padded, buyer_bbox), diagnostics


def _anchor_rois(
    page: PageImage,
    anchor: BoundingBox,
) -> list[tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """Use page-relative margins so 150/300 DPI scans behave consistently."""

    anchor_page = (
        int(anchor.x_min),
        int(anchor.y_min),
        int(anchor.x_max),
        int(anchor.y_max),
    )
    rois = [
        (
            "compact",
            _clip_bounds(
                page,
                (
                    anchor.x_min - page.width * 0.22,
                    anchor.y_min - page.height * 0.035,
                    anchor.x_max + page.width * 0.22,
                    anchor.y_max + page.height * 0.22,
                ),
            ),
            anchor_page,
        ),
        (
            "expanded",
            _clip_bounds(
                page,
                (
                    anchor.x_min - page.width * 0.36,
                    anchor.y_min - page.height * 0.08,
                    anchor.x_max + page.width * 0.32,
                    anchor.y_max + page.height * 0.32,
                ),
            ),
            anchor_page,
        ),
        (
            "full_signature_row",
            _clip_bounds(
                page,
                (
                    page.width * 0.03,
                    anchor.y_min - page.height * 0.05,
                    page.width * 0.98,
                    anchor.y_max + page.height * 0.30,
                ),
            ),
            anchor_page,
        ),
        (
            "anchor_left_extended",
            _clip_bounds(
                page,
                (
                    anchor.x_min - page.width * 0.45,
                    anchor.y_min - page.height * 0.09,
                    anchor.x_max + page.width * 0.12,
                    anchor.y_max + page.height * 0.34,
                ),
            ),
            anchor_page,
        ),
        (
            "anchor_right_extended",
            _clip_bounds(
                page,
                (
                    anchor.x_min - page.width * 0.12,
                    anchor.y_min - page.height * 0.09,
                    page.width * 0.99,
                    anchor.y_max + page.height * 0.34,
                ),
            ),
            anchor_page,
        ),
    ]
    return rois


def _fallback_roi(
    page: PageImage,
) -> tuple[str, tuple[int, int, int, int], None]:
    return (
        "fallback_lower_half",
        _clip_bounds(
            page,
            (
                page.width * 0.35,
                page.height * 0.50,
                page.width * 0.99,
                page.height * 0.99,
            ),
        ),
        None,
    )


def _fallback_lower_right_signature_roi(
    page: PageImage,
) -> tuple[str, tuple[int, int, int, int], None]:
    """OCR-independent buyer row for scans with an empty supplier block."""

    return (
        "fallback_lower_right_signature_row",
        _clip_bounds(
            page,
            (
                page.width * 0.50,
                page.height * 0.72,
                page.width * 0.99,
                page.height * 0.99,
            ),
        ),
        None,
    )


def _relative_anchor(
    anchor: tuple[int, int, int, int] | None,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if anchor is None:
        return None
    x1, y1, _, _ = bounds
    return (
        anchor[0] - x1,
        anchor[1] - y1,
        anchor[2] - x1,
        anchor[3] - y1,
    )


def _absolute_candidate_box(
    artifact: BuyerROIArtifact,
    page: PageImage,
    attribute: str = "seal",
) -> BoundingBox | None:
    detection = getattr(artifact, attribute)
    if detection.candidate_bbox is None:
        return None
    rx1, ry1, rx2, ry2 = detection.candidate_bbox
    x1, y1, _, _ = artifact.bounds
    return BoundingBox(
        x_min=x1 + rx1,
        y_min=y1 + ry1,
        x_max=x1 + rx2,
        y_max=y1 + ry2,
        page_width=page.width,
        page_height=page.height,
    )


def _best_detection(
    artifacts: list[BuyerROIArtifact],
    attribute: str,
) -> BuyerROIArtifact:
    """A strong PRESENT is never averaged down by a failed page or fallback ROI."""

    present = [
        artifact
        for artifact in artifacts
        if getattr(artifact, attribute).status is PresenceStatus.PRESENT
    ]
    anchored_present = [
        artifact for artifact in present if artifact.anchor_bbox is not None
    ]
    if anchored_present:
        present = anchored_present
    pool = present or [
        artifact
        for artifact in artifacts
        if getattr(artifact, attribute).status is PresenceStatus.UNCERTAIN
    ]
    pool = pool or artifacts
    return max(pool, key=lambda item: getattr(item, attribute).confidence)


def _trusted_lower_right_seal(
    artifact: BuyerROIArtifact,
    config: dict,
) -> bool:
    """Accept either a clean ring or a dark logo seal using independent evidence."""

    if (
        artifact.name != "fallback_lower_right_signature_row"
        or artifact.seal.status is not PresenceStatus.PRESENT
    ):
        return False
    features = artifact.seal.features
    no_false_positive_penalties = (
        float(features.get("qr_penalty", 1.0)) < 0.5
        and float(features.get("rectangular_stamp_penalty", 1.0)) < 0.5
        and float(features.get("supplier_location_penalty", 1.0)) < 0.5
    )
    if not no_false_positive_penalties:
        return False
    minimum = float(config["buyer_visual_checks"]["min_seal_confidence"])
    clean_ring = (
        artifact.seal.confidence >= minimum + 0.05
        and float(features.get("shape_score", 0.0)) >= 0.85
        and float(features.get("ring_score", 0.0)) >= 0.85
        and float(features.get("stroke_density_score", 0.0)) >= 0.80
    )
    logo_ring = (
        artifact.seal.confidence >= minimum
        and float(features.get("shape_score", 0.0)) >= 0.72
        and float(features.get("ring_score", 0.0)) >= 0.95
        and float(features.get("stroke_density_score", 0.0)) >= 0.90
        and float(features.get("angular_coverage", 0.0)) >= 0.80
        and float(features.get("internal_stroke_density", 0.0)) >= 0.18
        and float(features.get("multi_profile_consistency_score", 0.0)) >= 0.65
        and int(features.get("hough_profile_count", 0)) >= 2
        and max(
            float(features.get("color_score", 0.0)),
            float(features.get("ink_contrast_score", 0.0)),
        )
        >= 0.75
    )
    return clean_ring or logo_ring


def _trusted_broad_fallback_seal(
    artifact: BuyerROIArtifact,
    signature_artifact: BuyerROIArtifact,
    config: dict,
) -> bool:
    """Trust a strong broad-fallback seal only when a real signature corroborates it."""

    if (
        artifact.name != "fallback_lower_half"
        or artifact.seal.status is not PresenceStatus.PRESENT
        or signature_artifact.page_number != artifact.page_number
        or signature_artifact.signature.status is not PresenceStatus.PRESENT
        or artifact.seal.candidate_bbox is None
        or signature_artifact.signature.candidate_bbox is None
    ):
        return False
    features = artifact.seal.features
    minimum_seal = float(config["buyer_visual_checks"]["min_seal_confidence"])
    minimum_signature = float(
        config["buyer_visual_checks"]["min_signature_confidence"]
    )
    independently_strong = (
        artifact.seal.confidence >= minimum_seal + 0.05
        and signature_artifact.signature.confidence >= minimum_signature
        and float(features.get("shape_score", 0.0)) >= 0.85
        and float(features.get("ring_score", 0.0)) >= 0.85
        and float(features.get("stroke_density_score", 0.0)) >= 0.85
        and float(features.get("multi_profile_consistency_score", 0.0)) >= 0.72
        and float(features.get("angular_coverage", 0.0)) >= 0.70
        and int(features.get("hough_profile_count", 0)) >= 3
        and float(features.get("watermark_penalty", 1.0)) < 0.25
        and float(features.get("qr_penalty", 1.0)) < 0.25
        and float(features.get("rectangular_stamp_penalty", 1.0)) < 0.25
        and float(features.get("supplier_location_penalty", 1.0)) < 0.25
    )
    if not independently_strong:
        return False

    ax1, ay1, _, _ = artifact.bounds
    sx1, sy1, sx2, sy2 = artifact.seal.candidate_bbox
    seal_box = (ax1 + sx1, ay1 + sy1, ax1 + sx2, ay1 + sy2)
    bx1, by1, _, _ = signature_artifact.bounds
    hx1, hy1, hx2, hy2 = signature_artifact.signature.candidate_bbox
    signature_box = (bx1 + hx1, by1 + hy1, bx1 + hx2, by1 + hy2)
    intersection = max(
        0, min(seal_box[2], signature_box[2]) - max(seal_box[0], signature_box[0])
    ) * max(
        0, min(seal_box[3], signature_box[3]) - max(seal_box[1], signature_box[1])
    )
    seal_area = max(1, (seal_box[2] - seal_box[0]) * (seal_box[3] - seal_box[1]))
    signature_area = max(
        1,
        (signature_box[2] - signature_box[0])
        * (signature_box[3] - signature_box[1]),
    )
    return intersection / min(seal_area, signature_area) >= 0.12


def _signature_over_seal_confidence(
    artifact: BuyerROIArtifact,
    config: dict,
) -> float | None:
    """Confirm long curved strokes extending outside a trusted buyer seal."""

    if not _trusted_lower_right_seal(artifact, config):
        return None
    signature = artifact.signature
    features = signature.features
    if (
        signature.status is not PresenceStatus.UNCERTAIN
        or signature.confidence
        < float(config["buyer_visual_checks"]["min_signature_confidence"]) - 0.011
        or int(features.get("elongated_components", 0)) < 3
        or float(features.get("curvature_score", 0.0)) < 0.45
        or float(features.get("candidate_span_ratio", 0.0)) < 0.55
        or float(features.get("multi_profile_consistency_score", 0.0)) < 0.50
        or float(features.get("seal_overlap_score", 0.0)) < 0.5
        or float(features.get("qr_penalty", 1.0)) >= 0.5
        or float(features.get("rectangular_stamp_penalty", 1.0)) >= 0.5
        or signature.candidate_bbox is None
        or artifact.seal.candidate_bbox is None
    ):
        return None
    sx1, sy1, sx2, sy2 = signature.candidate_bbox
    rx1, ry1, rx2, ry2 = artifact.seal.candidate_bbox
    signature_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    intersection = max(0, min(sx2, rx2) - max(sx1, rx1)) * max(
        0, min(sy2, ry2) - max(sy1, ry1)
    )
    outside_ratio = 1.0 - intersection / signature_area
    if outside_ratio < 0.20:
        return None
    confidence = (
        0.84
        + 0.06 * float(features.get("curvature_score", 0.0))
        + 0.04 * float(features.get("multi_profile_consistency_score", 0.0))
        + 0.04 * outside_ratio
        + 0.02 * artifact.seal.confidence
    )
    return min(0.96, max(0.88, confidence))


def _roi_summary(artifact: BuyerROIArtifact) -> dict:
    return {
        "page": artifact.page_number,
        "roi_name": artifact.name,
        "roi_bbox": list(artifact.bounds),
        "anchor_bbox": list(artifact.anchor_bbox) if artifact.anchor_bbox else None,
        "status": artifact.seal.status.value,
        "confidence": round(artifact.seal.confidence, 6),
        "candidate_bbox_roi": artifact.seal.candidate_bbox,
        "candidate_count": artifact.seal.candidate_count,
        "features": artifact.seal.features,
        "profile_scores": artifact.seal.profile_scores,
        "signature": {
            "status": artifact.signature.status.value,
            "confidence": round(artifact.signature.confidence, 6),
            "candidate_bbox_roi": artifact.signature.candidate_bbox,
            "candidate_count": artifact.signature.candidate_count,
            "features": artifact.signature.features,
            "profile_scores": artifact.signature.profile_scores,
        },
        "locator_strategy": artifact.locator_strategy,
        "locator_confidence": round(artifact.locator_confidence, 6),
        "selected_picture_bbox": (
            list(artifact.selected_picture_bbox)
            if artifact.selected_picture_bbox is not None
            else None
        ),
        "distance_to_buyer_anchor": artifact.distance_to_buyer_anchor,
        "distance_to_supplier_anchor": artifact.distance_to_supplier_anchor,
        "rejection_reasons": artifact.rejection_reasons,
    }


def extract_buyer_visuals(
    pages: list[PageImage],
    document: OCRDocumentResult,
    config: dict,
    seal_detector: SealDetector,
    signature_detector: SignatureDetector,
) -> tuple[ExtractedField, ExtractedField, BuyerVisualArtifacts | None]:
    ocr_by_page = {page.page_number: page for page in document.pages}
    page_by_number = {page.page_number: page for page in pages}
    candidates: list[BuyerROIArtifact] = []
    pages_without_anchor: list[int] = []

    for page in pages:
        ocr_page = ocr_by_page.get(page.page_number)
        anchors = _buyer_label_anchors(ocr_page)
        definitions: list[
            tuple[
                str,
                tuple[int, int, int, int],
                tuple[int, int, int, int] | None,
            ]
        ] = []
        layout_definition, layout_diagnostics = _layout_picture_roi(page, ocr_page)
        if layout_definition is not None:
            definitions.append(layout_definition)
        if anchors:
            # The lowest explicit label is normally the signatures label, not a body mention.
            anchor = max(anchors, key=lambda item: item.y_min)
            definitions.extend(_anchor_rois(page, anchor))
        else:
            pages_without_anchor.append(page.page_number)
        definitions.append(_fallback_lower_right_signature_roi(page))
        definitions.append(_fallback_roi(page))

        seen: set[tuple[int, int, int, int]] = set()
        for name, bounds, anchor_page in definitions:
            if bounds in seen:
                continue
            seen.add(bounds)
            roi = _crop(page, bounds)
            anchor_in_roi = _relative_anchor(anchor_page, bounds)
            context = SealDetectionContext(
                roi_name=name,
                anchor_bbox=anchor_in_roi,
                anchored=anchor_page is not None,
            )
            seal, profiles = seal_detector.detect_with_profiles(roi, context)
            signature_context = SignatureDetectionContext(
                roi_name=name,
                anchor_bbox=anchor_in_roi,
                anchored=anchor_page is not None,
            )
            signature, signature_profiles = signature_detector.detect_with_profiles(
                roi, signature_context
            )
            candidates.append(
                BuyerROIArtifact(
                    page_number=page.page_number,
                    name=name,
                    bounds=bounds,
                    anchor_bbox=anchor_page,
                    roi=roi,
                    seal=seal,
                    signature=signature,
                    preprocessing_profiles=profiles,
                    signature_profiles=signature_profiles,
                    locator_strategy=(
                        "layout_picture" if name == "layout_picture" else
                        "anchor_geometry" if anchor_page is not None else
                        "fallback_lower_right_geometry"
                        if name == "fallback_lower_right_signature_row" else
                        "fallback_lower_half"
                    ),
                    locator_confidence=(
                        float(layout_diagnostics["locator_confidence"])
                        if name == "layout_picture" else
                        0.82 if anchor_page is not None else
                        0.70 if name == "fallback_lower_right_signature_row" else
                        0.35
                    ),
                    selected_picture_bbox=(
                        tuple(layout_diagnostics["selected_picture_bbox"])
                        if name == "layout_picture"
                        and layout_diagnostics["selected_picture_bbox"] is not None
                        else None
                    ),
                    distance_to_buyer_anchor=(
                        layout_diagnostics["distance_to_buyer_anchor"]
                        if name == "layout_picture" else None
                    ),
                    distance_to_supplier_anchor=(
                        layout_diagnostics["distance_to_supplier_anchor"]
                        if name == "layout_picture" else None
                    ),
                    rejection_reasons=(
                        layout_diagnostics["rejection_reasons"]
                        if name == "layout_picture" else []
                    ),
                )
            )

    if not candidates:
        empty = extracted_field(
            "BUYER_SIGNATURE",
            None,
            PresenceStatus.UNCERTAIN,
            0.0,
            "buyer_anchor_multi_roi",
            warnings=["Нет страниц для визуальной проверки"],
        )
        return empty, empty.model_copy(update={"code": "BUYER_SEAL"}), None

    best_seal = _best_detection(candidates, "seal")
    best_signature = _best_detection(candidates, "signature")
    paired_signature_confidence = _signature_over_seal_confidence(
        best_signature, config
    )
    if paired_signature_confidence is not None:
        best_signature.signature = best_signature.signature.model_copy(
            update={
                "status": PresenceStatus.PRESENT,
                "confidence": paired_signature_confidence,
                "evidence": [
                    *best_signature.signature.evidence,
                    "Подпись подтверждена криволинейными штрихами, "
                    "выходящими за границы сильного кандидата печати",
                ],
                "warnings": [],
            }
        )
    seal_page = page_by_number[best_seal.page_number]
    seal_box = _absolute_candidate_box(best_seal, seal_page)
    signature_page = page_by_number[best_signature.page_number]
    signature_box = _absolute_candidate_box(
        best_signature, signature_page, "signature"
    )
    seal_raw = {
        "selected_page": best_seal.page_number,
        "selected_roi": best_seal.name,
        "selected_roi_bbox": list(best_seal.bounds),
        "candidate_bbox_page": (
            [
                int(seal_box.x_min),
                int(seal_box.y_min),
                int(seal_box.x_max),
                int(seal_box.y_max),
            ]
            if seal_box
            else None
        ),
        "features": best_seal.seal.features,
        "profile_scores": best_seal.seal.profile_scores,
        "roi_results": [_roi_summary(candidate) for candidate in candidates],
        "locator_strategy": best_seal.locator_strategy,
        "anchor_bbox": (
            list(best_seal.anchor_bbox) if best_seal.anchor_bbox else None
        ),
        "selected_picture_bbox": (
            list(best_seal.selected_picture_bbox)
            if best_seal.selected_picture_bbox else None
        ),
        "original_roi_bbox": list(best_seal.bounds),
        "distance_to_buyer_anchor": best_seal.distance_to_buyer_anchor,
        "distance_to_supplier_anchor": best_seal.distance_to_supplier_anchor,
        "rejection_reasons": best_seal.rejection_reasons,
        "locator_confidence": best_seal.locator_confidence,
    }
    seal_warnings = list(best_seal.seal.warnings)
    trusted_fallback_seal = _trusted_lower_right_seal(
        best_seal, config
    ) or _trusted_broad_fallback_seal(best_seal, best_signature, config)
    if not best_seal.anchor_bbox and not trusted_fallback_seal:
        seal_warnings.append(
            "Лучший кандидат получен из fallback ROI без OCR-якоря покупателя"
        )
        if best_seal.seal.status is not PresenceStatus.UNCERTAIN:
            best_seal.seal = best_seal.seal.model_copy(
                update={
                    "status": PresenceStatus.UNCERTAIN,
                    "confidence": min(
                        best_seal.seal.confidence,
                        float(config["buyer_visual_checks"]["min_seal_confidence"])
                        - 0.01,
                    ),
                }
            )
    elif trusted_fallback_seal:
        seal_warnings.append(
            "Печать подтверждена сильными независимыми признаками "
            "в строке подписей покупателя"
        )
    if pages_without_anchor:
        seal_warnings.append(
            "Якорь «Покупатель» не найден на страницах: "
            + ", ".join(str(page) for page in pages_without_anchor)
        )

    seal = extracted_field(
        "BUYER_SEAL",
        seal_raw,
        best_seal.seal.status,
        best_seal.seal.confidence,
        "opencv_seal_multi_profile_ensemble",
        best_seal.page_number,
        seal_box,
        evidence=best_seal.seal.evidence,
        warnings=seal_warnings,
    )
    signature = extracted_field(
        "BUYER_SIGNATURE",
        {
            "selected_page": best_signature.page_number,
            "selected_roi": best_signature.name,
            "selected_roi_bbox": list(best_signature.bounds),
            "candidate_bbox_page": (
                [
                    int(signature_box.x_min),
                    int(signature_box.y_min),
                    int(signature_box.x_max),
                    int(signature_box.y_max),
                ]
                if signature_box
                else None
            ),
            "features": best_signature.signature.features,
            "profile_scores": best_signature.signature.profile_scores,
            "evidence": best_signature.signature.evidence,
            "roi_results": [
                {
                    "page": candidate.page_number,
                    "roi_name": candidate.name,
                    "roi_bbox": list(candidate.bounds),
                    "anchor_bbox": (
                        list(candidate.anchor_bbox)
                        if candidate.anchor_bbox
                        else None
                    ),
                    "status": candidate.signature.status.value,
                    "confidence": round(candidate.signature.confidence, 6),
                    "candidate_bbox_roi": candidate.signature.candidate_bbox,
                    "features": candidate.signature.features,
                    "profile_scores": candidate.signature.profile_scores,
                }
                for candidate in candidates
            ],
        },
        best_signature.signature.status,
        best_signature.signature.confidence,
        "opencv_signature_multi_roi",
        best_signature.page_number,
        signature_box,
        evidence=best_signature.signature.evidence,
        warnings=best_signature.signature.warnings,
    )
    artifact = BuyerVisualArtifacts(
        page_number=best_seal.page_number,
        roi=best_seal.roi,
        seal_status=best_seal.seal.status,
        signature_status=best_signature.signature.status,
        selected_roi_name=best_seal.name,
        roi_candidates=candidates,
    )
    return signature, seal, artifact
