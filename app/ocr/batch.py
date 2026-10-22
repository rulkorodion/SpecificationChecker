"""Sequential OCR-only batch used for fast diagnostics and cache warm-up."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.document.loader import DocumentLoader
from app.document.preprocessing import preprocess
from app.document.quality import assess_page_quality
from app.document.renderer import DocumentRenderer
from app.extractors.application import extract_application
from app.extractors.edo import detect_edo_signing, edo_presence_fields
from app.extractors.delivery import extract_delivery
from app.extractors.payment import extract_payment
from app.extractors.buyer_block import extract_buyer_visuals
from app.extractors.totals import extract_totals
from app.extractors.transportation import extract_transportation
from app.layout.reading_order import normalized_document_text
from app.ocr.base import OCREngine
from app.ocr.cache import OCRCache, cache_fingerprint
from app.ocr.models import OCRDocumentResult
from app.vision.seal_detector import OpenCVSealDetector
from app.vision.signature_detector import OpenCVSignatureDetector


def _text_extraction_summary(
    document: OCRDocumentResult,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the cheap, reference-free extractors directly on cached OCR."""

    application_number, application_date = extract_application(document)
    total_quantity, total_with_vat, tables = extract_totals(document)
    transportation = extract_transportation(document, config, tables)
    delivery_deadline, delivery_type = extract_delivery(document, config)
    payment_type, deferment_days = extract_payment(document, config)
    fields = {
        field.code: field.model_dump(mode="json")
        for field in (
            application_number,
            application_date,
            total_quantity,
            total_with_vat,
            transportation,
            delivery_deadline,
            delivery_type,
            payment_type,
            deferment_days,
        )
    }
    return {
        "fields": fields,
        "tables": tables.model_dump(mode="json"),
        "edo_signing": detect_edo_signing(document).to_dict(),
    }


def _compact_visual_field(field) -> dict[str, Any]:
    raw = field.raw_value if isinstance(field.raw_value, dict) else {}
    return {
        "code": field.code,
        "normalized_value": (
            field.normalized_value.value
            if hasattr(field.normalized_value, "value")
            else field.normalized_value
        ),
        "confidence": field.confidence,
        "page_number": field.page_number,
        "bounding_box": (
            field.bounding_box.model_dump(mode="json")
            if field.bounding_box is not None
            else None
        ),
        "extraction_method": field.extraction_method,
        "selected_roi": raw.get("selected_roi"),
        "selected_roi_bbox": raw.get("selected_roi_bbox"),
        "candidate_bbox_page": raw.get("candidate_bbox_page"),
        "features": raw.get("features", {}),
        "profile_scores": raw.get("profile_scores", {}),
        "evidence": field.evidence,
        "warnings": field.warnings,
    }


def _recognize_document(
    path: Path,
    config: dict[str, Any],
    engine: OCREngine,
) -> tuple[OCRDocumentResult, dict[str, Any]]:
    started = time.perf_counter()
    loaded = DocumentLoader(config["document"]["supported_extensions"]).load(path)
    loading_seconds = time.perf_counter() - started

    started = time.perf_counter()
    pages = DocumentRenderer(
        dpi=int(config["document"]["render_dpi"]),
        max_pages=int(config["document"]["max_pages"]),
    ).render(loaded)
    rendering_seconds = time.perf_counter() - started

    started = time.perf_counter()
    quality = [assess_page_quality(page) for page in pages]
    quality_seconds = time.perf_counter() - started

    started = time.perf_counter()
    ocr_pages = []
    for page, page_quality in zip(pages, quality, strict=True):
        original = engine.recognize_page(
            page.image, page.page_number, "original"
        )
        selected = original
        if config["ocr"]["retry_with_preprocessing"] and (
            not bool(getattr(engine, "layout_aware", False))
            and (
            original.mean_confidence
            < float(config["ocr"]["min_text_confidence"])
            or page_quality.perspective_detected
            )
        ):
            profile = (
                "geometry_corrected"
                if page_quality.perspective_detected
                else "enhanced_gray"
            )
            enhanced = engine.recognize_page(
                preprocess(page.image, profile),
                page.page_number,
                profile,
            )
            original_score = original.mean_confidence * min(
                1.0, len(original.blocks) / 20
            )
            enhanced_score = enhanced.mean_confidence * min(
                1.0, len(enhanced.blocks) / 20
            )
            if page_quality.perspective_detected:
                enhanced_score *= 1.03
            selected = enhanced if enhanced_score > original_score else original
        ocr_pages.append(selected)

    document = OCRDocumentResult(
        pages=ocr_pages,
        normalized_full_text=normalized_document_text(ocr_pages),
        engine_name=ocr_pages[0].engine_name if ocr_pages else "unknown",
        warnings=[
            warning
            for item in quality
            for warning in item.warnings
        ],
        full_text="\n".join(
            str(page.processing_metadata.get("full_text", ""))
            for page in ocr_pages
            if page.processing_metadata.get("full_text")
        ),
        native_markdown="\n\n".join(
            page.native_markdown for page in ocr_pages if page.native_markdown
        ),
        native_tables=[table for page in ocr_pages for table in page.native_tables],
        model_name=getattr(engine, "model_id", None),
        model_version=getattr(engine, "revision", None),
        pipeline_version=(
            ocr_pages[0].processing_metadata.get("pipeline_version")
            if ocr_pages else None
        ),
        processing_metadata={
            "parsing_mode": (
                "layout_then_original_crops"
                if bool(getattr(engine, "layout_aware", False))
                else "page_ocr"
            )
        },
    )
    return document, {
        "document_sha256": loaded.sha256,
        "page_count": len(pages),
        "page_sizes": [
            {"page": page.page_number, "width": page.width, "height": page.height}
            for page in pages
        ],
        "page_quality": [
            {
                "page": item.page_number,
                "blur_score": item.blur_score,
                "contrast_score": item.contrast_score,
                "brightness_score": item.brightness_score,
                "skew_angle": item.skew_angle,
                "perspective_detected": item.perspective_detected,
                "warnings": item.warnings,
            }
            for item in quality
        ],
        "timings": {
            "loading_seconds": round(loading_seconds, 6),
            "rendering_seconds": round(rendering_seconds, 6),
            "preprocessing_seconds": round(quality_seconds, 6),
            "ocr_seconds": round(time.perf_counter() - started, 6),
        },
        "fingerprint": cache_fingerprint(loaded, config),
    }


def run_ocr_batch(
    document_paths: list[Path],
    output_path: Path,
    config: dict[str, Any],
    engine: OCREngine,
    cache: OCRCache,
    *,
    reuse_cache: bool = True,
    force_ocr: bool = False,
    include_visual_checks: bool = False,
) -> dict[str, Any]:
    """Warm the compact OCR cache without references, validation or debug images."""

    batch_started = time.perf_counter()
    records: list[dict[str, Any]] = []
    seal_detector = (
        OpenCVSealDetector(
            float(config["buyer_visual_checks"]["min_seal_confidence"])
        )
        if include_visual_checks
        else None
    )
    signature_detector = (
        OpenCVSignatureDetector(
            float(config["buyer_visual_checks"]["min_signature_confidence"])
        )
        if include_visual_checks
        else None
    )
    for path in document_paths:
        item_started = time.perf_counter()
        loaded = DocumentLoader(config["document"]["supported_extensions"]).load(path)
        fingerprint = cache_fingerprint(loaded, config)
        cached = None
        cache_metadata: dict[str, Any] = {}
        if reuse_cache and not force_ocr:
            cached, cache_metadata = cache.get(fingerprint)

        if cached is not None:
            document = cached
            metadata = {
                "document_sha256": loaded.sha256,
                "page_count": len(document.pages),
                "page_sizes": [
                    {
                        "page": page.page_number,
                        "width": page.width,
                        "height": page.height,
                    }
                    for page in document.pages
                ],
                "page_quality": [],
                "timings": {
                    "loading_seconds": 0.0,
                    "rendering_seconds": 0.0,
                    "preprocessing_seconds": 0.0,
                    "ocr_seconds": 0.0,
                },
                "fingerprint": fingerprint,
            }
        else:
            start_document = getattr(engine, "start_document", None)
            if callable(start_document):
                if getattr(engine, "layout_aware", False):
                    start_document(path)
                else:
                    start_document()
            document, metadata = _recognize_document(path, config, engine)
            cache_metadata = cache.put(
                metadata.pop("fingerprint"), document, source="ocr-batch"
            )
            finish_document = getattr(engine, "finish_document", None)
            if callable(finish_document):
                finish_document("completed")

        visual_summary = None
        visual_seconds = 0.0
        if include_visual_checks:
            visual_started = time.perf_counter()
            pages = DocumentRenderer(
                dpi=int(config["document"]["render_dpi"]),
                max_pages=int(config["document"]["max_pages"]),
            ).render(loaded)
            assert seal_detector is not None
            assert signature_detector is not None
            edo_evidence = detect_edo_signing(document)
            if edo_evidence.found:
                signature, seal = edo_presence_fields(edo_evidence)
            else:
                signature, seal, _ = extract_buyer_visuals(
                    pages,
                    document,
                    config,
                    seal_detector,
                    signature_detector,
                )
            visual_seconds = time.perf_counter() - visual_started
            visual_summary = {
                "buyer_signature": _compact_visual_field(signature),
                "buyer_seal": _compact_visual_field(seal),
                "elapsed_seconds": round(visual_seconds, 6),
            }
            del pages

        records.append(
            {
                "document": str(path),
                **metadata,
                "engine": document.engine_name,
                "blocks": sum(len(page.blocks) for page in document.pages),
                "mean_confidence": round(
                    sum(
                        block.effective_confidence
                        for page in document.pages
                        for block in page.blocks
                    )
                    / max(
                        1,
                        sum(len(page.blocks) for page in document.pages),
                    ),
                    6,
                ),
                "text_extraction": _text_extraction_summary(document, config),
                "visual_extraction": visual_summary,
                "cache": cache_metadata,
                "elapsed_seconds": round(time.perf_counter() - item_started, 6),
            }
        )
        del document
        gc.collect()

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": records,
        "document_count": len(records),
        "elapsed_seconds": round(time.perf_counter() - batch_started, 6),
        "engine_runtime": (
            engine.runtime_info()
            if callable(getattr(engine, "runtime_info", None))
            else {}
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
