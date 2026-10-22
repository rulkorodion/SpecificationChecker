"""Полный локальный pipeline: document -> OCR -> extraction -> checks -> evaluation."""

from __future__ import annotations

import json
import logging
import platform
import re
import time
from collections.abc import Mapping
from functools import wraps
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.checks.rule_engine import RuleEngine
from app.document.loader import DocumentLoader
from app.document.page_selector import (
    prepare_document_for_ocr,
    source_pdf_page_count,
)
from app.document.preprocessing import preprocess
from app.document.quality import assess_page_quality
from app.document.renderer import DocumentRenderer
from app.enums import FinalDecision, PresenceStatus
from app.exceptions import OCREngineError
from app.extractors.base import uncertain_field
from app.extractors.runner import extract_specification
from app.layout.reading_order import normalized_document_text
from app.ocr.base import OCREngine
from app.ocr.cache import OCRCache, cache_fingerprint
from app.ocr.models import OCRDocumentResult
from app.performance import process_memory_info
from app.reporting.console import format_console_report
from app.reporting.debug_artifacts import (
    save_debug_artifacts,
    save_document_info,
    save_error_artifacts,
    save_ocr_artifacts,
)
from app.reporting.json_report import save_json_report
from app.schemas import (
    ExtractedSpecification,
    GroundTruthData,
    ReferenceData,
    VerificationReport,
)
from app.evaluation.ground_truth import evaluate_extraction
from app.vision.seal_detector import OpenCVSealDetector, SealDetector
from app.vision.signature_detector import OpenCVSignatureDetector, SignatureDetector

LOGGER = logging.getLogger(__name__)


def possible_reference_pairing_warning(
    extracted: ExtractedSpecification,
    reference: ReferenceData,
    ocr_document: OCRDocumentResult,
) -> str | None:
    """Flag a likely PDF/reference mix-up without changing extraction or decision."""

    field = extracted.application_number
    actual = str(field.normalized_value or "").strip()
    expected = str(reference.application_number).strip()
    if (
        not actual
        or actual == expected
        or field.confidence < 0.90
        or len(actual) < 4
    ):
        return None
    text = ocr_document.normalized_full_text
    actual_hits = len(
        re.findall(rf"(?<![\w]){re.escape(actual)}(?![\w])", text)
    )
    expected_hits = len(
        re.findall(rf"(?<![\w]){re.escape(expected)}(?![\w])", text)
    )
    if actual_hits < 2 or expected_hits:
        return None
    return (
        "Вероятно, документ сопоставлен не с тем reference: "
        f"номер {actual} уверенно найден в PDF {actual_hits} раз(а), "
        f"а ожидаемый номер {expected} в OCR-тексте не найден. "
        "Решение RETURN сохраняется; проверьте пару PDF/JSON."
    )


def _runtime_temp_scope(function):
    """Гарантированно удалить per-run temp даже при исключении/прерывании."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        config = kwargs.get("config") or args[4]
        temp_root = Path(
            config.get("runtime", {}).get("temp_dir", "runtime/temp")
        )
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="specification-check-", dir=temp_root):
            return function(*args, **kwargs)

    return wrapped


def _read_model(path: Path, model_type):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(data)
    except Exception as exc:
        from app.exceptions import InputValidationError

        raise InputValidationError(f"Некорректный JSON {path}: {exc}") from exc


def _read_reference(
    source: Path | str | Mapping[str, Any] | ReferenceData,
) -> tuple[ReferenceData, str]:
    """Validate a path or in-memory reference after independent extraction."""

    if isinstance(source, ReferenceData):
        return source, f"<in-memory:{source.specification_id}>"
    if isinstance(source, Mapping):
        try:
            reference = ReferenceData.model_validate(dict(source))
        except Exception as exc:
            from app.exceptions import InputValidationError

            raise InputValidationError(
                f"Некорректные эталонные данные в памяти: {exc}"
            ) from exc
        return reference, f"<in-memory:{reference.specification_id}>"
    path = Path(source)
    return _read_model(path, ReferenceData), str(path)


def _empty_extraction(reason: str) -> ExtractedSpecification:
    names = {
        "application_number": "APPLICATION_NUMBER",
        "application_date": "APPLICATION_DATE",
        "total_quantity": "TOTAL_QUANTITY",
        "total_with_vat": "TOTAL_WITH_VAT",
        "transportation_costs_included": "TRANSPORTATION_COSTS",
        "delivery_deadline": "DELIVERY_DEADLINE",
        "delivery_type": "DELIVERY_TYPE",
        "payment_type": "PAYMENT_TYPE",
        "deferment_days": "DEFERMENT_DAYS",
        "buyer_signature_present": "BUYER_SIGNATURE",
        "buyer_seal_present": "BUYER_SEAL",
    }
    return ExtractedSpecification(
        **{name: uncertain_field(code, reason, "not_extracted") for name, code in names.items()},
        extraction_errors=[reason],
    )


def _software_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in [
        "pydantic",
        "PyYAML",
        "numpy",
        "Pillow",
        "PyMuPDF",
        "opencv-python-headless",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "huggingface-hub",
    ]:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


@_runtime_temp_scope
def run_check(
    document_path: Path,
    reference_path: Path | str | Mapping[str, Any] | ReferenceData,
    ground_truth_path: Path | None,
    output_dir: Path,
    config: dict[str, Any],
    ocr_engine: OCREngine,
    save_debug: bool = False,
    seal_detector: SealDetector | None = None,
    signature_detector: SignatureDetector | None = None,
    *,
    debug_level: str | None = None,
    ocr_cache: OCRCache | None = None,
    reuse_ocr_cache: bool = False,
    force_ocr: bool = False,
    emit_console: bool = True,
    job_id: str | None = None,
) -> VerificationReport:
    """Выполнить проверку документа и вернуть внутренний отчёт."""

    started = datetime.now(timezone.utc)
    timer = time.perf_counter()
    stage_timings: dict[str, float] = {}
    technical_errors: list[str] = []
    warnings: list[str] = []
    debug_level = "full" if save_debug else (debug_level or "none")
    if debug_level not in {"none", "errors", "full"}:
        raise ValueError(f"Неизвестный debug_level: {debug_level}")
    cache_info: dict[str, Any] = {
        "enabled": ocr_cache is not None,
        "requested_reuse": reuse_ocr_cache,
        "forced_ocr": force_ocr,
        "hit": False,
        "created_bytes": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Стадия: загрузка")
    stage_started = time.perf_counter()
    loader = DocumentLoader(config["document"]["supported_extensions"])
    source_loaded = loader.load(document_path)
    stage_timings["loading_seconds"] = round(
        time.perf_counter() - stage_started, 6
    )
    page_limit = int(config["document"].get("max_specification_pages", 2))
    processing_document_path = document_path
    page_selection = None
    source_page_count = source_pdf_page_count(document_path)
    if source_page_count is not None and source_page_count > page_limit:
        selector_settings = dict(config["document"].get("page_selector", {}))
        selector_settings.setdefault(
            "max_source_pages", int(config["document"]["max_pages"])
        )
        LOGGER.info(
            "Стадия: Page Selector (%s исходных страниц)", source_page_count
        )
        selector_started = time.perf_counter()
        selection = prepare_document_for_ocr(
            document_path,
            reference_source=reference_path,
            output_dir=output_dir,
            page_limit=page_limit,
            settings=selector_settings,
        )
        processing_document_path = selection.document_path
        page_selection = selection.metadata
        stage_timings["page_selection_seconds"] = round(
            time.perf_counter() - selector_started, 6
        )
    loaded = (
        source_loaded
        if processing_document_path == document_path
        else loader.load(processing_document_path)
    )
    LOGGER.info("Стадия: рендеринг")
    stage_started = time.perf_counter()
    pages = DocumentRenderer(
        dpi=int(config["document"]["render_dpi"]),
        max_pages=int(config["document"]["max_pages"]),
    ).render(loaded)
    stage_timings["rendering_seconds"] = round(
        time.perf_counter() - stage_started, 6
    )
    stage_started = time.perf_counter()
    quality = [assess_page_quality(page) for page in pages]
    stage_timings["preprocessing_seconds"] = round(
        time.perf_counter() - stage_started, 6
    )
    warnings.extend(warning for item in quality for warning in item.warnings)

    LOGGER.info("Стадия: OCR")
    stage_started = time.perf_counter()
    ocr_pages = []
    extraction_artifacts = None
    try:
        unsupported_page_count = len(pages) > page_limit
        fingerprint = cache_fingerprint(loaded, config)
        ocr_document = (
            OCRDocumentResult(
                pages=[],
                normalized_full_text="",
                engine_name="skipped-unsupported-page-count",
                warnings=[
                    f"OCR пропущен: страниц {len(pages)}, поддерживаемый максимум {page_limit}"
                ],
            )
            if unsupported_page_count
            else None
        )
        if unsupported_page_count:
            cache_info["skipped_reason"] = "unsupported_page_count"
            LOGGER.info(
                "OCR пропущен: страниц %s, поддерживаемый максимум %s",
                len(pages),
                page_limit,
            )
        if (
            not unsupported_page_count
            and ocr_cache is not None
            and reuse_ocr_cache
            and not force_ocr
        ):
            cache_lookup_started = time.perf_counter()
            ocr_document, lookup = ocr_cache.get(fingerprint)
            stage_timings["ocr_cache_lookup_seconds"] = round(
                time.perf_counter() - cache_lookup_started, 6
            )
            cache_info.update(lookup)
            if ocr_document is not None:
                LOGGER.info("OCR-кэш: HIT, OCR-движок не запускается")
                ocr_pages = list(ocr_document.pages)
        if ocr_document is None:
            LOGGER.info(
                "OCR-кэш: MISS, выполняется %s", type(ocr_engine).__name__
            )
            start_document = getattr(ocr_engine, "start_document", None)
            if callable(start_document):
                if getattr(ocr_engine, "layout_aware", False):
                    start_document(processing_document_path)
                else:
                    start_document()
            for page in pages:
                original = ocr_engine.recognize_page(
                    page.image, page.page_number, "original"
                )
                selected = original
                page_quality = quality[page.page_number - 1]
                if (
                    config["ocr"]["retry_with_preprocessing"]
                    and not bool(getattr(ocr_engine, "layout_aware", False))
                    and (
                        original.mean_confidence
                        < float(config["ocr"]["min_text_confidence"])
                        or page_quality.perspective_detected
                    )
                ):
                    retry_profile = (
                        "geometry_corrected"
                        if page_quality.perspective_detected
                        else "enhanced_gray"
                    )
                    enhanced_image = preprocess(page.image, retry_profile)
                    enhanced = ocr_engine.recognize_page(
                        enhanced_image, page.page_number, retry_profile
                    )
                    # Выбор опирается только на confidence и полноту OCR.
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
                warnings.extend(selected.warnings)
            ocr_document = OCRDocumentResult(
                pages=ocr_pages,
                normalized_full_text=normalized_document_text(ocr_pages),
                engine_name=ocr_pages[0].engine_name if ocr_pages else "unknown",
                full_text="\n".join(
                    str(page.processing_metadata.get("full_text", ""))
                    for page in ocr_pages
                    if page.processing_metadata.get("full_text")
                ),
                native_markdown="\n\n".join(
                    page.native_markdown for page in ocr_pages if page.native_markdown
                ),
                native_tables=[
                    table for page in ocr_pages for table in page.native_tables
                ],
                model_name=getattr(ocr_engine, "model_id", None),
                model_version=getattr(ocr_engine, "revision", None),
                pipeline_version=(
                    ocr_pages[0].processing_metadata.get("pipeline_version")
                    if ocr_pages else None
                ),
                processing_metadata={
                    "parsing_mode": (
                        "layout_then_original_crops"
                        if bool(getattr(ocr_engine, "layout_aware", False))
                        else "page_ocr"
                    )
                },
            )
            if ocr_cache is not None:
                stored = ocr_cache.put(fingerprint, ocr_document)
                cache_info.update(stored)
                cache_info["created_bytes"] = stored["size_bytes"]
            finish_document = getattr(ocr_engine, "finish_document", None)
            if callable(finish_document):
                finish_document("completed")
        stage_timings["ocr_seconds"] = round(time.perf_counter() - stage_started, 6)
        LOGGER.info("Стадия: extraction")
        stage_started = time.perf_counter()
        if len(pages) > page_limit:
            reason = (
                f"Неподдерживаемый формат: страниц {len(pages)}, "
                f"поддерживаемый максимум {page_limit}"
            )
            extracted = _empty_extraction(reason)
            extraction_artifacts = None
            buyer_artifact = None
            warnings.append(reason)
        else:
            seal_detector = seal_detector or OpenCVSealDetector(
                float(config["buyer_visual_checks"]["min_seal_confidence"])
            )
            signature_detector = signature_detector or OpenCVSignatureDetector(
                float(config["buyer_visual_checks"]["min_signature_confidence"])
            )
            extracted, extraction_artifacts = extract_specification(
                ocr_document, pages, config, seal_detector, signature_detector
            )
            buyer_artifact = extraction_artifacts.buyer
        stage_timings["extraction_seconds"] = round(
            time.perf_counter() - stage_started, 6
        )
    except OCREngineError as exc:
        finish_document = getattr(ocr_engine, "finish_document", None)
        if callable(finish_document):
            finish_document("failed")
        stage_timings["ocr_seconds"] = round(
            time.perf_counter() - stage_started, 6
        )
        technical_errors.append(str(exc))
        extracted = _empty_extraction(str(exc))
        buyer_artifact = None
        ocr_document = OCRDocumentResult(
            pages=ocr_pages,
            normalized_full_text=normalized_document_text(ocr_pages),
            engine_name=type(ocr_engine).__name__,
            warnings=[str(exc)],
        )

    # Эталон загружается только после завершения независимого extraction.
    reference, reference_label = _read_reference(reference_path)
    LOGGER.info("Стадия: проверки")
    checks, decision, reasons = RuleEngine(config).evaluate(
        reference, extracted, technical_errors, page_count=len(pages)
    )
    pairing_warning = possible_reference_pairing_warning(
        extracted, reference, ocr_document
    )
    if pairing_warning:
        warnings.append(pairing_warning)
    extraction_evaluation = None
    if ground_truth_path is not None:
        ground_truth = _read_model(ground_truth_path, GroundTruthData)
        extraction_evaluation = evaluate_extraction(extracted, ground_truth, ground_truth_path)

    finished = datetime.now(timezone.utc)
    device_info_method = getattr(ocr_engine, "runtime_info", None)
    device_info = device_info_method() if callable(device_info_method) else {
        "requested_device": "not-reported",
        "actual_device": "not-reported",
    }
    memory_info = process_memory_info()
    diagnostics = {
        "tables": (
            extraction_artifacts.tables.model_dump(mode="json")
            if extraction_artifacts
            else {}
        ),
        "transportation": extracted.transportation_costs_included.raw_value,
        "edo_signing": (
            extraction_artifacts.edo.to_dict()
            if extraction_artifacts and extraction_artifacts.edo
            else None
        ),
        "signature": extracted.buyer_signature_present.raw_value,
        "seal": extracted.buyer_seal_present.raw_value,
        "gpu": device_info,
        "ocr_cache": cache_info,
        "page_selection": page_selection,
    }
    if extraction_artifacts:
        stage_timings.update(extraction_artifacts.extractor_timings)
    report = VerificationReport(
        specification_id=reference.specification_id,
        document_path=str(document_path),
        reference_path=reference_label,
        ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
        started_at=started,
        finished_at=finished,
        processing_time_seconds=round(time.perf_counter() - timer, 3),
        document_hash_sha256=source_loaded.sha256,
        page_count=len(pages),
        document_quality=quality,
        reference_snapshot=reference.model_dump(mode="json"),
        extracted_fields=extracted,
        checks=checks,
        final_decision=decision,
        decision_reasons=reasons,
        warnings=warnings,
        technical_errors=technical_errors,
        extraction_evaluation=extraction_evaluation,
        software_versions=_software_versions(),
        config_snapshot=config,
        extraction_diagnostics=diagnostics,
        performance={
            "stage_timings": stage_timings,
            "device": device_info,
            "memory": memory_info,
            "ocr_cache": cache_info,
        },
    )
    LOGGER.info("Стадия: отчёт")
    save_json_report(report, output_dir)
    console_text = format_console_report(
        report,
        application_number=reference.application_number,
        job_id=job_id or str(uuid4()),
    )
    (output_dir / "console_report.txt").write_text(console_text, encoding="utf-8")
    if debug_level == "full":
        save_ocr_artifacts(ocr_document, output_dir)
        save_document_info(
            {
                "path": str(document_path),
                "processing_path": str(processing_document_path),
                "mime_type": source_loaded.mime_type,
                "size_bytes": source_loaded.size_bytes,
                "sha256": source_loaded.sha256,
                "source_page_count": source_page_count,
                "page_count": len(pages),
                "page_sizes": [{"width": p.width, "height": p.height} for p in pages],
            },
            output_dir,
        )
        save_debug_artifacts(
            pages,
            ocr_document,
            output_dir,
            buyer_artifact,
            extracted,
            extraction_artifacts.tables if extraction_artifacts else None,
        )
    elif debug_level == "errors":
        save_error_artifacts(output_dir, checks, buyer_artifact)
    report_files = {output_dir / "report.json", output_dir / "console_report.txt"}
    report_bytes = sum(path.stat().st_size for path in report_files if path.is_file())
    debug_bytes = sum(
        path.stat().st_size
        for path in output_dir.rglob("*")
        if path.is_file() and path not in report_files
    )
    if emit_console:
        print(
            "DISK USAGE: "
            f"report={report_bytes / 1024:.1f} KiB, "
            f"ocr_cache_added={cache_info['created_bytes'] / 1024:.1f} KiB, "
            f"debug={debug_bytes / 1024:.1f} KiB, "
            f"persistent={(report_bytes + debug_bytes + cache_info['created_bytes']) / 1024:.1f} KiB"
        )
        print(console_text, end="")
    return report
