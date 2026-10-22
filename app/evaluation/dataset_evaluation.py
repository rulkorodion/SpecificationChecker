"""Пакетный прогон с изоляцией ошибок документов."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from app.enums import ExtractionEvaluationStatus
from app.pipeline import run_check
from app.ocr.cache import OCRCache
from app.reporting.dataset_report import save_dataset_summary

LOGGER = logging.getLogger(__name__)


def _key(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    return match.group(1) if match else path.stem.casefold()


def evaluate_dataset(
    documents_dir: Path,
    reference_dir: Path,
    ground_truth_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    engine_factory: Callable[[], Any],
    debug_level: str,
    ocr_cache: OCRCache | None = None,
    reuse_ocr_cache: bool = False,
    force_ocr: bool = False,
    include_ids: set[str] | None = None,
) -> dict[str, Any]:
    documents = sorted(
        [
            p
            for p in documents_dir.iterdir()
            if p.suffix.lower() in config["document"]["supported_extensions"]
            and (include_ids is None or _key(p) in include_ids)
        ]
    )
    references = {_key(p): p for p in reference_dir.glob("*.json")}
    truths = {_key(p): p for p in ground_truth_dir.glob("*.json")}
    results = []
    decisions: Counter[str] = Counter()
    field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    incorrect_extractions: list[dict[str, Any]] = []
    not_extracted_fields: list[dict[str, Any]] = []
    uncertain_count = 0
    technical = 0
    processed = 0
    for document in documents:
        key = _key(document)
        reference = references.get(key)
        truth = truths.get(key)
        if reference is None or truth is None:
            results.append(
                {
                    "document": document.name,
                    "error": "Не найдена связанная reference или ground truth",
                }
            )
            technical += 1
            continue
        try:
            engine = engine_factory()
            start_document = getattr(engine, "start_document", None)
            if callable(start_document):
                start_document()
            report = run_check(
                document,
                reference,
                truth,
                output_dir / f"specification_{key}",
                config,
                engine,
                debug_level == "full",
                debug_level=debug_level,
                ocr_cache=ocr_cache,
                reuse_ocr_cache=reuse_ocr_cache,
                force_ocr=force_ocr,
            )
            processed += 1
            decisions[report.final_decision.value] += 1
            if report.technical_errors:
                technical += 1
            evaluation = report.extraction_evaluation
            uncertain_count += sum(item.status.value == "UNCERTAIN" for item in report.checks)
            if evaluation:
                for item in evaluation.fields:
                    if item.status is ExtractionEvaluationStatus.NOT_APPLICABLE:
                        continue
                    field_counts[item.code][1] += 1
                    category_counts[item.category][1] += 1
                    if item.status is ExtractionEvaluationStatus.CORRECT:
                        field_counts[item.code][0] += 1
                        category_counts[item.category][0] += 1
                    elif item.status is ExtractionEvaluationStatus.INCORRECT:
                        incorrect_extractions.append(
                            {
                                "document": document.name,
                                "field": item.code,
                                "extracted": item.extracted,
                                "ground_truth": item.ground_truth,
                            }
                        )
                    elif item.status is ExtractionEvaluationStatus.NOT_EXTRACTED:
                        not_extracted_fields.append(
                            {"document": document.name, "field": item.code}
                        )
            results.append(
                {
                    "document": document.name,
                    "decision": report.final_decision.value,
                    "field_accuracy": evaluation.field_accuracy if evaluation else None,
                    "technical_errors": report.technical_errors,
                    "processing_time_seconds": report.processing_time_seconds,
                    "performance": report.performance,
                    "fields": {
                        name: value.model_dump(mode="json")
                        for name, value in report.extracted_fields
                        if name not in {"warnings", "extraction_errors"}
                    },
                    "checks": [
                        item.model_dump(mode="json") for item in report.checks
                    ],
                }
            )
        except Exception as exc:
            LOGGER.error("Ошибка документа %s: %s", document.name, exc)
            technical += 1
            results.append({"document": document.name, "error": str(exc)})
    correct = sum(item[0] for item in field_counts.values())
    applicable = sum(item[1] for item in field_counts.values())
    accuracies = {
        code: good / total if total else 0.0
        for code, (good, total) in field_counts.items()
    }
    summary = {
        "documents_found": len(documents),
        "processed": processed,
        "skipped": len(documents) - processed,
        "technical_errors": technical,
        "decision_counts": dict(decisions),
        "field_accuracy": accuracies,
        "overall_field_accuracy": correct / applicable if applicable else 0.0,
        "numeric_field_accuracy": (
            category_counts["numeric"][0] / category_counts["numeric"][1]
            if category_counts["numeric"][1]
            else 0.0
        ),
        "categorical_field_accuracy": (
            category_counts["categorical"][0] / category_counts["categorical"][1]
            if category_counts["categorical"][1]
            else 0.0
        ),
        "visual_field_accuracy": (
            category_counts["visual"][0] / category_counts["visual"][1]
            if category_counts["visual"][1]
            else 0.0
        ),
        "uncertain_checks": uncertain_count,
        "worst_fields": sorted(accuracies, key=accuracies.get)[:5],
        "incorrect_extractions": incorrect_extractions,
        "not_extracted_fields": not_extracted_fields,
        "results": results,
    }
    save_dataset_summary(summary, output_dir)
    return summary
