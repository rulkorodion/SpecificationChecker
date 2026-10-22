"""Независимая оценка extraction; не участвует в распознавании и Rule Engine."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from app.enums import ExtractionEvaluationStatus, PaymentType, PresenceStatus
from app.schemas import (
    ExtractionEvaluation,
    ExtractionFieldEvaluation,
    ExtractedSpecification,
    GroundTruthData,
)

FIELDS: list[tuple[str, str, str]] = [
    ("application_number", "application_number", "categorical"),
    ("application_date", "application_date", "categorical"),
    ("total_quantity", "total_quantity", "numeric"),
    ("total_with_vat", "total_with_vat", "numeric"),
    ("transportation_costs_included", "transportation_costs_included", "categorical"),
    ("delivery_deadline", "delivery_deadline", "categorical"),
    ("delivery_type", "delivery_type", "categorical"),
    ("payment_type", "payment_type", "categorical"),
    ("deferment_days", "deferment_days", "numeric"),
    ("buyer_signature_present", "buyer_signature_present", "visual"),
    ("buyer_seal_present", "buyer_seal_present", "visual"),
]


def _visual_to_bool(value: Any) -> bool | None:
    if value in (PresenceStatus.PRESENT, PresenceStatus.PRESENT.value):
        return True
    if value in (PresenceStatus.ABSENT, PresenceStatus.ABSENT.value):
        return False
    return None


def evaluate_extraction(
    extracted: ExtractedSpecification,
    ground_truth: GroundTruthData,
    ground_truth_path: Path,
) -> ExtractionEvaluation:
    """Сравнить уже завершённое извлечение с ground truth."""

    results: list[ExtractionFieldEvaluation] = []
    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for code, gt_attribute, category in FIELDS:
        field = getattr(extracted, code)
        expected = getattr(ground_truth, gt_attribute)
        actual = field.normalized_value
        if code == "deferment_days" and ground_truth.payment_type is PaymentType.PREPAYMENT:
            status = ExtractionEvaluationStatus.NOT_APPLICABLE
            message = "Неприменимо при предоплате"
        else:
            if category == "visual":
                actual = _visual_to_bool(actual)
            if actual is None:
                status = ExtractionEvaluationStatus.NOT_EXTRACTED
                message = "Поле не извлечено"
            elif actual == expected:
                status = ExtractionEvaluationStatus.CORRECT
                message = "Извлечение корректно"
            else:
                status = ExtractionEvaluationStatus.INCORRECT
                message = "Извлечённое значение отличается от ground truth"
            category_counts[category][1] += 1
            if status is ExtractionEvaluationStatus.CORRECT:
                category_counts[category][0] += 1
        results.append(
            ExtractionFieldEvaluation(
                code=code,
                status=status,
                extracted=actual,
                ground_truth=expected,
                category=category,
                message=message,
            )
        )

    correct = sum(x.status is ExtractionEvaluationStatus.CORRECT for x in results)
    incorrect = sum(x.status is ExtractionEvaluationStatus.INCORRECT for x in results)
    missing = sum(x.status is ExtractionEvaluationStatus.NOT_EXTRACTED for x in results)
    not_applicable = sum(
        x.status is ExtractionEvaluationStatus.NOT_APPLICABLE for x in results
    )
    applicable = len(results) - not_applicable

    def accuracy(category: str) -> float:
        good, total = category_counts[category]
        return good / total if total else 0.0

    return ExtractionEvaluation(
        ground_truth_path=str(ground_truth_path),
        fields=results,
        correct_fields=correct,
        incorrect_fields=incorrect,
        not_extracted_fields=missing,
        not_applicable_fields=not_applicable,
        applicable_fields=applicable,
        field_accuracy=correct / applicable if applicable else 0.0,
        numeric_field_accuracy=accuracy("numeric"),
        categorical_field_accuracy=accuracy("categorical"),
        visual_field_accuracy=accuracy("visual"),
    )
