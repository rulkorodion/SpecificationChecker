"""Строгие Pydantic-модели входов, извлечения и отчётов."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.enums import (
    CheckStatus,
    DeliveryType,
    ExtractionEvaluationStatus,
    FinalDecision,
    PaymentType,
    PresenceStatus,
    TableType,
)


class StrictModel(BaseModel):
    """Общая модель, запрещающая неизвестные поля."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReferenceData(StrictModel):
    specification_id: str = Field(min_length=1)
    application_number: str = Field(min_length=1)
    total_quantity: Decimal
    total_with_vat: Decimal
    transportation_costs_included: bool
    delivery_deadline: date
    delivery_type: DeliveryType
    payment_type: PaymentType
    deferment_days: int | None

    @model_validator(mode="before")
    @classmethod
    def ignore_retired_reference_dates(cls, value: Any) -> Any:
        """Keep legacy references readable; runtime checks are not configurable."""

        if isinstance(value, dict):
            value = dict(value)
            value.pop("check_date", None)
            value.pop("application_date", None)
            value.pop("required_checks", None)
        return value

    @field_validator("payment_type", mode="before")
    @classmethod
    def normalize_payment_alias(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = " ".join(value.lower().replace("ё", "е").split())
            if "оплата по факту готовности" in normalized:
                return PaymentType.PREPAYMENT.value
        return value

    @field_validator("application_number", mode="before")
    @classmethod
    def keep_number_as_string(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("application_number должен быть строкой")
        return value.strip()

    @field_validator("total_quantity", "total_with_vat", mode="before")
    @classmethod
    def decimal_from_json_string(cls, value: Any) -> Decimal:
        if isinstance(value, float):
            raise ValueError("Decimal-поля нельзя передавать как float")
        text = str(value).strip()
        text = re.sub(r"[\s\u00a0\u202f]", "", text)
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError("Значение должно быть корректным десятичным числом") from exc
        if not parsed.is_finite():
            raise ValueError("Значение должно быть конечным десятичным числом")
        return parsed

    @model_validator(mode="after")
    def validate_payment(self) -> "ReferenceData":
        if self.payment_type is PaymentType.DEFERMENT:
            if self.deferment_days is None or self.deferment_days <= 0:
                raise ValueError("при DEFERMENT deferment_days должен быть положительным")
        elif self.deferment_days is not None:
            raise ValueError("при PREPAYMENT deferment_days должен быть null")
        return self


class GroundTruthData(StrictModel):
    application_number: str = Field(min_length=1)
    application_date: date
    total_quantity: Decimal
    total_with_vat: Decimal
    transportation_costs_included: bool
    delivery_deadline: date
    delivery_type: DeliveryType
    payment_type: PaymentType
    deferment_days: int | None
    buyer_signature_present: bool
    buyer_seal_present: bool

    @model_validator(mode="before")
    @classmethod
    def ignore_retired_clause_truth(cls, value: Any) -> Any:
        """Accept old datasets without exposing retired fields in evaluation."""

        if isinstance(value, dict):
            value = dict(value)
            value.pop("mandatory_clause_1_present", None)
            value.pop("mandatory_clause_2_present", None)
        return value

    @field_validator("application_number", mode="before")
    @classmethod
    def validate_number(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("application_number должен быть строкой")
        return value.strip()

    @field_validator("total_quantity", "total_with_vat", mode="before")
    @classmethod
    def validate_decimal(cls, value: Any) -> Decimal:
        if isinstance(value, float):
            raise ValueError("Decimal-поля нельзя передавать как float")
        return Decimal(str(value))

    @model_validator(mode="after")
    def validate_payment(self) -> "GroundTruthData":
        if self.payment_type is PaymentType.DEFERMENT:
            if self.deferment_days is None or self.deferment_days <= 0:
                raise ValueError("при DEFERMENT нужны положительные deferment_days")
        elif self.deferment_days is not None:
            raise ValueError("при PREPAYMENT deferment_days должен быть null")
        return self


class BoundingBox(StrictModel):
    """Прямоугольник [x_min, y_min, x_max, y_max] в пикселях страницы."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("некорректный порядок координат")
        return self

    def relative(self) -> list[float]:
        return [
            self.x_min / self.page_width,
            self.y_min / self.page_height,
            self.x_max / self.page_width,
            self.y_max / self.page_height,
        ]


class ExtractedField(StrictModel):
    code: str
    raw_value: Any = None
    normalized_value: Any = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    page_number: int | None = Field(default=None, ge=1)
    bounding_box: BoundingBox | None = None
    extraction_method: str
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_extracted(self) -> bool:
        return self.normalized_value is not None


class ExtractedSpecification(StrictModel):
    application_number: ExtractedField
    application_date: ExtractedField
    total_quantity: ExtractedField
    total_with_vat: ExtractedField
    transportation_costs_included: ExtractedField
    delivery_deadline: ExtractedField
    delivery_type: ExtractedField
    payment_type: ExtractedField
    deferment_days: ExtractedField
    buyer_signature_present: ExtractedField
    buyer_seal_present: ExtractedField
    warnings: list[str] = Field(default_factory=list)
    extraction_errors: list[str] = Field(default_factory=list)


class TableResult(StrictModel):
    """Одна независимо найденная табличная часть документа."""

    page_number: int = Field(ge=1)
    bounding_box: BoundingBox
    table_type: TableType
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    totals_row: list[str] = Field(default_factory=list)
    total_quantity: Decimal | None = None
    total_with_vat: Decimal | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_row_bounding_box: BoundingBox | None = None
    service_keywords_found: list[str] = Field(default_factory=list)


class TableAggregation(StrictModel):
    """Объяснимая агрегация всех найденных табличных частей."""

    tables: list[TableResult] = Field(default_factory=list)
    included_table_indexes: list[int] = Field(default_factory=list)
    excluded_candidates: list[dict[str, Any]] = Field(default_factory=list)
    quantity_table_indexes: list[int] = Field(default_factory=list)
    total_table_indexes: list[int] = Field(default_factory=list)
    aggregated_quantity: Decimal | None = None
    aggregated_total_with_vat: Decimal | None = None
    document_grand_total: Decimal | None = None
    extraction_method: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict: bool = False
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CheckResult(StrictModel):
    code: str
    title: str
    status: CheckStatus
    expected: Any = None
    actual: Any = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    message: str
    page_number: int | None = None
    bounding_box: BoundingBox | None = None
    evidence: list[str] = Field(default_factory=list)
    is_required: bool = True


class PageQualityReport(StrictModel):
    page_number: int
    width: int
    height: int
    blur_score: float
    contrast_score: float
    skew_angle: float
    perspective_detected: bool
    brightness_score: float
    empty_area_ratio: float
    warnings: list[str] = Field(default_factory=list)
    recommended_preprocessing_profile: str = "original"


class ExtractionFieldEvaluation(StrictModel):
    code: str
    status: ExtractionEvaluationStatus
    extracted: Any = None
    ground_truth: Any = None
    category: str
    message: str


class ExtractionEvaluation(StrictModel):
    ground_truth_path: str
    fields: list[ExtractionFieldEvaluation]
    correct_fields: int
    incorrect_fields: int
    not_extracted_fields: int
    not_applicable_fields: int
    applicable_fields: int
    field_accuracy: float
    numeric_field_accuracy: float
    categorical_field_accuracy: float
    visual_field_accuracy: float


class VerificationReport(StrictModel):
    specification_id: str
    document_path: str
    reference_path: str
    ground_truth_path: str | None = None
    started_at: datetime
    finished_at: datetime
    processing_time_seconds: float
    document_hash_sha256: str
    page_count: int
    document_quality: list[PageQualityReport]
    reference_snapshot: dict[str, Any]
    extracted_fields: ExtractedSpecification
    checks: list[CheckResult]
    final_decision: FinalDecision
    decision_reasons: list[str]
    warnings: list[str] = Field(default_factory=list)
    technical_errors: list[str] = Field(default_factory=list)
    extraction_evaluation: ExtractionEvaluation | None = None
    software_versions: dict[str, str] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    extraction_diagnostics: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_path", "reference_path", "ground_truth_path", mode="before")
    @classmethod
    def paths_to_strings(cls, value: Any) -> Any:
        return str(value) if isinstance(value, Path) else value
