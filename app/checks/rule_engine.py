"""Fail-closed Rule Engine, не зависящий от OCR, CV и файловой системы."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from app.checks.date_rules import BusinessCalendar, WeekdayBusinessCalendar
from app.enums import (
    CheckStatus,
    FinalDecision,
    PaymentType,
    PresenceStatus,
)
from app.schemas import CheckResult, ExtractedField, ExtractedSpecification, ReferenceData

FIELD_DEFINITIONS: list[tuple[str, str, str]] = [
    ("APPLICATION_NUMBER", "Номер приложения", "application_number"),
    ("APPLICATION_DATE", "Дата приложения", "application_date"),
    ("TOTAL_QUANTITY", "Итоговое количество", "total_quantity"),
    ("TOTAL_WITH_VAT", "Общая сумма с НДС", "total_with_vat"),
    ("TRANSPORTATION_COSTS", "Транспортные расходы", "transportation_costs_included"),
    ("DELIVERY_DEADLINE", "Срок поставки", "delivery_deadline"),
    ("DELIVERY_TYPE", "Вид поставки", "delivery_type"),
    ("PAYMENT_TYPE", "Условия оплаты", "payment_type"),
    ("DEFERMENT_DAYS", "Количество дней отсрочки", "deferment_days"),
    ("BUYER_SIGNATURE", "Подпись покупателя", "buyer_signature_present"),
    ("BUYER_SEAL", "Печать покупателя", "buyer_seal_present"),
]


class RuleEngine:
    """Сравнить независимое извлечение с reference и выбрать решение."""

    def __init__(
        self,
        config: dict[str, Any],
        calendar: BusinessCalendar | None = None,
        today_provider: Callable[[], date] | None = None,
    ) -> None:
        self.config = config
        self.calendar = calendar or WeekdayBusinessCalendar()
        timezone_name = str(
            self.config.get("date_rule", {}).get("timezone", "Europe/Moscow")
        )
        self.today_provider = today_provider or (
            lambda: datetime.now(ZoneInfo(timezone_name)).date()
        )

    def evaluate(
        self,
        reference: ReferenceData,
        extracted: ExtractedSpecification,
        technical_errors: list[str] | None = None,
        page_count: int | None = None,
    ) -> tuple[list[CheckResult], FinalDecision, list[str]]:
        page_limit = int(
            self.config.get("document", {}).get("max_specification_pages", 2)
        )
        if page_count is not None and page_count > page_limit:
            message = (
                f"Документ содержит страниц: {page_count}; поддерживаемый "
                f"максимум: {page_limit}"
            )
            check = CheckResult(
                code="DOCUMENT_PAGE_COUNT",
                title="Количество страниц",
                status=CheckStatus.UNCERTAIN,
                expected=f"не более {page_limit}",
                actual=page_count,
                confidence=1.0,
                message=message,
                is_required=True,
            )
            return [check], FinalDecision.MANUAL_REVIEW, [
                f"Количество страниц: {message}"
            ]
        checks: list[CheckResult] = []
        for code, title, attribute in FIELD_DEFINITIONS:
            field = getattr(extracted, attribute)
            if code == "APPLICATION_DATE":
                checks.append(self._date_staleness(field))
                continue
            checks.append(
                self._field_check(
                    code,
                    title,
                    field,
                    reference,
                    True,
                    extracted.payment_type,
                )
            )

        decision, reasons = self.choose_decision(checks, technical_errors or [])
        return checks, decision, reasons

    def _expected(self, code: str, reference: ReferenceData) -> Any:
        mapping = {
            "APPLICATION_NUMBER": reference.application_number,
            "TOTAL_QUANTITY": reference.total_quantity,
            "TOTAL_WITH_VAT": reference.total_with_vat,
            "TRANSPORTATION_COSTS": reference.transportation_costs_included,
            "DELIVERY_DEADLINE": reference.delivery_deadline,
            "DELIVERY_TYPE": reference.delivery_type,
            "PAYMENT_TYPE": reference.payment_type,
            "DEFERMENT_DAYS": reference.deferment_days,
            "BUYER_SIGNATURE": True,
            "BUYER_SEAL": True,
        }
        return mapping[code]

    def _field_check(
        self,
        code: str,
        title: str,
        field: ExtractedField,
        reference: ReferenceData,
        required: bool,
        extracted_payment: ExtractedField,
    ) -> CheckResult:
        expected = self._expected(code, reference)
        if not required:
            return self._result(code, title, CheckStatus.SKIPPED, expected, field, "Проверка отключена")
        field_threshold = float(self.config["extraction"]["field_confidence_threshold"])
        if (
            code == "DEFERMENT_DAYS"
            and extracted_payment.normalized_value is PaymentType.PREPAYMENT
            and extracted_payment.confidence >= field_threshold
        ):
            return self._result(
                code, title, CheckStatus.SKIPPED, None, field, "Неприменимо при предоплате"
            )

        threshold = float(
            self.config["extraction"][
                "visual_confidence_threshold"
                if code in {"BUYER_SIGNATURE", "BUYER_SEAL"}
                else "field_confidence_threshold"
            ]
        )
        actual = field.normalized_value
        if actual is None or field.confidence < threshold:
            return self._result(
                code,
                title,
                CheckStatus.UNCERTAIN,
                expected,
                field,
                "Поле не извлечено или уверенность ниже порога",
            )

        if code in {"BUYER_SIGNATURE", "BUYER_SEAL"}:
            try:
                presence = actual if isinstance(actual, PresenceStatus) else PresenceStatus(actual)
            except ValueError:
                presence = PresenceStatus.UNCERTAIN
            if presence is PresenceStatus.UNCERTAIN:
                status = CheckStatus.UNCERTAIN
                message = "Визуальный детектор не смог сделать уверенный вывод"
            elif presence is PresenceStatus.PRESENT:
                status = CheckStatus.PASS
                message = (
                    "Подтверждено подписанием через оператора ЭДО"
                    if field.extraction_method == "edo_operator_signing_evidence"
                    else "Признак присутствия уверенно найден"
                )
            else:
                status = CheckStatus.MISMATCH
                message = "Обязательный визуальный признак отсутствует"
            return self._result(code, title, status, expected, field, message)

        status = CheckStatus.PASS if actual == expected else CheckStatus.MISMATCH
        message = "Значение совпадает" if status is CheckStatus.PASS else "Значение отличается"
        return self._result(code, title, status, expected, field, message)

    def _date_staleness(self, application_date: ExtractedField) -> CheckResult:
        limit = int(
            self.config["date_rule"].get(
                "max_age_days",
                self.config["date_rule"].get("max_business_days", 2),
            )
        )
        threshold = float(self.config["extraction"]["field_confidence_threshold"])
        if (
            application_date.normalized_value is None
            or application_date.confidence < threshold
        ):
            return self._result(
                "DATE_STALENESS",
                "Дата приложения",
                CheckStatus.UNCERTAIN,
                f"не более {limit} календарных дней от текущей даты Москвы",
                application_date,
                "Дата приложения не извлечена с достаточной уверенностью",
            )
        check_date = self.today_provider()
        days = abs((check_date - application_date.normalized_value).days)
        status = CheckStatus.STOP if days > limit else CheckStatus.PASS
        return CheckResult(
            code="DATE_STALENESS",
            title="Дата приложения",
            status=status,
            expected=f"не более {limit} календарных дней от {check_date.isoformat()}",
            actual=days,
            confidence=application_date.confidence,
            message=(
                f"Между датой и проверкой {days} дней"
                if status is CheckStatus.PASS
                else f"Превышен лимит (между датой и проверкой {days} дней)"
            ),
            page_number=application_date.page_number,
            bounding_box=application_date.bounding_box,
            evidence=application_date.evidence,
            is_required=True,
        )

    @staticmethod
    def _result(
        code: str,
        title: str,
        status: CheckStatus,
        expected: Any,
        field: ExtractedField,
        message: str,
    ) -> CheckResult:
        return CheckResult(
            code=code,
            title=title,
            status=status,
            expected=expected,
            actual=field.normalized_value,
            confidence=field.confidence,
            message=message,
            page_number=field.page_number,
            bounding_box=field.bounding_box,
            evidence=field.evidence,
            is_required=status is not CheckStatus.SKIPPED,
        )

    def choose_decision(
        self, checks: list[CheckResult], technical_errors: list[str]
    ) -> tuple[FinalDecision, list[str]]:
        reasons = [
            f"{item.title}: {item.message}"
            for item in checks
            if item.status in {CheckStatus.UNCERTAIN, CheckStatus.MISMATCH, CheckStatus.STOP}
        ]
        if technical_errors:
            return (
                FinalDecision.MANUAL_REVIEW,
                [*technical_errors, *reasons] or ["Техническая ошибка"],
            )
        if any(item.status is CheckStatus.STOP for item in checks):
            return FinalDecision.RETURN, reasons
        if any(item.is_required and item.status is CheckStatus.MISMATCH for item in checks):
            return FinalDecision.RETURN, reasons
        if any(item.is_required and item.status is CheckStatus.UNCERTAIN for item in checks):
            return FinalDecision.MANUAL_REVIEW, reasons
        if all(item.status in {CheckStatus.PASS, CheckStatus.SKIPPED} for item in checks):
            return FinalDecision.APPROVE, ["Все обязательные проверки пройдены"]
        return FinalDecision.MANUAL_REVIEW, reasons or ["Неполный набор результатов"]
