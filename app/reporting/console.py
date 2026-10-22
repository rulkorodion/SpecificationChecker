"""Понятный русский отчёт, читаемый без ANSI-цветов."""

from __future__ import annotations

import re

from app.enums import CheckStatus, FinalDecision
from app.schemas import CheckResult, VerificationReport

RULE = "=" * 60
DAY_FORM_PATTERN = re.compile(r"\b(\d+)\s+д(?:ень|ня|ней)\b", re.IGNORECASE)
USER_DECISION_LABELS = {
    FinalDecision.APPROVE: "СОГЛАСОВАНО (APPROVE)",
    FinalDecision.RETURN: "ВОЗВРАТ (RETURN)",
    FinalDecision.MANUAL_REVIEW: "ТРЕБУЕТ РУЧНОЙ ПРОВЕРКИ (MANUAL_REVIEW)",
}
USER_REASON_MESSAGES = {
    "Значение отличается": "значение отличается",
    "Поле не извлечено или уверенность ниже порога": (
        "не удалось определить с достаточной уверенностью"
    ),
    "Визуальный детектор не смог сделать уверенный вывод": (
        "не удалось определить с достаточной уверенностью"
    ),
    "Дата приложения не извлечена с достаточной уверенностью": (
        "не удалось определить с достаточной уверенностью"
    ),
    "Обязательный визуальный признак отсутствует": (
        "обязательный признак отсутствует"
    ),
}


def format_console_report(
    report: VerificationReport,
    *,
    application_number: str | None = None,
    job_id: str | None = None,
) -> str:
    if (application_number is None) != (job_id is None):
        raise ValueError(
            "application_number и job_id должны быть переданы вместе"
        )
    user_format = application_number is not None and job_id is not None
    lines = (
        [
            f"ПРОВЕРКА СПЕЦИФИКАЦИИ №{application_number}",
            f"ID проверки: {job_id}",
            "",
        ]
        if user_format
        else [RULE, "ПРОВЕРКА СПЕЦИФИКАЦИИ", RULE]
    )
    lines.extend(
        f"{index}. {check.title}: {_compact_result(check)}"
        for index, check in enumerate(report.checks, start=1)
    )
    lines.append("")
    if user_format:
        decision_label = USER_DECISION_LABELS[report.final_decision]
        lines.extend([f"ИТОГОВОЕ РЕШЕНИЕ: {decision_label}", "", "Причины:"])
    else:
        lines.extend(
            [
                RULE,
                f"ИТОГОВОЕ РЕШЕНИЕ: {report.final_decision.value}",
                RULE,
                "Причины:",
            ]
        )
    lines.extend(
        f"{index}. {_normalize_day_forms(_format_user_reason(reason) if user_format else reason)}"
        for index, reason in enumerate(report.decision_reasons, start=1)
    )
    return "\n".join(lines) + "\n"


def _format_user_reason(reason: str) -> str:
    title, separator, message = reason.partition(":")
    if not separator:
        return reason
    normalized_message = message.strip()
    readable_message = USER_REASON_MESSAGES.get(normalized_message)
    if readable_message is None:
        readable_message = normalized_message[:1].lower() + normalized_message[1:]
    return f"{title.strip()}: {readable_message}"


def _compact_result(check: CheckResult) -> str:
    if check.code in {"DATE_STALENESS", "DOCUMENT_PAGE_COUNT"}:
        return _readable_message(check.message)
    if check.status is CheckStatus.PASS:
        if check.code in {"BUYER_SIGNATURE", "BUYER_SEAL"}:
            return "присутствует"
        if check.code == "TRANSPORTATION_COSTS":
            state = "включены" if check.actual is True else "не включены"
            return f"{state} — совпадает"
        return "совпадает"
    if check.status is CheckStatus.SKIPPED:
        return _readable_message(check.message)
    if check.status is CheckStatus.MISMATCH:
        return (
            "не совпадает "
            f"(в документе: {_display(check.actual)}; в эталоне: {_display(check.expected)})"
        )
    if check.status is CheckStatus.STOP:
        return _readable_message(check.message)
    return "не определено — требуется ручная проверка"


def _readable_message(message: str) -> str:
    lowered = message[:1].lower() + message[1:]
    return _normalize_day_forms(lowered)


def _normalize_day_forms(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        days = int(match.group(1))
        remainder = days % 100
        if 11 <= remainder <= 14:
            word = "дней"
        elif days % 10 == 1:
            word = "день"
        elif days % 10 in {2, 3, 4}:
            word = "дня"
        else:
            word = "дней"
        return f"{days} {word}"

    return DAY_FORM_PATTERN.sub(replace, text)


def _display(value) -> str:
    if value is None:
        return "не извлечено / неприменимо"
    if isinstance(value, bool):
        return "включены" if value else "не включены"
    return getattr(value, "value", str(value))
