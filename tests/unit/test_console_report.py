from __future__ import annotations

from types import SimpleNamespace

from app.enums import CheckStatus, FinalDecision
from app.reporting.console import format_console_report
from app.schemas import CheckResult


def test_console_report_is_compact_and_numbered():
    report = SimpleNamespace(
        checks=[
            CheckResult(
                code="APPLICATION_NUMBER",
                title="Номер приложения",
                status=CheckStatus.PASS,
                expected="123",
                actual="123",
                confidence=0.99,
                message="Значение совпадает",
            ),
            CheckResult(
                code="DATE_STALENESS",
                title="Дата приложения",
                status=CheckStatus.PASS,
                expected="не более 2 календарных дней",
                actual=0,
                confidence=0.99,
                message="Между датой и проверкой 0 дней",
            ),
        ],
        final_decision=FinalDecision.APPROVE,
        decision_reasons=["Все обязательные проверки пройдены"],
    )

    text = format_console_report(report)

    assert "1. Номер приложения: совпадает" in text
    assert "2. Дата приложения: между датой и проверкой 0 дней" in text
    assert "ИТОГОВОЕ РЕШЕНИЕ: APPROVE" in text
    assert "В документе:" not in text
    assert "В эталоне:" not in text
    assert "Уверенность:" not in text
    assert "ОЦЕНКА РАСПОЗНАВАНИЯ" not in text


def test_console_report_explains_transportation_value_and_stale_date():
    report = SimpleNamespace(
        checks=[
            CheckResult(
                code="DATE_STALENESS",
                title="Дата приложения",
                status=CheckStatus.STOP,
                expected="не более 2 календарных дней",
                actual=6,
                confidence=0.99,
                message="Превышен лимит (между датой и проверкой 6 дней)",
            ),
            CheckResult(
                code="TRANSPORTATION_COSTS",
                title="Транспортные расходы",
                status=CheckStatus.PASS,
                expected=False,
                actual=False,
                confidence=0.99,
                message="Значение совпадает",
            ),
        ],
        final_decision=FinalDecision.RETURN,
        decision_reasons=[
            "Дата приложения: Превышен лимит (между датой и проверкой 6 дней)"
        ],
    )

    text = format_console_report(report)

    assert (
        "1. Дата приложения: превышен лимит "
        "(между датой и проверкой 6 дней)" in text
    )
    assert "2. Транспортные расходы: не включены — совпадает" in text


def test_transportation_mismatch_uses_readable_boolean_values():
    report = SimpleNamespace(
        checks=[
            CheckResult(
                code="TRANSPORTATION_COSTS",
                title="Транспортные расходы",
                status=CheckStatus.MISMATCH,
                expected=True,
                actual=False,
                confidence=0.99,
                message="Значение отличается",
            )
        ],
        final_decision=FinalDecision.RETURN,
        decision_reasons=["Транспортные расходы: Значение отличается"],
    )

    text = format_console_report(report)

    assert "в документе: не включены; в эталоне: включены" in text


def test_console_report_uses_correct_russian_day_forms():
    report = SimpleNamespace(
        checks=[
            CheckResult(
                code="DATE_STALENESS",
                title=f"Дата {days}",
                status=CheckStatus.PASS,
                actual=days,
                confidence=1.0,
                message=f"Между датой и проверкой {days} дней",
            )
            for days in (1, 2, 4, 5, 11, 21)
        ],
        final_decision=FinalDecision.APPROVE,
        decision_reasons=["Все обязательные проверки пройдены"],
    )

    text = format_console_report(report)

    assert "между датой и проверкой 1 день" in text
    assert "между датой и проверкой 2 дня" in text
    assert "между датой и проверкой 4 дня" in text
    assert "между датой и проверкой 5 дней" in text
    assert "между датой и проверкой 11 дней" in text
    assert "между датой и проверкой 21 день" in text
