from datetime import date
from decimal import Decimal

import pytest

from app.checks.date_rules import WeekdayBusinessCalendar
from app.normalization.dates import normalize_date
from app.normalization.numbers import normalize_decimal


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1 255 184,00", Decimal("1255184.00")),
        ("1\u00a0255\u00a0184, 00 руб.", Decimal("1255184.00")),
        ("23, 113", Decimal("23.113")),
        ("23.113", Decimal("23.113")),
        ("1255184.00", Decimal("1255184.00")),
    ],
)
def test_normalize_decimal(raw, expected):
    assert normalize_decimal(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("21.07.2026", date(2026, 7, 21)),
        ("21 июля 2026 г.", date(2026, 7, 21)),
        ("2026-07-21", date(2026, 7, 21)),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_business_days_excludes_weekend():
    calendar = WeekdayBusinessCalendar()
    assert calendar.business_days_between(date(2026, 7, 17), date(2026, 7, 21)) == 2

