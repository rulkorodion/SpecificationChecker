"""Изолируемый календарь рабочих дней."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol


class BusinessCalendar(Protocol):
    def business_days_between(self, start: date, end: date) -> int:
        """Число рабочих дней после start до end включительно."""


class WeekdayBusinessCalendar:
    """Пилотный календарь: понедельник-пятница, без праздников."""

    def business_days_between(self, start: date, end: date) -> int:
        if start == end:
            return 0
        direction = 1 if end > start else -1
        current = start
        count = 0
        while current != end:
            current += timedelta(days=direction)
            if current.weekday() < 5:
                count += direction
        return abs(count)

