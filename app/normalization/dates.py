"""Нормализация русских и ISO-дат."""

from __future__ import annotations

import re
from datetime import date

MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def normalize_date(value: str | date) -> date:
    """Преобразовать DD.MM.YYYY, русскую дату или ISO в date."""

    if isinstance(value, date):
        return value
    text = str(value).lower().strip().replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso:
        return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    numeric = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", text)
    if numeric:
        return date(int(numeric.group(3)), int(numeric.group(2)), int(numeric.group(1)))
    russian = re.search(
        rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS)})\s+(\d{{4}})\b", text
    )
    if russian:
        return date(int(russian.group(3)), MONTHS[russian.group(2)], int(russian.group(1)))
    raise ValueError(f"Неоднозначная или неподдерживаемая дата: {value!r}")

