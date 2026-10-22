"""Нормализация количеств и денежных сумм без float."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def normalize_decimal(value: str | int | Decimal) -> Decimal:
    """Преобразовать OCR-представление числа в Decimal без округления."""

    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    text = str(value).strip().replace("\u00a0", " ").replace("\u202f", " ")
    text = re.sub(r"(?i)(руб(?:\.|лей)?|₽)", "", text)
    text = re.sub(r"(?<=\d)[OoОо](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)[IlІ|](?=\d)", "1", text)
    text = re.sub(r"\s*([,.])\s*", r"\1", text)
    text = re.sub(r"[^\d,.\-]", "", text)
    if not text or text in {"-", ".", ","}:
        raise ValueError(f"Не найдено число: {value!r}")

    comma = text.rfind(",")
    dot = text.rfind(".")
    if comma >= 0 and dot >= 0:
        decimal_pos = max(comma, dot)
    elif comma >= 0:
        decimal_pos = comma
    else:
        decimal_pos = dot

    if decimal_pos >= 0:
        integer = re.sub(r"[,.]", "", text[:decimal_pos])
        fraction = re.sub(r"[,.]", "", text[decimal_pos + 1 :])
        normalized = f"{integer}.{fraction}" if fraction else integer
    else:
        normalized = text
    try:
        return Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"Некорректное число: {value!r}") from exc

