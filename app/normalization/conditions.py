"""Нормализация категориальных условий."""

from __future__ import annotations

import re

from app.enums import DeliveryType, PaymentType
from app.normalization.text import normalize_text


def classify_keywords(
    text: str, positive: list[str], negative: list[str]
) -> bool | None:
    """Вернуть True/False только при непротиворечивых признаках."""

    normalized = normalize_text(text)
    pos = any(normalize_text(item) in normalized for item in positive)
    neg = any(normalize_text(item) in normalized for item in negative)
    if pos == neg:
        return None
    return pos


def classify_delivery(text: str, config: dict) -> DeliveryType | None:
    value = classify_keywords(
        text, config["delivery"]["delivery_keywords"], config["delivery"]["pickup_keywords"]
    )
    if value is None:
        return None
    return DeliveryType.DELIVERY if value else DeliveryType.PICKUP


def classify_payment(text: str, config: dict) -> PaymentType | None:
    # Исправление применяется только внутри уже выделенного блока оплаты.
    # Это не глобальная замена по документу: обычные слова с похожими
    # сочетаниями символов остаются нетронутыми.
    text = re.sub(
        r"\bпредп[оа]лата\b",
        "предоплата",
        text,
        flags=re.IGNORECASE,
    )
    normalized = normalize_text(text)
    if "оплата по факту готовности" in normalized:
        return PaymentType.PREPAYMENT
    value = classify_keywords(
        text, config["payment"]["prepayment_keywords"], config["payment"]["deferment_keywords"]
    )
    if value is None:
        return None
    return PaymentType.PREPAYMENT if value else PaymentType.DEFERMENT


_PAYMENT_NUMBER_CONTEXT = (
    "дней",
    "календар",
    "оплата",
    "отсроч",
    "момента отгрузки",
)
_NUMBER_LIKE_TOKEN = re.compile(r"(?<![\w])([зЗ3Iil][0ОоOo4])(?![\w])")


def normalize_payment_numeric_tokens(text: str) -> tuple[str, list[str]]:
    """Исправить похожие символы только в числовых токенах условий оплаты."""

    replacements: list[str] = []
    translation = str.maketrans(
        {
            "з": "3",
            "З": "3",
            "I": "1",
            "i": "1",
            "l": "1",
            "О": "0",
            "о": "0",
            "O": "0",
            "o": "0",
        }
    )

    def replace(match: re.Match[str]) -> str:
        left = max(0, match.start() - 80)
        right = min(len(text), match.end() + 80)
        surrounding = normalize_text(text[left:right])
        if not any(marker in surrounding for marker in _PAYMENT_NUMBER_CONTEXT):
            return match.group(0)
        normalized = match.group(1).translate(translation)
        replacements.append(f"{match.group(1)}→{normalized}")
        return normalized

    return _NUMBER_LIKE_TOKEN.sub(replace, text), replacements
