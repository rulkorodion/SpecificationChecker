"""Безопасная нормализация OCR-текста."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    """Нормализовать пробелы, регистр и типографские варианты без угадывания смысла."""

    text = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"[«»„“”]", '"', text)
    text = re.sub(r"\b[nн]\s*(?=\d)", "№ ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_application_number(value: str) -> str:
    """Нормализовать номер без преобразования в integer."""

    return re.sub(r"\s+", "", normalize_text(value)).upper()

