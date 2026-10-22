"""Offline comparison of a saved local MonkeyOCR trace with a web result."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.ocr.models import OCRDocumentResult


KEY_PHRASES = [
    "подписанием настоящего приложения",
    "подписание настоящего приложения на странице 1",
    "275-фз",
    "44-фз",
    "223-фз",
    "специальных счетов",
    "бюджетными средствами",
    "транспортные расходы",
    "самовывоз",
    "календарных дней с момента отгрузки",
]


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


def _web_payload(path: Path) -> tuple[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return text, []
    payload = json.loads(text)
    if isinstance(payload, dict):
        web_text = str(
            payload.get("native_markdown")
            or payload.get("markdown")
            or payload.get("text")
            or ""
        )
        layout = payload.get("layout_elements") or payload.get("layout") or []
        return web_text, layout if isinstance(layout, list) else []
    return text, []


def compare_saved_monkey(local_result: Path, web_result: Path) -> dict[str, Any]:
    local = OCRDocumentResult.model_validate_json(local_result.read_text(encoding="utf-8"))
    web_text, web_layout = _web_payload(web_result)
    local_text = local.native_markdown or local.full_text or local.normalized_full_text
    left_chars = list(_normalized(local_text))
    right_chars = list(_normalized(web_text))
    left_words = _normalized(local_text).split()
    right_words = _normalized(web_text).split()
    local_layout = [
        item.model_dump(mode="json")
        for page in local.pages
        for item in page.layout_elements
    ]
    local_types: dict[str, int] = {}
    for item in local_layout:
        kind = str(item["block_type"])
        local_types[kind] = local_types.get(kind, 0) + 1
    web_types: dict[str, int] = {}
    for item in web_layout:
        kind = str(item.get("block_type") or item.get("type") or item.get("label") or "unknown")
        web_types[kind] = web_types.get(kind, 0) + 1
    return {
        "local_result": str(local_result),
        "web_result": str(web_result),
        "local_layout_elements": len(local_layout),
        "web_layout_elements": len(web_layout),
        "local_layout_types": local_types,
        "web_layout_types": web_types,
        "cer": _distance(left_chars, right_chars) / max(1, len(right_chars)),
        "wer": _distance(left_words, right_words) / max(1, len(right_words)),
        "key_phrases": [
            {
                "phrase": phrase,
                "local": phrase in _normalized(local_text),
                "web": phrase in _normalized(web_text),
            }
            for phrase in KEY_PHRASES
        ],
        "local_tables": len(local.native_tables),
        "local_buyer_pictures": [
            item["original_pixel_bbox"]
            for item in local_layout
            if item["block_type"] == "Picture"
            and item["original_pixel_bbox"][1]
            > item["original_page_height"] * 0.45
        ],
    }
