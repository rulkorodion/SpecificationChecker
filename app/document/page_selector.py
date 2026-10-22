"""Fast CPU-only page selection for multi-application PDF bundles."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from app.exceptions import PageSelectionError

LOGGER = logging.getLogger(__name__)

_APPLICATION_PATTERN = re.compile(
    r"(?:ПРИЛОЖЕН\w*|APPLICATION)\s*(?:№|N(?:O|0)?|#)?\s*([0-9][0-9\s]{3,15})",
    re.IGNORECASE,
)
_PAGE_PATTERN = re.compile(
    r"(?:СТР\w*|PAGE)\s*[.:]?\s*(\d{1,2})\s*(?:ИЗ|OF)\s*(\d{1,2})",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PageMarker:
    application_number: str
    logical_page_number: int
    total_pages: int
    zone: str


@dataclass(frozen=True, slots=True)
class PageSelectionResult:
    document_path: Path
    metadata: dict[str, Any] | None


def application_number_from_reference(source: Any) -> str:
    """Read only the selector key without validating the full reference early."""

    if isinstance(source, Mapping):
        value = source.get("application_number")
    elif hasattr(source, "application_number"):
        value = getattr(source, "application_number")
    else:
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise PageSelectionError(
                "reference_unreadable",
                f"Page Selector не смог прочитать reference: {exc}",
            ) from exc
        value = payload.get("application_number") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise PageSelectionError(
            "target_application_number_missing",
            "В reference отсутствует строковый application_number для Page Selector.",
        )
    return value.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _normalize_text(value: str) -> str:
    return " ".join(
        value.replace("\u00a0", " ").replace("\u202f", " ").split()
    )


def parse_page_markers(text: str, zone: str) -> list[PageMarker]:
    """Extract application/page markers from a short header or footer OCR text."""

    normalized = _normalize_text(text)
    applications = [
        (match.start(), re.sub(r"\D", "", match.group(1)))
        for match in _APPLICATION_PATTERN.finditer(normalized)
    ]
    pages = [
        (match.start(), int(match.group(1)), int(match.group(2)))
        for match in _PAGE_PATTERN.finditer(normalized)
    ]
    markers: list[PageMarker] = []
    for app_position, application_number in applications:
        following = [item for item in pages if item[0] >= app_position]
        if not following:
            continue
        _, logical_page, total_pages = min(
            following, key=lambda item: item[0] - app_position
        )
        markers.append(
            PageMarker(
                application_number=application_number,
                logical_page_number=logical_page,
                total_pages=total_pages,
                zone=zone,
            )
        )
    return markers


def _resolve_tesseract(command: str) -> str:
    configured = command.strip() or "tesseract"
    resolved = shutil.which(configured)
    if resolved:
        return resolved
    candidate = Path(configured)
    if candidate.is_file():
        return str(candidate)
    if os.name == "nt":
        windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if windows_default.is_file():
            return str(windows_default)
    raise PageSelectionError(
        "selector_unavailable",
        "Page Selector не запущен: исполняемый файл Tesseract не найден.",
    )


def _ocr_clip(
    page: fitz.Page,
    *,
    zone: str,
    ratio: float,
    dpi: int,
    command: str,
    language: str,
    timeout_seconds: int,
    tessdata_dir: str | None,
) -> str:
    rect = page.rect
    if zone == "top":
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * ratio)
    else:
        clip = fitz.Rect(
            rect.x0,
            rect.y1 - rect.height * ratio,
            rect.x1,
            rect.y1,
        )
    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=clip,
        alpha=False,
        colorspace=fitz.csGRAY,
    )
    args = [command, "stdin", "stdout"]
    if tessdata_dir:
        args.extend(["--tessdata-dir", tessdata_dir])
    args.extend(["-l", language, "--psm", "6"])
    try:
        completed = subprocess.run(
            args,
            input=pixmap.tobytes("png"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise PageSelectionError(
            "selector_timeout",
            f"Tesseract не завершил распознавание {zone}-crop за {timeout_seconds} с.",
        ) from exc
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PageSelectionError(
            "selector_ocr_failed",
            f"Tesseract завершился с кодом {completed.returncode}: {message}",
        )
    return completed.stdout.decode("utf-8", errors="replace")


def _page_markers(
    page: fitz.Page,
    *,
    settings: dict[str, Any],
    command: str,
) -> list[PageMarker]:
    common = {
        "dpi": int(settings.get("dpi", 300)),
        "command": command,
        "language": str(settings.get("language", "rus")),
        "timeout_seconds": int(settings.get("timeout_seconds", 20)),
        "tessdata_dir": settings.get("tessdata_dir"),
    }
    bottom_ratio = float(settings.get("bottom_crop_ratio", 0.18))
    top_ratio = float(settings.get("top_crop_ratio", 0.24))
    rect = page.rect
    bottom_clip = fitz.Rect(
        rect.x0,
        rect.y1 - rect.height * bottom_ratio,
        rect.x1,
        rect.y1,
    )
    markers = parse_page_markers(
        page.get_text("text", clip=bottom_clip), "bottom-native"
    )
    if markers:
        return markers
    footer = _ocr_clip(
        page,
        zone="bottom",
        ratio=bottom_ratio,
        **common,
    )
    markers = parse_page_markers(footer, "bottom")
    if markers:
        return markers
    top_clip = fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x1,
        rect.y0 + rect.height * top_ratio,
    )
    markers = parse_page_markers(page.get_text("text", clip=top_clip), "top-native")
    if markers:
        return markers
    header = _ocr_clip(
        page,
        zone="top",
        ratio=top_ratio,
        **common,
    )
    return parse_page_markers(header, "top")


def _selection_failure(
    *,
    output_path: Path,
    metadata: dict[str, Any],
    code: str,
    message: str,
) -> None:
    metadata["status"] = "FAILED"
    metadata["diagnostic"] = {"code": code, "message": message}
    _write_json(output_path, metadata)
    raise PageSelectionError(code, message, artifact_path=output_path)


def select_application_pages(
    source_pdf: Path,
    *,
    target_application_number: str,
    output_dir: Path,
    page_limit: int,
    settings: dict[str, Any],
) -> PageSelectionResult:
    """Select and reorder exactly one application from a multi-page PDF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "page_selection.json"
    target = re.sub(r"\D", "", str(target_application_number))
    if not target:
        raise PageSelectionError(
            "target_application_number_missing",
            "В reference отсутствует корректный application_number для Page Selector.",
            artifact_path=metadata_path,
        )
    started = time.perf_counter()
    command = _resolve_tesseract(str(settings.get("command", "tesseract")))
    try:
        source = fitz.open(source_pdf)
    except Exception as exc:
        raise PageSelectionError(
            "source_pdf_unreadable", f"Page Selector не смог открыть PDF: {exc}"
        ) from exc
    try:
        if source.needs_pass:
            raise PageSelectionError(
                "source_pdf_password_protected",
                "Page Selector не поддерживает PDF, защищённый паролем.",
            )
        metadata: dict[str, Any] = {
            "schema_version": "1.0",
            "status": "PROCESSING",
            "target_application_number": target,
            "source_document": source_pdf.name,
            "source_page_count": source.page_count,
            "selected_document": "selected_document.pdf",
            "selected_page_count": 0,
            "selected_pages": [],
            "selector": {
                "engine": "tesseract",
                "command": Path(command).name,
                "language": str(settings.get("language", "rus")),
                "dpi": int(settings.get("dpi", 300)),
                "top_crop_ratio": float(settings.get("top_crop_ratio", 0.24)),
                "bottom_crop_ratio": float(settings.get("bottom_crop_ratio", 0.18)),
            },
            "scanned_pages": [],
            "diagnostic": None,
        }
        max_source_pages = int(settings.get("max_source_pages", 20))
        if source.page_count > max_source_pages:
            _selection_failure(
                output_path=metadata_path,
                metadata=metadata,
                code="source_page_limit_exceeded",
                message=(
                    f"В исходном PDF {source.page_count} страниц; "
                    f"безопасный максимум — {max_source_pages}."
                ),
            )
        candidates: list[tuple[int, PageMarker]] = []
        for index, page in enumerate(source, start=1):
            markers = _page_markers(page, settings=settings, command=command)
            metadata["scanned_pages"].append(
                {
                    "source_page_number": index,
                    "markers": [
                        {
                            "application_number": marker.application_number,
                            "logical_page_number": marker.logical_page_number,
                            "total_pages": marker.total_pages,
                            "zone": marker.zone,
                        }
                        for marker in markers
                    ],
                }
            )
            candidates.extend(
                (index, marker)
                for marker in markers
                if marker.application_number == target
            )

        if not candidates:
            _selection_failure(
                output_path=metadata_path,
                metadata=metadata,
                code="target_not_found",
                message=(
                    f"Приложение № {target} не найдено в верхних/нижних "
                    f"областях {source.page_count} страниц PDF."
                ),
            )
        totals = {marker.total_pages for _, marker in candidates}
        if len(totals) != 1:
            _selection_failure(
                output_path=metadata_path,
                metadata=metadata,
                code="inconsistent_total_pages",
                message=f"Для приложения № {target} найдены противоречивые значения 'из N'.",
            )
        expected_total = next(iter(totals))
        if expected_total < 1 or expected_total > page_limit:
            _selection_failure(
                output_path=metadata_path,
                metadata=metadata,
                code="selected_application_too_long",
                message=(
                    f"Приложение № {target} содержит {expected_total} страниц; "
                    f"поддерживаемый максимум — {page_limit}."
                ),
            )
        by_logical_page: dict[int, list[int]] = {}
        for source_page, marker in candidates:
            by_logical_page.setdefault(marker.logical_page_number, []).append(source_page)
        expected_pages = set(range(1, expected_total + 1))
        if set(by_logical_page) != expected_pages or any(
            len(source_pages) != 1 for source_pages in by_logical_page.values()
        ):
            _selection_failure(
                output_path=metadata_path,
                metadata=metadata,
                code="ambiguous_page_set",
                message=(
                    f"Страницы приложения № {target} выбраны неоднозначно: "
                    f"ожидались номера 1–{expected_total} без повторов."
                ),
            )

        selected = [
            {
                "source_page_number": by_logical_page[logical_page][0],
                "logical_page_number": logical_page,
                "total_pages": expected_total,
            }
            for logical_page in range(1, expected_total + 1)
        ]
        destination_path = output_dir / "selected_document.pdf"
        destination = fitz.open()
        try:
            for item in selected:
                source_index = int(item["source_page_number"]) - 1
                destination.insert_pdf(source, from_page=source_index, to_page=source_index)
            destination.save(destination_path, garbage=4, deflate=True)
        finally:
            destination.close()
        metadata["status"] = "COMPLETED"
        metadata["selected_page_count"] = len(selected)
        metadata["selected_pages"] = selected
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_json(metadata_path, metadata)
        LOGGER.info(
            "Page Selector: application=%s source_pages=%s selected_order=%s elapsed=%.3fs",
            target,
            source.page_count,
            [item["source_page_number"] for item in selected],
            metadata["elapsed_seconds"],
        )
        return PageSelectionResult(destination_path, metadata)
    finally:
        source.close()


def prepare_document_for_ocr(
    source_path: Path,
    *,
    reference_source: Any,
    output_dir: Path,
    page_limit: int,
    settings: dict[str, Any],
) -> PageSelectionResult:
    """Keep short inputs unchanged and select an application only when required."""

    page_count = source_pdf_page_count(source_path)
    if page_count is None or page_count <= page_limit:
        return PageSelectionResult(source_path, None)
    if not bool(settings.get("enabled", True)):
        raise PageSelectionError(
            "selector_disabled",
            "Многостраничный PDF требует Page Selector, но он отключён.",
        )
    return select_application_pages(
        source_path,
        target_application_number=application_number_from_reference(reference_source),
        output_dir=output_dir,
        page_limit=page_limit,
        settings=settings,
    )


def source_pdf_page_count(path: Path) -> int | None:
    """Return a PDF page count; non-PDF inputs do not enter Page Selector."""

    if path.suffix.lower() != ".pdf":
        return None
    try:
        with fitz.open(path) as document:
            return document.page_count
    except Exception:
        return None
