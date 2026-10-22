"""One-off selective MonkeyOCR batch runner for scanned RAG pages.

The script is deliberately external to the production API/queue.  It reuses the
installed worker image and application OCR engine without rebuilding the image.

Commands:
  prepare  Validate the source PDFs and copy only the listed documents.
  run      Process one deterministic shard of manifest pages with one model/GPU.
  merge    Verify page results and build aggregate text/JSONL outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Manifest item {line_number} must be an object")
        records.append(value)
    return records


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes configured root: {relative}")
    return candidate


def _expand_pages(value: Any) -> list[int]:
    if isinstance(value, list):
        pages = [int(item) for item in value]
    elif isinstance(value, str):
        pages = []
        for part in value.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                start, finish = int(left), int(right)
                if start > finish:
                    raise ValueError(f"Invalid page range: {token}")
                pages.extend(range(start, finish + 1))
            else:
                pages.append(int(token))
    else:
        raise ValueError("pages must be a list or a range string")
    if not pages or min(pages) < 1 or len(set(pages)) != len(pages):
        raise ValueError(f"Invalid or duplicate page numbers: {pages}")
    return pages


def _result_path(output_root: Path, item: dict[str, Any]) -> Path:
    document_id = str(item["document_id"])
    version_id = str(item["version_id"])
    page_number = int(item["page_number"])
    for label, value in (("document_id", document_id), ("version_id", version_id)):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Unsafe {label}: {value!r}")
    return output_root / "pages" / document_id / version_id / f"page_{page_number:04d}.json"


def _validate_unique_tasks(tasks: Iterable[dict[str, Any]]) -> None:
    seen: set[tuple[str, str, int]] = set()
    for item in tasks:
        key = (
            str(item["document_id"]),
            str(item["version_id"]),
            int(item["page_number"]),
        )
        if key in seen:
            raise ValueError(f"Duplicate manifest page key: {key}")
        seen.add(key)


def prepare(args: argparse.Namespace) -> int:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for the prepare command") from exc

    source_root = args.source_root.resolve()
    output_root = args.output.resolve()
    plan = _read_json(args.plan)
    documents = plan.get("documents") if isinstance(plan, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("Plan must contain a non-empty documents list")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output must not be created inside the immutable source root")

    input_documents = output_root / "documents"
    input_documents.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    document_summary: list[dict[str, Any]] = []

    for index, spec in enumerate(documents, start=1):
        if not isinstance(spec, dict):
            raise ValueError(f"Document plan item {index} must be an object")
        relative_path = str(spec["relative_path"])
        pages = _expand_pages(spec["pages"])
        source_path = _safe_child(source_root, relative_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source PDF not found: {source_path}")
        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF sources are supported: {source_path}")

        with fitz.open(source_path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"Password-protected PDF: {relative_path}")
            page_count = pdf.page_count
        if max(pages) > page_count:
            raise ValueError(
                f"{relative_path}: requested page {max(pages)}, PDF has {page_count}"
            )

        source_hash = _sha256(source_path)
        document_id = str(spec.get("document_id") or f"doc-{index:03d}")
        version_id = str(spec.get("version_id") or source_hash[:16])
        input_file = f"documents/{document_id}.pdf"
        destination = _safe_child(output_root, input_file)

        if destination.is_file() and _sha256(destination) == source_hash:
            copy_status = "reused"
        else:
            if destination.exists() and not args.force:
                raise FileExistsError(
                    f"Destination exists with another hash: {destination}; use --force"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            if _sha256(destination) != source_hash:
                raise OSError(f"Copied PDF hash mismatch: {destination}")
            copy_status = "copied"

        for page_number in pages:
            tasks.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "document_id": document_id,
                    "version_id": version_id,
                    "relative_path": relative_path.replace("\\", "/"),
                    "page_number": page_number,
                    "source_sha256": source_hash,
                    "input_file": input_file,
                }
            )
        document_summary.append(
            {
                "document_id": document_id,
                "relative_path": relative_path.replace("\\", "/"),
                "source_sha256": source_hash,
                "pdf_page_count": page_count,
                "selected_pages": pages,
                "copy_status": copy_status,
            }
        )

    _validate_unique_tasks(tasks)
    expected_pages = int(plan.get("expected_pages", len(tasks)))
    if len(tasks) != expected_pages:
        raise ValueError(
            f"Plan expected {expected_pages} pages, expanded manifest has {len(tasks)}"
        )

    manifest_path = output_root / "manifest.jsonl"
    manifest_text = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in tasks
    )
    _write_text_atomic(manifest_path, manifest_text)
    _write_json_atomic(
        output_root / "prepare_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "source_root": str(source_root),
            "document_count": len(document_summary),
            "page_count": len(tasks),
            "manifest_sha256": _sha256(manifest_path),
            "documents": document_summary,
        },
    )
    print(
        f"PREPARED: documents={len(document_summary)} pages={len(tasks)} "
        f"output={output_root}"
    )
    return 0


def _ordered_block_text(page: Any) -> str:
    blocks = sorted(
        page.blocks,
        key=lambda block: (
            block.reading_order if block.reading_order is not None else 10**12,
            block.bounding_box.y_min,
            block.bounding_box.x_min,
        ),
    )
    return "\n".join(block.text.strip() for block in blocks if block.text.strip())


def _render_pdf_page(path: Path, page_number: int, dpi: int):
    import cv2
    import fitz
    import numpy as np

    with fitz.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError(f"Password-protected PDF: {path}")
        if page_number < 1 or page_number > pdf.page_count:
            raise ValueError(
                f"Page {page_number} outside PDF page count {pdf.page_count}: {path}"
            )
        scale = dpi / 72.0
        pix = pdf[page_number - 1].get_pixmap(
            matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB
        )
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run(args: argparse.Namespace) -> int:
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")

    # App imports are intentionally delayed so prepare/merge also work outside
    # the worker image without Torch or MonkeyOCR dependencies.
    from app.config import load_config
    from app.ocr.factory import build_ocr_engine

    manifest_path = args.manifest.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    tasks = _read_jsonl(manifest_path)
    _validate_unique_tasks(tasks)
    selected = [
        (index, item)
        for index, item in enumerate(tasks)
        if index % args.shard_count == args.shard_index
    ]
    if not selected:
        raise ValueError(f"Shard {args.shard_index}/{args.shard_count} has no tasks")

    config = load_config(args.config)
    config["document"]["render_dpi"] = args.dpi
    monkey = config["ocr"].setdefault("monkeyocr_v2", {})
    monkey["pdf_render_dpi"] = args.dpi
    monkey["element_cache_dir"] = str(
        args.cache_root.resolve() / f"shard-{args.shard_index}" / "monkey_elements"
    )
    monkey["modules_cache_dir"] = str(
        args.cache_root.resolve() / f"shard-{args.shard_index}" / "huggingface_modules"
    )
    monkey["model_call_journal"] = str(
        output_root / f"model_calls_shard_{args.shard_index}.jsonl"
    )
    engine = build_ocr_engine(config, config["ocr"].get("language"))
    runtime_info = engine.prepare()
    inference_keys = (
        "model_id",
        "revision",
        "profile",
        "dtype",
        "attention_backend",
        "max_pixels",
        "layout_max_pixels",
        "recognition_max_pixels",
        "transformers_use_kv_cache",
        "do_sample",
        "layout_max_tokens",
        "element_max_tokens",
        "layout_max_seconds",
        "element_max_seconds",
        "retry_truncated_element",
        "coordinate_scale",
        "crop_padding_ratio",
        "prompt_version",
        "adapter_version",
        "pipeline_version",
        "parsing_mode",
        "pdf_render_dpi",
    )
    settings = {
        "model_id": getattr(engine, "model_id", None),
        "revision": getattr(engine, "revision", None),
        "pipeline_version": "layout-crops-v1",
        "dpi": args.dpi,
        # Runtime/cache paths are deliberately excluded: changing the shard or
        # output directory must not invalidate a successfully recognized page.
        "monkeyocr": {
            key: monkey.get(key) for key in inference_keys if key in monkey
        },
    }
    settings_hash = _canonical_hash(settings)
    source_hash_cache: dict[Path, str] = {}
    counts = {"COMPLETED": 0, "PARTIAL": 0, "FAILED": 0, "SKIPPED": 0}
    shard_started = time.perf_counter()

    for position, (_, item) in enumerate(selected, start=1):
        result_path = _result_path(output_root, item)
        fingerprint = {
            "document_id": str(item["document_id"]),
            "version_id": str(item["version_id"]),
            "page_number": int(item["page_number"]),
            "source_sha256": str(item["source_sha256"]),
            "ocr_settings_sha256": settings_hash,
        }
        if result_path.is_file():
            try:
                previous = _read_json(result_path)
            except (OSError, ValueError):
                previous = {}
            if (
                previous.get("status") == "COMPLETED"
                and previous.get("fingerprint") == fingerprint
            ):
                counts["SKIPPED"] += 1
                print(
                    f"[{position}/{len(selected)}] SKIP "
                    f"{item['document_id']} page={item['page_number']}"
                )
                continue

        started = time.perf_counter()
        source_path: Path | None = None
        start_document = getattr(engine, "start_document", None)
        finish_document = getattr(engine, "finish_document", None)
        try:
            source_path = _safe_child(input_root, str(item["input_file"]))
            if not source_path.is_file():
                raise FileNotFoundError(f"Input PDF not found: {source_path}")
            actual_hash = source_hash_cache.get(source_path)
            if actual_hash is None:
                actual_hash = _sha256(source_path)
                source_hash_cache[source_path] = actual_hash
            if actual_hash != item["source_sha256"]:
                raise ValueError(
                    f"Source SHA-256 mismatch for {source_path}: "
                    f"expected={item['source_sha256']} actual={actual_hash}"
                )

            page_number = int(item["page_number"])
            if callable(start_document):
                start_document(f"{source_path}#page={page_number}")
            image = _render_pdf_page(source_path, page_number, args.dpi)
            height, width = image.shape[:2]
            page = engine.recognize_page(image, page_number, "original")
            normalized_text = _ordered_block_text(page)
            native_markdown = page.native_markdown.strip()
            text_for_rag = native_markdown or normalized_text
            truncated = bool(
                page.processing_metadata.get("possible_output_truncation", False)
            )
            empty = not text_for_rag.strip()
            status = "PARTIAL" if truncated or empty else "COMPLETED"
            warnings = list(page.warnings)
            if empty:
                warnings.append("MonkeyOCR returned no textual content for the selected page")
            result = {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "status": status,
                "fingerprint": fingerprint,
                "document_id": item["document_id"],
                "version_id": item["version_id"],
                "relative_path": item["relative_path"],
                "page_number": page_number,
                "source_sha256": item["source_sha256"],
                "input_file": item["input_file"],
                "page": {"width": width, "height": height, "dpi": args.dpi},
                "text": {
                    "text_for_rag": text_for_rag,
                    "normalized_block_text": normalized_text,
                    "main_text": str(page.processing_metadata.get("full_text", "")),
                    "native_markdown": native_markdown,
                },
                "blocks": [block.model_dump(mode="json") for block in page.blocks],
                "layout_elements": [
                    element.model_dump(mode="json") for element in page.layout_elements
                ],
                "tables": page.native_tables,
                "warnings": warnings,
                "confidence": None,
                "confidence_source": "model_unscored",
                "ocr": {
                    "engine_name": page.engine_name,
                    "model_id": getattr(engine, "model_id", None),
                    "revision": getattr(engine, "revision", None),
                    "settings_sha256": settings_hash,
                    "runtime": runtime_info,
                    "processing_metadata": page.processing_metadata,
                },
                "processing_time_seconds": round(time.perf_counter() - started, 6),
                "error": None,
            }
            _write_json_atomic(result_path, result)
            counts[status] += 1
            if callable(finish_document):
                finish_document(status.lower())
            print(
                f"[{position}/{len(selected)}] {status} "
                f"{item['document_id']} page={page_number} "
                f"seconds={result['processing_time_seconds']}"
            )
        except KeyboardInterrupt:
            if callable(finish_document):
                finish_document("interrupted")
            raise
        except Exception as exc:  # keep the remaining one-off batch alive
            if callable(finish_document):
                finish_document("failed")
            result = {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "status": "FAILED",
                "fingerprint": fingerprint,
                "document_id": item.get("document_id"),
                "version_id": item.get("version_id"),
                "relative_path": item.get("relative_path"),
                "page_number": item.get("page_number"),
                "source_sha256": item.get("source_sha256"),
                "input_file": item.get("input_file"),
                "processing_time_seconds": round(time.perf_counter() - started, 6),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            _write_json_atomic(result_path, result)
            counts["FAILED"] += 1
            print(
                f"[{position}/{len(selected)}] FAILED "
                f"{item.get('document_id')} page={item.get('page_number')}: {exc}",
                file=sys.stderr,
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_tasks": len(selected),
        "counts": counts,
        "elapsed_seconds": round(time.perf_counter() - shard_started, 6),
        "ocr_settings_sha256": settings_hash,
        "runtime": runtime_info,
    }
    _write_json_atomic(output_root / f"run_summary_shard_{args.shard_index}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if counts["FAILED"] == 0 and counts["PARTIAL"] == 0 else 2


def _deduplicate_adjacent_lines(parts: list[str]) -> str:
    lines: list[str] = []
    for part in parts:
        for raw_line in part.splitlines():
            line = raw_line.strip()
            if line and (not lines or line != lines[-1]):
                lines.append(line)
    return "\n".join(lines)


def _plain_text_fallback(
    engine: Any,
    image: Any,
    bbox: list[int],
    *,
    page_number: int,
    reading_order: int,
    original_label: str,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recognize one problematic element as plain text, then tiled text."""

    import cv2
    from PIL import Image

    from app.ocr.monkey_engine import OFFICIAL_PROMPTS

    x1, y1, x2, y2 = (int(value) for value in bbox)
    crop_bgr = image[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        raise ValueError(f"Empty fallback crop: {bbox}")
    crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    attempts: list[dict[str, Any]] = []

    # Whole-crop plain text is the least destructive fallback. If the model
    # repeats again, horizontal tiling changes the visual prefix and bounds the
    # amount of text generated by each call.
    for tile_count in (1, 3, 5):
        segments: list[dict[str, Any]] = []
        strategy_ok = True
        overlap = max(2, int(round(crop.height * 0.015))) if tile_count > 1 else 0
        for tile_index in range(tile_count):
            local_y1 = max(0, int(round(tile_index * crop.height / tile_count)) - overlap)
            local_y2 = min(
                crop.height,
                int(round((tile_index + 1) * crop.height / tile_count)) + overlap,
            )
            tile = crop.crop((0, local_y1, crop.width, local_y2))
            output = engine._infer(  # noqa: SLF001 - intentional one-off adapter reuse
                tile,
                OFFICIAL_PROMPTS["Text"],
                max_tokens=max_tokens,
                stage="rag_plain_text_fallback",
                page_number=page_number,
                element_index=reading_order,
                element_label=f"{original_label}-plain-text-{tile_count}",
            )
            text = output.text.strip()
            attempts.append(
                {
                    "tile_count": tile_count,
                    "tile_index": tile_index,
                    "finish_reason": output.finish_reason,
                    "truncated": output.truncated,
                    "generated_tokens": output.generated_tokens,
                }
            )
            if output.truncated:
                strategy_ok = False
                break
            if text:
                segments.append(
                    {
                        "text": text,
                        "bbox": [x1, y1 + local_y1, x2, y1 + local_y2],
                        "tile_index": tile_index,
                        "tile_count": tile_count,
                    }
                )
        if strategy_ok and segments:
            return segments, attempts
    return [], attempts


def retry_partials(args: argparse.Namespace) -> int:
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")

    from app.config import load_config
    from app.ocr.factory import build_ocr_engine

    manifest_path = args.manifest.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    tasks = _read_jsonl(manifest_path)
    _validate_unique_tasks(tasks)
    partial_tasks: list[tuple[int, dict[str, Any], Path, dict[str, Any]]] = []
    for index, item in enumerate(tasks):
        if index % args.shard_count != args.shard_index:
            continue
        result_path = _result_path(output_root, item)
        if not result_path.is_file():
            continue
        record = _read_json(result_path)
        if record.get("status") == "PARTIAL":
            partial_tasks.append((index, item, result_path, record))
    if not partial_tasks:
        print(f"RETRY: shard={args.shard_index} partial_pages=0")
        return 0

    config = load_config(args.config)
    config["document"]["render_dpi"] = args.dpi
    monkey = config["ocr"].setdefault("monkeyocr_v2", {})
    monkey["pdf_render_dpi"] = args.dpi
    monkey["element_cache_dir"] = str(
        args.cache_root.resolve()
        / f"fallback-shard-{args.shard_index}"
        / "monkey_elements"
    )
    monkey["modules_cache_dir"] = str(
        args.cache_root.resolve()
        / f"fallback-shard-{args.shard_index}"
        / "huggingface_modules"
    )
    monkey["model_call_journal"] = str(
        output_root / f"fallback_model_calls_shard_{args.shard_index}.jsonl"
    )
    engine = build_ocr_engine(config, config["ocr"].get("language"))
    runtime_info = engine.prepare()
    source_hash_cache: dict[Path, str] = {}
    counts = {"COMPLETED": 0, "PARTIAL": 0, "FAILED": 0}
    retry_started = time.perf_counter()

    for position, (_, item, result_path, record) in enumerate(partial_tasks, start=1):
        page_started = time.perf_counter()
        start_document = getattr(engine, "start_document", None)
        finish_document = getattr(engine, "finish_document", None)
        try:
            source_path = _safe_child(input_root, str(item["input_file"]))
            actual_hash = source_hash_cache.get(source_path)
            if actual_hash is None:
                actual_hash = _sha256(source_path)
                source_hash_cache[source_path] = actual_hash
            if actual_hash != item["source_sha256"]:
                raise ValueError(f"Source SHA-256 mismatch: {source_path}")
            page_number = int(item["page_number"])
            if callable(start_document):
                start_document(f"{source_path}#page={page_number}:fallback")
            image = _render_pdf_page(source_path, page_number, args.dpi)
            metadata = ((record.get("ocr") or {}).get("processing_metadata") or {})
            truncated = list(metadata.get("truncated_elements") or [])
            elements = list(record.get("layout_elements") or [])
            blocks = list(record.get("blocks") or [])
            tables = list(record.get("tables") or [])
            resolved_orders: set[int] = set()
            fallback_details: list[dict[str, Any]] = []

            for problem in truncated:
                order = int(problem["reading_order"])
                element = next(
                    (entry for entry in elements if int(entry.get("reading_order", -1)) == order),
                    None,
                )
                if element is None:
                    fallback_details.append(
                        {"reading_order": order, "resolved": False, "error": "layout element missing"}
                    )
                    continue
                label = str(element.get("block_type") or problem.get("label") or "Text")
                bbox = [int(value) for value in element["original_pixel_bbox"]]
                segments, attempts = _plain_text_fallback(
                    engine,
                    image,
                    bbox,
                    page_number=page_number,
                    reading_order=order,
                    original_label=label,
                    max_tokens=args.max_tokens,
                )
                if not segments:
                    fallback_details.append(
                        {
                            "reading_order": order,
                            "label": label,
                            "resolved": False,
                            "attempts": attempts,
                        }
                    )
                    continue

                fallback_text = _deduplicate_adjacent_lines(
                    [str(segment["text"]) for segment in segments]
                )
                if not fallback_text:
                    continue
                resolved_orders.add(order)
                element["text"] = fallback_text
                element["raw_response"] = fallback_text
                element["native_content"] = fallback_text
                element_metadata = dict(element.get("metadata") or {})
                element_metadata.update(
                    {
                        "truncated": False,
                        "possible_output_truncation": False,
                        "fallback_method": "plain_text_horizontal_tiles",
                        "fallback_original_label": label,
                        "fallback_tile_count": segments[0]["tile_count"],
                    }
                )
                element["metadata"] = element_metadata

                blocks = [
                    block
                    for block in blocks
                    if int((block.get("metadata") or {}).get("layout_reading_order", -1))
                    != order
                ]
                page_height, page_width = image.shape[:2]
                for segment_index, segment in enumerate(segments):
                    sx1, sy1, sx2, sy2 = segment["bbox"]
                    blocks.append(
                        {
                            "text": segment["text"].strip(),
                            "confidence": None,
                            "bounding_box": {
                                "x_min": sx1,
                                "y_min": sy1,
                                "x_max": sx2,
                                "y_max": sy2,
                                "page_width": page_width,
                                "page_height": page_height,
                            },
                            "page_number": page_number,
                            "preprocessing_profile": "original",
                            "engine_name": (record.get("ocr") or {}).get("engine_name", "MonkeyOCR"),
                            "block_type": label,
                            "source": "monkeyocr-plain-text-fallback",
                            "reading_order": order * 1000 + segment_index,
                            "metadata": {
                                "layout_reading_order": order,
                                "model_confidence": None,
                                "fallback_method": "plain_text_horizontal_tiles",
                                "fallback_original_label": label,
                                "fallback_tile_index": segment["tile_index"],
                                "fallback_tile_count": segment["tile_count"],
                            },
                        }
                    )
                if label == "Table":
                    table = next(
                        (entry for entry in tables if int(entry.get("reading_order", -1)) == order),
                        None,
                    )
                    if table is None:
                        table = {"page_number": page_number, "reading_order": order, "bbox": bbox}
                        tables.append(table)
                    table.update(
                        {
                            "otsl": "",
                            "html": "",
                            "raw_response": fallback_text,
                            "fallback_plain_text": fallback_text,
                            "fallback_method": "plain_text_horizontal_tiles",
                        }
                    )
                fallback_details.append(
                    {
                        "reading_order": order,
                        "label": label,
                        "resolved": True,
                        "tile_count": segments[0]["tile_count"],
                        "attempts": attempts,
                    }
                )

            remaining = [
                problem
                for problem in truncated
                if int(problem["reading_order"]) not in resolved_orders
            ]
            blocks.sort(key=lambda block: int(block.get("reading_order") or 0))
            normalized_text = "\n".join(
                str(block.get("text") or "").strip()
                for block in blocks
                if str(block.get("text") or "").strip()
            )
            markdown = "\n\n".join(
                str(element.get("native_content") or element.get("text") or "").strip()
                for element in sorted(elements, key=lambda entry: int(entry.get("reading_order", 0)))
                if element.get("block_type") not in {"Picture", "Page-header", "Page-footer"}
                and str(element.get("native_content") or element.get("text") or "").strip()
            )
            main_text = "\n".join(
                str(element.get("text") or "").strip()
                for element in sorted(elements, key=lambda entry: int(entry.get("reading_order", 0)))
                if element.get("block_type")
                in {"Caption", "List-item", "Section-header", "Text", "Title"}
                and str(element.get("text") or "").strip()
            )
            status = "COMPLETED" if not remaining and (markdown or normalized_text) else "PARTIAL"
            record["status"] = status
            record["blocks"] = blocks
            record["layout_elements"] = elements
            record["tables"] = tables
            record["text"] = {
                "text_for_rag": markdown or normalized_text,
                "normalized_block_text": normalized_text,
                "main_text": main_text,
                "native_markdown": markdown,
            }
            metadata["truncated_elements"] = remaining
            metadata["possible_output_truncation"] = bool(remaining)
            metadata["rag_plain_text_fallback"] = fallback_details
            record.setdefault("ocr", {})["processing_metadata"] = metadata
            record.setdefault("warnings", []).append(
                "RAG fallback retried truncated layout elements as plain text; "
                f"resolved={len(resolved_orders)} remaining={len(remaining)}"
            )
            record["fallback"] = {
                "created_at": _utc_now(),
                "method": "plain_text_horizontal_tiles",
                "runtime": runtime_info,
                "details": fallback_details,
                "processing_time_seconds": round(time.perf_counter() - page_started, 6),
            }
            record["processing_time_seconds"] = round(
                float(record.get("processing_time_seconds") or 0.0)
                + float(record["fallback"]["processing_time_seconds"]),
                6,
            )
            _write_json_atomic(result_path, record)
            counts[status] += 1
            if callable(finish_document):
                finish_document(f"fallback-{status.lower()}")
            print(
                f"[{position}/{len(partial_tasks)}] {status} "
                f"{item['document_id']} page={item['page_number']} "
                f"resolved={len(resolved_orders)} remaining={len(remaining)}"
            )
        except KeyboardInterrupt:
            if callable(finish_document):
                finish_document("fallback-interrupted")
            raise
        except Exception as exc:
            counts["FAILED"] += 1
            record["fallback_error"] = {
                "created_at": _utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json_atomic(result_path, record)
            if callable(finish_document):
                finish_document("fallback-failed")
            print(
                f"[{position}/{len(partial_tasks)}] FALLBACK FAILED "
                f"{item['document_id']} page={item['page_number']}: {exc}",
                file=sys.stderr,
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "partial_pages_selected": len(partial_tasks),
        "counts": counts,
        "elapsed_seconds": round(time.perf_counter() - retry_started, 6),
    }
    _write_json_atomic(
        output_root / f"fallback_summary_shard_{args.shard_index}.json", summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if counts["PARTIAL"] == 0 and counts["FAILED"] == 0 else 2


def merge(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    tasks = _read_jsonl(manifest_path)
    _validate_unique_tasks(tasks)
    counts = {"COMPLETED": 0, "PARTIAL": 0, "FAILED": 0, "MISSING": 0}
    all_records: list[dict[str, Any]] = []
    text_sections: list[str] = []

    for item in tasks:
        path = _result_path(output_root, item)
        if not path.is_file():
            counts["MISSING"] += 1
            all_records.append(
                {
                    "status": "MISSING",
                    "document_id": item["document_id"],
                    "version_id": item["version_id"],
                    "relative_path": item["relative_path"],
                    "page_number": item["page_number"],
                }
            )
            continue
        try:
            record = _read_json(path)
        except (OSError, ValueError) as exc:
            record = {
                "status": "FAILED",
                "document_id": item["document_id"],
                "version_id": item["version_id"],
                "relative_path": item["relative_path"],
                "page_number": item["page_number"],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        status = str(record.get("status", "FAILED"))
        counts[status if status in counts else "FAILED"] += 1
        all_records.append(record)
        text = str((record.get("text") or {}).get("text_for_rag") or "").strip()
        if text:
            text_sections.append(
                "\n".join(
                    [
                        "=" * 80,
                        f"DOCUMENT_ID: {item['document_id']}",
                        f"SOURCE: {item['relative_path']}",
                        f"PAGE: {item['page_number']}",
                        f"STATUS: {status}",
                        "=" * 80,
                        text,
                    ]
                )
            )

    jsonl = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in all_records
    )
    _write_text_atomic(output_root / "all_pages.jsonl", jsonl)
    _write_text_atomic(output_root / "all_text.txt", "\n\n".join(text_sections) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "expected_pages": len(tasks),
        "counts": counts,
        "all_pages_sha256": _sha256(output_root / "all_pages.jsonl"),
        "all_text_sha256": _sha256(output_root / "all_text.txt"),
        "complete": counts["COMPLETED"] == len(tasks),
    }
    _write_json_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build a selective input batch")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    run_parser = subparsers.add_parser("run", help="Run one OCR shard")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--input-root", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--cache-root", type=Path, required=True)
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--dpi", type=int, default=300)
    run_parser.add_argument("--shard-index", type=int, required=True)
    run_parser.add_argument("--shard-count", type=int, default=2)
    run_parser.set_defaults(handler=run)

    retry_parser = subparsers.add_parser(
        "retry-partials", help="Retry only truncated elements as plain text"
    )
    retry_parser.add_argument("--manifest", type=Path, required=True)
    retry_parser.add_argument("--input-root", type=Path, required=True)
    retry_parser.add_argument("--output-root", type=Path, required=True)
    retry_parser.add_argument("--cache-root", type=Path, required=True)
    retry_parser.add_argument("--config", type=Path)
    retry_parser.add_argument("--dpi", type=int, default=300)
    retry_parser.add_argument("--max-tokens", type=int, default=2500)
    retry_parser.add_argument("--shard-index", type=int, required=True)
    retry_parser.add_argument("--shard-count", type=int, default=2)
    retry_parser.set_defaults(handler=retry_partials)

    merge_parser = subparsers.add_parser("merge", help="Verify and aggregate OCR results")
    merge_parser.add_argument("--manifest", type=Path, required=True)
    merge_parser.add_argument("--output-root", type=Path, required=True)
    merge_parser.set_defaults(handler=merge)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
