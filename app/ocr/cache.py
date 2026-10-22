"""Компактный gzip-кэш результатов дорогой стадии OCR."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.document.loader import LoadedDocument
from app.ocr.models import OCRDocumentResult, OCRPageResult

CACHE_FORMAT_VERSION = 3


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def cache_fingerprint(loaded: LoadedDocument, config: dict[str, Any]) -> dict[str, Any]:
    """Вернуть все параметры, изменение которых делает OCR-кэш невалидным."""

    document = config["document"]
    ocr = config["ocr"]
    backend = str(ocr.get("backend", "monkeyocr_v2")).lower().replace("-", "_")
    ocr_fingerprint: dict[str, Any] = {
        "backend": backend,
        "language": str(ocr["language"]),
    }
    if backend == "monkeyocr_v2":
        monkey = ocr.get("monkeyocr_v2", {})
        ocr_fingerprint.update(
            {
                "model_id": str(
                    monkey.get("model_id", "zenosai/MonkeyOCRv2-B-Parsing")
                ),
                "revision": str(monkey.get("revision", "main")),
                "runtime": str(monkey.get("runtime", "auto")),
                "pipeline_version": str(
                    monkey.get("pipeline_version", "layout-crops-v1")
                ),
                "parsing_mode": str(
                    monkey.get("parsing_mode", "layout_then_original_crops")
                ),
                "prompt_version": int(monkey.get("prompt_version", 2)),
                "adapter_version": int(monkey.get("adapter_version", 4)),
                "coordinate_scale": float(monkey.get("coordinate_scale", 1000)),
                "inference_profile": str(monkey.get("profile", "local_safe")),
                "dtype": str(monkey.get("dtype", "auto")),
                "attention_backend": str(
                    monkey.get("attention_backend", "sdpa")
                ),
                "image_input_policy": str(
                    monkey.get("image_input_policy", "custom_pre_resize")
                ),
                "processor_max_pixels": monkey.get("processor_max_pixels"),
                "do_sample": bool(monkey.get("do_sample", False)),
                "layout_max_tokens": int(monkey.get("layout_max_tokens", 4096)),
                "layout_max_seconds": float(monkey.get("layout_max_seconds", 300)),
                "transformers_use_kv_cache": bool(
                    monkey.get("transformers_use_kv_cache", True)
                ),
                "layout_max_pixels": int(
                    monkey.get("layout_max_pixels", 1003520)
                ),
                "element_max_tokens": int(monkey.get("element_max_tokens", 5000)),
                "element_max_seconds": float(monkey.get("element_max_seconds", 180)),
                "recognition_max_pixels": int(
                    monkey.get("recognition_max_pixels", 1003520)
                ),
                "crop_padding_ratio": float(
                    monkey.get("crop_padding_ratio", 0.0025)
                ),
                "processor_use_fast": True,
                "transformers_version": _package_version("transformers"),
                "torch_version": _package_version("torch"),
            }
        )
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "document_sha256": loaded.sha256,
        "rendering": {
            "dpi": int(document["render_dpi"]),
            "max_pages": int(document["max_pages"]),
        },
        "preprocessing": {
            "implementation_version": int(ocr.get("preprocessing_version", 1)),
            "retry_with_preprocessing": bool(ocr["retry_with_preprocessing"]),
            "min_text_confidence": float(ocr["min_text_confidence"]),
            "profiles": list(ocr.get("preprocessing_profiles", [])),
        },
        "ocr": ocr_fingerprint,
    }


def cache_key(fingerprint: dict[str, Any]) -> str:
    canonical = json.dumps(
        fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class OCRCache:
    """LRU/retention-кэш без полноразмерных изображений."""

    def __init__(
        self,
        root: Path,
        max_size_mb: int = 1024,
        retention_days: int = 30,
        cleanup_on_start: bool = True,
    ) -> None:
        self.root = root
        self.max_size_bytes = max(0, int(max_size_mb)) * 1024 * 1024
        self.retention_days = max(0, int(retention_days))
        self.root.mkdir(parents=True, exist_ok=True)
        if cleanup_on_start:
            self.cleanup()

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json.gz"

    def get(
        self, fingerprint: dict[str, Any]
    ) -> tuple[OCRDocumentResult | None, dict[str, Any]]:
        key = cache_key(fingerprint)
        path = self.path_for(key)
        metadata = {"key": key, "path": str(path), "hit": False, "size_bytes": 0}
        if not path.is_file():
            return None, metadata
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("fingerprint") != fingerprint:
                return None, metadata
            document = OCRDocumentResult.model_validate(payload["ocr_document"])
            now = time.time()
            os.utime(path, (now, path.stat().st_mtime))
            metadata.update(
                {
                    "hit": True,
                    "size_bytes": path.stat().st_size,
                    "created_at": payload.get("created_at"),
                }
            )
            return document, metadata
        except (OSError, ValueError, KeyError):
            return None, metadata

    def put(
        self,
        fingerprint: dict[str, Any],
        document: OCRDocumentResult,
        source: str = "ocr",
    ) -> dict[str, Any]:
        key = cache_key(fingerprint)
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "fingerprint": fingerprint,
            "page_sizes": [
                {"page": page.page_number, "width": page.width, "height": page.height}
                for page in document.pages
            ],
            "ocr_document": document.model_dump(mode="json"),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(path)
        self.cleanup()
        return {
            "key": key,
            "path": str(path),
            "hit": False,
            "created": True,
            "size_bytes": path.stat().st_size,
        }

    def import_saved_run(
        self,
        loaded: LoadedDocument,
        config: dict[str, Any],
        saved_output: Path,
    ) -> dict[str, Any]:
        """Импортировать ранее сохранённые page_*.json без запуска OCR-модели."""

        ocr_dir = saved_output / "ocr"
        page_paths = sorted(ocr_dir.glob("page_*.json"))
        if not page_paths:
            raise ValueError(f"В {ocr_dir} нет сохранённых OCR page_*.json")
        pages = [
            OCRPageResult.model_validate_json(path.read_text(encoding="utf-8"))
            for path in page_paths
        ]
        full_text_path = ocr_dir / "full_text.txt"
        full_text = (
            full_text_path.read_text(encoding="utf-8")
            if full_text_path.is_file()
            else "\n".join(block.text for page in pages for block in page.blocks)
        )
        document = OCRDocumentResult(
            pages=pages,
            normalized_full_text=full_text,
            engine_name=pages[0].engine_name if pages else "saved-ocr",
        )
        return self.put(cache_fingerprint(loaded, config), document, "saved-run-import")

    def cleanup(self, dry_run: bool = False) -> dict[str, int]:
        files = list(self.root.glob("*/*.json.gz"))
        now = time.time()
        expired_before = (
            now - self.retention_days * 86400 if self.retention_days else None
        )
        victims: list[Path] = []
        if expired_before is not None:
            victims.extend(
                path for path in files if path.stat().st_mtime < expired_before
            )
        survivors = [path for path in files if path not in victims]
        survivor_size = sum(path.stat().st_size for path in survivors)
        if self.max_size_bytes and survivor_size > self.max_size_bytes:
            for path in sorted(survivors, key=lambda item: item.stat().st_atime):
                if survivor_size <= self.max_size_bytes:
                    break
                victims.append(path)
                survivor_size -= path.stat().st_size
        freed = sum(path.stat().st_size for path in victims)
        if not dry_run:
            for path in victims:
                path.unlink(missing_ok=True)
        return {"removed_files": len(victims), "freed_bytes": freed}

    def total_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.glob("*/*.json.gz"))
