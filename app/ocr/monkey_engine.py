"""Layout-aware MonkeyOCRv2 document parsing.

The implementation follows the official two-stage parsing pipeline: a page is
used only for layout detection, then every textual element is cropped from the
original page and recognized independently.  The legacy END2END whole-page
generation path intentionally does not exist in this module.
"""

from __future__ import annotations

import ast
import gc
import gzip
import hashlib
import html
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.exceptions import OCREngineError
from app.ocr.base import OCREngine
from app.ocr.models import OCRLayoutElement, OCRPageResult, OCRTextBlock
from app.schemas import BoundingBox

LOGGER = logging.getLogger(__name__)

OFFICIAL_PROMPTS = {
    "Caption": "Please output the text content from the image.",
    "List-item": "Please output the text content from the image.",
    "Page-footer": "Please output the text content from the image.",
    "Page-header": "Please output the text content from the image.",
    "Section-header": "Please output the text content from the image.",
    "Text": "Please output the text content from the image.",
    "Title": "Please output the text content from the image.",
    "Formula": "Please write out the expression of the formula in the image using LaTeX format.",
    "Table": "Please extract the table from the image and represent it in OTSL format.",
    "LAYOUT": "Please output the categories and coordinates of the document elements in reading order.",
}

PIPELINE_VERSION = "layout-crops-v1"
PROMPT_VERSION = 2
ADAPTER_VERSION = 4
TEXT_LABELS = {
    "Caption",
    "List-item",
    "Page-footer",
    "Page-header",
    "Section-header",
    "Text",
    "Title",
    "Formula",
    "Table",
}
MAIN_TEXT_LABELS = {"Caption", "List-item", "Section-header", "Text", "Title"}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _balanced_fragments(text: str, left: str, right: str) -> list[str]:
    result: list[str] = []
    depth = 0
    start = -1
    for index, character in enumerate(text):
        if character == left:
            if depth == 0:
                start = index
            depth += 1
        elif character == right and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                result.append(text[start : index + 1])
                start = -1
    return result


def has_complete_layout_payload(raw: str) -> bool:
    """Return True only after a complete, usable layout list was generated.

    MonkeyOCR can omit EOS after the closing layout bracket. Waiting for EOS
    is especially expensive with ``use_cache=False`` on a small GPU. Isolated
    or repaired dictionaries deliberately do not count as completion.
    """

    for fragment in _balanced_fragments(raw, "[", "]"):
        try:
            value = json.loads(fragment)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(fragment)
            except (SyntaxError, ValueError):
                continue
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if (
                isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and item.get("label")
            ):
                return True
    return False


def parse_layout_output(raw: str) -> tuple[list[dict[str, Any]], bool]:
    """Safely parse the official Python/JSON-like LAYOUT response."""

    text = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    candidates = [text, *_balanced_fragments(text, "[", "]")]
    parsed: list[Any] = []
    repaired = False
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            parsed = value
            repaired = candidate != text
            break
    if not parsed:
        repaired = True
        for fragment in _balanced_fragments(text, "{", "}"):
            try:
                value = json.loads(fragment)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(fragment)
                except (SyntaxError, ValueError):
                    continue
            if isinstance(value, dict):
                parsed.append(value)

    items: list[dict[str, Any]] = []
    for value in parsed:
        if not isinstance(value, dict):
            continue
        bbox = value.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            numeric_bbox = [float(coordinate) for coordinate in bbox]
        except (TypeError, ValueError):
            continue
        items.append(
            {
                "bbox": numeric_bbox,
                "label": str(value.get("label") or "Text"),
            }
        )
    return items, repaired


# Kept as a compatibility name for saved diagnostics/tests. In the new parser
# it parses layout only; content is always recognized from original-resolution crops.
parse_monkey_output = parse_layout_output


def layout_bbox_to_original(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    bbox_source_width: int,
    bbox_source_height: int,
    original_page_width: int,
    original_page_height: int,
    coordinate_scale: float = 1000.0,
    padding_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    """Map normalized layout coordinates to a clamped original-page bbox."""

    if coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be positive")
    if min(
        bbox_source_width,
        bbox_source_height,
        original_page_width,
        original_page_height,
    ) <= 0:
        raise ValueError("page dimensions must be positive")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    source_x1 = x1 / coordinate_scale * bbox_source_width
    source_y1 = y1 / coordinate_scale * bbox_source_height
    source_x2 = x2 / coordinate_scale * bbox_source_width
    source_y2 = y2 / coordinate_scale * bbox_source_height
    if source_x1 > source_x2:
        source_x1, source_x2 = source_x2, source_x1
    if source_y1 > source_y2:
        source_y1, source_y2 = source_y2, source_y1
    scale_x = original_page_width / bbox_source_width
    scale_y = original_page_height / bbox_source_height
    mapped = [
        source_x1 * scale_x,
        source_y1 * scale_y,
        source_x2 * scale_x,
        source_y2 * scale_y,
    ]
    pad_x = max(0.0, padding_ratio) * original_page_width
    pad_y = max(0.0, padding_ratio) * original_page_height
    mapped = [mapped[0] - pad_x, mapped[1] - pad_y, mapped[2] + pad_x, mapped[3] + pad_y]
    out_x1 = max(0, min(original_page_width - 1, int(round(mapped[0]))))
    out_y1 = max(0, min(original_page_height - 1, int(round(mapped[1]))))
    out_x2 = max(out_x1 + 1, min(original_page_width, int(round(mapped[2]))))
    out_y2 = max(out_y1 + 1, min(original_page_height, int(round(mapped[3]))))
    return out_x1, out_y1, out_x2, out_y2


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"[ \t]+", " ", value).strip()


def _otsl_rows(content: str) -> list[list[str]]:
    if "<fcel>" not in content and "<ecel>" not in content:
        return []
    rows: list[list[str]] = []
    for raw_row in content.split("<nl>"):
        cells = [
            _clean_text(match.group(1))
            for match in re.finditer(
                r"<(?:fcel|ecel)>(.*?)(?=<(?:fcel|ecel|lcel|ucel|xcel)>|$)",
                raw_row,
                flags=re.S,
            )
        ]
        if cells:
            rows.append(cells)
    return rows


def otsl_to_html(content: str) -> str:
    """Small, safe OTSL renderer preserving the native OTSL alongside HTML."""

    rows = _otsl_rows(content)
    if not rows:
        return content.strip()
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    ) + "</table>"


def _subdivide_lines(
    content: str, bbox: tuple[int, int, int, int]
) -> list[tuple[str, tuple[float, float, float, float]]]:
    lines = [_clean_text(line) for line in content.splitlines()]
    lines = [line for line in lines if line]
    if not lines and _clean_text(content):
        lines = [_clean_text(content)]
    x1, y1, x2, y2 = bbox
    height = (y2 - y1) / max(1, len(lines))
    return [
        (line, (x1, y1 + index * height, x2, y1 + (index + 1) * height))
        for index, line in enumerate(lines)
    ]


def _subdivide_table(
    rows: list[list[str]], bbox: tuple[int, int, int, int]
) -> list[tuple[str, tuple[float, float, float, float]]]:
    x1, y1, x2, y2 = bbox
    row_height = (y2 - y1) / max(1, len(rows))
    pieces: list[tuple[str, tuple[float, float, float, float]]] = []
    for row_index, row in enumerate(rows):
        column_width = (x2 - x1) / max(1, len(row))
        for column_index, text in enumerate(row):
            if text:
                pieces.append(
                    (
                        text,
                        (
                            x1 + column_index * column_width,
                            y1 + row_index * row_height,
                            x1 + (column_index + 1) * column_width,
                            y1 + (row_index + 1) * row_height,
                        ),
                    )
                )
    return pieces


def _repeat_tail_detected(value: str) -> bool:
    tail = (value or "")[-500:]
    for size in range(1, min(100, len(tail) // 2) + 1):
        fragment = tail[-size:]
        if fragment and tail.endswith(fragment * 6):
            return True
    return False


@dataclass(slots=True)
class InferenceOutput:
    text: str
    finish_reason: str | None = None
    truncated: bool = False
    generated_tokens: int | None = None
    input_size: tuple[int, int] | None = None
    processor_input_size: tuple[int, int] | None = None
    visual_tokens: int | None = None
    max_new_tokens: int | None = None


class CropRecognitionCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json.gz"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return None

    def put(self, key: str, payload: dict[str, Any]) -> int:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(path)
        return path.stat().st_size


class MonkeyOCRParsingEngine(OCREngine):
    """Official-prompt, two-stage MonkeyOCR parser for one page at a time."""

    layout_aware = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.profile = str(self.config.get("profile", "local_safe"))
        self.model_id = str(self.config.get("model_id", "zenosai/MonkeyOCRv2-B-Parsing"))
        self.revision = str(self.config.get("revision", "main"))
        self.runtime = "transformers"
        self.requested_device = str(self.config.get("device", "auto"))
        self.actual_device = "uninitialized"
        self.coordinate_scale = float(self.config.get("coordinate_scale", 1000))
        self.layout_max_tokens = int(self.config.get("layout_max_tokens", 4096))
        self.layout_max_seconds = float(self.config.get("layout_max_seconds", 300))
        self.layout_max_pixels = int(self.config.get("layout_max_pixels", 1003520))
        self.element_max_tokens = int(self.config.get("element_max_tokens", 5000))
        self.element_max_seconds = float(self.config.get("element_max_seconds", 180))
        self.recognition_max_pixels = int(self.config.get("recognition_max_pixels", 1003520))
        self.official_processor_max_pixels: int | None = None
        self.image_input_policy = "custom_pre_resize"
        self.crop_padding_ratio = float(self.config.get("crop_padding_ratio", 0.0025))
        self.timeout = int(self.config.get("request_timeout_seconds", 600))
        self.retry_truncated_element = bool(self.config.get("retry_truncated_element", True))
        self.use_kv_cache = True
        self.do_sample = False
        self.requested_dtype = str(self.config.get("dtype", "auto")).lower()
        self.effective_dtype = "uninitialized"
        self.requested_attention_backend = "sdpa"
        self.effective_attention_backend = "sdpa"
        self.pdf_render_dpi = int(self.config.get("pdf_render_dpi", 300))
        self.runtime_warnings: list[str] = []
        self.input_observations: list[dict[str, Any]] = []
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self.initialization_time_seconds = 0.0
        self.ocr_time_seconds = 0.0
        self.layout_time_seconds = 0.0
        self.recognition_time_seconds = 0.0
        self.model_calls = 0
        self.crop_cache_hits = 0
        self.current_document: str | None = None
        cache_dir = Path(self.config.get("element_cache_dir", "runtime/cache/monkey_elements"))
        self.element_cache = CropRecognitionCache(cache_dir.expanduser().resolve())
        self.journal_path = Path(
            self.config.get("model_call_journal", "runtime/logs/monkeyocr_model_calls.jsonl")
        ).expanduser().resolve()

    @property
    def engine_name(self) -> str:
        return f"MonkeyOCRv2-B-Parsing ({PIPELINE_VERSION};{self.runtime}:{self.actual_device})"

    def start_document(self, document: str | Path | None = None) -> None:
        self.layout_time_seconds = 0.0
        self.recognition_time_seconds = 0.0
        self.model_calls = 0
        self.crop_cache_hits = 0
        self.input_observations = []
        self.current_document = str(document) if document is not None else None
        self._journal({
            "document": self.current_document,
            "page": None,
            "element_or_full_document": "full_document",
            "stage": "document",
            "full_model_run": True,
            "cache_hit": False,
            "reason": "layout-aware parsing started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration": 0.0,
            "result": "started",
            "inference_profile": self.profile,
        })

    def finish_document(self, result: str = "completed") -> None:
        self._journal({
            "document": self.current_document,
            "page": None,
            "element_or_full_document": "full_document",
            "stage": "document",
            "full_model_run": False,
            "cache_hit": False,
            "reason": "layout-aware parsing finished",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration": round(self.layout_time_seconds + self.recognition_time_seconds, 6),
            "result": result,
        })

    def _runtime_warning(self, message: str) -> None:
        if message not in self.runtime_warnings:
            self.runtime_warnings.append(message)
        LOGGER.warning("MonkeyOCR: %s", message)

    def _resolve_dtype(self, torch: Any, use_cuda: bool) -> tuple[Any, str]:
        requested = self.requested_dtype
        if not use_cuda:
            if requested != "auto":
                self._runtime_warning(
                    f"dtype={requested} requires CUDA; using float32 on CPU"
                )
            return torch.float32, "float32"
        if requested == "float16":
            return torch.float16, "float16"
        bf16_supported = bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        if requested == "bfloat16" and not bf16_supported:
            self._runtime_warning(
                "bfloat16 is unsupported by this GPU/runtime; using float16"
            )
            return torch.float16, "float16"
        if requested in {"auto", "bfloat16"} and bf16_supported:
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"

    def _prepare_transformers(self) -> None:
        if self._model is not None:
            return
        configured_cache = str(self.config.get("cache_dir", "")).strip()
        if configured_cache:
            cache_root = Path(configured_cache).expanduser().resolve()
        else:
            local_app_data = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
            cache_root = local_app_data / "specification-checker" / "huggingface"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root))
        modules_cache = Path(
            self.config.get("modules_cache_dir", "runtime/cache/huggingface_modules")
        ).expanduser().resolve()
        modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(modules_cache)
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise OCREngineError(
                "Для MonkeyOCR установите зависимости из requirements-monkeyocr.txt"
            ) from exc
        use_cuda = self.requested_device != "cpu" and torch.cuda.is_available()
        if self.requested_device == "gpu:0" and not use_cuda:
            raise OCREngineError("MonkeyOCR: GPU запрошена, но PyTorch CUDA недоступна")
        self.actual_device = "gpu:0" if use_cuda else "cpu"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        dtype, self.effective_dtype = self._resolve_dtype(torch, use_cuda)
        attention_backend = "sdpa"
        local_files_only = bool(self.config.get("local_files_only", True))
        try:
            model_source = self.model_id
            if local_files_only:
                model_source = snapshot_download(
                    repo_id=self.model_id,
                    revision=self.revision,
                    local_files_only=True,
                )
            self._processor = AutoProcessor.from_pretrained(
                model_source,
                revision=None if local_files_only else self.revision,
                trust_remote_code=True,
                local_files_only=local_files_only,
                use_fast=True,
            )
            self.official_processor_max_pixels = int(
                getattr(
                    getattr(self._processor, "image_processor", None),
                    "max_pixels",
                    0,
                )
                or 0
            )

            def load_model(selected_attention: str):
                model_config = AutoConfig.from_pretrained(
                    model_source,
                    revision=None if local_files_only else self.revision,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                setattr(model_config, "_attn_implementation", selected_attention)
                vision_config = getattr(model_config, "vision_config", None)
                if vision_config is not None:
                    setattr(vision_config, "attn_implementation", selected_attention)
                return AutoModelForCausalLM.from_pretrained(
                    model_source,
                    revision=None if local_files_only else self.revision,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                    device_map="auto" if use_cuda else None,
                    dtype=dtype,
                    attn_implementation=selected_attention,
                    config=model_config,
                )

            self._model = load_model(attention_backend)
            if not use_cuda:
                self._model.to("cpu")
            self._model.eval()
            self._torch = torch
            device_map = getattr(self._model, "hf_device_map", {})
            if use_cuda and any(str(value) == "cpu" for value in device_map.values()):
                self.actual_device = "gpu:0+cpu-offload"
            info = self.runtime_info()
            LOGGER.info(
                "MonkeyOCR ready: profile=%s dtype=%s attention=%s use_cache=%s "
                "do_sample=%s processor_max_pixels=%s max_new_tokens=%s/%s "
                "GPU=%s VRAM=%s MiB",
                self.profile,
                self.effective_dtype,
                self.effective_attention_backend,
                self.use_kv_cache,
                self.do_sample,
                self.official_processor_max_pixels,
                self.layout_max_tokens,
                self.element_max_tokens,
                info.get("gpu_model") or "n/a",
                info.get("gpu_memory_mib") or "n/a",
            )
        except Exception as exc:
            self._model = None
            self._processor = None
            raise OCREngineError(f"MonkeyOCR не инициализировался: {exc}") from exc

    def prepare(self) -> dict[str, Any]:
        started = time.perf_counter()
        self._prepare_transformers()
        self.initialization_time_seconds += time.perf_counter() - started
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        gpu_name = None
        gpu_memory = None
        peak_vram = None
        try:
            import torch
            if torch.cuda.is_available():
                properties = torch.cuda.get_device_properties(0)
                gpu_name = properties.name
                gpu_memory = int(properties.total_memory / (1024**2))
                peak_vram = int(torch.cuda.max_memory_allocated(0) / (1024**2))
        except ImportError:
            pass
        return {
            "backend": "monkeyocr_v2",
            "pipeline_version": PIPELINE_VERSION,
            "parsing_mode": "layout_then_original_crops",
            "runtime": self.runtime,
            "model_id": self.model_id,
            "model_revision": self.revision,
            "inference_profile": self.profile,
            "requested_dtype": self.requested_dtype,
            "effective_dtype": self.effective_dtype,
            "requested_attention_backend": self.requested_attention_backend,
            "effective_attention_backend": self.effective_attention_backend,
            "image_input_policy": self.image_input_policy,
            "custom_layout_max_pixels": self.layout_max_pixels,
            "custom_recognition_max_pixels": self.recognition_max_pixels,
            "official_processor_max_pixels": self.official_processor_max_pixels,
            "pdf_render_dpi": self.pdf_render_dpi,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "gpu_model": gpu_name,
            "gpu_memory_mib": gpu_memory,
            "peak_vram_mib": peak_vram,
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "processor_use_fast": True,
            "transformers_use_kv_cache": self.use_kv_cache,
            "do_sample": self.do_sample,
            "layout_max_new_tokens": self.layout_max_tokens,
            "element_max_new_tokens": self.element_max_tokens,
            "layout_max_seconds": self.layout_max_seconds,
            "element_max_seconds": self.element_max_seconds,
            "confidence_source": "model-unscored",
            "model_initialization_seconds": round(self.initialization_time_seconds, 6),
            "ocr_seconds": round(self.ocr_time_seconds, 6),
            "layout_seconds": round(self.layout_time_seconds, 6),
            "recognition_seconds": round(self.recognition_time_seconds, 6),
            "model_calls": self.model_calls,
            "crop_cache_hits": self.crop_cache_hits,
            "runtime_warnings": list(self.runtime_warnings),
            "fallback_reason": "; ".join(self.runtime_warnings) or None,
            "input_observations": list(self.input_observations),
        }

    def _journal(self, payload: dict[str, Any]) -> None:
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _prepare_element_image(self, image: Image.Image) -> Image.Image:
        pixels = image.width * image.height
        if self.recognition_max_pixels > 0 and pixels > self.recognition_max_pixels:
            scale = (self.recognition_max_pixels / pixels) ** 0.5
            return image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    def _prepare_layout_image(self, image: Image.Image) -> Image.Image:
        """Resize only the all-page layout preview; never use it for text crops."""

        pixels = image.width * image.height
        if self.layout_max_pixels > 0 and pixels > self.layout_max_pixels:
            scale = (self.layout_max_pixels / pixels) ** 0.5
            return image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    def _processor_metrics(
        self, inputs: dict[str, Any]
    ) -> tuple[tuple[int, int] | None, int | None]:
        grid = inputs.get("image_grid_thw")
        if grid is None or not hasattr(grid, "tolist"):
            return None, None
        rows = grid.tolist()
        if not rows:
            return None, None
        image_processor = getattr(self._processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 14))
        merge_size = max(1, int(getattr(image_processor, "merge_size", 2)))
        visual_tokens = 0
        processor_size: tuple[int, int] | None = None
        for row in rows:
            if len(row) < 3:
                continue
            temporal, grid_height, grid_width = (int(value) for value in row[:3])
            visual_tokens += (
                temporal * grid_height * grid_width // (merge_size * merge_size)
            )
            if processor_size is None:
                processor_size = (
                    grid_width * patch_size,
                    grid_height * patch_size,
                )
        return processor_size, visual_tokens or None

    def _infer(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_tokens: int,
        stage: str,
        page_number: int,
        element_index: int | None = None,
        element_label: str | None = None,
    ) -> InferenceOutput:
        original_size = image.size
        if stage == "layout":
            image = self._prepare_layout_image(image)
        else:
            image = self._prepare_element_image(image)
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        self.model_calls += 1
        call_result = "failed"
        output: InferenceOutput | None = None
        observation: dict[str, Any] | None = None
        try:
            self._prepare_transformers()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            processor_input_size, visual_tokens = self._processor_metrics(inputs)
            observation = {
                    "stage": stage,
                    "page_number": page_number,
                    "element_index": element_index,
                    "element_label": element_label,
                    "original_input_size": list(original_size),
                    "application_input_size": list(image.size),
                    "processor_input_size": (
                        list(processor_input_size) if processor_input_size else None
                    ),
                    "visual_tokens": visual_tokens,
                    "max_new_tokens": max_tokens,
                }
            self.input_observations.append(observation)
            LOGGER.info(
                    "MonkeyOCR input: profile=%s stage=%s original=%sx%s app=%sx%s "
                    "processor=%s visual_tokens=%s max_new_tokens=%s",
                    self.profile,
                    stage,
                    original_size[0],
                    original_size[1],
                    image.size[0],
                    image.size[1],
                    processor_input_size or "n/a",
                    visual_tokens or "n/a",
                    max_tokens,
                )
            model_device = next(self._model.parameters()).device
            inputs = {
                    key: value.to(model_device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            input_length = int(inputs["input_ids"].shape[1])
            generation_options: dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    "do_sample": self.do_sample,
                    # The model's checkpoint defaults this to False, which
                    # makes every generated token recompute the full visual
                    # prefix. The cache is configurable, but enabled locally:
                    # for a 1 MP spec5 layout its calculated footprint is only
                    # about 300 MiB and it avoids quadratic decoding time.
                    "use_cache": self.use_kv_cache,
                }
            from transformers import StoppingCriteria, StoppingCriteriaList

            processor = self._processor
            stopping_criteria: list[StoppingCriteria] = []

            class _RepeatedTailCriteria(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs) -> bool:
                        generated = input_ids[0, input_length:]
                        token_count = int(generated.shape[0])
                        if token_count < 96 or token_count % 16:
                            return False
                        current = processor.decode(
                            generated,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        return _repeat_tail_detected(current)

            stopping_criteria.append(_RepeatedTailCriteria())
            if stage == "layout":

                class _CompleteLayoutCriteria(StoppingCriteria):
                        def __call__(self, input_ids, scores, **kwargs) -> bool:
                            generated = input_ids[0, input_length:]
                            current = processor.decode(
                                generated,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                            return has_complete_layout_payload(current)

                stopping_criteria.append(_CompleteLayoutCriteria())
                if self.layout_max_seconds > 0:
                    generation_options["max_time"] = self.layout_max_seconds
            elif self.element_max_seconds > 0:
                generation_options["max_time"] = self.element_max_seconds
            generation_options["stopping_criteria"] = StoppingCriteriaList(
                stopping_criteria
            )
            generation_started = time.perf_counter()
            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    **generation_options,
                )
            generation_elapsed = time.perf_counter() - generation_started
            generated_ids = output_ids[:, input_length:]
            generated_tokens = int(generated_ids.shape[1])
            text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            repeated = _repeat_tail_detected(text)
            layout_complete = stage != "layout" or has_complete_layout_payload(text)
            configured_time_limit = (
                self.layout_max_seconds if stage == "layout" else self.element_max_seconds
            )
            time_limit_reached = (
                configured_time_limit > 0
                and generation_elapsed >= configured_time_limit * 0.98
            )
            truncated = (
                generated_tokens >= max_tokens
                or repeated
                or time_limit_reached
                or not layout_complete
            )
            output = InferenceOutput(
                text=text,
                finish_reason=(
                    "repetition" if repeated
                    else "time_limit" if time_limit_reached
                    else "incomplete" if stage == "layout" and not layout_complete
                    else "length" if truncated
                    else "structured_stop" if stage == "layout"
                    else "stop"
                ),
                truncated=truncated,
                generated_tokens=generated_tokens,
                input_size=image.size,
                processor_input_size=processor_input_size,
                visual_tokens=visual_tokens,
                max_new_tokens=max_tokens,
            )
            observation["generated_tokens"] = generated_tokens
            observation["finish_reason"] = output.finish_reason
            observation["possible_truncation"] = output.truncated
            del generated_ids, output_ids, inputs
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            call_result = "completed"
            return output
        finally:
            elapsed = time.perf_counter() - started
            self.ocr_time_seconds += elapsed
            if stage == "layout":
                self.layout_time_seconds += elapsed
            else:
                self.recognition_time_seconds += elapsed
            journal_record = {
                "document": self.current_document,
                "page": page_number,
                "element_or_full_document": (
                    f"element_{element_index}_{element_label}" if element_index is not None else "page_layout"
                ),
                "stage": stage,
                "full_model_run": False,
                "cache_hit": False,
                "reason": "layout detection" if stage == "layout" else "element recognition",
                "started_at": started_at.isoformat(),
                "duration": round(elapsed, 6),
                "result": call_result,
                "original_input_size": list(original_size),
                "model_input_size": list(image.size),
                "processor_input_size": (
                    list(output.processor_input_size)
                    if output is not None and output.processor_input_size
                    else None
                ),
                "visual_tokens": output.visual_tokens if output is not None else None,
                "max_new_tokens": max_tokens,
                "use_kv_cache": self.use_kv_cache,
                "do_sample": self.do_sample,
                "generated_tokens": output.generated_tokens if output is not None else 0,
                "finish_reason": output.finish_reason if output is not None else None,
                "structured_complete": (
                    has_complete_layout_payload(output.text)
                    if stage == "layout" and output is not None
                    else None
                ),
            }
            if stage == "layout" and output is not None and output.truncated:
                journal_record["raw_incomplete_response"] = output.text
            self._journal(journal_record)

    def _crop_cache_key(self, crop_sha256: str, label: str, prompt: str) -> str:
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "runtime_prompt": prompt,
            "label": label,
            "crop_sha256": crop_sha256,
            "recognition_max_pixels": self.recognition_max_pixels,
            "element_max_tokens": self.element_max_tokens,
            "transformers_use_kv_cache": self.use_kv_cache,
            "do_sample": self.do_sample,
            "profile": self.profile,
            "dtype": self.requested_dtype,
            "attention_backend": self.requested_attention_backend,
            "image_input_policy": self.image_input_policy,
            "processor_use_fast": True,
            "transformers_version": _package_version("transformers"),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _crop_hash(crop: Image.Image) -> str:
        buffer = io.BytesIO()
        crop.convert("RGB").save(buffer, format="PNG", optimize=True)
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    def _recognize_element(
        self,
        crop: Image.Image,
        label: str,
        page_number: int,
        element_index: int,
    ) -> tuple[InferenceOutput, str, bool]:
        prompt = OFFICIAL_PROMPTS[label]
        crop_sha256 = self._crop_hash(crop)
        key = self._crop_cache_key(crop_sha256, label, prompt)
        cached = self.element_cache.get(key)
        if cached is not None:
            self.crop_cache_hits += 1
            self._journal({
                "document": self.current_document,
                "page": page_number,
                "element_or_full_document": f"element_{element_index}_{label}",
                "stage": "recognition",
                "full_model_run": False,
                "cache_hit": True,
                "reason": "unchanged crop hash",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "duration": 0.0,
                "result": "reused",
            })
            return InferenceOutput(**cached["output"]), crop_sha256, True
        output = self._infer(
            crop,
            prompt,
            max_tokens=self.element_max_tokens,
            stage="recognition",
            page_number=page_number,
            element_index=element_index,
            element_label=label,
        )
        if (
            output.truncated
            and output.finish_reason == "length"
            and self.retry_truncated_element
            and label in {"Table", "Formula"}
        ):
            output = self._infer(
                crop,
                prompt,
                max_tokens=min(self.element_max_tokens * 2, 10000),
                stage="recognition",
                page_number=page_number,
                element_index=element_index,
                element_label=label,
            )
        if not output.truncated:
            self.element_cache.put(
                key,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "output": {
                        "text": output.text,
                        "finish_reason": output.finish_reason,
                        "truncated": output.truncated,
                        "generated_tokens": output.generated_tokens,
                        "input_size": output.input_size,
                        "processor_input_size": output.processor_input_size,
                        "visual_tokens": output.visual_tokens,
                        "max_new_tokens": output.max_new_tokens,
                    },
                },
            )
        return output, crop_sha256, False

    @staticmethod
    def _native_content(label: str, raw: str) -> str:
        content = (raw or "").strip()
        if label == "Table":
            return otsl_to_html(content)
        if label == "Formula":
            return f"$$\n{content.strip('$').strip()}\n$$" if content else ""
        if label == "Title":
            return "# " + content.replace("\n", "\n# ") if content else ""
        if label == "Section-header":
            return "## " + content.replace("\n", "\n## ") if content else ""
        return content

    def recognize_page(
        self, image: np.ndarray, page_number: int, preprocessing_profile: str
    ) -> OCRPageResult:
        height, width = image.shape[:2]
        original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            layout_output = self._infer(
                original,
                OFFICIAL_PROMPTS["LAYOUT"],
                max_tokens=self.layout_max_tokens,
                stage="layout",
                page_number=page_number,
            )
        except OCREngineError:
            raise
        except Exception as exc:
            raise OCREngineError(
                f"Ошибка MonkeyOCR layout на странице {page_number}: {exc}"
            ) from exc
        if layout_output.truncated:
            raise OCREngineError(
                "MonkeyOCR не завершил корректный layout JSON "
                f"на странице {page_number} (finish_reason={layout_output.finish_reason})"
            )
        items, repaired = parse_layout_output(layout_output.text)
        if not items:
            raise OCREngineError(
                f"MonkeyOCR не вернул layout-элементы на странице {page_number}"
            )

        blocks: list[OCRTextBlock] = []
        elements: list[OCRLayoutElement] = []
        native_tables: list[dict[str, Any]] = []
        markdown_parts: list[str] = []
        main_text_parts: list[str] = []
        element_times: list[dict[str, Any]] = []
        truncated_elements: list[dict[str, Any]] = []
        layout_source_width, layout_source_height = (
            layout_output.input_size or original.size
        )
        for index, item in enumerate(items):
            label = item["label"]
            bbox = layout_bbox_to_original(
                item["bbox"],
                bbox_source_width=layout_source_width,
                bbox_source_height=layout_source_height,
                original_page_width=width,
                original_page_height=height,
                coordinate_scale=self.coordinate_scale,
                padding_ratio=self.crop_padding_ratio,
            )
            x1, y1, x2, y2 = bbox
            crop = original.crop(bbox)
            raw = ""
            crop_sha256 = self._crop_hash(crop)
            cache_hit = False
            finish_reason = None
            truncated = False
            recognition_started = time.perf_counter()
            if label in TEXT_LABELS:
                output, crop_sha256, cache_hit = self._recognize_element(
                    crop, label, page_number, index
                )
                diagnostic_raw = output.text.strip()
                raw = "" if output.truncated else diagnostic_raw
                finish_reason = output.finish_reason
                truncated = output.truncated
                if truncated:
                    truncated_elements.append(
                        {
                            "reading_order": index,
                            "label": label,
                            "finish_reason": output.finish_reason,
                            "generated_tokens": output.generated_tokens,
                            "max_new_tokens": output.max_new_tokens,
                        }
                    )
            else:
                diagnostic_raw = ""
            native = self._native_content(label, raw)
            metadata = {
                "finish_reason": finish_reason,
                "truncated": truncated,
                "crop_cache_hit": cache_hit,
                "crop_width": crop.width,
                "crop_height": crop.height,
                "recognition_input_policy": self.image_input_policy,
                "inference_profile": self.profile,
                "application_input_size": (
                    list(output.input_size)
                    if label in TEXT_LABELS and output.input_size
                    else None
                ),
                "processor_input_size": (
                    list(output.processor_input_size)
                    if label in TEXT_LABELS and output.processor_input_size
                    else None
                ),
                "visual_tokens": output.visual_tokens if label in TEXT_LABELS else None,
                "max_new_tokens": output.max_new_tokens if label in TEXT_LABELS else None,
                "possible_output_truncation": truncated,
            }
            element = OCRLayoutElement(
                block_type=label,
                page_number=page_number,
                reading_order=index,
                bbox_source_width=layout_source_width,
                bbox_source_height=layout_source_height,
                original_page_width=width,
                original_page_height=height,
                normalized_bbox=item["bbox"],
                original_pixel_bbox=list(bbox),
                text=_clean_text(raw) if label != "Table" else _clean_text(native),
                raw_response=diagnostic_raw,
                native_content=native,
                crop_sha256=crop_sha256,
                model_confidence=None,
                metadata=metadata,
            )
            elements.append(element)
            element_times.append({
                "reading_order": index,
                "label": label,
                "seconds": round(time.perf_counter() - recognition_started, 6),
                "cache_hit": cache_hit,
            })
            if label == "Picture":
                continue
            if label not in {"Page-header", "Page-footer"} and native:
                markdown_parts.append(native)
            if label in MAIN_TEXT_LABELS and raw:
                main_text_parts.append(_clean_text(raw))

            rows = _otsl_rows(raw) if label == "Table" else []
            if label == "Table":
                native_tables.append({
                    "page_number": page_number,
                    "reading_order": index,
                    "bbox": list(bbox),
                    "otsl": raw,
                    "html": native,
                    "raw_response": raw,
                })
            pieces = _subdivide_table(rows, bbox) if rows else _subdivide_lines(raw, bbox)
            for sub_index, (text, piece_bbox) in enumerate(pieces):
                blocks.append(OCRTextBlock(
                    text=text,
                    confidence=None,
                    bounding_box=BoundingBox(
                        x_min=piece_bbox[0], y_min=piece_bbox[1],
                        x_max=piece_bbox[2], y_max=piece_bbox[3],
                        page_width=width, page_height=height,
                    ),
                    page_number=page_number,
                    preprocessing_profile=preprocessing_profile,
                    engine_name=self.engine_name,
                    block_type=label,
                    source="monkeyocr-layout-crop",
                    reading_order=index * 1000 + sub_index,
                    metadata={
                        "layout_reading_order": index,
                        "raw_response": raw,
                        "native_content": native,
                        "model_confidence": None,
                        **metadata,
                    },
                ))

        warnings = [
            "MonkeyOCR не публикует confidence для layout/content блоков; model_confidence=null"
        ]
        if repaired:
            warnings.append("Layout-ответ MonkeyOCR был частично восстановлен")
        if truncated_elements:
            warnings.append(
                "MonkeyOCR reached a generation limit for one or more elements; "
                "their text was excluded as possibly truncated"
            )
        return OCRPageResult(
            page_number=page_number,
            width=width,
            height=height,
            blocks=blocks,
            preprocessing_profile=preprocessing_profile,
            engine_name=self.engine_name,
            warnings=warnings,
            native_markdown="\n\n".join(markdown_parts).strip(),
            native_tables=native_tables,
            layout_elements=elements,
            processing_metadata={
                "pipeline_version": PIPELINE_VERSION,
                "layout_raw": layout_output.text,
                "layout_repaired": repaired,
                "layout_finish_reason": layout_output.finish_reason,
                "layout_input_size": list(layout_output.input_size or original.size),
                "original_page_size": list(original.size),
                "inference_profile": self.profile,
                "inference_settings": self.runtime_info(),
                "possible_output_truncation": bool(truncated_elements),
                "truncated_elements": truncated_elements,
                "full_text": "\n".join(main_text_parts),
                "timings": {
                    "layout_seconds_total": round(self.layout_time_seconds, 6),
                    "recognition_seconds_total": round(self.recognition_time_seconds, 6),
                    "elements": element_times,
                },
            },
        )


# Public name retained for callers; it now points only to the two-stage parser.
MonkeyOCRv2Engine = MonkeyOCRParsingEngine
