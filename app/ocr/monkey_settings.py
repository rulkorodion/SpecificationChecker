"""Resolve the single verified MonkeyOCR inference configuration."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Mapping

from app.exceptions import InputValidationError

VALID_DTYPES = {"auto", "bfloat16", "float16"}


def _positive_integer(value: Any, *, name: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise InputValidationError(f"{name} must be a positive integer")
    return result


def resolve_monkeyocr_settings(
    config: dict[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply safe environment overrides to the verified local inference path.

    There is deliberately no profile switch here.  MonkeyOCR always uses SDPA,
    the KV cache, deterministic generation and the validated one-megapixel
    application-side resize for both layout and individual crops.
    """

    result = deepcopy(config)
    env = os.environ if environ is None else environ
    monkey = result.setdefault("ocr", {}).setdefault("monkeyocr_v2", {})

    dtype = str(env.get("MONKEYOCR_DTYPE", monkey.get("dtype", "auto"))).strip().lower()
    if dtype not in VALID_DTYPES:
        raise InputValidationError(
            "MONKEYOCR_DTYPE must be auto, bfloat16 or float16"
        )
    max_pixels = _positive_integer(
        env.get("MONKEYOCR_MAX_PIXELS", monkey.get("max_pixels", 1003520)),
        name="MONKEYOCR_MAX_PIXELS",
    )
    render_dpi = _positive_integer(
        env.get(
            "MONKEYOCR_PDF_RENDER_DPI",
            monkey.get(
                "pdf_render_dpi",
                result.get("document", {}).get("render_dpi", 300),
            ),
        ),
        name="MONKEYOCR_PDF_RENDER_DPI",
    )
    max_new_tokens_raw = env.get("MONKEYOCR_MAX_NEW_TOKENS", "").strip()
    if max_new_tokens_raw:
        max_new_tokens = _positive_integer(
            max_new_tokens_raw, name="MONKEYOCR_MAX_NEW_TOKENS"
        )
        layout_max_tokens = max_new_tokens
        element_max_tokens = max_new_tokens
    else:
        layout_max_tokens = _positive_integer(
            monkey.get("layout_max_tokens", 4096), name="layout_max_tokens"
        )
        element_max_tokens = _positive_integer(
            monkey.get("element_max_tokens", 5000), name="element_max_tokens"
        )

    monkey.update(
        {
            "profile": "local_safe",
            "dtype": dtype,
            "attention_backend": "sdpa",
            "image_input_policy": "custom_pre_resize",
            "max_pixels": max_pixels,
            "layout_max_pixels": max_pixels,
            "recognition_max_pixels": max_pixels,
            "transformers_use_kv_cache": True,
            "do_sample": False,
            "layout_max_tokens": layout_max_tokens,
            "element_max_tokens": element_max_tokens,
            "pdf_render_dpi": render_dpi,
        }
    )
    result.setdefault("document", {})["render_dpi"] = render_dpi
    return result
