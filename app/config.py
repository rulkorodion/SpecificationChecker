"""Загрузка и валидация YAML-конфигурации."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from app.exceptions import InputValidationError
from app.ocr.monkey_settings import resolve_monkeyocr_settings

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a runtime override without mutating the base config."""

    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Загрузить default.yaml и наложить пользовательский YAML."""

    try:
        default = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        if path is None or path.resolve() == DEFAULT_CONFIG_PATH.resolve():
            return resolve_monkeyocr_settings(default)
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise InputValidationError(f"Не удалось прочитать конфигурацию: {exc}") from exc
    if not isinstance(default, dict) or not isinstance(override, dict):
        raise InputValidationError("Корень YAML-конфигурации должен быть объектом")
    return resolve_monkeyocr_settings(merge_config(default, override))
