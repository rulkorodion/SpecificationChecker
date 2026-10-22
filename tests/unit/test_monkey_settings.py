from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from app.exceptions import InputValidationError
from app.ocr.monkey_engine import MonkeyOCRParsingEngine
from app.ocr.monkey_settings import resolve_monkeyocr_settings


ROOT = Path(__file__).resolve().parents[2]


def _base_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "default.yaml").read_text("utf-8"))


def test_verified_settings_are_the_only_inference_path() -> None:
    resolved = resolve_monkeyocr_settings(_base_config(), {})
    monkey = resolved["ocr"]["monkeyocr_v2"]
    assert monkey["profile"] == "local_safe"
    assert monkey["attention_backend"] == "sdpa"
    assert monkey["image_input_policy"] == "custom_pre_resize"
    assert monkey["layout_max_pixels"] == 1003520
    assert monkey["recognition_max_pixels"] == 1003520
    assert monkey["transformers_use_kv_cache"] is True
    assert monkey["do_sample"] is False
    assert monkey["layout_max_tokens"] == 4096
    assert monkey["element_max_tokens"] == 5000
    assert resolved["document"]["render_dpi"] == 300


def test_only_safe_quality_limits_can_be_overridden() -> None:
    resolved = resolve_monkeyocr_settings(
        _base_config(),
        {
            "MONKEYOCR_DTYPE": "float16",
            "MONKEYOCR_MAX_PIXELS": "800000",
            "MONKEYOCR_MAX_NEW_TOKENS": "6144",
            "MONKEYOCR_PDF_RENDER_DPI": "360",
        },
    )
    monkey = resolved["ocr"]["monkeyocr_v2"]
    assert monkey["profile"] == "local_safe"
    assert monkey["dtype"] == "float16"
    assert monkey["attention_backend"] == "sdpa"
    assert monkey["transformers_use_kv_cache"] is True
    assert monkey["do_sample"] is False
    assert monkey["layout_max_pixels"] == 800000
    assert monkey["recognition_max_pixels"] == 800000
    assert monkey["layout_max_tokens"] == monkey["element_max_tokens"] == 6144
    assert resolved["document"]["render_dpi"] == 360


@pytest.mark.parametrize("value", ["auto", "0", "-1", "not-a-number"])
def test_invalid_max_pixels_is_rejected(value: str) -> None:
    with pytest.raises(InputValidationError):
        resolve_monkeyocr_settings(
            _base_config(), {"MONKEYOCR_MAX_PIXELS": value}
        )


def test_bfloat16_falls_back_safely(monkeypatch, tmp_path) -> None:
    class Cuda:
        @staticmethod
        def is_bf16_supported():
            return False

    torch = SimpleNamespace(
        float32="float32", float16="float16", bfloat16="bfloat16", cuda=Cuda()
    )
    engine = MonkeyOCRParsingEngine(
        {
            "dtype": "bfloat16",
            "element_cache_dir": str(tmp_path / "elements"),
            "model_call_journal": str(tmp_path / "calls.jsonl"),
        }
    )
    dtype, name = engine._resolve_dtype(torch, True)
    assert (dtype, name) == ("float16", "float16")
    assert any("bfloat16" in item for item in engine.runtime_warnings)


def test_layout_and_crop_use_the_verified_pixel_limit(tmp_path) -> None:
    engine = MonkeyOCRParsingEngine(
        {
            "layout_max_pixels": 1003520,
            "recognition_max_pixels": 1003520,
            "element_cache_dir": str(tmp_path / "elements"),
            "model_call_journal": str(tmp_path / "calls.jsonl"),
        }
    )
    image = Image.new("RGB", (2480, 3508), "white")
    assert engine._prepare_layout_image(image).size == (842, 1191)
    assert engine._prepare_element_image(image).size == (842, 1191)


def test_generate_always_receives_deterministic_baseline_options(
    monkeypatch, tmp_path
) -> None:
    torch = pytest.importorskip("torch")

    class StoppingCriteria:
        pass

    class StoppingCriteriaList(list):
        pass

    transformers = types.ModuleType("transformers")
    transformers.StoppingCriteria = StoppingCriteria
    transformers.StoppingCriteriaList = StoppingCriteriaList
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class Processor:
        image_processor = SimpleNamespace(patch_size=14, merge_size=2)

        def apply_chat_template(self, *_args, **_kwargs):
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "image_grid_thw": torch.tensor([[1, 20, 30]]),
            }

        def decode(self, *_args, **_kwargs):
            return "recognized"

        def batch_decode(self, *_args, **_kwargs):
            return ["recognized"]

    class Model:
        def __init__(self):
            self.options = None

        def parameters(self):
            return iter([torch.zeros(1)])

        def generate(self, **kwargs):
            self.options = kwargs
            suffix = torch.tensor([[20, 21, 22]], device=kwargs["input_ids"].device)
            return torch.cat([kwargs["input_ids"], suffix], dim=1)

    engine = MonkeyOCRParsingEngine(
        {
            "layout_max_pixels": 1003520,
            "recognition_max_pixels": 1003520,
            "element_cache_dir": str(tmp_path / "elements"),
            "model_call_journal": str(tmp_path / "calls.jsonl"),
        }
    )
    engine._torch = torch
    engine._processor = Processor()
    engine._model = Model()
    output = engine._infer(
        Image.new("RGB", (2100, 180), "white"),
        "prompt",
        max_tokens=3,
        stage="recognition",
        page_number=1,
    )
    assert engine._model.options["max_new_tokens"] == 3
    assert engine._model.options["do_sample"] is False
    assert engine._model.options["use_cache"] is True
    assert output.truncated is True
    assert output.finish_reason == "length"
    assert output.visual_tokens == 150
    assert output.processor_input_size == (420, 280)
    assert engine.input_observations[-1]["possible_truncation"] is True
