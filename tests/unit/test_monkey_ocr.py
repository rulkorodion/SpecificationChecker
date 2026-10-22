from __future__ import annotations

import numpy as np
from PIL import Image

from app.ocr.monkey_engine import (
    InferenceOutput,
    MonkeyOCRParsingEngine,
    has_complete_layout_payload,
    layout_bbox_to_original,
    parse_layout_output,
)
from app.ocr.models import OCRDocumentResult
from app.ocr.monkey_compare import compare_saved_monkey


class StubParsingEngine(MonkeyOCRParsingEngine):
    def __init__(self, tmp_path, layout: str, recognition: dict[str, str]) -> None:
        super().__init__(
            {
                "runtime": "transformers",
                "device": "cpu",
                "crop_padding_ratio": 0.0,
                "element_cache_dir": str(tmp_path / "elements"),
                "model_call_journal": str(tmp_path / "calls.jsonl"),
            }
        )
        self.layout_raw = layout
        self.recognition = recognition
        self.calls: list[dict] = []
        self.actual_device = "cpu"

    def _infer(self, image: Image.Image, prompt: str, **kwargs) -> InferenceOutput:
        stage = kwargs["stage"]
        label = kwargs.get("element_label")
        self.calls.append({"stage": stage, "label": label, "size": image.size})
        text = self.layout_raw if stage == "layout" else self.recognition[label]
        return InferenceOutput(text=text, finish_reason="stop", input_size=image.size)


class SequencedInferenceEngine(MonkeyOCRParsingEngine):
    def __init__(self, tmp_path, outputs: list[InferenceOutput]) -> None:
        super().__init__(
            {
                "runtime": "transformers",
                "element_cache_dir": str(tmp_path / "elements"),
                "model_call_journal": str(tmp_path / "calls.jsonl"),
            }
        )
        self.outputs = list(outputs)
        self.infer_calls = 0

    def _infer(self, image: Image.Image, prompt: str, **kwargs) -> InferenceOutput:
        self.infer_calls += 1
        return self.outputs.pop(0)


def test_parse_layout_output_accepts_safe_python_payload_and_discards_content():
    items, repaired = parse_layout_output(
        "[{'bbox': [10, 20, 900, 80], 'label': 'Text', 'content': 'ignored'}]"
    )
    assert repaired is False
    assert items == [{"bbox": [10.0, 20.0, 900.0, 80.0], "label": "Text"}]


def test_parse_layout_output_recovers_a_complete_dict_without_execution():
    items, repaired = parse_layout_output(
        "noise {'bbox': [1, 2, 3, 4], 'label': 'Text'} tail"
    )
    assert repaired is True
    assert items[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_layout_early_stop_requires_complete_balanced_list():
    complete = "```json\n[{'bbox': [1, 2, 3, 4], 'label': 'Text'}]"
    assert has_complete_layout_payload(complete) is True
    assert has_complete_layout_payload(complete[:-1]) is False
    assert has_complete_layout_payload(
        "noise {'bbox': [1, 2, 3, 4], 'label': 'Text'}"
    ) is False


def test_local_transformers_kv_cache_is_enabled_by_default(tmp_path):
    engine = MonkeyOCRParsingEngine(
        {
            "runtime": "transformers",
            "element_cache_dir": str(tmp_path / "elements"),
            "model_call_journal": str(tmp_path / "calls.jsonl"),
        }
    )
    assert engine.use_kv_cache is True


def test_repetitive_text_is_not_retried_or_cached(tmp_path):
    engine = SequencedInferenceEngine(
        tmp_path,
        [InferenceOutput("повтор " * 100, "repetition", True, 192, (400, 50))],
    )
    output, _, cache_hit = engine._recognize_element(
        Image.new("RGB", (400, 50), "white"), "Text", 1, 7
    )
    assert output.finish_reason == "repetition"
    assert engine.infer_calls == 1
    assert cache_hit is False
    assert list((tmp_path / "elements").rglob("*.json.gz")) == []


def test_only_length_truncated_table_gets_one_retry(tmp_path):
    engine = SequencedInferenceEngine(
        tmp_path,
        [
            InferenceOutput("partial", "length", True, 5000, (800, 400)),
            InferenceOutput("<fcel>Итого<fcel>10", "stop", False, 20, (800, 400)),
        ],
    )
    output, _, _ = engine._recognize_element(
        Image.new("RGB", (800, 400), "white"), "Table", 1, 8
    )
    assert output.truncated is False
    assert engine.infer_calls == 2
    assert len(list((tmp_path / "elements").rglob("*.json.gz"))) == 1


def test_coordinate_transform_maps_to_original_with_padding_and_clamp():
    assert layout_bbox_to_original(
        [-10, 100, 1010, 900],
        bbox_source_width=1000,
        bbox_source_height=1000,
        original_page_width=2480,
        original_page_height=3508,
        padding_ratio=0.01,
    ) == (0, 316, 2480, 3192)


def test_text_is_recognized_from_original_resolution_crop(tmp_path):
    engine = StubParsingEngine(
        tmp_path,
        "[{'bbox': [100, 200, 900, 400], 'label': 'Text'}]",
        {"Text": "Условия оплаты\nПредоплата 100%"},
    )
    page = engine.recognize_page(
        np.full((1000, 2000, 3), 255, dtype=np.uint8), 1, "original"
    )
    assert engine.calls == [
        {"stage": "layout", "label": None, "size": (2000, 1000)},
        {"stage": "recognition", "label": "Text", "size": (1600, 200)},
    ]
    assert [block.text for block in page.blocks] == [
        "Условия оплаты",
        "Предоплата 100%",
    ]
    assert page.blocks[0].confidence is None
    assert page.layout_elements[0].original_pixel_bbox == [200, 200, 1800, 400]
    assert page.layout_elements[0].bbox_source_width == 2000
    assert page.native_markdown == "Условия оплаты\nПредоплата 100%"


def test_picture_is_a_visual_zone_and_is_not_sent_to_generation(tmp_path):
    engine = StubParsingEngine(
        tmp_path,
        "[{'bbox': [500, 600, 900, 900], 'label': 'Picture'}]",
        {},
    )
    page = engine.recognize_page(
        np.full((1000, 1000, 3), 255, dtype=np.uint8), 1, "original"
    )
    assert len(engine.calls) == 1
    assert engine.calls[0]["stage"] == "layout"
    assert page.blocks == []
    assert page.layout_elements[0].block_type == "Picture"
    assert page.layout_elements[0].raw_response == ""


def test_native_table_and_geometric_cells_are_both_preserved(tmp_path):
    otsl = (
        "<fcel>Наименование<fcel>Количество<fcel>Всего с НДС<nl>"
        "<fcel>Товар<fcel>12.177<fcel>775944.27<nl>"
        "<fcel>Итого<fcel>12.177<fcel>775944.27"
    )
    engine = StubParsingEngine(
        tmp_path,
        "[{'bbox': [100, 100, 900, 500], 'label': 'Table'}]",
        {"Table": otsl},
    )
    page = engine.recognize_page(
        np.full((1000, 1000, 3), 255, dtype=np.uint8), 1, "original"
    )
    assert len(page.blocks) == 9
    assert page.native_tables[0]["otsl"] == otsl
    assert page.native_tables[0]["html"].startswith("<table>")
    assert page.layout_elements[0].raw_response == otsl
    quantity = next(block for block in page.blocks if block.text == "12.177")
    amount = next(block for block in page.blocks if block.text == "775944.27")
    assert quantity.bounding_box.x_min < amount.bounding_box.x_min


def test_reading_order_and_list_items_are_preserved(tmp_path):
    engine = StubParsingEngine(
        tmp_path,
        "[{'bbox':[20,20,980,80],'label':'Section-header'},"
        " {'bbox':[20,100,980,160],'label':'List-item'}]",
        {"Section-header": "Условия", "List-item": "275-ФЗ не применяется"},
    )
    page = engine.recognize_page(
        np.full((1000, 1000, 3), 255, dtype=np.uint8), 1, "original"
    )
    assert [item.block_type for item in page.layout_elements] == [
        "Section-header",
        "List-item",
    ]
    assert [item.reading_order for item in page.layout_elements] == [0, 1]
    assert page.native_markdown == "## Условия\n\n275-ФЗ не применяется"


def test_crop_hash_cache_avoids_second_recognition_call(tmp_path):
    layout = "[{'bbox': [100, 200, 900, 400], 'label': 'Text'}]"
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    first = StubParsingEngine(tmp_path, layout, {"Text": "Текст"})
    first.recognize_page(image, 1, "original")
    second = StubParsingEngine(tmp_path, layout, {"Text": "Не должен использоваться"})
    page = second.recognize_page(image, 1, "original")
    assert [call["stage"] for call in second.calls] == ["layout"]
    assert page.blocks[0].text == "Текст"
    assert page.layout_elements[0].metadata["crop_cache_hit"] is True


def test_unstructured_layout_fails_closed(tmp_path):
    engine = StubParsingEngine(tmp_path, "not a layout", {})
    try:
        engine.recognize_page(
            np.full((100, 100, 3), 255, dtype=np.uint8), 1, "original"
        )
    except Exception as exc:
        assert "layout-элементы" in str(exc)
    else:
        raise AssertionError("unstructured layout must fail closed")


def test_compare_saved_result_never_invokes_model(tmp_path):
    engine = StubParsingEngine(
        tmp_path,
        "[{'bbox':[20,20,980,80],'label':'Text'}]",
        {"Text": "Подписанием настоящего Приложения"},
    )
    page = engine.recognize_page(
        np.full((1000, 1000, 3), 255, dtype=np.uint8), 1, "original"
    )
    document = OCRDocumentResult(
        pages=[page],
        normalized_full_text="подписанием настоящего приложения",
        full_text="Подписанием настоящего Приложения",
        native_markdown=page.native_markdown,
        engine_name=page.engine_name,
        pipeline_version="layout-crops-v1",
    )
    local = tmp_path / "ocr_document.json"
    local.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    web = tmp_path / "web.md"
    web.write_text("Подписанием настоящего Приложения", encoding="utf-8")
    calls_before = list(engine.calls)
    result = compare_saved_monkey(local, web)
    assert result["cer"] == 0.0
    assert engine.calls == calls_before
