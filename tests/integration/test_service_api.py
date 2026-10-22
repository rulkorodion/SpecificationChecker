from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import SpecificationCheckerService
from app.cli import _check
from app.enums import PresenceStatus
from app.ocr.base import FakeOCREngine
from app.ocr.models import OCRPageResult
from app.vision.models import VisualDetection
from app.vision.seal_detector import FakeSealDetector
from app.vision.signature_detector import FakeSignatureDetector


class CountingFakeOCREngine(FakeOCREngine):
    layout_aware = True

    def __init__(self, pages):
        super().__init__(pages)
        self.calls = 0

    def recognize_page(self, image, page_number, preprocessing_profile):
        self.calls += 1
        return super().recognize_page(image, page_number, preprocessing_profile)


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("testdata/specifications/documents/Спецификация 35.jpg").is_file(),
    reason="private document corpus is not available",
)
def test_service_returns_json_payload_and_reuses_ocr_engine(
    reference, tmp_path
):
    page = OCRPageResult(
        page_number=1,
        width=100,
        height=100,
        blocks=[],
        preprocessing_profile="original",
        engine_name="CountingFakeOCR",
    )
    engine = CountingFakeOCREngine({1: page})
    present = VisualDetection(
        status=PresenceStatus.PRESENT,
        confidence=0.99,
    )
    service = SpecificationCheckerService(
        {
            "config_overrides": {
                "ocr_cache": {"enabled": False},
                "runtime": {
                    "temp_dir": str(tmp_path / "temp"),
                    "reports_dir": str(tmp_path / "reports"),
                },
            }
        },
        ocr_engine=engine,
        seal_detector=FakeSealDetector(present),
        signature_detector=FakeSignatureDetector(present),
    )
    document = Path("testdata/specifications/documents/Спецификация 35.jpg")

    first = service.check_specification(
        document,
        reference.model_dump(mode="json"),
        {
            "output_dir": tmp_path / "reports" / "first",
            "force_ocr": True,
        },
    )
    second = service.check_specification(
        document,
        reference,
        {
            "output_dir": tmp_path / "reports" / "second",
            "force_ocr": True,
        },
    )

    json.dumps(first, ensure_ascii=False)
    assert first["decision"] in {"APPROVE", "RETURN", "MANUAL_REVIEW"}
    assert set(first) == {
        "decision",
        "extracted_values",
        "field_details",
        "checks",
        "reasons",
        "comment_1c",
        "confidence",
        "metadata",
    }
    assert first["comment_1c"]
    assert "overall" in first["confidence"]
    assert "application_number" in first["extracted_values"]
    assert first["metadata"]["reference_source"] == "<in-memory:test-001>"
    assert first["metadata"]["processing_time_seconds"] >= 0
    assert service.ocr_engine is engine
    assert engine.calls == 2
    assert second["metadata"]["reference_source"] == "<in-memory:test-001>"


def test_cli_check_uses_specification_service(monkeypatch, tmp_path):
    calls = []

    class FakeService:
        def __init__(self, options):
            calls.append(("init", options))
            self.config = {
                "ocr": {"backend": "monkeyocr_v2", "device": "auto"}
            }

        def prepare(self):
            raise AssertionError("prepare must stay deferred for cache reuse")

        def check_specification(self, document, reference, options):
            calls.append(("check", document, reference, options))
            return {"decision": "APPROVE"}

    monkeypatch.setattr("app.cli.SpecificationCheckerService", FakeService)
    args = SimpleNamespace(
        config=None,
        device="auto",
        ocr_backend="monkeyocr-v2",
        ocr_language=None,
        ocr_cache_dir=None,
        reuse_ocr_cache=True,
        force_ocr=False,
        debug_level="none",
        save_debug=False,
        ground_truth=None,
        output_dir=tmp_path / "report",
        document=Path("document.pdf"),
        reference=Path("reference.json"),
    )

    exit_code = _check(args)

    assert exit_code == 0
    assert [item[0] for item in calls] == ["init", "check"]
