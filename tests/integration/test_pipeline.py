from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.enums import PresenceStatus
from app.ocr.base import FakeOCREngine
from app.ocr.models import OCRPageResult, OCRTextBlock
from app.pipeline import run_check
from app.exceptions import InputValidationError, PageSelectionError
from app.reporting.console import format_console_report
from app.schemas import BoundingBox
from app.vision.models import VisualDetection
from app.vision.seal_detector import FakeSealDetector
from app.vision.signature_detector import FakeSignatureDetector


PRIVATE_SPEC_1 = Path("testdata/specifications/documents/Спецификация 1.pdf")
PRIVATE_REFERENCE_1 = Path("testdata/specifications/reference/specification_1.json")
PRIVATE_SPEC_20 = Path("testdata/specifications/documents/Спецификация 20.pdf")
PRIVATE_REFERENCE_20 = Path("testdata/specifications/reference/specification_20.json")


def make_block(text, x, y):
    return OCRTextBlock(
        text=text,
        confidence=0.99,
        bounding_box=BoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + 300,
            y_max=y + 20,
            page_width=595,
            page_height=842,
        ),
        page_number=1,
        preprocessing_profile="original",
        engine_name="FakeOCR",
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not (PRIVATE_SPEC_1.is_file() and PRIVATE_REFERENCE_1.is_file()),
    reason="private specification corpus is not available",
)
def test_pipeline_creates_report_with_fake_ocr(config, tmp_path):
    changed = deepcopy(config)
    changed["document"]["render_dpi"] = 72
    clause_1 = (
        "Подписанием настоящего Приложения, Стороны подтверждают, что они ознакомлены "
        "с содержанием всех страниц Приложения, понимают их значение и обязуются их выполнять"
    )
    clause_2 = (
        "Подписание настоящего Приложения на странице 1 означает полное и безоговорочное "
        "согласие Сторон со всеми условиями, изложенными на всех Страницах Приложения"
    )
    today = datetime.now(ZoneInfo("Europe/Moscow")).date()
    blocks = [
        make_block(
            f"Приложение № 76930 от {today.strftime('%d.%m.%Y')} г.",
            120,
            50,
        ),
        make_block("Наименование товара", 40, 160),
        make_block("Ед. изм. Кол-во Цена за ед. Всего с НДС", 300, 160),
        make_block("Итого:", 40, 200),
        make_block("69,284", 360, 200),
        make_block("5 272 993,70", 500, 200),
        make_block(clause_1, 40, 260),
        make_block(clause_2, 40, 300),
        make_block(
            "В общую сумму Приложения включены расходы по оплате транспортных услуг "
            "и иные расходы, связанные с транспортировкой товара",
            40,
            350,
        ),
        make_block("Срок поставки: не позднее 24.04.2026 г.", 40, 400),
        make_block("Товар поставляется на условиях: доставка ж/д транспортом CPT", 40, 450),
        make_block("Условия оплаты: Предоплата 100%, до поставки.", 40, 500),
        make_block("Покупатель:", 350, 650),
    ]
    page = OCRPageResult(
        page_number=1,
        width=595,
        height=842,
        blocks=blocks,
        preprocessing_profile="original",
        engine_name="FakeOCR",
    )
    present = VisualDetection(status=PresenceStatus.PRESENT, confidence=0.99)
    output = tmp_path / "run"
    report = run_check(
        document_path=__import__("pathlib").Path("testdata/specifications/documents/Спецификация 1.pdf"),
        reference_path=__import__("pathlib").Path("testdata/specifications/reference/specification_1.json"),
        ground_truth_path=None,
        output_dir=output,
        config=changed,
        ocr_engine=FakeOCREngine({1: page}),
        seal_detector=FakeSealDetector(present),
        signature_detector=FakeSignatureDetector(present),
        job_id="173c3a42-b809-4760-bfec-1c566450267b",
    )
    assert (output / "report.json").exists()
    assert not (output / "ocr").exists()
    assert report.final_decision.value == "APPROVE"
    console_text = (output / "console_report.txt").read_text(encoding="utf-8")
    assert console_text == format_console_report(
        report,
        application_number="76930",
        job_id="173c3a42-b809-4760-bfec-1c566450267b",
    )
    assert console_text.startswith(
        "ПРОВЕРКА СПЕЦИФИКАЦИИ №76930\n"
        "ID проверки: 173c3a42-b809-4760-bfec-1c566450267b\n\n"
    )
    assert "ИТОГОВОЕ РЕШЕНИЕ: СОГЛАСОВАНО (APPROVE)" in console_text
    assert "=" not in console_text


def test_runtime_temp_is_cleaned_when_pipeline_raises(config, tmp_path):
    changed = deepcopy(config)
    changed["runtime"]["temp_dir"] = str(tmp_path / "runtime-temp")
    with pytest.raises(InputValidationError):
        run_check(
            document_path=tmp_path / "missing.pdf",
            reference_path=tmp_path / "missing-reference.json",
            ground_truth_path=None,
            output_dir=tmp_path / "run",
            config=changed,
            ocr_engine=FakeOCREngine({}),
        )
    temp_root = tmp_path / "runtime-temp"
    assert temp_root.is_dir()
    assert list(temp_root.iterdir()) == []


@pytest.mark.integration
@pytest.mark.skipif(
    not (PRIVATE_SPEC_20.is_file() and PRIVATE_REFERENCE_20.is_file()),
    reason="private specification corpus is not available",
)
def test_more_than_two_pages_requires_enabled_page_selector(config, tmp_path):
    changed = deepcopy(config)
    changed["document"]["render_dpi"] = 36
    changed["document"]["page_selector"]["enabled"] = False
    output = tmp_path / "compound-document"

    # No fake OCR pages are supplied: the test would fail if MonkeyOCR were called.
    with pytest.raises(PageSelectionError) as raised:
        run_check(
            document_path=__import__("pathlib").Path(
                "testdata/specifications/documents/Спецификация 20.pdf"
            ),
            reference_path=__import__("pathlib").Path(
                "testdata/specifications/reference/specification_20.json"
            ),
            ground_truth_path=None,
            output_dir=output,
            config=changed,
            ocr_engine=FakeOCREngine({}),
        )

    assert raised.value.diagnostic_code == "selector_disabled"
