from __future__ import annotations

from app.enums import PresenceStatus
from app.extractors.edo import detect_edo_signing, edo_presence_fields
from app.extractors.runner import extract_specification
from app.ocr.models import OCRDocumentResult, OCRPageResult, OCRTextBlock
from app.schemas import BoundingBox


def _document(lines: list[str]) -> OCRDocumentResult:
    blocks = [
        OCRTextBlock(
            text=text,
            confidence=0.98,
            bounding_box=BoundingBox(
                x_min=80,
                y_min=100 + index * 40,
                x_max=1500,
                y_max=130 + index * 40,
                page_width=1600,
                page_height=2200,
            ),
            page_number=1,
            preprocessing_profile="original",
            engine_name="fake",
        )
        for index, text in enumerate(lines)
    ]
    page = OCRPageResult(
        page_number=1,
        width=1600,
        height=2200,
        blocks=blocks,
        preprocessing_profile="original",
        engine_name="fake",
    )
    return OCRDocumentResult(
        pages=[page],
        normalized_full_text=" ".join(lines),
        engine_name="fake",
    )


def test_specification_12_style_edo_phrase_confirms_both_visual_fields():
    document = _document(
        [
            "Документ подписан и передан через оператора ЭДО АО «ПФ «СКБ Контур»",
            "Подписи отправителя:",
            "Подписи получателя:",
            "Подпись соответствует файлу документа",
        ]
    )

    evidence = detect_edo_signing(document)
    signature, seal = edo_presence_fields(evidence)

    assert evidence.found is True
    assert evidence.recipient_marker_found is True
    assert evidence.confidence == 0.99
    assert signature.normalized_value is PresenceStatus.PRESENT
    assert seal.normalized_value is PresenceStatus.PRESENT
    assert signature.extraction_method == "edo_operator_signing_evidence"
    assert seal.extraction_method == "edo_operator_signing_evidence"


def test_diadoc_transport_footer_alone_is_not_signing_evidence():
    evidence = detect_edo_signing(
        _document(["Передан через Диадок 12.07.2026, идентификатор документа 123"])
    )
    assert evidence.found is False


def test_incidental_edo_operator_mention_is_not_signing_evidence():
    evidence = detect_edo_signing(
        _document(["Оператор ЭДО оказывает услуги электронного документооборота"])
    )
    assert evidence.found is False


def test_phrase_may_be_split_across_ocr_lines():
    evidence = detect_edo_signing(
        _document(
            [
                "Документ подписан и передан",
                "через оператора ЭДО",
            ]
        )
    )
    assert evidence.found is True
    assert evidence.page_number == 1
    assert evidence.bounding_box is not None


def test_edo_evidence_bypasses_ink_detectors(config):
    class DetectorThatMustNotRun:
        def detect(self, *args, **kwargs):
            raise AssertionError("OpenCV detector must be skipped for signed EDO document")

    document = _document(
        [
            "Документ подписан и передан через оператора ЭДО",
            "Подписи получателя:",
        ]
    )
    extracted, artifacts = extract_specification(
        document,
        [],
        config,
        DetectorThatMustNotRun(),
        DetectorThatMustNotRun(),
    )

    assert extracted.buyer_signature_present.normalized_value is PresenceStatus.PRESENT
    assert extracted.buyer_seal_present.normalized_value is PresenceStatus.PRESENT
    assert artifacts.buyer is None
    assert artifacts.edo is not None
