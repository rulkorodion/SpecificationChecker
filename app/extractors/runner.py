"""Оркестрация extractors только над OCR и изображениями документа."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.document.renderer import PageImage
from app.extractors.application import extract_application
from app.extractors.buyer_block import BuyerVisualArtifacts, extract_buyer_visuals
from app.extractors.delivery import extract_delivery
from app.extractors.edo import EDOSigningEvidence, detect_edo_signing, edo_presence_fields
from app.extractors.payment import extract_payment
from app.extractors.totals import extract_totals
from app.extractors.transportation import extract_transportation
from app.ocr.models import OCRDocumentResult
from app.schemas import ExtractedSpecification, TableAggregation
from app.vision.seal_detector import SealDetector
from app.vision.signature_detector import SignatureDetector


@dataclass(slots=True)
class ExtractionArtifacts:
    buyer: BuyerVisualArtifacts | None
    edo: EDOSigningEvidence | None
    tables: TableAggregation
    extractor_timings: dict[str, float] = field(default_factory=dict)


def extract_specification(
    document: OCRDocumentResult,
    pages: list[PageImage],
    config: dict,
    seal_detector: SealDetector,
    signature_detector: SignatureDetector,
) -> tuple[ExtractedSpecification, ExtractionArtifacts]:
    """Выполнить независимое извлечение без reference и ground truth."""

    timings: dict[str, float] = {}
    started = time.perf_counter()
    application_number, application_date = extract_application(document)
    timings["application_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    total_quantity, total_with_vat, table_analysis = extract_totals(document)
    timings["tables_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    transportation = extract_transportation(document, config, table_analysis)
    timings["transportation_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    delivery_deadline, delivery_type = extract_delivery(document, config)
    timings["delivery_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    payment_type, deferment_days = extract_payment(document, config)
    timings["payment_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    edo_evidence = detect_edo_signing(document)
    timings["edo_detection_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if edo_evidence.found:
        signature, seal = edo_presence_fields(edo_evidence)
        artifact = None
    else:
        signature, seal, artifact = extract_buyer_visuals(
            pages, document, config, seal_detector, signature_detector
        )
    timings["visual_detection_seconds"] = round(
        time.perf_counter() - started, 6
    )
    extracted = ExtractedSpecification(
        application_number=application_number,
        application_date=application_date,
        total_quantity=total_quantity,
        total_with_vat=total_with_vat,
        transportation_costs_included=transportation,
        delivery_deadline=delivery_deadline,
        delivery_type=delivery_type,
        payment_type=payment_type,
        deferment_days=deferment_days,
        buyer_signature_present=signature,
        buyer_seal_present=seal,
    )
    return extracted, ExtractionArtifacts(
        buyer=artifact,
        edo=edo_evidence if edo_evidence.found else None,
        tables=table_analysis,
        extractor_timings=timings,
    )
