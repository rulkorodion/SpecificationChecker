"""Evidence that a document was signed through an EDI/EDO operator."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.enums import PresenceStatus
from app.extractors.base import extracted_field
from app.extractors.utils import union_box
from app.layout.reading_order import page_lines
from app.normalization.text import normalize_text
from app.ocr.models import OCRDocumentResult
from app.schemas import BoundingBox, ExtractedField


@dataclass(frozen=True, slots=True)
class EDOSigningEvidence:
    found: bool
    confidence: float
    page_number: int | None = None
    bounding_box: BoundingBox | None = None
    matched_text: str = ""
    recipient_marker_found: bool = False
    detection_method: str = "edo_signed_document_phrase"

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.bounding_box is not None:
            data["bounding_box"] = self.bounding_box.model_dump(mode="json")
        return data


def _signed_via_edo(text: str) -> tuple[bool, bool]:
    """Return (strong match, exact canonical phrase)."""

    normalized = normalize_text(text)
    exact = "документ подписан и передан через оператора эдо" in normalized
    document = bool(re.search(r"\bдокумент\w*\b", normalized))
    signed = bool(re.search(r"\bподписан\w*\b", normalized))
    transferred = bool(re.search(r"\bпередан\w*\b", normalized))
    operator = bool(re.search(r"\bоператор\w*\b", normalized))
    edo = bool(re.search(r"\bэдо\b", normalized)) or (
        "электрон" in normalized and "документооборот" in normalized
    )
    return document and signed and transferred and operator and edo, exact


def _recipient_marker(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        marker in normalized
        for marker in (
            "подписи получателя",
            "подпись получателя",
            "сертификат получателя",
        )
    )


def detect_edo_signing(document: OCRDocumentResult) -> EDOSigningEvidence:
    """Find document-level EDO signing evidence without using reference data."""

    candidates: list[EDOSigningEvidence] = []
    for page in document.pages:
        page_text = " ".join(block.text for block in page.blocks)
        recipient = _recipient_marker(
            " ".join((page_text, page.native_markdown or ""))
        )
        lines = page_lines(page)
        for start in range(len(lines)):
            for size in range(1, min(6, len(lines) - start) + 1):
                blocks = [block for line in lines[start : start + size] for block in line]
                text = " ".join(block.text for block in blocks)
                matched, exact = _signed_via_edo(text)
                if not matched:
                    continue
                confidence = 0.99 if exact and recipient else 0.98 if exact else 0.95
                candidates.append(
                    EDOSigningEvidence(
                        found=True,
                        confidence=confidence,
                        page_number=page.page_number,
                        bounding_box=union_box(blocks),
                        matched_text=text,
                        recipient_marker_found=recipient,
                    )
                )
                break

        # Layout-aware parsers may preserve a phrase in native Markdown even if
        # it cannot be represented as a conventional OCR line/block.
        native_text = page.native_markdown or ""
        matched, exact = _signed_via_edo(native_text)
        if matched:
            confidence = 0.99 if exact and recipient else 0.98 if exact else 0.95
            candidates.append(
                EDOSigningEvidence(
                    found=True,
                    confidence=confidence,
                    page_number=page.page_number,
                    matched_text=native_text,
                    recipient_marker_found=recipient,
                    detection_method="edo_signed_document_phrase_native_markdown",
                )
            )

    if not candidates:
        return EDOSigningEvidence(found=False, confidence=0.0)
    return max(
        candidates,
        key=lambda item: (
            item.confidence,
            item.bounding_box is not None,
            -len(item.matched_text),
        ),
    )


def edo_presence_fields(
    evidence: EDOSigningEvidence,
) -> tuple[ExtractedField, ExtractedField]:
    """Apply the business rule that verified EDO signing replaces ink marks."""

    if not evidence.found:
        raise ValueError("EDO presence fields require positive signing evidence")
    raw = {
        "edo_signing": evidence.to_dict(),
        "business_rule": (
            "Документ, подписанный и переданный через оператора ЭДО, "
            "считается имеющим подпись и печать покупателя"
        ),
    }
    common = {
        "raw": raw,
        "normalized": PresenceStatus.PRESENT,
        "confidence": evidence.confidence,
        "method": "edo_operator_signing_evidence",
        "page": evidence.page_number,
        "box": evidence.bounding_box,
        "evidence": [
            "Найдена фраза о подписании и передаче документа через оператора ЭДО",
            f"OCR: {evidence.matched_text}",
        ],
    }
    return (
        extracted_field("BUYER_SIGNATURE", **common),
        extracted_field("BUYER_SEAL", **common),
    )
