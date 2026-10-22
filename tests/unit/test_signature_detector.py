from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.enums import PresenceStatus
from app.vision.signature_detector import (
    OpenCVSignatureDetector,
    SignatureDetectionContext,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "seal"


def _load(name: str) -> np.ndarray:
    image = cv2.imdecode(
        np.fromfile(str(FIXTURES / name), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert image is not None
    return image


@pytest.mark.parametrize(
    "name",
    ["signature_only.png", "seal_overlapped_by_signature.png"],
)
def test_real_handwriting_masks_are_present(name):
    result = OpenCVSignatureDetector().detect(_load(name))
    assert result.status is PresenceStatus.PRESENT
    assert result.confidence >= 0.85
    assert result.features["diagonal_stroke_score"] >= 0.30


@pytest.mark.parametrize(
    "name",
    [
        "empty_buyer_block.png",
        "qr_code.png",
        "rectangular_stamp.png",
        "buyer_block_with_seal.png",
        "seal_crossed_by_line.png",
        "watermark_only.png",
    ],
)
def test_non_signature_visuals_never_produce_present(name):
    result = OpenCVSignatureDetector().detect(_load(name))
    assert result.status in {PresenceStatus.ABSENT, PresenceStatus.UNCERTAIN}


def test_qr_and_rectangle_have_explicit_penalties():
    qr = OpenCVSignatureDetector().detect(_load("qr_code.png"))
    rectangle = OpenCVSignatureDetector().detect(_load("rectangular_stamp.png"))
    assert qr.features["qr_penalty"] > 0
    assert rectangle.features["rectangular_stamp_penalty"] > 0


def test_trusted_lower_right_geometry_does_not_turn_seal_only_into_signature():
    result = OpenCVSignatureDetector().detect(
        _load("buyer_block_with_seal.png"),
        SignatureDetectionContext(
            roi_name="fallback_lower_right_signature_row",
            anchored=False,
        ),
    )
    assert result.status in {PresenceStatus.ABSENT, PresenceStatus.UNCERTAIN}


def test_trusted_lower_right_geometry_does_not_override_a_real_qr():
    result = OpenCVSignatureDetector().detect(
        _load("qr_code.png"),
        SignatureDetectionContext(
            roi_name="fallback_lower_right_signature_row",
            anchored=False,
        ),
    )

    assert result.status in {PresenceStatus.ABSENT, PresenceStatus.UNCERTAIN}
    assert result.features["qr_signature_override"] is False
