from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.enums import PresenceStatus
from app.vision.seal_detector import OpenCVSealDetector, SealDetectionContext


FIXTURES = Path(__file__).parents[1] / "fixtures" / "seal"
DEFAULT_CONTEXT = SealDetectionContext(
    roi_name="compact",
    anchor_bbox=(380, 35, 535, 90),
    anchored=True,
)


def load_fixture(name: str):
    image = cv2.imdecode(
        np.fromfile(str(FIXTURES / name), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert image is not None, f"Missing generated fixture: {name}"
    return image


@pytest.mark.parametrize(
    "fixture_name,context",
    [
        ("buyer_block_with_seal.png", DEFAULT_CONTEXT),
        ("pale_blue_seal.png", DEFAULT_CONTEXT),
        ("grayscale_seal.png", DEFAULT_CONTEXT),
        ("seal_crossed_by_line.png", DEFAULT_CONTEXT),
        (
            "partially_cropped_seal.png",
            SealDetectionContext(
                roi_name="expanded",
                anchor_bbox=(250, 35, 405, 90),
                anchored=True,
            ),
        ),
        ("seal_overlapped_by_signature.png", DEFAULT_CONTEXT),
    ],
)
def test_positive_seal_variants_are_present(fixture_name, context):
    detection = OpenCVSealDetector().detect(load_fixture(fixture_name), context)
    assert detection.status is PresenceStatus.PRESENT
    assert detection.confidence >= 0.85
    assert detection.candidate_bbox is not None
    assert detection.features["ring_score"] >= 0.56
    assert detection.features["multi_profile_consistency_score"] >= 0.48


@pytest.mark.parametrize(
    "fixture_name",
    [
        "signature_only.png",
        "qr_code.png",
        "watermark_only.png",
        "rectangular_stamp.png",
        "round_graphic_only.png",
        "supplier_seal_outside_buyer_area.png",
        "empty_buyer_block.png",
    ],
)
def test_negative_buyer_blocks_never_produce_present(fixture_name):
    detection = OpenCVSealDetector().detect(
        load_fixture(fixture_name), DEFAULT_CONTEXT
    )
    assert detection.status in {PresenceStatus.ABSENT, PresenceStatus.UNCERTAIN}


def test_qr_candidate_has_explicit_penalty_when_localized():
    detection = OpenCVSealDetector().detect(
        load_fixture("qr_code.png"), DEFAULT_CONTEXT
    )
    if detection.candidate_bbox is not None:
        assert detection.features["qr_penalty"] > 0
