from __future__ import annotations

import numpy as np
import yaml

from app.enums import PresenceStatus
from app.extractors.buyer_block import (
    BuyerROIArtifact,
    _signature_over_seal_confidence,
    _trusted_broad_fallback_seal,
    _trusted_lower_right_seal,
)
from app.vision.models import VisualDetection


def _config() -> dict:
    with open("config/default.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _logo_seal_artifact() -> BuyerROIArtifact:
    seal = VisualDetection(
        status=PresenceStatus.PRESENT,
        confidence=0.858591,
        candidate_bbox=[0, 107, 640, 823],
        candidate_count=3,
        features={
            "shape_score": 0.7667,
            "ring_score": 0.9884,
            "stroke_density_score": 1.0,
            "color_score": 1.0,
            "ink_contrast_score": 1.0,
            "multi_profile_consistency_score": 0.7199,
            "angular_coverage": 0.8889,
            "internal_stroke_density": 0.3669,
            "hough_profile_count": 2,
            "qr_penalty": 0.0,
            "rectangular_stamp_penalty": 0.0,
            "supplier_location_penalty": 0.0,
        },
    )
    signature = VisualDetection(
        status=PresenceStatus.UNCERTAIN,
        confidence=0.84,
        candidate_bbox=[0, 0, 909, 892],
        candidate_count=21,
        features={
            "elongated_components": 21,
            "curvature_score": 0.621,
            "candidate_span_ratio": 0.7475,
            "multi_profile_consistency_score": 0.75,
            "seal_overlap_score": 1.0,
            "qr_penalty": 0.0,
            "rectangular_stamp_penalty": 0.0,
        },
    )
    return BuyerROIArtifact(
        page_number=1,
        name="fallback_lower_right_signature_row",
        bounds=(1240, 2526, 2149, 3418),
        anchor_bbox=None,
        roi=np.zeros((892, 909, 3), dtype=np.uint8),
        seal=seal,
        signature=signature,
    )


def test_dark_logo_seal_is_trusted_from_independent_features():
    artifact = _logo_seal_artifact()

    assert _trusted_lower_right_seal(artifact, _config())


def test_signature_crossing_trusted_seal_is_confirmed():
    artifact = _logo_seal_artifact()

    confidence = _signature_over_seal_confidence(artifact, _config())

    assert confidence is not None
    assert confidence >= 0.85


def test_seal_alone_does_not_confirm_signature():
    artifact = _logo_seal_artifact()
    artifact.signature = artifact.signature.model_copy(
        update={
            "candidate_bbox": [0, 107, 640, 823],
            "features": {
                **artifact.signature.features,
                "elongated_components": 1,
                "curvature_score": 0.1,
            },
        }
    )

    assert _signature_over_seal_confidence(artifact, _config()) is None


def test_strong_broad_fallback_seal_is_trusted_with_overlapping_signature():
    seal_artifact = _logo_seal_artifact()
    seal_artifact.name = "fallback_lower_half"
    seal_artifact.bounds = (868, 1754, 2455, 3472)
    seal_artifact.seal = seal_artifact.seal.model_copy(
        update={
            "confidence": 0.920849,
            "candidate_bbox": [0, 448, 816, 1306],
            "features": {
                **seal_artifact.seal.features,
                "shape_score": 0.90,
                "ring_score": 0.8719,
                "stroke_density_score": 1.0,
                "multi_profile_consistency_score": 0.7938,
                "angular_coverage": 0.7778,
                "hough_profile_count": 3,
                "watermark_penalty": 0.0,
            },
        }
    )
    signature_artifact = _logo_seal_artifact()
    signature_artifact.bounds = (1240, 2525, 2455, 3472)
    signature_artifact.signature = signature_artifact.signature.model_copy(
        update={
            "status": PresenceStatus.PRESENT,
            "confidence": 0.98,
            "candidate_bbox": [0, 0, 781, 825],
        }
    )

    assert _trusted_broad_fallback_seal(
        seal_artifact, signature_artifact, _config()
    )


def test_broad_fallback_seal_is_not_trusted_without_present_signature():
    seal_artifact = _logo_seal_artifact()
    seal_artifact.name = "fallback_lower_half"
    seal_artifact.seal = seal_artifact.seal.model_copy(update={"confidence": 0.95})
    signature_artifact = _logo_seal_artifact()

    assert not _trusted_broad_fallback_seal(
        seal_artifact, signature_artifact, _config()
    )
