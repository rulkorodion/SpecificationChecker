"""Explainable multi-profile OpenCV detector for buyer seals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.enums import PresenceStatus
from app.vision.models import VisualDetection


@dataclass(frozen=True, slots=True)
class SealDetectionContext:
    """Geometry connecting a candidate to the OCR buyer-label anchor."""

    roi_name: str = "unknown"
    anchor_bbox: tuple[int, int, int, int] | None = None
    anchored: bool = False


class SealDetector(ABC):
    @abstractmethod
    def detect(
        self,
        roi: np.ndarray,
        context: SealDetectionContext | None = None,
    ) -> VisualDetection:
        """Detect a seal without identifying a person or an organisation."""

    def detect_with_profiles(
        self,
        roi: np.ndarray,
        context: SealDetectionContext | None = None,
    ) -> tuple[VisualDetection, dict[str, np.ndarray]]:
        """Return detection and optional debug preprocessing profiles."""

        return self.detect(roi, context), {}


@dataclass(slots=True)
class _CircleCandidate:
    x: float
    y: float
    radius: float
    sources: set[str]


class OpenCVSealDetector(SealDetector):
    """Detect intact and interrupted round/oval seals using independent evidence."""

    _MAX_ANALYSIS_SIDE = 640

    def __init__(self, min_confidence: float = 0.85) -> None:
        self.min_confidence = min_confidence

    @staticmethod
    def preprocessing_profiles(roi: np.ndarray) -> dict[str, np.ndarray]:
        """Build materially different views; no one threshold is trusted alone."""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
        adaptive = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            11,
        )

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        hsv_blue = (
            (hue >= 80)
            & (hue <= 155)
            & (saturation >= 22)
            & (value <= 253)
        )

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lab_b = lab[:, :, 2]
        b, g, r = cv2.split(roi)
        blue_minus_red = b.astype(np.int16) - r.astype(np.int16)
        blue_minus_green = b.astype(np.int16) - g.astype(np.int16)
        lab_blue = (
            (
                (lab_b.astype(np.int16) < 122)
                & (saturation >= 12)
                & (blue_minus_red > 8)
            )
            | (
                (blue_minus_red > 14)
                & (blue_minus_green > -4)
                & (saturation >= 14)
            )
        )

        hsv_blue_mask = np.uint8(hsv_blue) * 255
        lab_color_mask = np.uint8(lab_blue) * 255
        color_mask = cv2.bitwise_or(hsv_blue_mask, lab_color_mask)
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        )
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        saturation_boosted = hsv.copy()
        saturation_boosted[:, :, 1] = cv2.convertScaleAbs(
            saturation, alpha=1.8, beta=10
        )
        saturation_enhanced = cv2.cvtColor(saturation_boosted, cv2.COLOR_HSV2BGR)

        strong_dark = np.uint8(gray < 155) * 255
        watermark_suppressed_mask = cv2.bitwise_or(color_mask, strong_dark)
        watermark_suppressed = np.full_like(roi, 255)
        watermark_suppressed[watermark_suppressed_mask > 0] = roi[
            watermark_suppressed_mask > 0
        ]

        base_ink = cv2.bitwise_or(color_mask, np.uint8(clahe < 150) * 255)
        line_width = max(25, int(roi.shape[1] * 0.12))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 1))
        horizontal_lines = cv2.morphologyEx(
            base_ink, cv2.MORPH_OPEN, horizontal_kernel
        )
        line_suppressed = cv2.subtract(base_ink, horizontal_lines)
        line_suppressed = cv2.morphologyEx(
            line_suppressed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        combined = cv2.bitwise_or(color_mask, line_suppressed)

        return {
            "original_rgb": roi.copy(),
            "grayscale": gray,
            "clahe": clahe,
            "adaptive_threshold": adaptive,
            "hsv_blue_mask": hsv_blue_mask,
            "lab_color_separation": lab_color_mask,
            "saturation_enhanced": saturation_enhanced,
            "watermark_suppressed": watermark_suppressed,
            "line_suppressed": line_suppressed,
            "combined_mask": combined,
        }

    @staticmethod
    def _scaled_profiles(
        profiles: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float]:
        height, width = profiles["grayscale"].shape[:2]
        scale = min(1.0, OpenCVSealDetector._MAX_ANALYSIS_SIDE / max(height, width))
        if scale == 1.0:
            return profiles, scale
        scaled: dict[str, np.ndarray] = {}
        for name, image in profiles.items():
            interpolation = (
                cv2.INTER_NEAREST
                if name.endswith("mask")
                or name in {"adaptive_threshold", "line_suppressed"}
                else cv2.INTER_AREA
            )
            scaled[name] = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=interpolation
            )
        return scaled, scale

    @staticmethod
    def _hough_candidates(
        image: np.ndarray,
        source: str,
    ) -> list[_CircleCandidate]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minimum = min(image.shape[:2])
        if minimum < 80:
            return []
        blurred = cv2.GaussianBlur(image, (5, 5), 1.2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.3,
            minDist=max(24, minimum * 0.16),
            param1=90,
            param2=30,
            minRadius=max(14, int(minimum * 0.075)),
            maxRadius=max(22, int(minimum * 0.38)),
        )
        if circles is None:
            return []
        return [
            _CircleCandidate(float(x), float(y), float(radius), {source})
            for x, y, radius in circles[0]
        ]

    @staticmethod
    def _merge_candidates(
        candidates: list[_CircleCandidate],
    ) -> list[_CircleCandidate]:
        merged: list[_CircleCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.radius):
            match: _CircleCandidate | None = None
            for existing in merged:
                distance = float(np.hypot(candidate.x - existing.x, candidate.y - existing.y))
                radius_delta = abs(candidate.radius - existing.radius)
                if (
                    distance <= max(candidate.radius, existing.radius) * 0.22
                    and radius_delta <= max(candidate.radius, existing.radius) * 0.30
                ):
                    match = existing
                    break
            if match is None:
                merged.append(candidate)
                continue
            old_weight = len(match.sources)
            new_sources = candidate.sources - match.sources
            if new_sources:
                new_weight = len(new_sources)
                total_weight = old_weight + new_weight
                match.x = (match.x * old_weight + candidate.x * new_weight) / total_weight
                match.y = (match.y * old_weight + candidate.y * new_weight) / total_weight
                match.radius = (
                    match.radius * old_weight + candidate.radius * new_weight
                ) / total_weight
            match.sources.update(candidate.sources)
        return merged

    @staticmethod
    def _angular_coverage(
        mask: np.ndarray,
        x: float,
        y: float,
        radius: float,
        sectors: int = 36,
    ) -> tuple[float, float]:
        yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        annulus = (distance >= radius * 0.70) & (distance <= radius * 1.16)
        if not np.any(annulus):
            return 0.0, 0.0
        angles = (np.arctan2(yy - y, xx - x) + 2 * np.pi) % (2 * np.pi)
        densities: list[float] = []
        for sector in range(sectors):
            lower = sector * 2 * np.pi / sectors
            upper = (sector + 1) * 2 * np.pi / sectors
            sector_mask = annulus & (angles >= lower) & (angles < upper)
            densities.append(
                float(np.mean(mask[sector_mask] > 0))
                if np.any(sector_mask)
                else 0.0
            )
        coverage = float(np.mean(np.asarray(densities) >= 0.015))
        density = float(np.mean(mask[annulus] > 0))
        return coverage, density

    @staticmethod
    def _circle_density(
        mask: np.ndarray,
        x: float,
        y: float,
        radius: float,
    ) -> float:
        yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= (radius * 0.72) ** 2
        return float(np.mean(mask[inside] > 0)) if np.any(inside) else 0.0

    @staticmethod
    def _location_score(
        candidate: _CircleCandidate,
        context: SealDetectionContext | None,
        shape: tuple[int, int],
        scale: float,
    ) -> tuple[float, float]:
        if context is None or context.anchor_bbox is None:
            return 0.40, 0.0
        x1, y1, x2, y2 = context.anchor_bbox
        anchor_x = (x1 + x2) * 0.5 * scale
        anchor_y = (y1 + y2) * 0.5 * scale
        height, width = shape
        dx = abs(candidate.x - anchor_x) / max(width * 0.42, 1)
        vertical_offset = (candidate.y - anchor_y) / max(height * 0.45, 1)
        horizontal = max(0.0, 1.0 - dx)
        vertical = max(0.0, 1.0 - abs(vertical_offset - 0.45) * 0.75)
        score = float(np.clip(0.65 * horizontal + 0.35 * vertical, 0.0, 1.0))
        # A full signature-row ROI may contain the supplier's genuine round seal.
        # Distance from the explicit buyer label is therefore a hard negative,
        # not merely a weak ranking hint.
        supplier_penalty = float(np.clip((dx - 0.55) * 1.35, 0.0, 0.70))
        return score, supplier_penalty

    @staticmethod
    def _qr_pattern_penalty(
        adaptive_mask: np.ndarray,
        candidate: _CircleCandidate,
    ) -> float:
        x1 = max(0, int(candidate.x - candidate.radius * 1.05))
        y1 = max(0, int(candidate.y - candidate.radius * 1.05))
        x2 = min(adaptive_mask.shape[1], int(candidate.x + candidate.radius * 1.05))
        y2 = min(adaptive_mask.shape[0], int(candidate.y + candidate.radius * 1.05))
        crop = adaptive_mask[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 30:
            return 0.0
        contours, hierarchy = cv2.findContours(
            crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            return 0.0
        nested_squares = 0
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < crop.size * 0.004:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, perimeter * 0.04, True)
            _, _, width, height = cv2.boundingRect(contour)
            child = hierarchy[0][index][2]
            if (
                len(polygon) == 4
                and 0.70 <= width / max(height, 1) <= 1.30
                and child >= 0
            ):
                nested_squares += 1
        binary = crop > 0
        horizontal_transitions = float(np.mean(binary[:, 1:] != binary[:, :-1]))
        vertical_transitions = float(np.mean(binary[1:, :] != binary[:-1, :]))
        transition_density = (horizontal_transitions + vertical_transitions) * 0.5
        if nested_squares >= 3:
            return 0.72
        if nested_squares >= 2 and transition_density >= 0.03:
            return 0.48
        return 0.0

    @staticmethod
    def _rectangular_penalty(
        line_mask: np.ndarray,
        candidate: _CircleCandidate,
    ) -> float:
        x1 = max(0, int(candidate.x - candidate.radius * 1.1))
        y1 = max(0, int(candidate.y - candidate.radius * 1.1))
        x2 = min(line_mask.shape[1], int(candidate.x + candidate.radius * 1.1))
        y2 = min(line_mask.shape[0], int(candidate.y + candidate.radius * 1.1))
        crop = line_mask[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(9, crop.shape[1] // 3), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(9, crop.shape[0] // 3))
        )
        horizontal = cv2.morphologyEx(crop, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN, vertical_kernel)

        def long_components(mask: np.ndarray, horizontal_axis: bool) -> int:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            count = 0
            for contour in contours:
                _, _, width, height = cv2.boundingRect(contour)
                bounding_area = max(1, width * height)
                rectangularity = cv2.contourArea(contour) / bounding_area
                if (
                    horizontal_axis
                    and width >= crop.shape[1] * 0.80
                    and rectangularity >= 0.72
                ):
                    count += 1
                if (
                    not horizontal_axis
                    and height >= crop.shape[0] * 0.80
                    and rectangularity >= 0.72
                ):
                    count += 1
            return count

        horizontal_count = long_components(horizontal, True)
        vertical_count = long_components(vertical, False)
        if horizontal_count >= 2 and vertical_count >= 2:
            return 0.68
        if horizontal_count >= 2 and vertical_count >= 1:
            return 0.52
        if horizontal_count >= 1 and vertical_count >= 2:
            return 0.45
        if horizontal_count >= 3:
            return 0.35
        return 0.0

    def _score_candidate(
        self,
        candidate: _CircleCandidate,
        profiles: dict[str, np.ndarray],
        context: SealDetectionContext | None,
        scale: float,
        global_qr_penalty: float,
    ) -> tuple[float, dict[str, Any], dict[str, float]]:
        supporting_profiles = {
            "hsv_blue_mask": profiles["hsv_blue_mask"],
            "lab_color_separation": profiles["lab_color_separation"],
            "line_suppressed": profiles["line_suppressed"],
            "combined_mask": profiles["combined_mask"],
            "adaptive_threshold": profiles["adaptive_threshold"],
        }
        profile_scores: dict[str, float] = {}
        coverages: list[float] = []
        densities: list[float] = []
        for name, mask in supporting_profiles.items():
            coverage, density = self._angular_coverage(
                mask, candidate.x, candidate.y, candidate.radius
            )
            profile_scores[name] = round(
                float(np.clip(0.65 * coverage + 0.35 * min(1.0, density / 0.16), 0, 1)),
                4,
            )
            coverages.append(coverage)
            densities.append(density)

        best_coverage = max(coverages, default=0.0)
        best_density = max(densities, default=0.0)
        profile_consistency = float(np.mean(list(profile_scores.values())))
        hough_consistency = min(1.0, len(candidate.sources) / 3.0)
        multi_profile_consistency = float(
            np.clip(0.65 * profile_consistency + 0.35 * hough_consistency, 0, 1)
        )
        ring_score = float(
            np.clip(
                0.65 * np.clip((best_coverage - 0.28) / 0.62, 0, 1)
                + 0.35 * min(1.0, best_density / 0.17),
                0,
                1,
            )
        )
        shape_score = float(
            np.clip(0.55 * hough_consistency + 0.45 * best_coverage, 0, 1)
        )
        stroke_density = self._circle_density(
            profiles["combined_mask"], candidate.x, candidate.y, candidate.radius
        )
        dense_factor = (
            1.0
            if stroke_density <= 0.75
            else float(np.clip((0.98 - stroke_density) / 0.23, 0.40, 1.0))
        )
        stroke_density_score = float(
            np.clip(stroke_density / 0.12, 0, 1)
            * dense_factor
        )
        color_density = self._circle_density(
            cv2.bitwise_or(
                profiles["hsv_blue_mask"], profiles["lab_color_separation"]
            ),
            candidate.x,
            candidate.y,
            candidate.radius,
        )
        color_score = float(np.clip(color_density / 0.10, 0, 1))
        achromatic_contrast_score = float(
            np.mean(
                [
                    profile_scores["line_suppressed"],
                    profile_scores["combined_mask"],
                    profile_scores["adaptive_threshold"],
                ]
            )
        )
        ink_contrast_score = max(color_score, achromatic_contrast_score)
        location_score, supplier_penalty = self._location_score(
            candidate,
            context,
            profiles["grayscale"].shape[:2],
            scale,
        )
        radius_ratio = candidate.radius / max(
            1.0, min(profiles["grayscale"].shape[:2])
        )
        size_score = float(
            np.clip(1.0 - abs(radius_ratio - 0.24) / 0.18, 0.0, 1.0)
        )
        qr_penalty = max(
            global_qr_penalty,
            self._qr_pattern_penalty(
                profiles["adaptive_threshold"], candidate
            ),
        )
        rectangular_stamp_penalty = self._rectangular_penalty(
            profiles["combined_mask"], candidate
        )
        saturation = profiles["hsv_blue_mask"]
        dark = np.uint8(profiles["grayscale"] < 155) * 255
        color_or_dark = cv2.bitwise_or(saturation, dark)
        evidence_density = self._circle_density(
            color_or_dark, candidate.x, candidate.y, candidate.radius
        )
        watermark_penalty = float(
            0.18
            if evidence_density < 0.018 and color_density < 0.008
            else 0.0
        )

        weighted = (
            0.22 * shape_score
            + 0.24 * ring_score
            + 0.12 * stroke_density_score
            + 0.12 * ink_contrast_score
            + 0.10 * location_score
            + 0.10 * multi_profile_consistency
            + 0.10 * size_score
        )
        touches_side_border = (
            candidate.x - candidate.radius <= 2
            or candidate.x + candidate.radius >= profiles["grayscale"].shape[1] - 2
        )
        partial_arc_bonus = (
            0.055
            if touches_side_border
            and ring_score >= 0.70
            and multi_profile_consistency >= 0.65
            else 0.0
        )
        penalties = (
            watermark_penalty
            + qr_penalty
            + rectangular_stamp_penalty
            + supplier_penalty
        )
        confidence = float(
            np.clip(weighted + partial_arc_bonus - penalties, 0.0, 0.99)
        )
        features: dict[str, Any] = {
            "shape_score": round(shape_score, 4),
            "ring_score": round(ring_score, 4),
            "stroke_density_score": round(stroke_density_score, 4),
            "color_score": round(color_score, 4),
            "ink_contrast_score": round(ink_contrast_score, 4),
            "location_score": round(location_score, 4),
            "size_score": round(size_score, 4),
            "multi_profile_consistency_score": round(
                multi_profile_consistency, 4
            ),
            "watermark_penalty": round(watermark_penalty, 4),
            "qr_penalty": round(qr_penalty, 4),
            "rectangular_stamp_penalty": round(
                rectangular_stamp_penalty, 4
            ),
            "supplier_location_penalty": round(supplier_penalty, 4),
            "partial_arc_bonus": round(partial_arc_bonus, 4),
            "angular_coverage": round(best_coverage, 4),
            "annular_ink_density": round(best_density, 4),
            "internal_stroke_density": round(stroke_density, 4),
            "blue_color_density": round(color_density, 4),
            "hough_profile_count": len(candidate.sources),
            "roi_name": context.roi_name if context else "unknown",
            "anchor_linked": bool(context and context.anchored),
        }
        return confidence, features, profile_scores

    def _analyze(
        self,
        roi: np.ndarray,
        context: SealDetectionContext | None,
    ) -> tuple[VisualDetection, dict[str, np.ndarray]]:
        if roi.size == 0 or min(roi.shape[:2]) < 40:
            return (
                VisualDetection(
                    status=PresenceStatus.UNCERTAIN,
                    confidence=0.0,
                    warnings=["Пустая или слишком маленькая ROI покупателя"],
                ),
                {},
            )

        full_profiles = self.preprocessing_profiles(roi)
        profiles, scale = self._scaled_profiles(full_profiles)
        candidates: list[_CircleCandidate] = []
        hough_inputs = {
            "hsv": profiles["hsv_blue_mask"],
            "lab": profiles["lab_color_separation"],
            "line": profiles["line_suppressed"],
            "clahe": profiles["clahe"],
        }
        for source, image in hough_inputs.items():
            candidates.extend(self._hough_candidates(image, source)[:12])
        merged = self._merge_candidates(candidates)
        merged = sorted(
            merged,
            key=lambda item: (len(item.sources), item.radius),
            reverse=True,
        )[:24]
        profile_height, profile_width = profiles["grayscale"].shape[:2]
        global_qr_penalty = self._qr_pattern_penalty(
            profiles["adaptive_threshold"],
            _CircleCandidate(
                x=profile_width * 0.5,
                y=profile_height * 0.5,
                radius=float(max(profile_width, profile_height)),
                sources={"global_qr_screen"},
            ),
        )

        scored: list[
            tuple[float, _CircleCandidate, dict[str, Any], dict[str, float]]
        ] = []
        for candidate in merged:
            confidence, features, profile_scores = self._score_candidate(
                candidate, profiles, context, scale, global_qr_penalty
            )
            scored.append((confidence, candidate, features, profile_scores))

        if not scored:
            ink_ratio = float(np.mean(profiles["combined_mask"] > 0))
            status = PresenceStatus.ABSENT if ink_ratio < 0.002 else PresenceStatus.UNCERTAIN
            confidence = 0.90 if status is PresenceStatus.ABSENT else min(
                0.55, self.min_confidence - 0.01
            )
            return (
                VisualDetection(
                    status=status,
                    confidence=confidence,
                    candidate_count=0,
                    features={
                        "combined_ink_ratio": round(ink_ratio, 5),
                        "anchor_linked": bool(context and context.anchored),
                    },
                    evidence=[
                        "circle_candidates=0",
                        f"combined_ink_ratio={ink_ratio:.5f}",
                    ],
                    warnings=(
                        []
                        if status is PresenceStatus.ABSENT
                        else ["Есть штрихи, но нет устойчивого круглого кандидата"]
                    ),
                ),
                full_profiles,
            )

        confidence, candidate, features, profile_scores = max(
            scored, key=lambda item: item[0]
        )
        inverse_scale = 1.0 / scale
        x1 = max(0, int((candidate.x - candidate.radius * 1.12) * inverse_scale))
        y1 = max(0, int((candidate.y - candidate.radius * 1.12) * inverse_scale))
        x2 = min(
            roi.shape[1], int((candidate.x + candidate.radius * 1.12) * inverse_scale)
        )
        y2 = min(
            roi.shape[0], int((candidate.y + candidate.radius * 1.12) * inverse_scale)
        )
        structural_evidence = (
            float(features["ring_score"]) >= 0.56
            and float(features["shape_score"]) >= 0.52
            and float(features["multi_profile_consistency_score"]) >= 0.48
            and float(features["qr_penalty"]) < 0.35
            and float(features["supplier_location_penalty"]) < 0.25
        )
        if confidence >= self.min_confidence and structural_evidence:
            status = PresenceStatus.PRESENT
            warnings: list[str] = []
        else:
            status = PresenceStatus.UNCERTAIN
            confidence = min(confidence, self.min_confidence - 0.01)
            warnings = [
                "Круговой кандидат найден, но независимых визуальных доказательств недостаточно"
            ]
        evidence = [
            f"shape_score={float(features['shape_score']):.4f}",
            f"ring_score={float(features['ring_score']):.4f}",
            f"stroke_density_score={float(features['stroke_density_score']):.4f}",
            f"color_score={float(features['color_score']):.4f}",
            f"ink_contrast_score={float(features['ink_contrast_score']):.4f}",
            f"location_score={float(features['location_score']):.4f}",
            f"size_score={float(features['size_score']):.4f}",
            "multi_profile_consistency_score="
            f"{float(features['multi_profile_consistency_score']):.4f}",
            f"watermark_penalty={float(features['watermark_penalty']):.4f}",
            f"qr_penalty={float(features['qr_penalty']):.4f}",
            "rectangular_stamp_penalty="
            f"{float(features['rectangular_stamp_penalty']):.4f}",
            f"partial_arc_bonus={float(features['partial_arc_bonus']):.4f}",
            f"candidate_bbox={x1},{y1},{x2},{y2}",
        ]
        return (
            VisualDetection(
                status=status,
                confidence=confidence,
                evidence=evidence,
                warnings=warnings,
                candidate_bbox=[x1, y1, x2, y2],
                candidate_count=len(merged),
                features=features,
                profile_scores=profile_scores,
            ),
            full_profiles,
        )

    def detect(
        self,
        roi: np.ndarray,
        context: SealDetectionContext | None = None,
    ) -> VisualDetection:
        detection, _ = self._analyze(roi, context)
        return detection

    def detect_with_profiles(
        self,
        roi: np.ndarray,
        context: SealDetectionContext | None = None,
    ) -> tuple[VisualDetection, dict[str, np.ndarray]]:
        return self._analyze(roi, context)


class FakeSealDetector(SealDetector):
    def __init__(self, result: VisualDetection) -> None:
        self.result = result

    def detect(
        self,
        roi: np.ndarray,
        context: SealDetectionContext | None = None,
    ) -> VisualDetection:
        return self.result
