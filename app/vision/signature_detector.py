"""Независимый многопрофильный детектор рукописной подписи."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import atan2
from statistics import median

import cv2
import numpy as np

from app.enums import PresenceStatus
from app.vision.models import VisualDetection


@dataclass(frozen=True, slots=True)
class SignatureDetectionContext:
    roi_name: str = "unknown"
    anchor_bbox: tuple[int, int, int, int] | None = None
    anchored: bool = False


class SignatureDetector(ABC):
    @abstractmethod
    def detect(
        self,
        roi: np.ndarray,
        context: SignatureDetectionContext | None = None,
    ) -> VisualDetection:
        """Определить наличие подписи, не устанавливая личность."""

    def detect_with_profiles(
        self,
        roi: np.ndarray,
        context: SignatureDetectionContext | None = None,
    ) -> tuple[VisualDetection, dict[str, np.ndarray]]:
        return self.detect(roi, context), {}


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    scale = min(1.0, 1000.0 / max(height, width))
    source = (
        cv2.resize(
            mask,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else mask.copy()
    )
    skeleton = np.zeros_like(source)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(source) and cv2.countNonZero(skeleton) < source.size:
        opened = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(source, opened))
        source = cv2.erode(source, kernel)
    if scale < 1.0:
        return cv2.resize(skeleton, (width, height), interpolation=cv2.INTER_NEAREST)
    return skeleton


class OpenCVSignatureDetector(SignatureDetector):
    """Ансамбль признаков нерегулярных рукописных штрихов."""

    def __init__(self, min_confidence: float = 0.85) -> None:
        self.min_confidence = min_confidence

    @staticmethod
    def preprocessing_profiles(roi: np.ndarray) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        saturation = cv2.createCLAHE(2.0, (8, 8)).apply(hsv[:, :, 1])
        saturation_mask = cv2.threshold(
            saturation, 34, 255, cv2.THRESH_BINARY
        )[1]
        blue_mask = cv2.inRange(hsv, np.array([75, 28, 45]), np.array([145, 255, 255]))
        dark_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        combined = cv2.bitwise_or(dark_mask, saturation_mask)
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(25, roi.shape[1] // 9), 1)
        )
        horizontal_lines = cv2.morphologyEx(
            combined, cv2.MORPH_OPEN, horizontal_kernel
        )
        # Удаляется только прямолинейное ядро; кривые пересекающие штрихи остаются.
        line_suppressed = cv2.subtract(
            combined,
            cv2.dilate(
                horizontal_lines,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
            ),
        )
        line_suppressed = cv2.morphologyEx(
            line_suppressed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2)),
        )
        seal_suppressed = line_suppressed.copy()
        circle_scale = min(1.0, 1200.0 / max(gray.shape[:2]))
        circle_gray = (
            cv2.resize(
                gray,
                (
                    max(1, int(gray.shape[1] * circle_scale)),
                    max(1, int(gray.shape[0] * circle_scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            if circle_scale < 1.0
            else gray
        )
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(circle_gray, (7, 7), 1.4),
            cv2.HOUGH_GRADIENT,
            dp=1.3,
            minDist=max(25, min(circle_gray.shape[:2]) // 4),
            param1=90,
            param2=35,
            minRadius=max(12, min(circle_gray.shape[:2]) // 14),
            maxRadius=max(18, min(circle_gray.shape[:2]) // 2),
        )
        if circles is not None:
            for x, y, radius in np.round(circles[0]).astype(int)[:4]:
                if circle_scale < 1.0:
                    x = int(round(x / circle_scale))
                    y = int(round(y / circle_scale))
                    radius = int(round(radius / circle_scale))
                ring = np.zeros_like(gray)
                cv2.circle(ring, (x, y), radius, 255, max(2, radius // 18))
                seal_suppressed = cv2.subtract(seal_suppressed, ring)
        skeleton = _skeletonize(line_suppressed)
        return {
            "original_color": roi.copy(),
            "grayscale": gray,
            "saturation_enhanced": saturation,
            "blue_ink_mask": blue_mask,
            "dark_ink_mask": dark_mask,
            "horizontal_lines": horizontal_lines,
            "horizontal_line_suppressed": line_suppressed,
            "seal_suppressed": seal_suppressed,
            "skeletonized_strokes": skeleton,
            "combined_handwriting_mask": cv2.bitwise_or(
                line_suppressed, cv2.bitwise_or(blue_mask, seal_suppressed)
            ),
        }

    @staticmethod
    def _analysis_region(
        shape: tuple[int, ...],
        context: SignatureDetectionContext,
    ) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        if context.anchor_bbox is None:
            return 0, 0, width, height
        ax1, ay1, ax2, ay2 = context.anchor_bbox
        return (
            max(0, int(ax1 - width * 0.18)),
            max(0, int(ay1 - height * 0.10)),
            width,
            min(height, int(ay2 + height * 0.78)),
        )

    @staticmethod
    def _candidate_features(
        mask: np.ndarray,
        horizontal_lines: np.ndarray,
        context: SignatureDetectionContext,
        detailed: bool = True,
    ) -> dict[str, float | int | list[int] | None]:
        x0, y0, x1, y1 = OpenCVSignatureDetector._analysis_region(
            mask.shape, context
        )
        working = np.zeros_like(mask)
        working[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        contours, _ = cv2.findContours(
            working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        total = float(max(1, (y1 - y0) * (x1 - x0)))
        selected: list[np.ndarray] = []
        orientations: list[float] = []
        curvature_values: list[float] = []
        longest_ratio = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            if (
                area / total < 0.000035
                or width < max(6, mask.shape[1] * 0.025)
                or height < 2
            ):
                continue
            aspect = width / max(height, 1)
            perimeter = cv2.arcLength(contour, False)
            diagonal = float(np.hypot(width, height))
            circularity = (
                4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
            )
            if 0.75 <= aspect <= 1.35 and circularity > 0.62:
                continue
            curvature = min(1.0, max(0.0, perimeter / max(diagonal, 1.0) - 1.0) / 5.0)
            if aspect < 1.20 and curvature < 0.18:
                continue
            selected.append(contour)
            longest_ratio = max(longest_ratio, width / max(mask.shape[1], 1))
            curvature_values.append(curvature)
            if len(contour) >= 5:
                vx, vy, _, _ = cv2.fitLine(
                    contour, cv2.DIST_L2, 0, 0.01, 0.01
                ).reshape(-1)
                orientations.append(float(atan2(float(vy), float(vx))))

        candidate_bbox: list[int] | None = None
        candidate_span_ratio = 0.0
        candidate_mask = np.zeros_like(mask)
        if selected:
            for contour in selected:
                cv2.drawContours(candidate_mask, [contour], -1, 255, 2)
            points = np.vstack(selected)
            x, y, width, height = cv2.boundingRect(points)
            candidate_bbox = [x, y, x + width, y + height]
            candidate_span_ratio = width / max(mask.shape[1], 1)
        irregular_score = min(1.0, len(selected) / 5.0)
        curvature_score = (
            float(median(curvature_values)) if curvature_values else 0.0
        )
        orientation_variance = (
            min(1.0, float(np.std(orientations)) / 0.75)
            if len(orientations) >= 2
            else 0.0
        )
        dilated_lines = cv2.dilate(
            horizontal_lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        )
        intersection = cv2.bitwise_and(candidate_mask, dilated_lines)
        candidate_pixels = max(1, cv2.countNonZero(candidate_mask))
        crossing_score = min(
            1.0, cv2.countNonZero(intersection) / candidate_pixels * 8.0
        )
        ink_ratio = float(np.mean(working > 0))
        density_score = min(1.0, ink_ratio / 0.035) if ink_ratio <= 0.12 else max(
            0.0, 1.0 - (ink_ratio - 0.12) * 4.0
        )
        hough_scale = min(1.0, 1200.0 / max(working.shape[:2]))
        hough_mask = (
            cv2.resize(
                working,
                (
                    max(1, int(working.shape[1] * hough_scale)),
                    max(1, int(working.shape[0] * hough_scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            if hough_scale < 1.0
            else working
        )
        lines = cv2.HoughLinesP(
            hough_mask,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=max(12, int(hough_mask.shape[1] * 0.06)),
            maxLineGap=12,
        ) if detailed else None
        line_angles: list[float] = []
        if lines is not None:
            for lx1, ly1, lx2, ly2 in lines[:, 0]:
                line_angles.append(
                    abs(float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))) % 180
                )
        diagonal = sum(
            15 < angle < 75 or 105 < angle < 165 for angle in line_angles
        )
        axis_aligned = sum(
            angle <= 8
            or angle >= 172
            or 82 <= angle <= 98
            for angle in line_angles
        )
        diagonal_score = diagonal / len(line_angles) if line_angles else 0.0
        rectangular_penalty = (
            axis_aligned / len(line_angles)
            if len(line_angles) >= 20 and axis_aligned / len(line_angles) >= 0.82
            else 0.0
        )
        return {
            "elongated_components": len(selected),
            "longest_width_ratio": longest_ratio,
            "irregular_stroke_score": irregular_score,
            "curvature_score": curvature_score,
            "orientation_variance_score": orientation_variance,
            "crossing_score": crossing_score,
            "baseline_overlap_score": crossing_score,
            "handwriting_density_score": density_score,
            "ink_ratio": ink_ratio,
            "candidate_span_ratio": candidate_span_ratio,
            "diagonal_stroke_score": diagonal_score,
            "hough_line_count": len(line_angles),
            "rectangular_stamp_penalty": rectangular_penalty,
            "candidate_bbox": candidate_bbox,
        }

    def detect(
        self,
        roi: np.ndarray,
        context: SignatureDetectionContext | None = None,
    ) -> VisualDetection:
        detection, _ = self.detect_with_profiles(roi, context)
        return detection

    def detect_with_profiles(
        self,
        roi: np.ndarray,
        context: SignatureDetectionContext | None = None,
    ) -> tuple[VisualDetection, dict[str, np.ndarray]]:
        if roi.size == 0:
            return (
                VisualDetection(
                    status=PresenceStatus.UNCERTAIN,
                    confidence=0.0,
                    warnings=["Пустая ROI покупателя"],
                ),
                {},
            )
        context = context or SignatureDetectionContext()
        profiles = self.preprocessing_profiles(roi)
        primary = self._candidate_features(
            profiles["combined_handwriting_mask"],
            profiles["horizontal_lines"],
            context,
        )
        consistency_hits = 0
        for name in (
            "blue_ink_mask",
            "dark_ink_mask",
            "horizontal_line_suppressed",
            "seal_suppressed",
        ):
            features = self._candidate_features(
                profiles[name], profiles["horizontal_lines"], context, detailed=False
            )
            if int(features["elongated_components"]) >= 2:
                consistency_hits += 1
        consistency = consistency_hits / 4.0

        removed_ring_pixels = cv2.countNonZero(
            cv2.subtract(
                profiles["horizontal_line_suppressed"],
                profiles["seal_suppressed"],
            )
        )
        seal_overlap = 1.0 if removed_ring_pixels > roi.size * 0.00015 else 0.0

        qr_detected = False
        try:
            qr_scale = min(1.0, 1200.0 / max(roi.shape[:2]))
            qr_image = (
                cv2.resize(
                    roi,
                    (
                        max(1, int(roi.shape[1] * qr_scale)),
                        max(1, int(roi.shape[0] * qr_scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                if qr_scale < 1.0
                else roi
            )
            qr_detected, _ = cv2.QRCodeDetector().detect(qr_image)
        except Exception:
            qr_detected = False
        # Большое число линий встречается и у настоящей подписи поверх печати
        # или водяного знака, поэтому само по себе не является признаком QR.
        qr_penalty = 1.0 if qr_detected else 0.0
        # OpenCV QRCodeDetector occasionally treats dense lettering inside a
        # round blue seal as a QR marker.  In the trusted lower-right signing
        # row, ignore that false positive only when several independent
        # handwriting features confirm long curved strokes crossing the seal.
        qr_signature_override = (
            qr_penalty >= 0.5
            and context.roi_name == "fallback_lower_right_signature_row"
            and seal_overlap >= 0.5
            and int(primary["elongated_components"]) >= 3
            and float(primary["curvature_score"]) >= 0.45
            and float(primary["candidate_span_ratio"]) >= 0.65
            and float(primary["crossing_score"]) >= 0.30
            and consistency >= 0.75
            and float(primary["rectangular_stamp_penalty"]) < 0.5
        )
        effective_qr_penalty = 0.0 if qr_signature_override else qr_penalty

        broad_handwriting = (
            int(primary["elongated_components"]) >= 1
            and float(primary["candidate_span_ratio"]) >= 0.45
            and float(primary["diagonal_stroke_score"]) >= 0.40
            and float(primary["curvature_score"]) >= 0.30
        )
        compact_complex_handwriting = (
            int(primary["elongated_components"]) >= 2
            and float(primary["candidate_span_ratio"]) >= 0.20
            and float(primary["diagonal_stroke_score"]) >= 0.65
            and float(primary["curvature_score"]) >= 0.35
            and float(primary["orientation_variance_score"]) >= 0.25
        )
        wide_curved_handwriting = (
            int(primary["elongated_components"]) >= 3
            and float(primary["candidate_span_ratio"]) >= 0.55
            and float(primary["curvature_score"]) >= 0.40
            and float(primary["orientation_variance_score"]) >= 0.25
            and float(primary["ink_ratio"]) <= 0.18
            and float(primary["rectangular_stamp_penalty"]) < 0.5
        )
        baseline_crossing_handwriting = (
            int(primary["elongated_components"]) >= 3
            and float(primary["candidate_span_ratio"]) >= 0.35
            and float(primary["curvature_score"]) >= 0.35
            and float(primary["crossing_score"]) >= 0.20
            and float(primary["ink_ratio"]) <= 0.12
            and float(primary["rectangular_stamp_penalty"]) < 0.5
        )
        clear_handwriting_bonus = (
            0.15
            if (
                broad_handwriting
                or compact_complex_handwriting
                or wide_curved_handwriting
                or baseline_crossing_handwriting
            )
            else 0.0
        )
        score = max(0.0, min(
            0.98,
            0.30
            + 0.14 * float(primary["irregular_stroke_score"])
            + 0.10 * float(primary["curvature_score"])
            + 0.12 * min(1.0, float(primary["longest_width_ratio"]) / 0.20)
            + 0.08 * float(primary["orientation_variance_score"])
            + 0.07 * float(primary["handwriting_density_score"])
            + 0.08 * consistency
            + 0.05 * float(primary["crossing_score"])
            + 0.03 * float(primary["baseline_overlap_score"])
            + 0.10 * float(primary["candidate_span_ratio"])
            + 0.08 * float(primary["diagonal_stroke_score"])
            + clear_handwriting_bonus
            - 0.28 * effective_qr_penalty
            - 0.28 * float(primary["rectangular_stamp_penalty"]),
        ))
        trusted_buyer_geometry = (
            context.roi_name == "fallback_lower_right_signature_row"
        )
        anchored = (
            context.anchored
            or context.roi_name == "unknown"
            or trusted_buyer_geometry
        )
        present = (
            anchored
            and score >= self.min_confidence
            and int(primary["elongated_components"]) >= 1
            and float(primary["irregular_stroke_score"]) >= 0.20
            and (
                float(primary["candidate_span_ratio"]) >= 0.35
                or broad_handwriting
                or compact_complex_handwriting
                or wide_curved_handwriting
                or baseline_crossing_handwriting
            )
            and (
                float(primary["diagonal_stroke_score"]) >= 0.30
                or wide_curved_handwriting
                or baseline_crossing_handwriting
                or qr_signature_override
            )
            and effective_qr_penalty < 0.5
            and float(primary["rectangular_stamp_penalty"]) < 0.5
        )
        if present:
            status = PresenceStatus.PRESENT
            warnings: list[str] = []
        elif (
            float(primary["ink_ratio"]) < 0.0025
            and int(primary["elongated_components"]) == 0
        ):
            status = PresenceStatus.ABSENT
            score = max(0.86, 1.0 - float(primary["ink_ratio"]) * 20.0)
            warnings = []
        else:
            status = PresenceStatus.UNCERTAIN
            score = min(score, self.min_confidence - 0.01)
            warnings = [
                "Рукописные штрихи не отделены от печати/линии с достаточной уверенностью"
            ]
        features = {
            key: value
            for key, value in primary.items()
            if key != "candidate_bbox"
        }
        features.update(
            {
                "stroke_length_score": min(
                    1.0, float(primary["longest_width_ratio"]) / 0.20
                ),
                "seal_overlap_score": seal_overlap,
                "qr_penalty": qr_penalty,
                "effective_qr_penalty": effective_qr_penalty,
                "qr_signature_override": qr_signature_override,
                "multi_profile_consistency_score": consistency,
                "clear_handwriting_bonus": clear_handwriting_bonus,
                "wide_curved_handwriting": wide_curved_handwriting,
                "baseline_crossing_handwriting": baseline_crossing_handwriting,
                "roi_name": context.roi_name,
                "anchored": context.anchored,
            }
        )
        evidence = [
            f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in features.items()
        ]
        return (
            VisualDetection(
                status=status,
                confidence=score,
                evidence=evidence,
                warnings=warnings,
                candidate_bbox=primary["candidate_bbox"],
                candidate_count=int(primary["elongated_components"]),
                features=features,
                profile_scores={
                    "combined_handwriting_mask": score,
                    "multi_profile_consistency": consistency,
                },
            ),
            profiles,
        )


class FakeSignatureDetector(SignatureDetector):
    def __init__(self, result: VisualDetection) -> None:
        self.result = result

    def detect(
        self,
        roi: np.ndarray,
        context: SignatureDetectionContext | None = None,
    ) -> VisualDetection:
        return self.result
