"""Умеренные OpenCV-профили без безусловной агрессивной бинаризации."""

from __future__ import annotations

import cv2
import numpy as np


def _order_quad(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def detect_page_quad(image: np.ndarray) -> np.ndarray | None:
    """Найти границы листа/крупной таблицы без подгонки под фиксированные координаты."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(1.0, 1200 / max(height, width))
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    page_area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        if cv2.contourArea(contour) < page_area * 0.42:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return _order_quad(polygon.reshape(4, 2).astype(np.float32) / scale)
    return None


def perspective_is_material(image: np.ndarray, quad: np.ndarray | None) -> bool:
    if quad is None:
        return False
    height, width = image.shape[:2]
    expected = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    deviation = np.linalg.norm((quad - expected) / np.array([width, height]), axis=1)
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    opposite_mismatch = max(
        abs(top - bottom) / max(top, bottom, 1),
        abs(left - right) / max(left, right, 1),
    )
    return float(deviation.max()) > 0.025 or opposite_mismatch > 0.035


def _perspective_correct(image: np.ndarray) -> np.ndarray:
    quad = detect_page_quad(image)
    if not perspective_is_material(image, quad):
        return image
    assert quad is not None
    height, width = image.shape[:2]
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _deskew(gray: np.ndarray) -> np.ndarray:
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    points = np.column_stack(np.where(inverted > 0))
    if len(points) < 100:
        return gray
    angle = cv2.minAreaRect(points.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 10:
        return gray
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=255
    )


def preprocess(image: np.ndarray, profile: str) -> np.ndarray:
    """Вернуть один из детерминированных OCR-профилей."""

    if profile == "original":
        return image.copy()
    source = _perspective_correct(image) if profile == "geometry_corrected" else image
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    if profile in {"enhanced_gray", "geometry_corrected"}:
        return enhanced
    if profile == "thresholded":
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    raise ValueError(f"Неизвестный профиль предобработки: {profile}")
