"""Детерминированные показатели качества страницы."""

from __future__ import annotations

import cv2
import numpy as np

from app.document.preprocessing import detect_page_quad, perspective_is_material
from app.document.renderer import PageImage
from app.schemas import PageQualityReport


def assess_page_quality(page: PageImage) -> PageQualityReport:
    """Оценить размытие, контраст, яркость, наклон и долю пустого фона."""

    gray = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    empty_ratio = float(np.mean(gray > 245))
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, max(100, page.width // 4))
    angles: list[float] = []
    if lines is not None:
        for line in lines[:30]:
            theta = float(line[0][1])
            angle = theta * 180 / np.pi - 90
            if abs(angle) <= 15:
                angles.append(angle)
    skew = float(np.median(angles)) if angles else 0.0
    perspective = perspective_is_material(page.image, detect_page_quad(page.image))

    warnings: list[str] = []
    if min(page.width, page.height) < 1000:
        warnings.append("Низкое разрешение страницы")
    if blur < 80:
        warnings.append("Возможное размытие")
    if contrast < 30:
        warnings.append("Низкий контраст")
    if brightness > 245:
        warnings.append("Возможный пересвет")
    if brightness < 80:
        warnings.append("Возможное затемнение")
    if abs(skew) > 2:
        warnings.append("Обнаружен наклон")
    if perspective:
        warnings.append("Обнаружено перспективное искажение")
    profile = (
        "geometry_corrected"
        if perspective
        else ("enhanced_gray" if warnings else "original")
    )
    return PageQualityReport(
        page_number=page.page_number,
        width=page.width,
        height=page.height,
        blur_score=round(blur, 3),
        contrast_score=round(contrast, 3),
        skew_angle=round(skew, 3),
        perspective_detected=perspective,
        brightness_score=round(brightness, 3),
        empty_area_ratio=round(empty_ratio, 4),
        warnings=warnings,
        recommended_preprocessing_profile=profile,
    )
