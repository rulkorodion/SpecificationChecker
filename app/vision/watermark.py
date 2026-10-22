"""Осторожное подавление низкоконтрастного серого водяного знака."""

from __future__ import annotations

import cv2
import numpy as np


def suppress_gray_watermark(image: np.ndarray) -> np.ndarray:
    """Ослабить только светлые малонасыщенные пиксели, сохраняя чернила."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_saturation = (hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 150) & (hsv[:, :, 2] < 245)
    result = image.copy()
    result[low_saturation] = np.minimum(255, result[low_saturation].astype(np.int16) + 45)
    return result.astype(np.uint8)

