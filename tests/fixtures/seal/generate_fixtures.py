"""Generate deterministic buyer-block images for seal-detector regression tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


WIDTH = 1000
HEIGHT = 620
OUTPUT = Path(__file__).resolve().parent


def canvas() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "BUYER:",
        (380, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.line(image, (250, 440), (850, 440), (45, 45, 45), 3, cv2.LINE_AA)
    return image


def draw_seal(
    image: np.ndarray,
    center: tuple[int, int] = (430, 320),
    radius: int = 145,
    color: tuple[int, int, int] = (205, 70, 30),
) -> None:
    cv2.circle(image, center, radius, color, 6, cv2.LINE_AA)
    cv2.circle(image, center, int(radius * 0.78), color, 4, cv2.LINE_AA)
    cv2.circle(image, center, int(radius * 0.28), color, 3, cv2.LINE_AA)
    for angle in range(0, 360, 24):
        radians = np.deg2rad(angle)
        inner = int(radius * 0.80)
        outer = int(radius * 0.94)
        p1 = (
            int(center[0] + np.cos(radians) * inner),
            int(center[1] + np.sin(radians) * inner),
        )
        p2 = (
            int(center[0] + np.cos(radians) * outer),
            int(center[1] + np.sin(radians) * outer),
        )
        cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)
    cv2.putText(
        image,
        "COMPANY",
        (center[0] - 82, center[1] + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        color,
        3,
        cv2.LINE_AA,
    )


def draw_signature(image: np.ndarray, color: tuple[int, int, int] = (170, 45, 25)) -> None:
    points = np.array(
        [
            [290, 390],
            [350, 310],
            [385, 410],
            [430, 275],
            [470, 400],
            [540, 300],
            [610, 385],
            [700, 325],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [points], False, color, 6, cv2.LINE_AA)
    cv2.line(image, (330, 370), (750, 350), color, 4, cv2.LINE_AA)


def draw_qr(image: np.ndarray) -> None:
    modules = 29
    cell = 12
    left, top = 330, 160
    rng = np.random.default_rng(76930)
    matrix = rng.integers(0, 2, size=(modules, modules), dtype=np.uint8)

    def finder(row: int, column: int) -> None:
        matrix[row : row + 7, column : column + 7] = 1
        matrix[row + 1 : row + 6, column + 1 : column + 6] = 0
        matrix[row + 2 : row + 5, column + 2 : column + 5] = 1

    finder(0, 0)
    finder(0, modules - 7)
    finder(modules - 7, 0)
    for row in range(modules):
        for column in range(modules):
            if matrix[row, column]:
                x1 = left + column * cell
                y1 = top + row * cell
                cv2.rectangle(
                    image, (x1, y1), (x1 + cell, y1 + cell), (20, 20, 20), -1
                )


def save(name: str, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Could not encode {name}")
    encoded.tofile(str(OUTPUT / name))


def generate() -> None:
    positive = canvas()
    draw_seal(positive)
    save("buyer_block_with_seal.png", positive)

    signature = canvas()
    draw_signature(signature)
    save("signature_only.png", signature)

    qr = canvas()
    draw_qr(qr)
    save("qr_code.png", qr)

    watermark = canvas()
    cv2.putText(
        watermark,
        "WATERMARK",
        (40, 365),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (215, 215, 215),
        16,
        cv2.LINE_AA,
    )
    save("watermark_only.png", watermark)

    rectangle = canvas()
    cv2.rectangle(rectangle, (260, 190), (750, 415), (145, 75, 50), 7)
    cv2.rectangle(rectangle, (285, 215), (725, 390), (145, 75, 50), 4)
    cv2.putText(
        rectangle,
        "APPROVED",
        (330, 325),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (145, 75, 50),
        5,
        cv2.LINE_AA,
    )
    save("rectangular_stamp.png", rectangle)

    pale = canvas()
    draw_seal(pale, color=(250, 225, 195))
    save("pale_blue_seal.png", pale)

    grayscale = canvas()
    draw_seal(grayscale, color=(105, 105, 105))
    save("grayscale_seal.png", grayscale)

    crossed = canvas()
    draw_seal(crossed)
    cv2.line(crossed, (180, 330), (790, 330), (25, 25, 25), 8, cv2.LINE_AA)
    save("seal_crossed_by_line.png", crossed)

    cropped = canvas()
    draw_seal(cropped, center=(85, 325), radius=150)
    save("partially_cropped_seal.png", cropped)

    overlapped = canvas()
    draw_seal(overlapped)
    draw_signature(overlapped)
    save("seal_overlapped_by_signature.png", overlapped)

    circle = canvas()
    cv2.circle(circle, (440, 320), 140, (80, 80, 80), 8, cv2.LINE_AA)
    save("round_graphic_only.png", circle)

    supplier = canvas()
    cv2.putText(
        supplier,
        "SUPPLIER",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    draw_seal(supplier, center=(105, 310), radius=92)
    save("supplier_seal_outside_buyer_area.png", supplier)

    save("empty_buyer_block.png", canvas())


if __name__ == "__main__":
    generate()
