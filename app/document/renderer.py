"""Рендеринг всех страниц PDF и растровых изображений."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.document.loader import LoadedDocument
from app.exceptions import DocumentProcessingError


@dataclass(slots=True)
class PageImage:
    page_number: int
    image: np.ndarray
    width: int
    height: int


class DocumentRenderer:
    """Преобразовать PDF/изображение в BGR-страницы."""

    def __init__(self, dpi: int = 300, max_pages: int = 20) -> None:
        self.dpi = dpi
        self.max_pages = max_pages

    def render(self, document: LoadedDocument) -> list[PageImage]:
        if document.extension == ".pdf":
            return self._render_pdf(document.path)
        return self._render_image(document.path)

    def _render_pdf(self, path: Path) -> list[PageImage]:
        try:
            pdf = fitz.open(path)
        except Exception as exc:
            raise DocumentProcessingError(f"PDF повреждён или не открывается: {exc}") from exc
        try:
            if pdf.needs_pass:
                raise DocumentProcessingError("PDF защищён паролем")
            if pdf.page_count < 1:
                raise DocumentProcessingError("PDF не содержит страниц")
            if pdf.page_count > self.max_pages:
                raise DocumentProcessingError(
                    f"Страниц {pdf.page_count}, превышен max_pages={self.max_pages}"
                )
            scale = self.dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            pages: list[PageImage] = []
            for index in range(pdf.page_count):
                pix = pdf[index].get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
                rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                pages.append(PageImage(index + 1, bgr, pix.width, pix.height))
            return pages
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(f"Ошибка рендеринга PDF: {exc}") from exc
        finally:
            pdf.close()

    def _render_image(self, path: Path) -> list[PageImage]:
        try:
            image = Image.open(path)
            pages: list[PageImage] = []
            frame = 0
            while True:
                image.seek(frame)
                oriented = ImageOps.exif_transpose(image.copy()).convert("RGBA")
                background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
                rgb = Image.alpha_composite(background, oriented).convert("RGB")
                array = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                height, width = array.shape[:2]
                pages.append(PageImage(frame + 1, array, width, height))
                frame += 1
                if frame >= self.max_pages:
                    try:
                        image.seek(frame)
                    except EOFError:
                        break
                    raise DocumentProcessingError(
                        f"Страниц больше max_pages={self.max_pages}"
                    )
        except EOFError:
            pass
        except (OSError, UnidentifiedImageError) as exc:
            raise DocumentProcessingError(f"Изображение повреждено: {exc}") from exc
        if not pages:
            raise DocumentProcessingError("Изображение не содержит страниц")
        return pages


def save_page(path: Path, image: np.ndarray) -> None:
    """Сохранить изображение с поддержкой Unicode-путей Windows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise DocumentProcessingError(f"Не удалось кодировать изображение: {path}")
    encoded.tofile(str(path))

