"""Понятные исключения прикладного уровня."""


class SpecificationCheckerError(Exception):
    """Базовая ошибка пилота."""


class InputValidationError(SpecificationCheckerError):
    """Некорректный входной файл или конфигурация."""


class DocumentProcessingError(SpecificationCheckerError):
    """Документ невозможно безопасно открыть или отрендерить."""


class PageSelectionError(DocumentProcessingError):
    """Нужное приложение нельзя однозначно выбрать из многостраничного PDF."""

    def __init__(
        self,
        diagnostic_code: str,
        message: str,
        *,
        artifact_path=None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_code = diagnostic_code
        self.artifact_path = artifact_path


class OCREngineError(SpecificationCheckerError):
    """OCR не инициализировался или не завершил распознавание."""
