"""Public package interface for Specification Checker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from app.service import SpecificationCheckerService, check_specification

__all__ = ["SpecificationCheckerService", "check_specification", "__version__"]


def __getattr__(name: str) -> Any:
    """Load the OCR service only when its public symbols are requested.

    Persistence-only processes such as Alembic and the lightweight API image
    can import ``app.*`` without importing NumPy/OpenCV or an OCR runtime.
    """

    if name in {"SpecificationCheckerService", "check_specification"}:
        from app import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
