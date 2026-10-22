"""Interface boundary for replaceable 1C connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.integrations.contracts import (
    CanonicalCheckResult,
    CanonicalDocument,
    DeliveryReceipt,
)
from app.schemas import ReferenceData


class OneCIntegrationError(RuntimeError):
    """External adapter conversion, retrieval or delivery failed."""


class OneCAdapter(ABC):
    """The only contract a future HTTP/1C implementation must implement."""

    name: str

    @abstractmethod
    def reference_to_internal(
        self, source: Mapping[str, Any] | str | Path
    ) -> ReferenceData:
        """Convert a 1C-specific payload into canonical ReferenceData."""

    @abstractmethod
    def get_document(self, source: Any) -> CanonicalDocument:
        """Retrieve a document and return its trusted canonical description."""

    @abstractmethod
    def build_result_payload(
        self, result: CanonicalCheckResult
    ) -> dict[str, Any]:
        """Map the canonical result into the concrete 1C wire payload."""

    @abstractmethod
    def send_result(self, payload: Mapping[str, Any]) -> DeliveryReceipt:
        """Deliver an already-built payload without invoking OCR."""

    def close(self) -> None:
        """Release adapter resources; mock and stateless adapters do nothing."""
