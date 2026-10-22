"""Replaceable adapters for external document systems such as 1C."""

from app.integrations.base import OneCAdapter, OneCIntegrationError
from app.integrations.contracts import (
    CanonicalCheckInput,
    CanonicalCheckResult,
    CanonicalCheckSummary,
    CanonicalDocument,
    DeliveryReceipt,
)
from app.integrations.factory import build_onec_adapter
from app.integrations.http_onec import HttpOneCAdapter
from app.integrations.mock_onec import MockOneCAdapter
from app.integrations.service import OneCIntegrationService
from app.integrations.settings import OneCIntegrationSettings

__all__ = [
    "CanonicalCheckInput",
    "CanonicalCheckResult",
    "CanonicalCheckSummary",
    "CanonicalDocument",
    "DeliveryReceipt",
    "HttpOneCAdapter",
    "MockOneCAdapter",
    "OneCAdapter",
    "OneCIntegrationError",
    "OneCIntegrationService",
    "OneCIntegrationSettings",
    "build_onec_adapter",
]
