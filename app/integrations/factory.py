"""Adapter factory: the worker depends only on OneCAdapter."""

from app.integrations.base import OneCAdapter, OneCIntegrationError
from app.integrations.http_onec import HttpOneCAdapter
from app.integrations.mock_onec import MockOneCAdapter
from app.integrations.settings import OneCIntegrationSettings


def build_onec_adapter(settings: OneCIntegrationSettings) -> OneCAdapter:
    if settings.adapter == "mock":
        return MockOneCAdapter(settings)
    if settings.adapter in {"http", "onec_http"}:
        return HttpOneCAdapter(settings)
    raise OneCIntegrationError(
        f"Адаптер ONEC_ADAPTER={settings.adapter!r} не реализован; "
        "доступны mock и http"
    )
