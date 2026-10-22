"""HTTP callback adapter for the confirmed 1C result contract."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.integrations.base import OneCIntegrationError
from app.integrations.contracts import CanonicalCheckResult, DeliveryReceipt
from app.integrations.mock_onec import MockOneCAdapter
from app.integrations.settings import OneCIntegrationSettings


class HttpOneCAdapter(MockOneCAdapter):
    """POST the human-readable result to the 1C HTTP publication."""

    name = "http"

    def __init__(self, settings: OneCIntegrationSettings) -> None:
        super().__init__(settings)
        api_url = (settings.api_url or "").strip()
        parsed = urlsplit(api_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OneCIntegrationError(
                "Для HTTP-адаптера требуется полный ONEC_API_URL с http/https"
            )
        self.api_url = api_url

    def build_result_payload(
        self, result: CanonicalCheckResult
    ) -> dict[str, Any]:
        order_id = (result.external_id or "").strip()
        if not order_id:
            raise OneCIntegrationError(
                "Невозможно отправить ответ в 1С: отсутствует external_id заказа"
            )
        recommendation = (result.comment_1c or "").strip()
        if not recommendation:
            raise OneCIntegrationError(
                "Невозможно отправить ответ в 1С: отсутствует comment_1c"
            )
        return {
            "order_id": order_id,
            # Field spelling is fixed by the 1C contract.
            "recomendation": recommendation,
        }

    def send_result(self, payload: Mapping[str, Any]) -> DeliveryReceipt:
        body = json.dumps(
            dict(payload), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        if self.settings.auth_username is not None:
            assert self.settings.auth_password is not None
            credentials = (
                f"{self.settings.auth_username}:"
                f"{self.settings.auth_password.get_secret_value()}"
            ).encode("utf-8")
            headers["Authorization"] = (
                "Basic " + base64.b64encode(credentials).decode("ascii")
            )
        elif self.settings.auth_token is not None:
            headers["Authorization"] = (
                f"Bearer {self.settings.auth_token.get_secret_value()}"
            )
        request = Request(
            self.api_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                status_code = int(response.getcode())
                response_text = response.read(65536).decode(
                    "utf-8", errors="replace"
                )
        except HTTPError as exc:
            raise OneCIntegrationError(
                f"1С вернула HTTP {exc.code} вместо 200"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OneCIntegrationError(
                "Не удалось доставить результат в 1С"
            ) from exc

        if status_code != 200:
            raise OneCIntegrationError(
                f"1С вернула HTTP {status_code} вместо 200"
            )
        order_id = str(payload.get("order_id", "")).strip()
        return DeliveryReceipt(
            adapter=self.name,
            accepted=True,
            sent_at=datetime.now(timezone.utc),
            external_message_id=order_id or None,
            details={
                "status_code": status_code,
                "response_body": response_text,
            },
        )
