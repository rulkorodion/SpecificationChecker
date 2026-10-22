from __future__ import annotations

import base64
import json
from uuid import uuid4

import pytest

from app.enums import FinalDecision
from app.integrations import (
    CanonicalCheckResult,
    HttpOneCAdapter,
    OneCIntegrationError,
    OneCIntegrationSettings,
    build_onec_adapter,
)
from app.persistence import JobStatus


def canonical_result(**updates) -> CanonicalCheckResult:
    values = {
        "job_id": uuid4(),
        "external_id": "order-uid-42",
        "status": JobStatus.COMPLETED,
        "decision": FinalDecision.RETURN,
        "comment_1c": "ПРОВЕРКА СПЕЦИФИКАЦИИ №42\nРезультат проверки",
    }
    values.update(updates)
    return CanonicalCheckResult(**values)


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"") -> None:
        self.status_code = status_code
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def getcode(self) -> int:
        return self.status_code

    def read(self, _limit: int) -> bytes:
        return self.body


def test_http_adapter_builds_exact_confirmed_onec_payload():
    adapter = HttpOneCAdapter(
        OneCIntegrationSettings(api_url="http://192.168.100.202/API/v1/sendAiAnswer")
    )

    payload = adapter.build_result_payload(canonical_result())

    assert payload == {
        "order_id": "order-uid-42",
        "recomendation": "ПРОВЕРКА СПЕЦИФИКАЦИИ №42\nРезультат проверки",
    }
    assert "decision" not in payload


def test_http_adapter_posts_utf8_json_and_accepts_only_200(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(200, b'{"accepted":true}')

    monkeypatch.setattr("app.integrations.http_onec.urlopen", fake_urlopen)
    settings = OneCIntegrationSettings(
        api_url="http://192.168.100.202/API/v1/sendAiAnswer",
        timeout_seconds=15,
    )
    adapter = HttpOneCAdapter(settings)
    payload = adapter.build_result_payload(canonical_result())

    receipt = adapter.send_result(payload)

    request = captured["request"]
    assert request.full_url == settings.api_url
    assert request.method == "POST"
    assert request.get_header("Content-type") == "application/json; charset=utf-8"
    assert request.get_header("Authorization") is None
    assert captured["timeout"] == 15
    assert json.loads(request.data.decode("utf-8")) == payload
    assert receipt.accepted is True
    assert receipt.external_message_id == "order-uid-42"
    assert receipt.details == {
        "status_code": 200,
        "response_body": '{"accepted":true}',
    }


def test_http_adapter_uses_basic_auth_when_credentials_are_configured(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(200, "ОК".encode("utf-8"))

    monkeypatch.setattr("app.integrations.http_onec.urlopen", fake_urlopen)
    settings = OneCIntegrationSettings(
        api_url="http://onec.local/API/v1/sendAiAnswer",
        auth_username="test-user",
        auth_password="test-password",
        timeout_seconds=15,
    )
    adapter = HttpOneCAdapter(settings)

    receipt = adapter.send_result(adapter.build_result_payload(canonical_result()))

    expected = base64.b64encode(b"test-user:test-password").decode("ascii")
    assert captured["request"].get_header("Authorization") == f"Basic {expected}"
    assert captured["timeout"] == 15
    assert receipt.details["response_body"] == "ОК"


def test_onec_settings_require_complete_unambiguous_authentication():
    with pytest.raises(ValueError, match="должны быть заданы вместе"):
        OneCIntegrationSettings(auth_username="test-user")
    with pytest.raises(ValueError, match="Нельзя одновременно"):
        OneCIntegrationSettings(
            auth_token="token",
            auth_username="test-user",
            auth_password="test-password",
        )


def test_http_adapter_rejects_non_200_response(monkeypatch):
    monkeypatch.setattr(
        "app.integrations.http_onec.urlopen",
        lambda _request, timeout: FakeResponse(202),
    )
    adapter = HttpOneCAdapter(
        OneCIntegrationSettings(api_url="http://onec.local/API/v1/sendAiAnswer")
    )

    with pytest.raises(OneCIntegrationError, match="HTTP 202 вместо 200"):
        adapter.send_result(
            {"order_id": "uid", "recomendation": "report"}
        )


def test_http_adapter_requires_url_order_id_and_comment():
    with pytest.raises(OneCIntegrationError, match="полный ONEC_API_URL"):
        HttpOneCAdapter(OneCIntegrationSettings(adapter="http"))

    adapter = HttpOneCAdapter(
        OneCIntegrationSettings(api_url="http://onec.local/API/v1/sendAiAnswer")
    )
    with pytest.raises(OneCIntegrationError, match="external_id"):
        adapter.build_result_payload(canonical_result(external_id=None))
    with pytest.raises(OneCIntegrationError, match="comment_1c"):
        adapter.build_result_payload(canonical_result(comment_1c=None))


def test_factory_builds_http_adapter():
    settings = OneCIntegrationSettings(
        adapter="http",
        api_url="http://onec.local/API/v1/sendAiAnswer",
    )

    assert isinstance(build_onec_adapter(settings), HttpOneCAdapter)
