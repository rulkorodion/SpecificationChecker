"""Environment/config settings for the isolated 1C integration layer."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} должен быть true/false")


class OneCIntegrationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    adapter: str = "mock"
    api_url: str | None = None
    auth_token: SecretStr | None = None
    auth_username: str | None = None
    auth_password: SecretStr | None = None
    timeout_seconds: float = Field(default=15.0, gt=0, le=600)
    max_delivery_attempts: int = Field(default=3, ge=1, le=20)
    mock_outbox_dir: Path = Path("runtime/onec/mock_outbox")

    @field_validator("adapter")
    @classmethod
    def normalize_adapter(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("ONEC_ADAPTER не может быть пустым")
        return normalized

    @field_validator("auth_username", mode="before")
    @classmethod
    def normalize_auth_username(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def validate_authentication(self) -> "OneCIntegrationSettings":
        has_username = self.auth_username is not None
        has_password = self.auth_password is not None
        if has_username != has_password:
            raise ValueError(
                "ONEC_AUTH_USERNAME и ONEC_AUTH_PASSWORD должны быть заданы вместе"
            )
        if self.auth_token is not None and has_username:
            raise ValueError(
                "Нельзя одновременно задавать Bearer-токен и Basic-аутентификацию"
            )
        return self

    @classmethod
    def from_environment(
        cls, config: Mapping[str, Any] | None = None
    ) -> "OneCIntegrationSettings":
        section = dict((config or {}).get("onec_integration", {}))
        token = os.getenv("ONEC_AUTH_TOKEN")
        username = os.getenv("ONEC_AUTH_USERNAME")
        password = os.getenv("ONEC_AUTH_PASSWORD")
        api_url = os.getenv("ONEC_API_URL", str(section.get("api_url", ""))).strip()
        return cls(
            enabled=_env_bool(
                "ONEC_INTEGRATION_ENABLED", bool(section.get("enabled", False))
            ),
            adapter=os.getenv("ONEC_ADAPTER", str(section.get("adapter", "mock"))),
            api_url=api_url or None,
            auth_token=SecretStr(token) if token else None,
            auth_username=username or None,
            auth_password=SecretStr(password) if password else None,
            timeout_seconds=float(
                os.getenv("ONEC_TIMEOUT_SECONDS", section.get("timeout_seconds", 15))
            ),
            max_delivery_attempts=int(
                os.getenv(
                    "ONEC_MAX_DELIVERY_ATTEMPTS",
                    section.get("max_delivery_attempts", 3),
                )
            ),
            mock_outbox_dir=Path(
                os.getenv(
                    "ONEC_MOCK_OUTBOX_DIR",
                    str(section.get("mock_outbox_dir", "runtime/onec/mock_outbox")),
                )
            ),
        )
