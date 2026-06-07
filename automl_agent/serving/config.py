from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class GoogleAuthSettings:
    client_id: Optional[str]
    client_secret: Optional[str]
    session_secret: Optional[str]
    enabled: bool
    allowed_domains: List[str] = field(default_factory=list)
    success_redirect: str = "/schema"
    redirect_uri: Optional[str] = None
    secure_cookies: bool = False

    @classmethod
    def from_env(cls) -> "GoogleAuthSettings":
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        explicit_enabled = os.getenv("GOOGLE_AUTH_ENABLED")
        enabled = _env_bool(explicit_enabled) if explicit_enabled is not None else bool(client_id and client_secret)
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            session_secret=os.getenv("SESSION_SECRET_KEY"),
            enabled=enabled,
            allowed_domains=_csv_env("GOOGLE_ALLOWED_DOMAINS"),
            success_redirect=os.getenv("AUTH_SUCCESS_REDIRECT", "/schema"),
            redirect_uri=os.getenv("GOOGLE_REDIRECT_URI"),
            secure_cookies=_env_bool(os.getenv("SESSION_SECURE_COOKIES"), default=False),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        missing = [
            key
            for key, value in {
                "GOOGLE_CLIENT_ID": self.client_id,
                "GOOGLE_CLIENT_SECRET": self.client_secret,
                "SESSION_SECRET_KEY": self.session_secret,
            }.items()
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Google auth is enabled, but these env vars are missing: {joined}")


@dataclass(frozen=True)
class ServingSettings:
    model_bundle_path: Path
    google_auth: GoogleAuthSettings

    @classmethod
    def from_env(cls) -> "ServingSettings":
        return cls(
            model_bundle_path=Path(os.getenv("AUTOML_MODEL_BUNDLE", "artifacts/run/model_bundle.joblib")),
            google_auth=GoogleAuthSettings.from_env(),
        )

    def validate(self) -> None:
        self.google_auth.validate()


def _csv_env(name: str) -> List[str]:
    return [item.strip().lower() for item in os.getenv(name, "").split(",") if item.strip()]


def _env_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

