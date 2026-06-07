from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from authlib.integrations.base_client import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse


GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


@dataclass
class GoogleAuthSettings:
    client_id: Optional[str]
    client_secret: Optional[str]
    session_secret: Optional[str]
    enabled: bool
    allowed_domains: List[str]
    success_redirect: str
    redirect_uri: Optional[str]
    secure_cookies: bool

    @classmethod
    def from_env(cls) -> "GoogleAuthSettings":
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        session_secret = os.getenv("SESSION_SECRET_KEY")
        explicit_enabled = os.getenv("GOOGLE_AUTH_ENABLED")
        enabled = _env_bool(explicit_enabled) if explicit_enabled is not None else bool(client_id and client_secret)
        allowed_domains = [
            domain.strip().lower()
            for domain in os.getenv("GOOGLE_ALLOWED_DOMAINS", "").split(",")
            if domain.strip()
        ]
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            session_secret=session_secret,
            enabled=enabled,
            allowed_domains=allowed_domains,
            success_redirect=os.getenv("AUTH_SUCCESS_REDIRECT", "/schema"),
            redirect_uri=os.getenv("GOOGLE_REDIRECT_URI"),
            secure_cookies=_env_bool(os.getenv("SESSION_SECURE_COOKIES"), default=False),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        missing = []
        if not self.client_id:
            missing.append("GOOGLE_CLIENT_ID")
        if not self.client_secret:
            missing.append("GOOGLE_CLIENT_SECRET")
        if not self.session_secret:
            missing.append("SESSION_SECRET_KEY")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Google auth is enabled, but these env vars are missing: {joined}")


def configure_google_auth(app, settings: Optional[GoogleAuthSettings] = None) -> GoogleAuthSettings:
    settings = settings or GoogleAuthSettings.from_env()
    settings.validate()
    app.state.google_auth_settings = settings

    if not settings.enabled:
        return settings

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        https_only=settings.secure_cookies,
        same_site="lax",
    )

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        server_metadata_url=GOOGLE_DISCOVERY_URL,
        client_kwargs={"scope": "openid profile email"},
    )
    app.state.oauth = oauth

    @app.get("/auth/login")
    async def auth_login(request: Request):
        redirect_uri = settings.redirect_uri or str(request.url_for("auth_callback"))
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        try:
            token = await oauth.google.authorize_access_token(request)
        except OAuthError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.error) from exc

        user_info = token.get("userinfo")
        if not user_info:
            user_info = await oauth.google.userinfo(token=token)
        user = _normalize_user(dict(user_info))
        _validate_user(user, settings.allowed_domains)
        request.session["user"] = user
        return RedirectResponse(settings.success_redirect)

    @app.get("/auth/me")
    async def auth_me(user: Dict[str, Any] = Depends(require_google_user)):
        return user

    @app.post("/auth/logout")
    async def auth_logout(request: Request):
        request.session.clear()
        return {"status": "logged_out"}

    return settings


def require_google_user(request: Request) -> Dict[str, Any]:
    settings = getattr(request.app.state, "google_auth_settings", GoogleAuthSettings.from_env())
    if not settings.enabled:
        return {"auth": "disabled"}
    user = request.session.get("user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google login required.",
            headers={"WWW-Authenticate": "Google"},
        )
    return user


def _normalize_user(user_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sub": user_info.get("sub"),
        "email": user_info.get("email"),
        "email_verified": bool(user_info.get("email_verified")),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "hd": user_info.get("hd"),
    }


def _validate_user(user: Dict[str, Any], allowed_domains: List[str]) -> None:
    email = str(user.get("email") or "").lower()
    if not user.get("email_verified"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Google email is not verified.")
    if allowed_domains:
        domain = email.rsplit("@", 1)[-1] if "@" in email else ""
        hosted_domain = str(user.get("hd") or "").lower()
        if domain not in allowed_domains and hosted_domain not in allowed_domains:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Google account domain is not allowed.")


def _env_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

