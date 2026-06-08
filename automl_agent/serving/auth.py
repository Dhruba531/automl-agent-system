from __future__ import annotations

from typing import Any, Dict, List

from authlib.integrations.base_client import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from automl_agent.serving.config import GoogleAuthSettings


GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


def configure_google_auth(app, settings: GoogleAuthSettings) -> GoogleAuthSettings:
    settings.validate()
    app.state.google_auth_settings = settings

    @app.get("/auth/status")
    async def auth_status(request: Request):
        user = request.session.get("user") if settings.enabled else None
        return {
            "enabled": settings.enabled,
            "authenticated": bool(user),
            "user": user,
        }

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
    settings = getattr(request.app.state, "google_auth_settings", None)
    if settings is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Authentication is not configured.")
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
