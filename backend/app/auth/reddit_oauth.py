"""
Módulo de autenticación OAuth 2.0 con Reddit.

Flujo:
1. GET /auth/reddit/login  -> redirige al usuario a Reddit para autorizar
2. Reddit redirige a /auth/reddit/callback con un "code"
3. Intercambiamos el code por un access_token (y refresh_token)
4. Guardamos el token en la sesión firmada (cookie), NUNCA en disco/BD.

Consentimiento explícito: el scope solicitado es el mínimo necesario
(identity, history, read) y se muestra al usuario en la propia pantalla
de autorización de Reddit, que es quien gestiona el consentimiento real.
"""
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.auth.dynamic_origin import frontend_target
from app.config import settings

router = APIRouter(prefix="/auth/reddit", tags=["auth"])

REDDIT_AUTH_URL = "https://www.reddit.com/api/v1/authorize"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

# Scopes mínimos: identidad básica + historial público de posts/comentarios
SCOPES = "identity history read"


@router.get("/login")
async def login(request: Request):
    """Redirige al usuario a la pantalla de consentimiento de Reddit."""
    state = secrets.token_urlsafe(24)
    request.session["reddit_oauth_state"] = state

    params = {
        "client_id": settings.reddit_client_id,
        "response_type": "code",
        "state": state,
        "redirect_uri": settings.reddit_redirect_uri,
        "duration": "temporary",  # no pedimos acceso permanente
        "scope": SCOPES,
    }
    return RedirectResponse(f"{REDDIT_AUTH_URL}?{urlencode(params)}")


@router.get(
    "/callback",
    responses={
        400: {"description": "Estado OAuth inválido (posible CSRF)."},
        502: {"description": "Reddit no devolvió un token de acceso válido."},
    },
)
async def callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        # El usuario denegó el consentimiento explícitamente
        return RedirectResponse(f"{frontend_target(request, settings.frontend_origin)}/?auth_error={error}")

    saved_state = request.session.get("reddit_oauth_state")
    if not state or state != saved_state:
        raise HTTPException(status_code=400, detail="Estado OAuth inválido (posible CSRF)")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            REDDIT_TOKEN_URL,
            auth=(settings.reddit_client_id, settings.reddit_client_secret),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.reddit_redirect_uri,
            },
            headers={"User-Agent": settings.reddit_user_agent},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="No se pudo obtener el token de Reddit")

    token_data = resp.json()
    # Guardamos SOLO en la sesión firmada del navegador (no hay BD ni fichero)
    request.session["reddit_access_token"] = token_data["access_token"]
    request.session.pop("reddit_oauth_state", None)

    return RedirectResponse(f"{frontend_target(request, settings.frontend_origin)}/dashboard")


@router.post("/logout")
async def logout(request: Request):
    """Borra solo las claves de Reddit de la sesión (no afecta a Instagram,
    ya que ambos módulos pueden convivir en la misma sesión de navegador)."""
    request.session.pop("reddit_access_token", None)
    request.session.pop("reddit_oauth_state", None)
    return {"status": "ok"}


@router.get("/status")
async def status(request: Request):
    token = request.session.get("reddit_access_token")
    return {"authenticated": token is not None}
