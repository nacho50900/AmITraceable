"""
Módulo de autenticación con Instagram, vía "Business Login for Instagram".

Requiere que la cuenta del usuario sea Instagram Business o Creator (las
cuentas personales no tienen acceso a esta API — ver README). No requiere
vincular una Página de Facebook, a diferencia de "Facebook Login for
Business".

Flujo (documentado en developers.facebook.com/docs/instagram-platform):
1. GET /auth/instagram/login    -> redirige a Instagram para autorizar
2. Instagram redirige a /auth/instagram/callback con un "code"
3. Intercambiamos el code por un access_token de corta duración
4. Lo cambiamos por uno de larga duración (60 días)
5. Guardamos el token en la sesión firmada (cookie), NUNCA en disco/BD —
   mismo principio de diseño que el módulo de Reddit.

Consentimiento explícito: el único scope solicitado es
`instagram_business_basic` (perfil + lectura de media pública propia). No
se piden permisos de mensajería ni de publicación, que no hacen falta para
este análisis y ampliarían innecesariamente lo que el usuario autoriza.
"""
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.config import settings

router = APIRouter(prefix="/auth/instagram", tags=["auth"])

IG_AUTH_URL = "https://www.instagram.com/oauth/authorize"
IG_SHORT_TOKEN_URL = "https://api.instagram.com/oauth/access_token"
IG_LONG_TOKEN_URL = "https://graph.instagram.com/access_token"

SCOPES = "instagram_business_basic"


def _require_configured():
    if not (settings.instagram_app_id and settings.instagram_app_secret and settings.instagram_redirect_uri):
        raise HTTPException(
            status_code=503,
            detail="Instagram no está configurado en este servidor (faltan credenciales).",
        )


@router.get("/login")
async def login(request: Request):
    """Redirige al usuario a la pantalla de consentimiento de Instagram."""
    _require_configured()

    state = secrets.token_urlsafe(24)
    request.session["instagram_oauth_state"] = state

    params = {
        "client_id": settings.instagram_app_id,
        "redirect_uri": settings.instagram_redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
    }
    return RedirectResponse(f"{IG_AUTH_URL}?{urlencode(params)}")


@router.get("/callback")
async def callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    _require_configured()

    if error:
        # El usuario denegó el consentimiento explícitamente
        return RedirectResponse(f"{settings.frontend_origin}/?auth_error={error}")

    saved_state = request.session.get("instagram_oauth_state")
    if not state or state != saved_state:
        raise HTTPException(status_code=400, detail="Estado OAuth inválido (posible CSRF)")

    async with httpx.AsyncClient() as client:
        # 1. Code -> token de corta duración
        short_resp = await client.post(
            IG_SHORT_TOKEN_URL,
            data={
                "client_id": settings.instagram_app_id,
                "client_secret": settings.instagram_app_secret,
                "grant_type": "authorization_code",
                "redirect_uri": settings.instagram_redirect_uri,
                "code": code,
            },
        )
        if short_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="No se pudo obtener el token de Instagram")

        short_data = short_resp.json()
        short_token = short_data["access_token"]
        ig_user_id = short_data["user_id"]

        # 2. Token corto -> token largo (60 días)
        long_resp = await client.get(
            IG_LONG_TOKEN_URL,
            params={
                "grant_type": "ig_exchange_token",
                "client_secret": settings.instagram_app_secret,
                "access_token": short_token,
            },
        )
        if long_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="No se pudo ampliar el token de Instagram")

        long_token = long_resp.json()["access_token"]

    # Guardamos SOLO en la sesión firmada del navegador (no hay BD ni fichero)
    request.session["instagram_access_token"] = long_token
    request.session["instagram_user_id"] = str(ig_user_id)
    request.session.pop("instagram_oauth_state", None)

    return RedirectResponse(f"{settings.frontend_origin}/dashboard?platform=instagram")


@router.post("/logout")
async def logout(request: Request):
    """Borra solo las claves de Instagram de la sesión (no afecta a Reddit,
    ya que ambos módulos pueden convivir en la misma sesión de navegador)."""
    request.session.pop("instagram_access_token", None)
    request.session.pop("instagram_user_id", None)
    request.session.pop("instagram_oauth_state", None)
    return {"status": "ok"}


@router.get("/status")
async def status(request: Request):
    token = request.session.get("instagram_access_token")
    return {"authenticated": token is not None}
