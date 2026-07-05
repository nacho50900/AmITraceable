"""
Endpoint de análisis: orquesta extracción -> fingerprint -> inferencia de
atributos -> scoring -> informe, para cualquier plataforma soportada.

Todo ocurre en memoria durante esta única petición HTTP. No se escribe
nada a disco ni a base de datos en ningún punto del pipeline.

Diseño: una única ruta `/api/analyze/{platform}` en vez de una ruta
"principal" (p. ej. `/api/analyze` para Reddit) con el resto de plataformas
colgando de ella como casos especiales. Cada plataforma se registra como una
entrada más en `_PLATFORM_CLIENT_FACTORIES`, con el mismo peso estructural
que las demás. Añadir una plataforma nueva es añadir una función factory y
una entrada en el diccionario — no tocar la lógica del endpoint ni la de
ninguna otra plataforma.
"""
from typing import Callable

from fastapi import APIRouter, HTTPException, Request

from app.instagram_client import InstagramClient
from app.models.schemas import ExposureReport, SocialProfile
from app.nlp.attribute_inference import infer_attributes
from app.nlp.fingerprint import build_fingerprint
from app.reddit_client import RedditClient
from app.report.generator import generate_report
from app.scoring.privacy_score import compute_score

router = APIRouter(prefix="/api", tags=["analysis"])


def _reddit_client_from_session(request: Request) -> RedditClient:
    access_token = request.session.get("reddit_access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="No autenticado con Reddit")
    return RedditClient(access_token)


def _instagram_client_from_session(request: Request) -> InstagramClient:
    access_token = request.session.get("instagram_access_token")
    ig_user_id = request.session.get("instagram_user_id")
    if not access_token or not ig_user_id:
        raise HTTPException(status_code=401, detail="No autenticado con Instagram")
    return InstagramClient(access_token, ig_user_id)


# Cada entrada construye el cliente ya autenticado a partir de la sesión, o
# lanza 401 si falta el token de esa plataforma. Todas las entradas tienen
# el mismo peso: ninguna es "la principal".
_PLATFORM_CLIENT_FACTORIES: dict[str, Callable[[Request], object]] = {
    "reddit": _reddit_client_from_session,
    "instagram": _instagram_client_from_session,
}


def _build_report(profile: SocialProfile) -> ExposureReport:
    """Ejecuta el pipeline común (fingerprint -> inferencia -> scoring ->
    informe) sobre un perfil ya normalizado, sea cual sea su origen."""
    if not profile.posts:
        raise HTTPException(
            status_code=422,
            detail="No se encontró actividad pública suficiente para analizar",
        )

    fingerprint = build_fingerprint(profile.posts)
    inferred_attributes = infer_attributes(profile.posts)
    score = compute_score(profile.posts, fingerprint, inferred_attributes)

    return generate_report(
        platform=profile.platform,
        username=profile.username,
        posts=profile.posts,
        fingerprint=fingerprint,
        inferred_attributes=inferred_attributes,
        score=score,
    )


@router.post("/analyze/{platform}", response_model=ExposureReport)
async def analyze(platform: str, request: Request):
    factory = _PLATFORM_CLIENT_FACTORIES.get(platform)
    if factory is None:
        raise HTTPException(status_code=404, detail=f"Plataforma no soportada: {platform}")

    client = factory(request)
    profile = await client.fetch_profile()
    return _build_report(profile)
