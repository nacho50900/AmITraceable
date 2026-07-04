"""
Endpoints de análisis: orquestan extracción -> fingerprint -> inferencia de
atributos -> scoring -> informe, para cada plataforma soportada.

Todo ocurre en memoria durante esta única petición HTTP. No se escribe
nada a disco ni a base de datos en ningún punto del pipeline.

Nota: `/api/analyze` (Reddit) se mantiene con ese nombre por compatibilidad
con el frontend ya existente. El endpoint de Instagram se añade aparte como
`/api/analyze/instagram` en vez de renombrar el primero, para no romper el
contrato ya probado. Si en el futuro se soportan más plataformas, valdría
la pena unificar esto bajo `/api/analyze/{platform}`.
"""
from fastapi import APIRouter, HTTPException, Request

from app.instagram_client import InstagramClient
from app.models.schemas import ExposureReport, SocialProfile
from app.nlp.attribute_inference import infer_attributes
from app.nlp.fingerprint import build_fingerprint
from app.reddit_client import RedditClient
from app.report.generator import generate_report
from app.scoring.privacy_score import compute_score

router = APIRouter(prefix="/api", tags=["analysis"])


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


@router.post("/analyze", response_model=ExposureReport)
async def analyze_reddit(request: Request):
    access_token = request.session.get("reddit_access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="No autenticado con Reddit")

    client = RedditClient(access_token)
    profile = await client.fetch_profile()
    return _build_report(profile)


@router.post("/analyze/instagram", response_model=ExposureReport)
async def analyze_instagram(request: Request):
    access_token = request.session.get("instagram_access_token")
    ig_user_id = request.session.get("instagram_user_id")
    if not access_token or not ig_user_id:
        raise HTTPException(status_code=401, detail="No autenticado con Instagram")

    client = InstagramClient(access_token, ig_user_id)
    profile = await client.fetch_profile()
    return _build_report(profile)
