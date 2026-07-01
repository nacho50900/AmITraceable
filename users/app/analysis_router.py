"""
Endpoint principal de análisis: orquesta extracción -> fingerprint ->
inferencia de atributos -> scoring -> informe.

Todo ocurre en memoria durante esta única petición HTTP. No se escribe
nada a disco ni a base de datos en ningún punto del pipeline.
"""
from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import ExposureReport
from app.nlp.attribute_inference import infer_attributes
from app.nlp.fingerprint import build_fingerprint
from app.reddit_client import RedditClient
from app.report.generator import generate_report
from app.scoring.privacy_score import compute_score

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=ExposureReport)
async def analyze(request: Request):
    access_token = request.session.get("reddit_access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="No autenticado con Reddit")

    client = RedditClient(access_token)
    profile = await client.fetch_profile()

    if not profile.posts:
        raise HTTPException(
            status_code=422,
            detail="No se encontró actividad pública suficiente para analizar",
        )

    fingerprint = build_fingerprint(profile.posts)
    inferred_attributes = infer_attributes(profile.posts)
    score = compute_score(profile.posts, fingerprint, inferred_attributes)

    report = generate_report(
        username=profile.username,
        posts=profile.posts,
        fingerprint=fingerprint,
        inferred_attributes=inferred_attributes,
        score=score,
    )
    return report
