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
import asyncio
import json
from typing import Annotated, Callable

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.ai_analysis import AiAnalysisUnavailable, analyze_report_with_ai
from app.instagram_client import InstagramClient
from app.models.schemas import ExposureReport, SocialProfile
from app.nlp.attribute_inference import infer_attributes
from app.nlp.fingerprint import build_fingerprint
from app.progress import ProgressCallback, emit_progress
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


async def _build_report(profile: SocialProfile, progress_callback: ProgressCallback | None = None) -> ExposureReport:
    """Ejecuta el pipeline común (fingerprint -> inferencia -> scoring ->
    informe) sobre un perfil ya normalizado, sea cual sea su origen.

    `progress_callback` es opcional y no cambia el comportamiento si se
    omite (usado por `POST /api/analyze/{platform}`, incluidos los tests
    existentes); solo lo usa el endpoint de streaming de progreso en vivo,
    más abajo."""
    if not profile.posts:
        raise HTTPException(
            status_code=422,
            detail="No se encontró actividad pública suficiente para analizar",
        )

    await emit_progress(progress_callback, "Analizando tu forma de escribir...")
    fingerprint = build_fingerprint(profile.posts)

    await emit_progress(progress_callback, "Detectando atributos personales...")
    inferred_attributes = infer_attributes(profile.posts)

    await emit_progress(progress_callback, "Calculando el riesgo de privacidad...")
    score = compute_score(profile.posts, fingerprint, inferred_attributes)

    return await generate_report(
        platform=profile.platform,
        username=profile.username,
        posts=profile.posts,
        fingerprint=fingerprint,
        inferred_attributes=inferred_attributes,
        score=score,
        progress_callback=progress_callback,
    )


@router.post(
    "/analyze/ai-summary",
    responses={
        503: {"description": "El análisis con IA no está disponible (sin API key, cuota agotada, o error del proveedor)."},
    },
)
async def ai_summary(report: Annotated[ExposureReport, Body(...)]):
    """
    Endpoint AISLADO del pipeline principal: recibe un ExposureReport ya
    generado (el mismo JSON que el frontend ya tiene tras el análisis, se
    lo reenvía tal cual) y pide a Mistral AI conclusiones priorizadas.

    Deliberadamente NO se recalcula ni se vuelve a tocar sesión/tokens de
    Reddit o Instagram aquí -- este endpoint solo sabe leer un informe ya
    hecho, nada más. Esto mantiene la función de IA totalmente opcional y
    desacoplada: si Mistral falla o no está configurado, el resto del
    análisis (que ya se completó antes de llegar aquí) no se ve afectado
    en absoluto.

    Devuelve 503 (no 500) cuando la IA no está disponible -- falta de API
    key, cuota del tier gratuito agotada, o error del proveedor -- para que
    el frontend pueda distinguir claramente "esto es temporal/opcional" de
    un fallo real de la aplicación.

    IMPORTANTE: esta ruta está registrada ANTES que `POST /analyze/{platform}`
    a propósito. FastAPI resuelve las rutas en el orden en que se registran;
    si `/analyze/{platform}` fuera primero, "ai-summary" haría match como si
    fuera el nombre de una plataforma (capturado por el parámetro `platform`)
    y esta ruta nunca llegaría a ejecutarse -- ver test_analysis_router.py,
    donde este orden está verificado explícitamente para evitar que alguien
    lo deshaga sin darse cuenta en el futuro.
    """
    try:
        conclusions = await analyze_report_with_ai(report)
    except AiAnalysisUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {"conclusions": conclusions}


@router.post(
    "/analyze/{platform}",
    response_model=ExposureReport,
    responses={
        401: {"description": "No autenticado con la plataforma solicitada."},
        404: {"description": "Plataforma no soportada."},
        422: {"description": "No se encontró actividad pública suficiente para analizar."},
    },
)
async def analyze(platform: str, request: Request):
    factory = _PLATFORM_CLIENT_FACTORIES.get(platform)
    if factory is None:
        raise HTTPException(status_code=404, detail=f"Plataforma no soportada: {platform}")

    client = factory(request)
    profile = await client.fetch_profile()
    return await _build_report(profile)


@router.get(
    "/analyze/{platform}/stream",
    responses={
        401: {"description": "No autenticado con la plataforma solicitada."},
        404: {"description": "Plataforma no soportada."},
        422: {
            "description": (
                "No se encontró actividad pública suficiente para analizar. "
                "NOTA: esta ruta nunca devuelve un 422 HTTP real -- el pipeline "
                "(_build_report) puede lanzar esta excepción internamente, pero "
                "run_pipeline() la captura y la entrega como evento SSE "
                "{'done': true, 'error': ...} dentro de una respuesta 200. Se "
                "documenta aquí igualmente porque el código que la origina es "
                "compartido con POST /analyze/{platform}, donde sí es un 422 real."
            )
        },
    },
)
async def analyze_stream(platform: str, request: Request):
    """
    Variante de streaming del mismo análisis, vía Server-Sent Events (SSE),
    para que el frontend pueda mostrar una pantalla de progreso en vivo
    ("Leyendo publicaciones...", "Analizando fotos...", contadores) en vez
    de una espera opaca. Emite hitos REALES del pipeline (no un temporizador
    simulado): cada `yield` corresponde a un paso que de verdad acaba de
    terminar (una llamada a la API de la plataforma, una foto analizada...).

    No sustituye a `POST /api/analyze/{platform}` (que sigue existiendo tal
    cual, sin streaming, para compatibilidad/tests); es una ruta adicional
    que hace el mismo trabajo y además informa del progreso mientras ocurre.

    Formato de cada evento (`data: <json>\\n\\n`):
      - En curso:  {"done": false, "stage": "...", "posts_analyzed": N, ...}
      - Éxito:     {"done": true, "report": {...ExposureReport...}}
      - Error:     {"done": true, "error": "..."}
    """
    factory = _PLATFORM_CLIENT_FACTORIES.get(platform)
    if factory is None:
        raise HTTPException(status_code=404, detail=f"Plataforma no soportada: {platform}")

    # Se construye (y por tanto se valida la sesión/401) ANTES de abrir el
    # stream, para poder devolver un 401/404 normal en vez de un evento SSE
    # de error si el usuario ni siquiera está autenticado.
    client = factory(request)

    queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(stage: str, counts: dict) -> None:
        await queue.put({"done": False, "stage": stage, **counts})

    async def run_pipeline() -> None:
        try:
            await on_progress("Conectando con la plataforma...", {})
            profile = await client.fetch_profile(progress_callback=on_progress)
            report = await _build_report(profile, progress_callback=on_progress)
            await queue.put({"done": True, "report": json.loads(report.model_dump_json())})
        except HTTPException as exc:
            await queue.put({"done": True, "error": exc.detail})
        except Exception as exc:  # pragma: no cover - red de seguridad ante fallos inesperados
            await queue.put({"done": True, "error": f"Error inesperado durante el análisis: {exc}"})

    pipeline_task = asyncio.create_task(run_pipeline())

    async def event_generator():
        try:
            while True:
                item = await queue.get()
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("done"):
                    break
        finally:
            # Si el cliente cierra la conexión antes de terminar, se cancela
            # el trabajo en curso en vez de dejarlo corriendo en segundo
            # plano sin que nadie vaya a leer el resultado.
            if not pipeline_task.done():
                pipeline_task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Evita que un proxy intermedio (p. ej. Nginx) almacene en búfer
            # la respuesta y rompa el streaming en tiempo real.
            "X-Accel-Buffering": "no",
        },
    )
