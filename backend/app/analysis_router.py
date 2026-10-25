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
import os
import time
from typing import Annotated, Callable

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.ai_analysis import AiAnalysisUnavailable, analyze_report_with_ai
from app.nlp.translation import source_language_for, translate_texts_local, translation_available
from app.log.analysis_run_log import log_analysis_run
from app.log.translation_log import log_translation_run
from app.analysis_timing import run_with_timer, timed_stage
from app.config import settings
from app.instagram_client import InstagramClient
from app.models.schemas import ExposureReport, RecalculateRequest, SocialProfile, TranslateDescriptionsRequest
from app.nlp.attribute_inference import infer_attributes
from app.nlp.fingerprint import build_fingerprint
from app.progress import ProgressCallback, emit_progress
from app.reddit_client import RedditClient
from app.report.generator import generate_report
from app.scoring.privacy_score import compute_score, compute_score_from_report
from app.skip_control import SkipControl, register_run, request_skip_for, unregister_run
from app.scoring.k_anonymity import _apply_proportion, final_remaining_population, PopulationNarrowingStep
from app.data.ine_reference import EYE_COLOR_DISTRIBUTION, HAIR_COLOR_DISTRIBUTION, SKIN_TONE_DISTRIBUTION, TOTAL_POPULATION_ES
from app import stages

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


def _skip_kwargs(skip: SkipControl | None) -> dict:
    """`{"skip": skip}` solo si hay un SkipControl real -- sin él (endpoint
    clásico `POST /analyze/{platform}`, tests) las funciones de las pistas
    largas se llaman exactamente igual que antes de existir el botón de
    saltar, sin ningún argumento nuevo."""
    return {} if skip is None else {"skip": skip}


async def _geolocate_after(
    prerequisite: asyncio.Task, posts, avatar_url=None, progress_callback=None, skip: SkipControl | None = None
):
    """Espera a que `prerequisite` (la comprobación de cuentas por username
    de Maigret) TERMINE -- bien, con error o cancelada -- y solo entonces
    lanza el análisis de fotos.

    Motivo: correr ambos a la vez saturaba la máquina anfitriona. Maigret
    abre decenas de conexiones simultáneas (y el DNS de Docker/WSL2 se
    ahoga: timeouts masivos que además dejan el barrido sin resultados
    fiables) mientras Qwen + DINOv2 ocupan GPU, VRAM y RAM; en el equipo
    de desarrollo (16GB, WSL2 con 13GB) esa suma coincidió con un bugcheck
    MEMORY_MANAGEMENT de Windows (causa exacta no confirmada). Maigret es casi puro I/O de red y las
    fotos son puro cómputo, así que no ganan nada importante compitiendo.

    `asyncio.wait()` (no `await prerequisite`) a propósito: espera sin
    relanzar la excepción de la tarea -- un fallo de la comprobación de
    cuentas ya se gestiona donde se recoge su resultado
    (generate_report) y no debe impedir analizar las fotos.

    El import de `estimate_locations_for_posts` es perezoso y por módulo
    (igual que en `_build_report`) para que los tests puedan sustituirlo
    con monkeypatch."""
    await asyncio.wait([prerequisite])
    from app.vision import geolocation

    return await geolocation.estimate_locations_for_posts(
        posts, avatar_url=avatar_url, progress_callback=progress_callback, **_skip_kwargs(skip)
    )


async def _build_report(
    profile: SocialProfile,
    progress_callback: ProgressCallback | None = None,
    fetch_seconds: float | None = None,
    skip: SkipControl | None = None,
) -> ExposureReport:
    """Ejecuta el pipeline común (fingerprint -> inferencia -> scoring ->
    informe) sobre un perfil ya normalizado, sea cual sea su origen.

    `progress_callback` es opcional y no cambia el comportamiento si se
    omite (usado por `POST /api/analyze/{platform}`, incluidos los tests
    existentes); solo lo usa el endpoint de streaming de progreso en vivo,
    más abajo.

    `skip` (ver app/skip_control.py), si se da, permite al usuario saltar el
    resto de las dos pistas LARGAS en segundo plano (comprobación de
    cuentas y análisis de fotos): al activarse, ambas devuelven lo que
    lleven hasta ese momento y el pipeline sigue con normalidad -- la IA
    (autodeclaraciones, inferencias) y el informe se generan igualmente
    sobre ese resultado parcial. Sin él, comportamiento idéntico al de antes.

    `fetch_seconds`, si se da, es el tiempo que tardó `fetch_profile()` en
    el llamador (ver `analyze`/`analyze_stream` más abajo) -- se registra
    como una etapa más en el log general de análisis (ver
    app/log/analysis_run_log.py), aunque ocurra fuera de esta función, para
    que el log refleje el tiempo total real de principio a fin y no solo
    el del pipeline interno."""
    if not profile.posts:
        raise HTTPException(
            status_code=422,
            detail="No se encontró actividad pública suficiente para analizar",
        )

    # Comprobación de cuentas por username (ver ADR-44/ADR-48,
    # app/osint/username_correlation.py): ~5000 peticiones HTTP, del orden
    # de minutos. Detrás de `settings.enable_username_correlation` (False
    # por defecto, ver app/config.py) -- ni siquiera se importa el módulo
    # si está desactivado, mismo criterio que `enable_scene_analysis`. Se
    # lanza YA, en paralelo con el resto del pipeline (fingerprint,
    # atributos...), y se recoge más adelante en generate_report
    # (parámetro `username_correlation_task`).
    #
    # Se crea ANTES que la tarea de fotos de más abajo a propósito: si
    # `settings.username_correlation_before_photos` está activo (por
    # defecto), las fotos esperan a que ésta termine -- ver el comentario
    # de `_geolocate_after` sobre por qué no corren a la vez.
    username_correlation_task: asyncio.Task | None = None
    if settings.enable_username_correlation:
        from app.osint.username_correlation import check_username_across_sites

        username_correlation_task = asyncio.create_task(
            check_username_across_sites(
                profile.username, progress_callback=progress_callback, **_skip_kwargs(skip)
            )
        )

    # El análisis de fotos (geolocalización vía DINOv2+FAISS y contenido
    # con Qwen) es, con diferencia, lo que más tarda de todo el pipeline.
    # Se lanza como tarea en segundo plano AQUÍ, lo antes posible, para que
    # corra en PARALELO con el resto del análisis (fingerprint, atributos,
    # autodeclaraciones con IA). El resultado se recoge más adelante,
    # dentro de generate_report (parámetro `geolocation_task`).
    #
    # Se crea ANTES de activar `run_with_timer()` a propósito -- ver el
    # docstring de app/analysis_timing.py sobre por qué esta tarea nunca
    # debe compartir el timer del pipeline principal (su tiempo se mide
    # aparte, por foto, en app/log/performance_log.py).
    geolocation_task: asyncio.Task | None = None
    if profile.platform == "instagram":
        if username_correlation_task is not None and settings.username_correlation_before_photos:
            geolocation_task = asyncio.create_task(
                _geolocate_after(
                    username_correlation_task,
                    profile.posts,
                    avatar_url=profile.avatar_url,
                    progress_callback=progress_callback,
                    skip=skip,
                )
            )
        else:
            from app.vision.geolocation import estimate_locations_for_posts

            geolocation_task = asyncio.create_task(
                estimate_locations_for_posts(
                    profile.posts,
                    avatar_url=profile.avatar_url,
                    progress_callback=progress_callback,
                    **_skip_kwargs(skip),
                )
            )

    # `asyncio.create_task` solo PROGRAMA las tareas de arriba -- no les
    # cede el control de verdad. El resto de este bloque (build_fingerprint,
    # infer_attributes, compute_score) es código SÍNCRONO que no hace
    # ningún `await` real, así que sin este `sleep(0)` el event loop no
    # tendría ninguna oportunidad de arrancarlas hasta que este bloque
    # síncrono termine del todo -- "empezarían en paralelo" solo de
    # nombre, no en la práctica. Este yield explícito les da su primer
    # turno real (arranca la descarga de la primera foto / la primera
    # petición HTTP de la comprobación de cuentas) antes de seguir.
    await asyncio.sleep(0)

    async with run_with_timer() as timer:
        if fetch_seconds is not None:
            timer.add("obtencion_perfil", fetch_seconds)

        pipeline_start = time.monotonic()

        async with timed_stage("huella_escritura"):
            await emit_progress(progress_callback, stages.ANALYZING_WRITING_STYLE)
            fingerprint = build_fingerprint(profile.posts)

        async with timed_stage("deteccion_atributos"):
            await emit_progress(progress_callback, stages.DETECTING_ATTRIBUTES)
            inferred_attributes = infer_attributes(profile.posts)

        async with timed_stage("scoring"):
            await emit_progress(progress_callback, stages.COMPUTING_SCORE)
            score = compute_score(profile.posts, fingerprint, inferred_attributes)

        report = await generate_report(
            platform=profile.platform,
            username=profile.username,
            posts=profile.posts,
            fingerprint=fingerprint,
            inferred_attributes=inferred_attributes,
            score=score,
            progress_callback=progress_callback,
            bio=profile.bio,
            full_name=profile.full_name,
            avatar_url=profile.avatar_url,
            geolocation_task=geolocation_task,
            username_correlation_task=username_correlation_task,
        )

        total_seconds = time.monotonic() - pipeline_start + (fetch_seconds or 0.0)

        # Recuentos agregados para el log -- sin ningún dato personal, ver
        # docstring de app/log/analysis_run_log.py. `n_photos` cuenta FICHEROS
        # de imagen (un carrusel de 5 fotos suma 5, no 1), igual criterio
        # que `total` en estimate_locations_for_posts().
        n_comments = sum(1 for p in profile.posts if p.type == "comment")
        n_media_items = sum(1 for p in profile.posts if p.type in ("image", "video", "carousel_album"))
        n_photos = sum(
            len(getattr(p, "media_urls", None) or [])
            for p in profile.posts
            if p.type in ("image", "carousel_album")
        )
        n_posts = len(profile.posts) - n_comments

        log_analysis_run(
            platform=profile.platform,
            n_posts=n_posts,
            n_comments=n_comments,
            n_media_items=n_media_items,
            n_photos=n_photos,
            ai_enabled=settings.ai_key_configured,
            scene_analysis_enabled=settings.enable_scene_analysis,
            geolocation_available=report.geolocation_available,
            total_seconds=total_seconds,
            timer=timer,
        )

    return report


@router.post(
    "/analyze/ai-summary",
    responses={
        503: {"description": "El análisis con IA no está disponible (sin API key, cuota agotada, o error del proveedor)."},
    },
)
async def ai_summary(report: Annotated[ExposureReport, Body(...)], lang: str = "es"):
    """
    Endpoint AISLADO del pipeline principal: recibe un ExposureReport ya
    generado (el mismo JSON que el frontend ya tiene tras el análisis, se
    lo reenvía tal cual) y pide a Mistral AI conclusiones priorizadas.

    `lang` (query param, "es" por defecto): idioma del veredicto/
    conclusiones generados por la IA -- ver `ai_analysis.SUPPORTED_LANGUAGES`.
    El frontend manda aquí su idioma de UI actual (ver webapp/src/i18n). Un
    valor no soportado se ignora silenciosamente y se sirve en español, no
    es un error -- este parámetro es una preferencia, no un contrato.

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
        result = await analyze_report_with_ai(report, lang=lang)
    except AiAnalysisUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return result


@router.post("/analyze/translate-descriptions")
async def translate_descriptions(request: Annotated[TranslateDescriptionsRequest, Body(...)], lang: str = "es"):
    """
    Endpoint AISLADO, registrado con el mismo cuidado de orden que
    `POST /analyze/ai-summary` justo arriba (ver el comentario en
    `ai_summary()`): recibe una lista de textos cortos ya generados por
    Moondream2 (afición o caption de fotos, ver
    app/vision/scene_analysis.py y ADR-30) y devuelve su traducción a
    `lang`.

    A diferencia de `ai_summary` (que sí usa Mistral, un servicio
    externo con cuota y posibles caídas), la traducción es LOCAL (ver
    ADR-31 y app/nlp/translation.py -- modelos MarianMT/CTranslate2, sin
    red ni API key) y NUNCA lanza: si los modelos no están convertidos
    en disco, o falla cualquier cosa durante la traducción, se devuelven
    los textos ORIGINALES sin cambios -- por eso este endpoint no
    necesita un camino de error 503 como sí tiene ai-summary, siempre
    responde 200 con algo útil.

    `lang` (query param, "es" por defecto): un valor no soportado, o
    "es" (nada que traducir hacia el idioma nativo del proyecto),
    devuelve los textos SIN CAMBIOS, no es un error.

    IMPORTANTE: igual que ai-summary, esta ruta está registrada ANTES que
    `POST /analyze/{platform}` a propósito -- mismo motivo de orden de
    resolución de rutas de FastAPI, ver el comentario en `ai_summary()`
    de arriba.
    """
    # Traducción local con CTranslate2 es CPU-bound (no I/O), así que se
    # despacha a un hilo aparte para no bloquear el event loop de FastAPI
    # mientras corre -- mismo patrón que ya usa todo el pipeline de
    # visión (ver asyncio.to_thread en app/vision/geolocation.py).
    start = time.monotonic()
    translations = await asyncio.to_thread(translate_texts_local, request.texts, lang)
    elapsed = time.monotonic() - start

    # `lang` no soportado no se registra -- es un no-op sin coste real de
    # traducción (ver translate_texts_local()), no aporta nada al log de
    # rendimiento. Nunca debe romper la respuesta: log_translation_run()
    # ya no lanza por sí sola (ver su docstring), pero se envuelve
    # igualmente por si acaso, con el mismo criterio best-effort que el
    # resto del logging de este proyecto.
    source_lang = source_language_for(lang)
    if source_lang is not None and request.texts:
        try:
            log_translation_run(
                num_texts=len(request.texts),
                source_lang=source_lang,
                target_lang=lang,
                cpu_count=os.cpu_count() or 4,
                total_seconds=elapsed,
                translation_available=translation_available(lang),
            )
        except Exception:
            pass

    return {"translations": translations}


# Categorías que solo pueden venir de los rasgos físicos manuales (ver
# ManualTraitsSelector en el frontend).
_MANUAL_TRAIT_CATEGORIES = ("color_ojos", "color_pelo", "color_piel")


@router.post("/analyze/recalculate", response_model=ExposureReport)
async def recalculate_report(request: Annotated[RecalculateRequest, Body(...)]):
    """
    Recalcula el informe de exposición añadiendo los rasgos físicos manuales proporcionados.
    Actualiza `population_narrowing` multiplicando las proporciones en cadena y
    recalcula el score de privacidad completo con la misma fórmula del pipeline.

    Es idempotente respecto a los rasgos manuales: quita los de llamadas
    anteriores antes de aplicar los nuevos, así que reenviar el informe ya
    recalculado no acumula nada, y `manual_attributes=[]` los elimina todos.

    NOTA: esta ruta estática DEBE registrarse antes que `/analyze/{platform}`
    (justo debajo) -- FastAPI/Starlette resuelve las rutas en el orden en que
    se registran, así que si `/analyze/{platform}` fuera antes, capturaría
    "recalculate" como valor del parámetro `platform` y esta ruta nunca se
    llegaría a ejecutar (404 "Plataforma no soportada: recalculate").
    """
    from app.models.schemas import InferredAttribute

    report = request.report
    manual_attributes = request.manual_attributes

    # 0. Partir del informe SIN los rasgos manuales de recálculos anteriores. El
    # frontend reenvía el último informe recalculado, que ya los lleva: sin
    # esto cada "Aplicar" volvía a multiplicar encima (y "Limpiar" no limpiaba
    # nada). Los rasgos que llegan ahora sustituyen a los anteriores, y una
    # lista vacía deja el informe como salió del análisis.
    report.inferred_attributes = [a for a in report.inferred_attributes if a.category not in _MANUAL_TRAIT_CATEGORIES]
    report.population_narrowing = [step for step in report.population_narrowing if step.source != "manual"]

    # 1. Añadir a inferred_attributes
    new_inferred = []
    for attr in manual_attributes:
        new_inferred.append(InferredAttribute(
            category=attr.category,
            value=attr.value,
            confidence=1.0,  # Autodeclaración manual es 100% fiable
            evidence=[],
        ))
    report.inferred_attributes.extend(new_inferred)

    # 2. Recalcular estrechamiento de población
    steps = report.population_narrowing
    last_remaining = final_remaining_population(steps)
    remaining = float(last_remaining) if last_remaining is not None else float(TOTAL_POPULATION_ES)

    for attr in manual_attributes:
        proportion = None
        label = ""
        if attr.category == "color_ojos":
            proportion = EYE_COLOR_DISTRIBUTION.get(attr.value)
            label = f"Color de ojos: {attr.value.title()}"
        elif attr.category == "color_pelo":
            proportion = HAIR_COLOR_DISTRIBUTION.get(attr.value)
            label = f"Color de pelo: {attr.value.title()}"
        elif attr.category == "color_piel":
            proportion = SKIN_TONE_DISTRIBUTION.get(attr.value)
            label = f"Color de piel: {attr.value.title()}"

        remaining, step = _apply_proportion(
            remaining,
            proportion,
            label,
            attr.category,
            [],
            source="manual",
            note="Rasgo físico añadido manualmente. Proporción estimada contextualmente.",
            note_code=None,
            value_raw=attr.value,
        )
        if step:
            steps.append(step)

    report.population_narrowing = steps
    report.remaining_population_all_traits = final_remaining_population(steps)
    report.remaining_population_all_traits_proportion = (
        report.remaining_population_all_traits / TOTAL_POPULATION_ES
        if report.remaining_population_all_traits is not None else None
    )

    # 4. Recalcular el score con la MISMA fórmula que el pipeline (ADR-56), ya
    # con los rasgos añadidos y la población estrechada: antes solo se
    # recalculaba inferable_data_risk con otra fórmula, y las tres barras
    # podían dar números incoherentes con los del análisis inicial.
    report.privacy_score = compute_score_from_report(report)

    return report


_SESSION_TOKEN_KEYS = {"reddit": "reddit_access_token", "instagram": "instagram_access_token"}


def _run_key(platform: str, request: Request) -> str:
    """Clave del análisis en curso de ESTE usuario y plataforma (ver
    app/skip_control.py). El token de acceso ya está en la sesión firmada
    (la factory del cliente lo exige antes de llegar aquí), así que no
    hace falta ningún identificador nuevo; solo vive en memoria."""
    return f"{platform}:{request.session.get(_SESSION_TOKEN_KEYS[platform])}"


@router.post(
    "/analyze/{platform}/skip",
    responses={
        401: {"description": "No autenticado con la plataforma solicitada."},
        404: {"description": "Plataforma no soportada."},
    },
)
async def skip_analysis(platform: str, request: Request):
    """Pide saltar el resto del análisis en curso (comprobación de cuentas y
    fotos pendientes): ambas pistas devuelven lo que lleven y el stream SSE
    sigue su curso -- IA sobre lo obtenido, informe final. Devuelve
    `{"skipped": true}` si había un análisis en curso para este usuario y
    `{"skipped": false}` si no (ya terminó, o nunca empezó): no es un error."""
    factory = _PLATFORM_CLIENT_FACTORIES.get(platform)
    if factory is None:
        raise HTTPException(status_code=404, detail=f"Plataforma no soportada: {platform}")
    factory(request)  # valida la sesión (401) igual que el resto de rutas
    return {"skipped": request_skip_for(_run_key(platform, request))}


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
    fetch_start = time.monotonic()
    profile = await client.fetch_profile()
    fetch_seconds = time.monotonic() - fetch_start
    return await _build_report(profile, fetch_seconds=fetch_seconds)


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
    # Interruptor de "saltar el resto del análisis": se registra bajo una
    # clave de plataforma+sesión para que `POST /analyze/{platform}/skip`
    # (otra petición HTTP) lo encuentre. Se da de baja al terminar.
    skip = SkipControl()
    run_key = _run_key(platform, request)
    register_run(run_key, skip)

    async def on_progress(stage: str, counts: dict) -> None:
        await queue.put({"done": False, "stage": stage, **counts})

    async def run_pipeline() -> None:
        try:
            await on_progress(stages.CONNECTING, {})
            fetch_start = time.monotonic()
            profile = await client.fetch_profile(progress_callback=on_progress)
            fetch_seconds = time.monotonic() - fetch_start
            report = await _build_report(
                profile, progress_callback=on_progress, fetch_seconds=fetch_seconds, skip=skip
            )
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
            unregister_run(run_key, skip)

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
