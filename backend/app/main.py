"""
Punto de entrada de la aplicación.

Diseño RGPD: no se conecta ninguna base de datos. El único estado entre
peticiones es la cookie de sesión firmada (SessionMiddleware), que contiene
los tokens de acceso (Reddit y/o Instagram) y nada más. Cerrar sesión /
borrar la cookie elimina cualquier rastro del usuario en el sistema.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.sessions import SessionMiddleware

from app.analysis_router import router as analysis_router
from app.auth.instagram_oauth import router as instagram_auth_router
from app.auth.reddit_oauth import router as reddit_auth_router
from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Reparto de hilos internos de PyTorch entre las inferencias que pueden
    # correr A LA VEZ, para evitar sobre-suscripción de CPU: por defecto,
    # PyTorch usa TODOS los núcleos disponibles en CADA llamada (da igual
    # que la hayas lanzado desde un hilo de Python aparte) -- con
    # `photo_analysis_concurrency` fotos analizándose a la vez, eso
    # significa varias inferencias peleándose por los mismos núcleos en
    # vez de repartírselos, lo que puede hacer que el conjunto vaya MÁS
    # LENTO que analizar las fotos de una en una. Se fija UNA VEZ aquí,
    # antes de que ningún modelo haga ninguna inferencia real (es un
    # ajuste de proceso completo, no por modelo ni por hilo) -- afecta por
    # igual a DINOv2 y a Moondream2, comparten el mismo runtime de
    # PyTorch. Dimensionado para el caso sostenido (varias fotos
    # analizándose a la vez, cada una con su propio Moondream2 en marcha),
    # que es el que de verdad importa: DINOv2 es rápido y su ventana de
    # solape con Moondream2 en la misma foto es breve, así que no compensa
    # complicar esto con un reparto dinámico por modelo.
    try:
        import os

        import torch

        cpu_count = os.cpu_count() or 4
        concurrency = max(1, settings.photo_analysis_concurrency)
        threads_per_inference = max(1, cpu_count // concurrency)
        torch.set_num_threads(threads_per_inference)
        # `photo_analysis_concurrency` ahora se autocalcula por defecto a
        # partir de `cpu_count` (ver `_default_photo_analysis_concurrency`
        # en config.py) -- se deja constancia aquí de los valores
        # resultantes en ESTA máquina para poder verificarlos en el log de
        # arranque, en vez de tener que inferirlos indirectamente.
        logger.info(
            "Reparto de hilos de PyTorch: %d núcleos detectados, "
            "concurrencia de fotos=%d, hilos por inferencia=%d "
            "(PHOTO_ANALYSIS_CONCURRENCY en .env para forzar otro valor)",
            cpu_count,
            concurrency,
            threads_per_inference,
        )

        # Log explícito de si hay GPU disponible: sin esto, la única forma
        # de saber si de verdad se está usando la GPU (en vez de haber
        # caído a CPU en silencio -- p. ej. porque falta la reserva de GPU
        # en docker-compose.yml, o el driver NVIDIA no tiene soporte WSL2)
        # era mirar el uso de VRAM por fuera del contenedor mientras corría
        # un análisis, indirecto y fácil de malinterpretar. `cuda.is_available()`
        # es la MISMA comprobación que ya hacen geolocation.py y
        # scene_analysis.py para decidir dónde cargar cada modelo -- este
        # log solo hace explícito, en el arranque, lo que esos dos módulos
        # ya deciden por su cuenta más abajo.
        if torch.cuda.is_available():
            logger.info(
                "GPU detectada: %s (CUDA %s) -- los modelos de vision (DINOv2, y "
                "Moondream2 si ENABLE_SCENE_ANALYSIS=true) se cargaran en GPU.",
                torch.cuda.get_device_name(0),
                torch.version.cuda,
            )
        else:
            logger.info(
                "GPU no detectada (torch.cuda.is_available()=False) -- los modelos "
                "de vision se cargaran en CPU. Si esperabas usar una GPU, revisa la "
                "reserva de GPU en docker-compose.yml, el driver NVIDIA (soporte "
                "WSL2 si estas en Windows) y que requirements-vision.txt este "
                "instalando el build de torch con CUDA, no el SOLO-CPU."
            )
    except ImportError:
        pass  # torch no instalado (WITH_GEOLOCATION=false) -- nada que ajustar

    # Precarga el modelo DINOv2 + el índice FAISS de geolocalización EN EL
    # ARRANQUE del contenedor, en vez de esperar a la primera petición de
    # análisis de Instagram. `_lazy_load()` ya cachea en variables de
    # módulo (una sola carga por vida del proceso) -- esto solo adelanta
    # CUÁNDO ocurre esa carga, para que el coste (descargar los pesos del
    # modelo si no están ya en caché, leer el índice de disco) lo pague el
    # arranque de `docker-compose up`, no el primer usuario que analice.
    #
    # Se salta por completo si el índice no está construido o
    # torch/faiss/transformers no están instaladas (WITH_GEOLOCATION=false
    # en el Dockerfile) -- en ese caso _geolocation_available() ya es
    # False y no tiene sentido intentar nada.
    from app.vision.geolocation import _geolocation_available, _lazy_load

    if _geolocation_available():
        logger.info("Precargando modelo de geolocalización (DINOv2 + índice FAISS)...")
        # _lazy_load() es bloqueante (E/S de red y disco, más CPU) -- se
        # ejecuta en un hilo aparte para no congelar el event loop de
        # asyncio mientras carga.
        await asyncio.to_thread(_lazy_load)
        logger.info("Modelo de geolocalización listo.")

    # Moondream2 (análisis de CONTENIDO, distinto de la geolocalización de
    # arriba) se precarga TAMBIÉN aquí, mismo motivo que DINOv2 -- antes NO
    # se precargaba, así que la primera foto analizada tras activar
    # ENABLE_SCENE_ANALYSIS pagaba el coste de carga completo (hasta ~350s
    # en frío, medido en producción) DENTRO de
    # `_SCENE_ANALYSIS_TIMEOUT_SECONDS` (30s, pensado para el tiempo de
    # INFERENCIA, no de carga) -- esa primera foto fallaba casi siempre por
    # timeout, sin ser un problema real de esa foto en concreto. Se salta
    # si el interruptor está desactivado (default): no tiene sentido pagar
    # ese coste de arranque si no se va a usar.
    if settings.enable_scene_analysis:
        from app.vision.scene_analysis import _lazy_load as _lazy_load_scene_analysis
        from app.vision.scene_analysis import _scene_analysis_available

        if _scene_analysis_available():
            logger.info("Precargando modelo de analisis de contenido (Moondream2)...")
            # try/except AÑADIDO: `_lazy_load_scene_analysis()` ahora mismo
            # incluye el "Intento nº7" (carga en GPU, ver scene_analysis.py)
            # que está bien razonado pero SIN VERIFICAR contra el modelo
            # real -- si falla, antes esto tumbaba el arranque ENTERO del
            # backend (ni Reddit ni Instagram sin Moondream2 funcionarían,
            # aunque el fallo sea solo de esta función opcional). Con esto,
            # un fallo aquí se registra con el traceback completo y el
            # arranque sigue -- mismo comportamiento de fallback silencioso
            # por foto que ya tiene `analyze_image_content()` cuando faltan
            # dependencias, ver el `else` de abajo.
            try:
                await asyncio.to_thread(_lazy_load_scene_analysis)
            except Exception:
                logger.exception(
                    "Fallo al cargar Moondream2 -- el analisis de contenido se "
                    "saltara silenciosamente en cada foto (geolocalizacion y el "
                    "resto de la app siguen funcionando con normalidad). Traceback "
                    "completo arriba -- si es el 'Intento nº7' (GPU) fallando, "
                    "pega este traceback para seguir depurando desde ahi."
                )
            else:
                logger.info("Modelo de analisis de contenido listo.")
        else:
            # Mismo aviso que ya usa analyze_image_content() en tiempo de
            # peticion -- ENABLE_SCENE_ANALYSIS=true sin las dependencias
            # instaladas (falta requirements-vision.txt completo, o
            # WITH_GEOLOCATION=false en el build) es una config
            # inconsistente, pero no debe tumbar el arranque: se avisa y
            # se sigue, igual que ya hace analyze_image_content() por foto.
            logger.warning(
                "ENABLE_SCENE_ANALYSIS=true pero faltan dependencias "
                "(torch/transformers/timm/einops, ver requirements-vision.txt) "
                "-- el analisis de contenido se saltara silenciosamente en cada foto."
            )

    # Aviso (no bloqueante) de tablas de distribución poblacional del INE
    # (app/data/ine_reference.py, usadas por k_anonymity.py) cuyo dato de
    # origen documentado ha superado su umbral de antigüedad esperado --
    # ver stale_tables() para el criterio. Es solo un log, no impide
    # arrancar ni cambia ningún cálculo: hay que revisar y actualizar la
    # tabla a mano si el aviso salta.
    from app.data.ine_reference import stale_tables

    stale = stale_tables()
    if stale:
        detalle = ", ".join(f"{name} ({days}d)" for name, _verified, days in stale)
        logger.warning(
            "Tablas de app/data/ine_reference.py con más antigüedad de la "
            "esperada desde su fecha de origen documentada -- revisar si "
            "el INE ha publicado cifras más recientes: %s",
            detalle,
        )

    yield


app = FastAPI(
    title="Herramienta de Análisis de Exposición de Identidad Digital",
    description="TFG - Análisis defensivo de huella digital propia mediante OSINT e IA (Reddit + Instagram)",
    version="0.2.0",
    lifespan=_lifespan,
)

# Métricas Prometheus en /metrics, equivalente al express-prom-bundle que
# usaba el servicio Node original. El docker-compose y el prometheus.yml
# ya apuntan a este contenedor ("backend:3000"), así que no hace falta
# tocar esa parte de la infraestructura.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret_key,
    same_site="none",
    https_only=True,
)

app.add_middleware(
    CORSMiddleware,
    # CORS se decide en el arranque, no puede ser dinámico por petición
    # (a diferencia del redirect_uri de Instagram o la redirección final
    # tras el login -- ver app/auth/dynamic_origin.py). Con webapp+backend
    # bajo el mismo origen (nginx de por medio) esto en la práctica no
    # llega a entrar en juego para esas rutas; solo importa de verdad en
    # desarrollo local sin proxy (frontend y backend en puertos distintos).
    allow_origins=[settings.frontend_origin or "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reddit_auth_router)
app.include_router(instagram_auth_router)
app.include_router(analysis_router)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "identity-exposure-tfg",
        "note": "Herramienta educativa/defensiva. Requiere consentimiento explícito vía OAuth.",
    }
