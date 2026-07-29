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
