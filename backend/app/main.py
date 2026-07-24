"""
Punto de entrada de la aplicación.

Diseño RGPD: no se conecta ninguna base de datos. El único estado entre
peticiones es la cookie de sesión firmada (SessionMiddleware), que contiene
los tokens de acceso (Reddit y/o Instagram) y nada más. Cerrar sesión /
borrar la cookie elimina cualquier rastro del usuario en el sistema.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.sessions import SessionMiddleware

from app.analysis_router import router as analysis_router
from app.auth.instagram_oauth import router as instagram_auth_router
from app.auth.reddit_oauth import router as reddit_auth_router
from app.config import settings

app = FastAPI(
    title="Herramienta de Análisis de Exposición de Identidad Digital",
    description="TFG - Análisis defensivo de huella digital propia mediante OSINT e IA (Reddit + Instagram)",
    version="0.2.0",
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
