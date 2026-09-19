"""
Endpoint standalone del frente de correlacion de cuentas por username (ver
docstring de app/osint/username_correlation.py para el alcance y las
razones de diseno).

Deliberadamente FUERA de analysis_router.py: no depende de una sesion
autenticada con ninguna plataforma ni de un perfil ya extraido -- solo
necesita el username que se quiere comprobar. Por eso es un router propio
en vez de una ruta mas dentro de `_PLATFORM_CLIENT_FACTORIES`.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    UsernameCorrelationReport,
    UsernameCorrelationRequest,
    UsernameSiteMatch,
)
from app.osint.username_correlation import check_username_across_sites

router = APIRouter(prefix="/api", tags=["osint"])


@router.post(
    "/username-correlation",
    response_model=UsernameCorrelationReport,
    responses={422: {"description": "El nombre de usuario no puede estar vacio."}},
)
async def username_correlation(payload: UsernameCorrelationRequest) -> UsernameCorrelationReport:
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="El nombre de usuario no puede estar vacio")

    results = await check_username_across_sites(username)

    return UsernameCorrelationReport(
        username=username,
        matches=[
            UsernameSiteMatch(site=result.site, url=result.url, exists=result.exists)
            for result in results
        ],
        checked_at=datetime.now(timezone.utc),
    )
