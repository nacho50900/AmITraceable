"""
Tipo de callback de progreso usado en todo el pipeline de análisis para
reportar hitos reales (no una barra de carga simulada) al endpoint de
streaming (`GET /api/analyze/{platform}/stream`, ver analysis_router.py).

Se pasa como parámetro opcional (`progress_callback=None` por defecto) a
fetch_profile de cada cliente, a generate_report y a
estimate_locations_for_posts, precisamente para que el endpoint clásico
`POST /api/analyze/{platform}` (usado por los tests existentes y por
cualquier cliente que no necesite progreso en vivo) siga funcionando
exactamente igual sin ningún cambio de comportamiento.
"""
from typing import Awaitable, Callable, Optional

# (mensaje_de_etapa, contadores_parciales) -> None
ProgressCallback = Callable[[str, dict], Awaitable[None]]


async def emit_progress(callback: Optional[ProgressCallback], stage: str, **counts) -> None:
    if callback is not None:
        await callback(stage, counts)
