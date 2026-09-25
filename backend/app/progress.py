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
import asyncio
from typing import Awaitable, Callable, Optional, TypeVar

# (mensaje_de_etapa, contadores_parciales) -> None
ProgressCallback = Callable[[str, dict], Awaitable[None]]

T = TypeVar("T")


async def emit_progress(callback: Optional[ProgressCallback], stage: str, **counts) -> None:
    if callback is not None:
        await callback(stage, counts)


async def run_with_heartbeat(
    coro: Awaitable[T],
    callback: Optional[ProgressCallback],
    stage: str,
    *,
    interval_seconds: float = 5.0,
) -> T:
    """Ejecuta `coro` mientras re-emite el mismo evento de progreso (`stage`)
    cada `interval_seconds` -- para que el stream SSE nunca quede en
    silencio más de ese margen mientras el paso en curso hace un trabajo
    largo de una sola tacada (típicamente una llamada a un LLM: ver el
    fallo real que motivó esto, 24/9 -- `extract_demographics_with_ai` en
    report/generator.py podía tardar minutos con Qwen3.5-4B local -carga
    inicial del modelo o inferencia lenta en una GPU modesta-, frente a
    los segundos que tardaba la llamada HTTP a Mistral/Gemini; ese
    silencio prolongado hacía que el `EventSource` del navegador diese la
    conexión por muerta y reconectase solo, REINICIANDO TODO EL PIPELINE
    desde el principio -- el "bucle" que parecía visto desde fuera).

    Si `callback` es `None` (mismo criterio que `emit_progress`, para el
    endpoint clásico sin streaming) no se emite nada, solo se espera
    `coro` tal cual, sin overhead de latido."""
    if callback is None:
        return await coro

    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval_seconds)
            if task in done:
                return task.result()
            await emit_progress(callback, stage)
    finally:
        if not task.done():
            task.cancel()
