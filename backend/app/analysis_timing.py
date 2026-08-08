"""
Cronometraje interno del pipeline de análisis, usado por el log GENERAL de
cada análisis (ver app/analysis_run_log.py) para saber cuánto tardó cada
etapa (huella de escritura, detección de atributos, autodeclaraciones con
IA, geolocalización de fotos, estrechamiento de población...), no solo el
tiempo total.

Se usa un `ContextVar` en vez de pasar un objeto "timer" de un lado a otro
por todas las firmas de función (build_fingerprint, generate_report,
_apply_ai_findings...) porque estas ya tienen bastantes parámetros
opcionales (progress_callback, geolocation_task...) y añadir uno más solo
para cronometrar tocaría muchas firmas sin cambiar el comportamiento real.
Con `ContextVar`, cualquier función que corra DENTRO DE LA MISMA TAREA de
asyncio (que es como corre un análisis completo, de principio a fin, salvo
la tarea aparte de geolocalización -- ver más abajo) puede registrar su
propio tramo con `timed_stage(...)` sin que quien la llama tenga que saber
nada de cronometraje.

Nota sobre `asyncio.create_task` (geolocalización de fotos, lanzada al
principio de analysis_router._build_report ANTES de activar el timer):
una tarea creada así copia el `contextvars.Context` en el momento de
crearse, así que si no hay ningún timer activo todavía en ese instante,
esa tarea nunca "ve" el timer que se active después en la tarea padre --
mejor así: el tiempo de esa tarea en segundo plano se mide aparte, con más
detalle, en app/performance_log.py (tiempo por FOTO, no de este pipeline
general).

Si no hay ningún timer activo (p. ej. un test que llama a
`generate_report` directamente, sin pasar por `_build_report`),
`timed_stage` no hace nada -- no cambia ningún comportamiento existente.
"""
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    stages: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        # Se SUMA en vez de sobreescribir: si alguna etapa llegara a
        # ejecutarse más de una vez dentro del mismo análisis (no ocurre
        # hoy, pero es más robusto que asumir que nunca pasará), el log
        # muestra el tiempo total de esa etapa, no solo la última vez.
        self.stages[name] = self.stages.get(name, 0.0) + seconds


_active_timer: ContextVar["StageTimer | None"] = ContextVar("_active_timer", default=None)


@asynccontextmanager
async def run_with_timer():
    """Activa un StageTimer nuevo para el resto de esta tarea de asyncio y
    lo expone (`as timer`) para leer `.stages` al terminar."""
    timer = StageTimer()
    token = _active_timer.set(timer)
    try:
        yield timer
    finally:
        _active_timer.reset(token)


@asynccontextmanager
async def timed_stage(name: str):
    """Mide cuánto tarda el bloque `async with` y lo suma a la etapa
    `name` del timer activo, si hay alguno."""
    start = time.monotonic()
    try:
        yield
    finally:
        timer = _active_timer.get()
        if timer is not None:
            timer.add(name, time.monotonic() - start)
