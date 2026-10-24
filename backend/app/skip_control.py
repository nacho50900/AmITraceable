"""
Control de "saltar el resto del análisis" para el pipeline de análisis en
streaming (`GET /api/analyze/{platform}/stream`, ver analysis_router.py).

Las dos pistas LARGAS del pipeline -- la comprobación de cuentas por username
(Maigret, ~5000 sitios, ver app/osint/username_correlation.py) y el análisis
de fotos (DINOv2 + Moondream2, ver app/vision/geolocation.py) -- pueden tardar
minutos. `SkipControl` es el interruptor que el usuario acciona desde la
pantalla de carga (botón "Saltar el resto del análisis", vía
`POST /api/analyze/{platform}/skip`) para decirles que paren YA y devuelvan lo
que lleven hasta el momento. NO salta el resto del pipeline: la extracción de
autodeclaraciones con IA, las inferencias sobre lo obtenido y la generación del
informe siguen ejecutándose con normalidad sobre ese resultado parcial.

Se pasa como parámetro opcional (`skip=None` por defecto), mismo patrón que
`progress_callback` en app/progress.py: sin él, todo se comporta exactamente
igual que antes (endpoint clásico `POST /api/analyze/{platform}`, tests...).
"""
import asyncio
from typing import Awaitable


class SkipControl:
    """Interruptor de un solo uso, por análisis en curso."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def skipped(self) -> bool:
        return self._event.is_set()

    def request_skip(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


async def wait_or_skip(task: "asyncio.Future", skip: SkipControl | None) -> bool:
    """Espera a que `task` termine O a que se pida saltar, lo que ocurra
    antes. Devuelve True si `task` terminó (con resultado o con error -- NO
    relanza nada aquí: quien llama lee `task.result()` y decide), False si
    se pidió saltar antes de que terminara (la tarea sigue en marcha: quien
    llama decide si cancelarla).

    Usa `asyncio.wait()` en vez de `await task` a propósito, mismo criterio
    que `_cancel_and_await` en app/osint/username_correlation.py: no relanza
    la excepción de la tarea, y una cancelación REAL de quien llama sí se
    propaga con normalidad."""
    if task.done():
        return True
    if skip is None:
        await asyncio.wait([task])
        return True
    skip_waiter = asyncio.ensure_future(skip.wait())
    try:
        await asyncio.wait({task, skip_waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        skip_waiter.cancel()
    return task.done()


async def cancel_pending(tasks: list["asyncio.Future"]) -> None:
    """Cancela las tareas que aún no han terminado y espera a que acaben de
    propagar la cancelación (sin relanzarla, ver `wait_or_skip`)."""
    pending = [t for t in tasks if not t.done()]
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending)


# Registro de análisis en curso, para que `POST .../skip` (otra petición HTTP
# distinta del stream SSE) encuentre el `SkipControl` del análisis de ESE
# usuario. La clave la decide analysis_router (plataforma + token de sesión);
# solo vive en memoria mientras dura el análisis -- nada se persiste.
_active_runs: dict[str, SkipControl] = {}


def register_run(key: str, control: SkipControl) -> None:
    _active_runs[key] = control


def unregister_run(key: str, control: SkipControl) -> None:
    # Solo se borra si la entrada sigue siendo la de ESTE análisis: si el
    # EventSource reconectó y arrancó otro análisis con la misma clave, no
    # hay que llevarse por delante el control del nuevo.
    if _active_runs.get(key) is control:
        del _active_runs[key]


def request_skip_for(key: str) -> bool:
    """True si había un análisis en curso para `key` y se le pidió saltar."""
    control = _active_runs.get(key)
    if control is None:
        return False
    control.request_skip()
    return True
