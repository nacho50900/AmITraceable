"""
Correlacion de cuentas por nombre de usuario -- comprobacion de EXISTENCIA
unicamente, alcance deliberadamente acotado.

Contexto de diseno (ver conversacion/registro de la sesion donde se decidio
esto, inspirada en Lermen, Paleka, Swanson, Aerni, Carlini, Tramer (2026,
ETH Zurich / Anthropic), "Large-scale online deanonymization with LLMs",
arXiv:2602.16800): ese estudio muestra que cruzar cuentas de una misma
persona entre plataformas es uno de los vectores mas fuertes de
deanonimizacion asistida por IA. Este modulo demuestra ese mismo vector
para el propio usuario que se analiza a si mismo, con limites explicitos
para no convertirse en la propia herramienta de deanonimizacion de
terceros que el proyecto pretende combatir.

DECISION DE DISENO (tercera vuelta sobre este modulo, ver historial): en
vez de mantener nuestra propia lista de sitios (v1, ~14 a mano) o
reimplementar el motor de deteccion de Maigret a partir de su fichero de
datos vendorizado (v2), se usa la libreria `maigret` (pip, MIT License,
github.com/soxoj/maigret) DIRECTAMENTE como dependencia, llamando a su
funcion de comprobacion de bajo nivel (`maigret.checking.maigret()`) con
las dos flags que desactivan la parte de "dossier":

- `is_parsing_enabled=False` -- Maigret NO extrae bio/avatar/ids/otros
  usernames de la pagina de perfil. Confirmado empiricamente: con esta
  flag, el `SiteResult` que devuelve ni siquiera incluye la clave
  `response_text` (ver test_content_is_never_extracted_or_stored).
- `is_enrich_enabled=False` -- Maigret NO hace peticiones adicionales a
  endpoints de API derivados de la URL del perfil (su modo "--enrich").

La RECURSION (`--no-recursion` en la CLI de Maigret) no hace falta
desactivarla explicitamente aqui: ese bucle (volver a lanzar
`maigret.checking.maigret()` con los usernames/ids nuevos que la
extraccion hubiera encontrado) vive en la orquestacion de su CLI
(`maigret/maigret.py`, la funcion `main()` interactiva), NO en la funcion
de libreria `maigret.checking.maigret()` que llamamos aqui directamente.
Al llamar a la funcion de bajo nivel UNA vez y no volver a invocarla
nosotros mismos con nada que ella hubiera "descubierto", no hay ningun
bucle de recursion que desactivar -- sencillamente no existe en este
camino de codigo.

Por que esto es mejor que reimplementar el motor (v2, descartada): Maigret
ya resuelve correctamente las plantillas de "engine" compartidas, el
`regexCheck` por sitio, `ignore403`, las paginas de error conocidas
(`errors`), las cabeceras especiales que algunos sitios necesitan, y los
protocolos no soportados -- reimplementar todo eso a mano (que es lo que
hacia la v2 de este modulo) es justo el tipo de logica fragil que ya
mantiene, con mas cobertura, la propia comunidad de Maigret. Aqui solo se
CONSUME esa libreria con el alcance mas estrecho, no se copia su motor.

Base de datos de sitios: se usa la que trae empaquetada la propia
distribucion de `maigret` instalada (fichero `resources/data.json` dentro
del paquete) -- NO se vendoriza una copia propia. Fijar la version exacta
de `maigret` en requirements.txt (ver ese fichero) ya congela que base de
datos se usa, igual que fijariamos la version de cualquier otra
dependencia; actualizarla es tan simple como subir esa version.

INTEGRACION EN LA PANTALLA DE CARGA Y EN EL INFORME (ver ADR-48): esta
comprobacion se lanza como una tarea mas en paralelo con el resto del
pipeline (analysis_router._build_report, mismo patron que
`geolocation_task`), y su resultado (solo las cuentas ENCONTRADAS,
`exists=True`) se incluye en `ExposureReport.related_accounts` (ver
schemas.py::UsernameCorrelationSummary). El progreso en vivo usa
`progress_callback` (ver app/progress.py), igual que el resto del
pipeline -- con una salvedad: `maigret.checking.maigret()` no expone un
callback async por sitio comprobado, sino un objeto `query_notify` cuyo
`.update()` se llama de forma SINCRONA una vez por sitio. Por eso aqui se
usa un `query_notify` propio que solo lleva la cuenta (sin async), y una
tarea de fondo separada que la sondea cada segundo y emite el progreso
real -- evita tanto tener que await-ear codigo sincrono como inundar el
stream SSE con ~5000 eventos individuales.
"""
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import maigret as _maigret_package
from maigret.checking import maigret as _maigret_check
from maigret.result import MaigretCheckStatus
from maigret.sites import MaigretDatabase

from app.progress import ProgressCallback, emit_progress

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_CONNECTIONS = 40

# Cada cuanto se sondea el contador de sitios ya comprobados para emitir un
# evento de progreso -- ver docstring del modulo. No tiene sentido bajar
# esto mucho: el frontend solo pinta "X/Y sitios comprobados", no hace
# falta granularidad de sub-segundo para que se perciba en marcha.
PROGRESS_POLL_INTERVAL_SECONDS = 1.0

# Protocolos que no son HTTP normal (unos pocos sitios .onion/.i2p o
# comprobaciones DNS) -- fuera de alcance para este TFG, igual que en la
# version anterior de este modulo.
_EXCLUDED_TAGS = ("tor", "i2p", "dns")

_STATUS_TO_EXISTS = {
    MaigretCheckStatus.CLAIMED: True,
    MaigretCheckStatus.AVAILABLE: False,
    MaigretCheckStatus.UNKNOWN: None,
    # MaigretCheckStatus.ILLEGAL ("username no valido para este sitio", p.
    # ej. no cumple el regexCheck del sitio) se filtra en
    # check_username_across_sites -- no es "existe" ni "no existe", el
    # sitio directamente no aplica a este username.
}

# Carga perezosa y cacheada del fichero de datos que trae `maigret`
# instalado -- parsear ~2.3MB de JSON en cada import (p.ej. en cada test
# que importa app.main) seria un coste de arranque innecesario.
_CACHED_DATABASE: MaigretDatabase | None = None


def _database_path() -> Path:
    return Path(_maigret_package.__file__).resolve().parent / "resources" / "data.json"


def get_database() -> MaigretDatabase:
    global _CACHED_DATABASE
    if _CACHED_DATABASE is None:
        _CACHED_DATABASE = MaigretDatabase().load_from_file(str(_database_path()))
    return _CACHED_DATABASE


def _default_site_dict() -> dict:
    return get_database().ranked_sites_dict(
        top=sys.maxsize,
        disabled=False,
        excluded_tags=list(_EXCLUDED_TAGS),
    )


@dataclass(frozen=True)
class UsernameSiteResult:
    site: str
    url: str
    # None = no se pudo determinar (timeout, error de red, bloqueo
    # anti-bot -- MaigretCheckStatus.UNKNOWN). Distinto de False, que
    # significa "se comprobo y no existe".
    exists: bool | None


class _CountingQueryNotify:
    """Objeto duck-typed compatible con el `query_notify` de
    `maigret.checking.maigret()` -- esa funcion llama a `.start()` una vez
    al principio, `.update()` una vez por CADA sitio comprobado (de forma
    sincrona, nunca `await`), y `.finish()`/`.warning()`/`.enrich()` en
    otros puntos. Aqui no se imprime nada (a diferencia de su
    `QueryNotifyPrint` por defecto) -- solo interesa CONTAR cuantos han
    terminado; el resultado real se lee de lo que devuelve `maigret()`,
    no de este objeto."""

    def __init__(self) -> None:
        self.checked = 0

    def start(self, *args, **kwargs) -> None:
        pass  # Nada que inicializar: `checked` ya arranca en 0 en __init__.

    def update(self, *args, **kwargs) -> None:
        self.checked += 1

    def finish(self, *args, **kwargs) -> None:
        pass  # El resultado se lee de lo que devuelve maigret(), no de aquí.

    def warning(self, *args, **kwargs) -> None:
        pass  # Solo interesa CONTAR comprobaciones (ver docstring de la clase); se descarta.

    def enrich(self, *args, **kwargs) -> None:
        pass  # is_enrich_enabled=False (ver check_username_across_sites): Maigret no debería llamar a esto nunca; se implementa igualmente porque el duck-typing lo exige.


async def _poll_progress(notifier: _CountingQueryNotify, total_sites: int, progress_callback: ProgressCallback) -> None:
    """Bucle de sondeo extraído de `check_username_across_sites` (antes una
    función anidada) para que su `while`/`if` no sumen a la complejidad
    cognitiva de esa función -- ver el issue de Sonar "Cognitive
    Complexity" sobre ese refactor. Cancelada externamente por
    `check_username_across_sites` vía `_cancel_and_await` cuando
    `_maigret_check` termina."""
    last_reported = -1
    while True:
        await asyncio.sleep(PROGRESS_POLL_INTERVAL_SECONDS)
        if notifier.checked != last_reported:
            last_reported = notifier.checked
            await emit_progress(
                progress_callback,
                "Comprobando cuentas relacionadas...",
                accounts_checked=notifier.checked,
                total_accounts=total_sites,
                track="correlacion_cuentas",
            )


async def _cancel_and_await(task: asyncio.Task) -> None:
    """Cancela `task` y espera a que termine de propagar la cancelación.

    El `asyncio.CancelledError` que llega aquí SIEMPRE viene de nuestro
    propio `task.cancel()` de la línea anterior -- `task` (el poll de
    `_poll_progress`) no se expone a nadie más, así que nunca puede haber
    una cancelación externa mezclada con esta. Por eso absorberlo aquí es
    correcto y no una excepción "tragada" a ciegas: re-lanzarlo
    propagaría hacia `check_username_across_sites` una cancelación que
    esa misma función inició para limpiar, no una cancelación real de
    quien la llamó a ella."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _raw_results_to_site_results(raw_results: dict) -> list[UsernameSiteResult]:
    """Convierte el dict que devuelve `maigret.checking.maigret()` a
    `UsernameSiteResult`, extraído de `check_username_across_sites` por el
    mismo motivo que `_poll_progress` (complejidad cognitiva)."""
    results: list[UsernameSiteResult] = []
    for name, site_result in raw_results.items():
        check_result = site_result.get("status")
        status = check_result.status if check_result is not None else None
        if status not in _STATUS_TO_EXISTS:
            continue  # ILLEGAL (username no aplica a este sitio) o sin resultado -- se omite
        results.append(
            UsernameSiteResult(
                site=name,
                url=site_result.get("url_user") or site_result.get("url_main") or "",
                exists=_STATUS_TO_EXISTS[status],
            )
        )
    return results


async def check_username_across_sites(
    username: str,
    sites: dict | None = None,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    progress_callback: ProgressCallback | None = None,
) -> list[UsernameSiteResult]:
    """Comprueba `username` contra `sites` (dict nombre -> MaigretSite; por
    defecto, la base de datos completa de Maigret tras excluir tor/i2p/dns,
    ~5000 sitios).

    Llama a maigret.checking.maigret() -- el motor real de Maigret -- con
    `is_parsing_enabled=False` e `is_enrich_enabled=False`: nunca se
    extrae ni se solicita contenido del perfil (bio, avatar, ids, otros
    usernames), solo existencia. Ver docstring del modulo para el porque
    esto tambien excluye la recursion sin necesidad de una flag aparte.

    `progress_callback` (ver app/progress.py) es opcional -- si se pasa,
    se emite un evento `track="correlacion_cuentas"` con
    `accounts_checked`/`total_accounts` aproximadamente cada
    PROGRESS_POLL_INTERVAL_SECONDS, mas uno final al terminar. Si se omite
    (p. ej. `POST /api/username-correlation`, uso independiente), el
    comportamiento es exactamente el de antes: sin overhead de polling.

    RENDIMIENTO: con la base completa, un barrido son varios miles de
    peticiones HTTP y tarda del orden de minutos incluso con
    `max_connections` alto -- mismo aviso que en la version anterior de
    este modulo, ver docstring de app/osint_router.py.

    Username vacio o solo espacios -> lista vacia, sin llamar a Maigret ni
    al `progress_callback`."""
    username = username.strip()
    if not username:
        return []

    site_dict = _default_site_dict() if sites is None else sites
    total_sites = len(site_dict)

    notifier = _CountingQueryNotify()
    report_progress = progress_callback is not None and total_sites
    progress_task: asyncio.Task | None = None
    if report_progress:
        progress_task = asyncio.create_task(_poll_progress(notifier, total_sites, progress_callback))

    try:
        raw_results = await _maigret_check(
            username=username,
            site_dict=site_dict,
            logger=logger,
            query_notify=notifier,
            timeout=timeout,
            is_parsing_enabled=False,
            is_enrich_enabled=False,
            max_connections=max_connections,
            no_progressbar=True,
        )
    finally:
        if progress_task is not None:
            await _cancel_and_await(progress_task)

    if report_progress:
        # Evento final con el total exacto -- por si el ultimo tramo de
        # comprobaciones termino entre dos sondeos y se quedo sin
        # reportar (el poll de arriba solo corre CADA
        # PROGRESS_POLL_INTERVAL_SECONDS, no justo al terminar), o si con
        # 0 sitios de margen el bucle nunca llego a ejecutarse una vez.
        await emit_progress(
            progress_callback,
            "Comprobando cuentas relacionadas...",
            accounts_checked=total_sites,
            total_accounts=total_sites,
            track="correlacion_cuentas",
        )

    return _raw_results_to_site_results(raw_results)

