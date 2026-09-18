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

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_CONNECTIONS = 40

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


async def check_username_across_sites(
    username: str,
    sites: dict | None = None,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[UsernameSiteResult]:
    """Comprueba `username` contra `sites` (dict nombre -> MaigretSite; por
    defecto, la base de datos completa de Maigret tras excluir tor/i2p/dns,
    ~5000 sitios).

    Llama a maigret.checking.maigret() -- el motor real de Maigret -- con
    `is_parsing_enabled=False` e `is_enrich_enabled=False`: nunca se
    extrae ni se solicita contenido del perfil (bio, avatar, ids, otros
    usernames), solo existencia. Ver docstring del modulo para el porque
    esto tambien excluye la recursion sin necesidad de una flag aparte.

    RENDIMIENTO: con la base completa, un barrido son varios miles de
    peticiones HTTP y tarda del orden de minutos incluso con
    `max_connections` alto -- mismo aviso que en la version anterior de
    este modulo, ver docstring de app/osint_router.py.

    Username vacio o solo espacios -> lista vacia, sin llamar a Maigret."""
    username = username.strip()
    if not username:
        return []

    site_dict = _default_site_dict() if sites is None else sites

    raw_results = await _maigret_check(
        username=username,
        site_dict=site_dict,
        logger=logger,
        timeout=timeout,
        is_parsing_enabled=False,
        is_enrich_enabled=False,
        max_connections=max_connections,
        no_progressbar=True,
    )

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
