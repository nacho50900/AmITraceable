"""
Log GENERAL de cada análisis completo -- a diferencia de
app/performance_log.py, que solo registra el tramo de análisis de fotos,
este registra el pipeline entero: fecha, plataforma, volumen de actividad
analizado (nº de posts/comentarios/fotos) y cuánto tardó cada etapa
(huella de escritura, detección de atributos, autodeclaraciones con IA,
espera de la geolocalización de fotos, estrechamiento de población...).

Pensado para la memoria del TFG (rendimiento real del sistema completo,
no solo del módulo de fotos) y para decidir, con datos y no con
intuición, qué etapa merece optimizarse antes.

Diseño RGPD (igual que performance_log.py, ver su docstring): ninguna
entrada identifica a la persona ni a su cuenta -- ni username, ni bio, ni
permalinks, ni IP, ni contenido de ningún post. Solo recuentos agregados y
tiempos.

Puede desactivarse por completo con ENABLE_PERFORMANCE_LOGGING=false (ver
Settings.enable_performance_logging en config.py) -- mismo interruptor que
usa performance_log.py, activado por defecto.
"""
import json
import logging
import time
from pathlib import Path

from app.analysis_timing import StageTimer
from app.config import settings

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).parent.parent / "data" / "performance"
_LOG_PATH = _LOG_DIR / "analysis_run_log.jsonl"

_warned_unwritable = False


def log_analysis_run(
    *,
    platform: str,
    n_posts: int,
    n_comments: int,
    n_media_items: int,
    n_photos: int,
    ai_enabled: bool,
    scene_analysis_enabled: bool,
    geolocation_available: bool,
    total_seconds: float,
    timer: StageTimer,
) -> None:
    """Añade una línea al log general de análisis. Nunca lanza excepción
    hacia el llamador: un fallo al escribir el log no debe tumbar ni
    degradar el análisis real (mismo criterio que performance_log.py)."""
    global _warned_unwritable

    if not settings.enable_performance_logging:
        return  # logging de rendimiento desactivado, ver ENABLE_PERFORMANCE_LOGGING

    entry = {
        "timestamp": time.time(),
        "platform": platform,
        "n_posts": n_posts,
        "n_comments": n_comments,
        "n_media_items": n_media_items,
        "n_photos": n_photos,
        "ai_enabled": ai_enabled,
        "scene_analysis_enabled": scene_analysis_enabled,
        "geolocation_available": geolocation_available,
        "total_seconds": round(total_seconds, 3),
        "stages_seconds": {name: round(seconds, 3) for name, seconds in timer.stages.items()},
    }

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        if not _warned_unwritable:
            logger.warning(
                "No se pudo escribir el log general de análisis en %s "
                "(sin permisos o directorio no accesible) -- se sigue sin "
                "registrar métricas, el análisis en sí no se ve afectado.",
                _LOG_PATH,
            )
            _warned_unwritable = True
