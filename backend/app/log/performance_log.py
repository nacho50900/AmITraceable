"""
Registro de métricas de RENDIMIENTO del análisis de fotos (DINOv2 +,
opcionalmente, Moondream2) -- pensado para poder analizar después, con
datos reales y no con intuición, si la concurrencia/nº de hilos elegidos
automáticamente (ver `_default_photo_analysis_concurrency` en config.py)
es en la práctica una buena elección, y para tener una fuente empírica
citable en la memoria del TFG (ver la sección "Plan de evaluación
pendiente" del README).

Diseño RGPD (coherente con el resto del proyecto, ver README): esto NO es
la base de datos que el proyecto evita a propósito. Cada entrada es
puramente técnica -- cuántas fotos, cuántos núcleos, cuánto tardó -- sin
ningún identificador de usuario, cuenta, permalink, URL ni contenido de
ninguna foto. No hace falta borrarlo al cerrar sesión porque no dice nada
sobre ninguna persona concreta.

Formato: JSON Lines (`.jsonl`), una línea por análisis -- se elige sobre
CSV porque el número de duraciones individuales por foto varía de un
análisis a otro (no encaja bien en columnas fijas) y sobre una base de
datos porque un fichero de texto append-only es suficiente para este caso
de uso (analizarlo offline con pandas, ver scripts/analyze_performance_log.py)
sin añadir una dependencia de infraestructura nueva.

Vive en `backend/data/performance/`, que ya cae dentro de `backend/data/`
en el .gitignore existente (mismo motivo que el índice FAISS: son datos
regenerables, no versionados). Si el directorio no se puede escribir (p.
ej. permisos en algún despliegue concreto), se avisa una vez por proceso y
se sigue sin romper el análisis -- este módulo es una herramienta de
observación, nunca debe ser motivo de que un análisis falle.

Puede desactivarse por completo con ENABLE_PERFORMANCE_LOGGING=false (ver
Settings.enable_performance_logging en config.py) -- activado por
defecto, ya que no guarda nada personal.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).parent.parent.parent / "data" / "performance"
_LOG_PATH = _LOG_DIR / "photo_analysis_log.jsonl"

_warned_unwritable = False


@dataclass
class PhotoAnalysisTiming:
    """Acumulador de tiempos durante UN análisis (una llamada a
    `estimate_locations_for_posts`). Los tres `record*()` se llaman una
    vez por foto procesada (con o sin resultado válido -- lo que importa
    aquí es cuánto tardó el intento, no si tuvo éxito).

    `per_photo_seconds` es el tiempo TOTAL de la foto (desde que arrancan
    los dos modelos hasta que los dos han terminado para ESA foto, ver
    `_process_photo` en app/vision/geolocation.py) -- se mantiene tal
    cual por compatibilidad con logs antiguos y porque sigue siendo la
    cifra relevante para "cuánto tardó esta foto en tener resultado
    completo".

    `dinov2_seconds`/`scene_seconds` son NUEVOS (ver ADR-29): tiempo de
    CADA modelo por separado, medido dentro de su propio semáforo. Desde
    ADR-29, ambos modelos corren en pipeline entre fotos distintas (el
    semáforo de uno se libera sin esperar al otro), así que la SUMA de
    `dinov2_seconds[i] + scene_seconds[i]` para una foto puede ser MAYOR
    que `per_photo_seconds[i]` si esa foto se solapó bien con sus
    vecinas -- eso es señal de que el pipeline está funcionando, no un
    error de medición. Sirven para diagnosticar, foto a foto, cuál de
    los dos modelos domina el tiempo y si el offload a iGPU (ADR-28)
    está ayudando o no en la práctica, cosa que `per_photo_seconds` por
    sí solo no permitía distinguir."""

    per_photo_seconds: list[float] = field(default_factory=list)
    dinov2_seconds: list[float] = field(default_factory=list)
    scene_seconds: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        self.per_photo_seconds.append(round(seconds, 3))

    def record_dinov2(self, seconds: float) -> None:
        self.dinov2_seconds.append(round(seconds, 3))

    def record_scene(self, seconds: float) -> None:
        self.scene_seconds.append(round(seconds, 3))


def log_photo_analysis_run(
    *,
    total_photos: int,
    cpu_count: int,
    configured_concurrency: int,
    actual_concurrency: int,
    threads_per_inference: int,
    enable_scene_analysis: bool,
    igpu_offload_used: bool,
    total_wall_seconds: float,
    per_photo_seconds: list[float],
    per_photo_dinov2_seconds: list[float],
    per_photo_scene_seconds: list[float],
) -> None:
    """Añade una línea al log de rendimiento. Nunca lanza excepción hacia
    el llamador: un fallo al escribir el log no debe tumbar ni degradar el
    análisis real."""
    global _warned_unwritable

    if not settings.enable_performance_logging:
        return  # logging de rendimiento desactivado, ver ENABLE_PERFORMANCE_LOGGING

    if total_photos == 0:
        return  # nada que registrar -- no hubo fotos que analizar

    avg = sum(per_photo_seconds) / len(per_photo_seconds) if per_photo_seconds else None
    avg_dinov2 = (
        sum(per_photo_dinov2_seconds) / len(per_photo_dinov2_seconds)
        if per_photo_dinov2_seconds
        else None
    )
    avg_scene = (
        sum(per_photo_scene_seconds) / len(per_photo_scene_seconds)
        if per_photo_scene_seconds
        else None
    )

    entry = {
        "timestamp": time.time(),
        "total_photos": total_photos,
        "cpu_count": cpu_count,
        "configured_concurrency": configured_concurrency,
        "actual_concurrency": actual_concurrency,
        "threads_per_inference": threads_per_inference,
        "enable_scene_analysis": enable_scene_analysis,
        # Estado REAL en el momento de este análisis, no si
        # ENABLE_IGPU_OFFLOAD estaba a true en la config: si el worker
        # falló o nunca respondió, esto sale False aunque el flag esté
        # activado -- ver `_igpu_worker_device_index`/`_igpu_worker_failed`
        # en app/vision/geolocation.py. Sin esto, una comparación de
        # rendimiento "con offload vs sin offload" no sería de fiar (un
        # fallback silencioso al modelo local contaminaría el grupo "con
        # offload" con tiempos que en realidad son de ejecución local).
        "igpu_offload_used": igpu_offload_used,
        "total_wall_seconds": round(total_wall_seconds, 3),
        "avg_seconds_per_photo": round(avg, 3) if avg is not None else None,
        # Desglose por modelo (ver ADR-29) -- desde el pipeline entre
        # fotos, la suma de estas dos medias puede ser MAYOR que
        # avg_seconds_per_photo si hay buen solapamiento entre fotos
        # vecinas; eso es la señal de que el pipeline funciona, no un
        # error de los datos.
        "avg_dinov2_seconds_per_photo": round(avg_dinov2, 3) if avg_dinov2 is not None else None,
        "avg_scene_seconds_per_photo": round(avg_scene, 3) if avg_scene is not None else None,
        "per_photo_seconds": per_photo_seconds,
        "per_photo_dinov2_seconds": per_photo_dinov2_seconds,
        "per_photo_scene_seconds": per_photo_scene_seconds,
    }

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        if not _warned_unwritable:
            logger.warning(
                "No se pudo escribir el log de rendimiento en %s "
                "(sin permisos o directorio no accesible) -- se sigue sin "
                "registrar métricas, el análisis en sí no se ve afectado.",
                _LOG_PATH,
            )
            _warned_unwritable = True
