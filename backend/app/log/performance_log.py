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

Este módulo solo cubre el TRAMO de análisis de fotos (DINOv2 +
Moondream2). La traducción local de descripciones (ver
app/nlp/translation.py, ADR-30/ADR-31) es una operación aparte y bajo
demanda -- no ligada a un análisis concreto -- y se registra en su propio
log, ver app/log/translation_log.py.
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
    dinov2_local_device: str | None,
    moondream_device: str | None,
    total_wall_seconds: float,
    per_photo_seconds: list[float],
    per_photo_dinov2_seconds: list[float],
    per_photo_scene_seconds: list[float],
) -> None:
    """Añade una línea al log de rendimiento. Nunca lanza excepción hacia
    el llamador: un fallo al escribir el log no debe tumbar ni degradar el
    análisis real.

    `dinov2_local_device`/`moondream_device`: dispositivo REAL donde
    corrió cada modelo en este proceso (ver
    `app.vision.geolocation.get_local_device()` y
    `app.vision.scene_analysis.get_device()`) -- IMPORTANTE, y motivo por
    el que existen estos dos parámetros en vez de asumir nada a partir de
    `igpu_offload_used`: sin offload activo, DINOv2 NO corre en CPU, sino
    en la MISMA GPU dedicada que Moondream2 (ver
    `_select_igpu_worker_device_index()` en geolocation.py -- el offload
    a iGPU existe justamente para dejar de compartirla). Un primer intento
    de esta función asumía "GPU si hay offload a iGPU, si no CPU" para
    los dos modelos, lo cual describía correctamente la traducción (ver
    app/log/translation_log.py, esa sí es CPU siempre) pero NO el
    análisis de fotos -- corregido tras confirmarlo directamente contra
    el código de selección de dispositivo (no era una suposición
    razonable, era un error de hecho). La escritura en sí (con su
    fallback de aviso único si el directorio no es escribible) vive en
    `_append_entry`."""
    if not settings.enable_performance_logging:
        return  # logging de rendimiento desactivado, ver ENABLE_PERFORMANCE_LOGGING

    if total_photos == 0:
        return  # nada que registrar -- no hubo fotos que analizar

    # `throughput_seconds_per_photo` (total_wall_seconds / total_photos) es
    # DISTINTO de `avg_seconds_per_photo` (media de `per_photo_seconds`,
    # que mide LATENCIA por foto: desde que arranca su intento hasta que
    # termina). Con concurrencia baja y una etapa mucho más lenta que la
    # otra, la LATENCIA de las fotos que quedan detrás en la cola del
    # semáforo crece de forma aproximadamente lineal con su posición, así
    # que la MEDIA de latencias puede acabar siendo varias veces mayor
    # que el tiempo real que cuesta añadir una foto más al análisis
    # (confirmado en producción -- ver ADR de rendimiento / conversación
    # del 21/8). El throughput SÍ es comparable entre configuraciones;
    # la latencia media es útil solo para estimar cuánto tarda en
    # aparecer el resultado de UNA foto concreta (progreso de UI).
    throughput = total_wall_seconds / total_photos

    # Tiempo de cómputo por DISPOSITIVO durante ESTE análisis de fotos (no
    # incluye traducción -- ver app/log/translation_log.py, es una
    # operación separada y bajo demanda, no atada a este análisis). TRES
    # categorías, no dos -- ver el docstring de arriba sobre por qué:
    #
    # - `cuda_gpu_seconds`: GPU DEDICADA (CUDA). Moondream2 corre ahí
    #   salvo que no haya CUDA en absoluto (`moondream_device == "cpu"`).
    #   DINOv2 corre ahí TAMBIÉN, compartiéndola, cuando NO hay offload a
    #   iGPU y sí hay CUDA (`dinov2_local_device == "cuda"`) -- este es
    #   el caso más común en la práctica (ver `_select_igpu_worker_
    #   device_index()`), y es justo la contención que ADR-28 intenta
    #   aliviar.
    # - `igpu_seconds`: iGPU vía DirectML (worker aparte, ver
    #   `_igpu_worker_device_index` en geolocation.py). Solo DINOv2, solo
    #   si `igpu_offload_used` es True.
    # - `cpu_seconds`: cualquiera de los dos modelos SIN ninguna GPU
    #   disponible (`dinov2_local_device`/`moondream_device == "cpu"`) --
    #   en la práctica solo ocurre en máquinas sin CUDA en absoluto (ver
    #   memoria: en el equipo de desarrollo, con GTX 1650, esto no
    #   debería pasar salvo error de detección).
    #
    # Con el pipeline entre fotos (ADR-29) estos tiempos pueden solaparse
    # entre sí, así que ningún par de estas tres cifras tiene por qué
    # sumar `total_wall_seconds` -- son SEGUNDOS DE CÓMPUTO consumidos,
    # no un cronómetro de pared exclusivo; el `_pct` es "cuántos segundos
    # de trabajo hubo en ese dispositivo por cada segundo de reloj", que
    # puede superar el 100% con buen solapamiento (señal de que el
    # pipeline aprovecha bien la concurrencia entre modelos).
    cuda_gpu_seconds, igpu_seconds, cpu_seconds = _compute_device_seconds(
        igpu_offload_used=igpu_offload_used,
        dinov2_local_device=dinov2_local_device,
        moondream_device=moondream_device,
        dinov2_total=sum(per_photo_dinov2_seconds),
        scene_total=sum(per_photo_scene_seconds),
    )

    cuda_gpu_usage_pct = _pct_of_wall_time(cuda_gpu_seconds, total_wall_seconds)
    igpu_usage_pct = _pct_of_wall_time(igpu_seconds, total_wall_seconds)
    cpu_usage_pct = _pct_of_wall_time(cpu_seconds, total_wall_seconds)

    avg = _average(per_photo_seconds)
    avg_dinov2 = _average(per_photo_dinov2_seconds)
    avg_scene = _average(per_photo_scene_seconds)

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
        # Dispositivo real por modelo -- ver docstring de la función.
        "dinov2_local_device": dinov2_local_device,
        "moondream_device": moondream_device,
        "total_wall_seconds": round(total_wall_seconds, 3),
        # Métrica principal para COMPARAR configuraciones -- ver el
        # comentario de arriba sobre por qué no vale usar
        # `avg_seconds_per_photo` para esto.
        "throughput_seconds_per_photo": round(throughput, 3),
        # Segundos de cómputo por dispositivo y su fracción respecto al
        # tiempo de reloj total -- ver comentario de arriba. Solo cubre
        # el análisis de fotos (DINOv2 + Moondream2); la traducción se
        # registra aparte en app/log/translation_log.py (siempre CPU).
        "cuda_gpu_seconds": round(cuda_gpu_seconds, 3),
        "igpu_seconds": round(igpu_seconds, 3),
        "cpu_seconds": round(cpu_seconds, 3),
        "cuda_gpu_usage_pct": round(cuda_gpu_usage_pct, 1),
        "igpu_usage_pct": round(igpu_usage_pct, 1),
        "cpu_usage_pct": round(cpu_usage_pct, 1),
        # Se mantiene por compatibilidad y porque sigue siendo útil para
        # otra cosa (estimar la latencia típica de UNA foto, no el
        # rendimiento agregado del análisis) -- ver comentario arriba.
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

    _append_entry(entry)


def _compute_device_seconds(
    *,
    igpu_offload_used: bool,
    dinov2_local_device: str | None,
    moondream_device: str | None,
    dinov2_total: float,
    scene_total: float,
) -> tuple[float, float, float]:
    """Reparte `dinov2_total`/`scene_total` (segundos de cómputo de cada
    modelo) entre las tres categorías de dispositivo -- ver el comentario
    detallado en `log_photo_analysis_run` sobre por qué son tres y no dos."""
    cuda_gpu_seconds = 0.0
    igpu_seconds = 0.0
    cpu_seconds = 0.0

    if igpu_offload_used:
        igpu_seconds += dinov2_total
    elif dinov2_local_device == "cuda":
        cuda_gpu_seconds += dinov2_total
    elif dinov2_local_device == "cpu":
        cpu_seconds += dinov2_total
    # dinov2_local_device is None (modelo nunca cargado, p. ej. 0 fotos
    # con geolocalización intentada): dinov2_total ya es 0.0, no aporta.

    if moondream_device == "cuda":
        cuda_gpu_seconds += scene_total
    elif moondream_device == "cpu":
        cpu_seconds += scene_total
    # moondream_device is None (enable_scene_analysis=False o modelo
    # nunca cargado): scene_total ya es 0.0, no aporta.

    return cuda_gpu_seconds, igpu_seconds, cpu_seconds


def _pct_of_wall_time(device_seconds: float, total_wall_seconds: float) -> float:
    return (device_seconds / total_wall_seconds * 100) if total_wall_seconds else 0.0


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _append_entry(entry: dict) -> None:
    """Escribe `entry` como una línea JSON en el log. Nunca lanza: un
    fallo de escritura no debe tumbar ni degradar el análisis real."""
    global _warned_unwritable

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
