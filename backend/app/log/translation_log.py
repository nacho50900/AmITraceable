"""
Log de RENDIMIENTO de la traducción local de descripciones (afición/caption
de Moondream2, ver app/nlp/translation.py y ADR-30/ADR-31) -- una entrada
por cada llamada a `POST /analyze/translate-descriptions`.

A diferencia de performance_log.py (que mide el análisis de fotos: DINOv2 +
Moondream2, una vez por análisis completo de Instagram), la traducción es
una operación INDEPENDIENTE y bajo demanda: el frontend la dispara cada vez
que se abre un informe o se cambia de idioma, así que puede pasar 0, 1 o
varias veces por análisis, en cualquier momento posterior -- no hay ningún
identificador de análisis/informe que las una de forma fiable
(`TranslateDescriptionsRequest`, ver app/models/schemas.py, solo lleva la
lista de textos). Por eso vive en su propio fichero en vez de intentar
encajarla dentro de una entrada de photo_analysis_log.jsonl.

CPU, siempre: `translate_texts_local()` carga el traductor CTranslate2 con
`device="cpu"` fijo (ver app/nlp/translation.py) -- a día de hoy no existe
ningún camino de GPU/iGPU para la traducción. IMPORTANTE: esto NO
significa que compita por CPU con Moondream2 en la práctica -- con una
GPU dedicada disponible, DINOv2 y Moondream2 corren los dos en esa GPU
(compartida si no hay offload a iGPU, ver ADR-28 y el docstring de
`log_photo_analysis_run()` en performance_log.py), así que la traducción
en CPU normalmente NO compite con ellos por el mismo recurso -- compite,
si acaso, con cualquier otra cosa que esté usando la CPU en ese momento
(el propio servidor FastAPI, por ejemplo). Este dato, cruzado con
`cuda_gpu_seconds`/`igpu_seconds`/`cpu_seconds` de performance_log.py
para el mismo periodo, es lo que permite ver el reparto agregado de
trabajo entre GPU dedicada / iGPU / CPU -- ver la sección de traducción
dentro de scripts/analyze_performance_log.py (fusionada ahí junto con el
análisis de fotos, no en un script aparte).

Mismo diseño RGPD, mismo formato JSON Lines, mismo directorio
(`backend/data/performance/`) y mismo interruptor
(ENABLE_PERFORMANCE_LOGGING) que performance_log.py -- ver su docstring
para el razonamiento completo, no se repite aquí. IMPORTANTE: esta entrada
NO guarda ninguno de los textos traducidos ni originales (podrían contener
aficiones o descripciones personales de la persona analizada) -- solo
cuántos textos había y cuánto tardó.
"""
import json
import logging
import time
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).parent.parent.parent / "data" / "performance"
_LOG_PATH = _LOG_DIR / "translation_log.jsonl"

_warned_unwritable = False


def log_translation_run(
    *,
    num_texts: int,
    source_lang: str,
    target_lang: str,
    cpu_count: int,
    total_seconds: float,
    translation_available: bool,
) -> None:
    """Añade una línea al log de rendimiento de traducción. Nunca lanza
    excepción hacia el llamador (mismo criterio que performance_log.py):
    un fallo al escribir el log no debe tumbar la traducción real.

    `translation_available`: si el modelo local estaba de verdad
    disponible en disco para esta dirección en el momento de la llamada
    (ver `app.nlp.translation.translation_available()`). Si es False,
    `total_seconds` mide solo el coste de comprobarlo y devolver los
    textos sin traducir (ruta de degradación, ver translation.py) -- se
    registra igual, con el flag a False, para poder filtrarlas al
    analizar en vez de que contaminen en silencio la media de "traducir
    de verdad"."""
    global _warned_unwritable

    if not settings.enable_performance_logging:
        return  # logging de rendimiento desactivado, ver ENABLE_PERFORMANCE_LOGGING

    if num_texts == 0:
        return  # nada que registrar -- lista vacía, no-op en translate_texts_local()

    entry = {
        "timestamp": time.time(),
        "num_texts": num_texts,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "direction": f"{source_lang}-{target_lang}",
        "cpu_count": cpu_count,
        # Fijo por ahora -- ver docstring del módulo. Campo pensado para
        # no tener que cambiar el esquema del log el día que exista un
        # camino de GPU para la traducción.
        "device": "cpu",
        "translation_available": translation_available,
        "total_seconds": round(total_seconds, 3),
        "avg_seconds_per_text": round(total_seconds / num_texts, 3),
    }

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        if not _warned_unwritable:
            logger.warning(
                "No se pudo escribir el log de traducción en %s "
                "(sin permisos o directorio no accesible) -- se sigue sin "
                "registrar métricas, la traducción en sí no se ve afectada.",
                _LOG_PATH,
            )
            _warned_unwritable = True
