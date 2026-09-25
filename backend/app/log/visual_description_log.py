"""
Registro OPCIONAL de las DESCRIPCIONES que genera Moondream2 para cada
foto, junto con qué variante del modelo las produjo (F16/Q8_0/Q4_K_M/...,
ver `app.vision.scene_analysis.get_model_variant()`) -- pensado para
poder comparar, foto a foto, qué dice cada backend sobre la MISMA imagen
exacta, no solo cuánto tarda (eso ya lo cubre performance_log.py). Nace
de la sesión de cuantización (17/9): medir velocidad sin poder comparar
calidad es una comparación coja para la memoria del TFG.

DISEÑO RGPD -- DISTINTO A PROPÓSITO del resto de logs de este paquete
(ver performance_log.py, translation_log.py): aquellos son "puramente
técnicos, sin nada personal" por diseño. ESTE NO PUEDE SERLO -- su
propósito completo es guardar el CONTENIDO real que Moondream2 extrae de
cada foto (si hay personas, indicios de pareja, matrícula, texto visible
tal cual aparece...). Para un proyecto cuyo objeto de estudio es
precisamente la exposición de privacidad de otras personas, acumular esto
sin pensarlo sería replicar exactamente el problema que la memoria
analiza. Por eso:

- Desactivado por defecto (`LOG_VISUAL_DESCRIPTIONS=false`, ver
  `Settings.log_visual_descriptions` en config.py) -- a diferencia de
  `enable_performance_logging`, que sí va activado por defecto. Actívalo
  solo para sesiones de comparación deliberadas (p. ej. las mismas fotos
  de prueba contra F16 vs Q8_0 vs Q4_K_M), no lo dejes encendido corriendo
  contra perfiles reales sin necesidad.
- Vive en su propio directorio (`backend/data/visual_descriptions/`, NO
  junto a `data/performance/`) para que sea visualmente imposible
  confundirlo con los logs técnicos al mirar la carpeta `data/`.
- No guarda NINGÚN identificador de la persona/cuenta de origen (ni
  usuario, ni permalink, ni plataforma) -- solo el hash de la imagen y lo
  que el modelo dijo sobre ella. Sigue siendo contenido personal (una
  descripción de una foto real es un dato personal aunque no lleve
  nombre), así que trátalo con el mismo cuidado que el resto de datos
  analizados por este proyecto: no lo subas a git (ya cae en
  `data/`, gitignored), bórralo cuando termines de comparar.

Identificador de imagen: SHA-256 de los píxeles crudos
(`image.tobytes()`, no del fichero JPEG/PNG original ni de una nueva
compresión) truncado a 16 caracteres hex -- determinista para el mismo
contenido de imagen sin importar cómo se cargó ni en qué formato se vaya
a guardar, así la MISMA foto da el MISMO id entre ejecuciones distintas
(F16 hoy, Q4_K_M mañana) y se pueden agrupar entradas de la misma foto
sin necesitar guardar la imagen en sí.

Formato: JSON Lines, igual que performance_log.py, mismos motivos (una
entrada por foto analizada, fácil de analizar después con pandas si hace
falta, sin dependencia de infraestructura nueva).
"""
import hashlib
import json
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).parent.parent.parent / "data" / "visual_descriptions"
_LOG_PATH = _LOG_DIR / "visual_description_log.jsonl"

_warned_unwritable = False


def image_content_id(image) -> str:
    """SHA-256 de los píxeles crudos de `image` (PIL.Image), truncado a
    16 caracteres hex -- ver docstring del módulo sobre por qué esto y no
    un hash del fichero original ni un perceptual hash (`imagehash`, ya
    usado en otra parte del proyecto para detectar fotos CASI iguales:
    aquí se necesita identidad EXACTA para comparar salidas de modelo
    sobre la misma imagen, no similitud)."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def log_visual_description(
    *,
    image_id: str,
    model_variant: str | None,
    caption: str | None,
    structured: str | None,
) -> None:
    """Registra una descripción generada, si `Settings.log_visual_descriptions`
    está activado (comprobado aquí, no por quien llama -- así una sola
    variable de entorno basta para activar/desactivar esto en cualquier
    punto de llamada sin tener que repetir la comprobación). Nunca lanza
    -- best-effort, mismo criterio que el resto de logs de este paquete:
    un fallo escribiendo esto no debe afectar al análisis real."""
    if not settings.log_visual_descriptions:
        return

    entry = {
        "image_id": image_id,
        "moondream_model_variant": model_variant,
        "caption": caption,
        "structured": structured,
    }
    _append_entry(entry)


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
                "No se pudo escribir el log de descripciones visuales en %s "
                "(sin permisos o directorio no accesible) -- se sigue sin "
                "registrar, el análisis en sí no se ve afectado.",
                _LOG_PATH,
            )
            _warned_unwritable = True
