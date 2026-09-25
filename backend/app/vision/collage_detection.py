"""
Detección heurística de fotos que son en realidad un COLLAGE (varias
sub-fotos combinadas en una sola imagen, típicamente con apps tipo
Instagram Layout, PicCollage, Instasize...) -- ver ADR-41.

Por qué importa: tanto DINOv2 (geolocation.py) como Moondream2
(scene_analysis.py) asumen que cada foto muestra UNA escena coherente. Un
collage rompe ese supuesto en los dos módulos, por motivos distintos:

- DINOv2/FAISS: el índice se construyó con imágenes de calle de UN solo
  lugar cada una (OSV-5M). El embedding de un collage de 4 sub-fotos de
  sitios distintos no se parece de forma coherente a NINGÚN vecino del
  índice -- el resultado más probable no es "sin match" (que sería lo
  honesto), sino un match de baja confianza pero sintácticamente válido:
  un falso positivo silencioso, peor que no analizar la foto en absoluto.
- Moondream2: el prompt (ver docstring de cabecera de scene_analysis.py)
  ya identifica como ambigüedad real el caso de varias personas de
  protagonismo similar en una misma foto -- un collage con sub-fotos de
  personas DISTINTAS es exactamente ese problema, pero peor: ni siquiera
  son necesariamente la misma escena ni el mismo momento.

Estrategia elegida: heurístico barato en CPU (numpy sobre la imagen ya
decodificada), NO otro modelo de ML -- coherente con las restricciones de
hardware del proyecto (GTX 1650, 4GB VRAM) y con no gastar cómputo de GPU
en analizar algo que de entrada no va a dar una señal fiable. Se ejecuta
ANTES de despachar la foto a DINOv2/Moondream2 (ver `_process_photo` en
geolocation.py), así que una foto detectada como collage ni siquiera llega
a cargar ninguno de los dos modelos para ella.

Qué detecta: bandas de color casi uniforme (blanco o negro, los colores de
margen que usan las apps de collage habituales) que dividen la imagen en
una cuadrícula, en el INTERIOR de la imagen -- una franja pegada al borde
EXTERIOR se descarta a propósito (letterboxing/padding normal, no dice
nada sobre si hay varias sub-fotos).

ALTERNATIVA CONSIDERADA Y DESCARTADA: leer el tag EXIF `Software` (algunas
apps de collage lo dejan escrito, p. ej. "Instagram Layout"). Se descarta
porque Instagram/Reddit re-codifican las imágenes al servirlas desde su
CDN y normalmente STRIPEAN los metadatos EXIF no esenciales en el proceso
-- en la práctica, para las fotos que este proyecto analiza (descargadas
de esas CDN, no del carrete original del usuario), ese tag casi nunca
sobrevive. Añadir el chequeo habría sido código muerto la mayoría de las
veces, sin aportar nada que el heurístico de píxeles no cubra ya mejor.

LIMITACIONES CONOCIDAS (documentadas a propósito, ver principio del
proyecto de documentar honestamente lo que no cubre este código):
- No detecta collages SIN margen visible entre sub-fotos (algunas
  plantillas no llevan borde, o usan un color de margen distinto del
  blanco/negro) -- esos casos se comportan como antes (foto normal,
  analizada igual).
- Sesgado a propósito hacia FALSOS NEGATIVOS antes que FALSOS POSITIVOS:
  los umbrales (franja fina, posición interior, color puro) buscan evitar
  marcar como collage una foto normal que tiene una franja ancha de color
  uniforme por motivos legítimos (cielo despejado, pared lisa, horizonte
  del mar) -- ese tipo de foto ya se trata aparte por otro mecanismo
  existente (`representative=False` en geolocation.py, ver
  `_MAX_NEIGHBOR_SPREAD_KM`). Perder una foto que SÍ era collage (se
  analiza igual, sin detectarla) es un coste menor que descartar sin
  motivo una foto normal que sí tenía señal real que aportar.
"""
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Imágenes más pequeñas que esto (en cualquier dimensión) no se analizan:
# ni son razonablemente un collage de verdad (demasiado pequeñas para
# distinguir sub-fotos), ni el resto de los umbrales de aquí abajo (que
# asumen fotos de tamaño normal) se comportan de forma sensata sobre algo
# tan pequeño como un icono o una miniatura corrupta.
_MIN_DIMENSION_PX = 20

# Franja candidata a "margen de collage": una fila/columna (o varias
# consecutivas) con muy poca variación de tono...
_UNIFORM_STD_THRESHOLD = 6.0
# ...Y además predominantemente blanca o negra (los dos colores de margen
# que usan, con diferencia, la mayoría de apps de collage habituales).
_WHITE_MEAN_THRESHOLD = 240.0
_BLACK_MEAN_THRESHOLD = 15.0

# Margen excluido en cada extremo (como fracción de alto/ancho): una franja
# uniforme PEGADA al borde exterior de la foto es letterboxing/padding
# normal (frecuente si Instagram rellena un formato distinto al original),
# no una línea de collage -- solo importan las franjas del INTERIOR.
_EDGE_MARGIN_FRACTION = 0.03

# Grosor máximo de una franja para seguir contando como "línea de collage"
# (como fracción de alto/ancho, respectivamente): los márgenes reales de
# apps de collage son de pocos píxeles. Una franja mucho más gruesa es más
# probable que sea una zona de color liso de la propia foto (pared, cielo)
# que un margen entre sub-fotos -- ver limitación documentada arriba sobre
# el sesgo deliberado hacia falsos negativos.
_MAX_BAND_THICKNESS_FRACTION = 0.05

# Grosor mínimo, en píxeles absolutos, para no reaccionar a una única fila
# que por casualidad (ruido de compresión JPEG) salga muy plana.
_MIN_BAND_THICKNESS_PX = 2


def detect_collage(image: Image.Image) -> bool:
    """True si `image` (ya decodificada, misma imagen que se pasaría a
    DINOv2/Moondream2) tiene toda la pinta de ser un collage de varias
    sub-fotos, por la presencia de una franja de margen blanca o negra en
    su interior (horizontal o vertical). Ver docstring de cabecera del
    módulo para el criterio completo y sus limitaciones conocidas.

    Nunca lanza: cualquier fallo inesperado (imagen en un modo de color
    exótico, error de conversión) se trata como "no se pudo determinar" y
    se devuelve False -- mismo criterio best-effort que el resto de este
    pipeline (ver analyze_image_content/estimate_location_from_image):
    esta es una optimización, no un requisito, así que un fallo aquí
    nunca debe impedir el análisis normal de la foto."""
    try:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    except Exception as exc:
        logger.warning("detect_collage: no se pudo convertir la imagen a escala de grises (%s): %s", type(exc).__name__, exc)
        return False

    height, width = gray.shape
    if height < _MIN_DIMENSION_PX or width < _MIN_DIMENSION_PX:
        return False

    return _has_uniform_interior_band(gray, orientation="horizontal") or _has_uniform_interior_band(
        gray, orientation="vertical"
    )


def _has_uniform_interior_band(gray: np.ndarray, orientation: str) -> bool:
    """orientation='horizontal': analiza FILAS (reduce a lo largo del
    ancho) -- detecta líneas horizontales, que dividirían la imagen en
    paneles apilados verticalmente (p. ej. un collage de 2 fotos, una
    encima de otra). orientation='vertical': analiza COLUMNAS (reduce a
    lo largo del alto) -- líneas verticales, paneles lado a lado. Mismo
    criterio en los dos casos, solo cambia el eje sobre el que se reduce."""
    if orientation == "horizontal":
        line_means = gray.mean(axis=1)  # una media por FILA
        line_stds = gray.std(axis=1)
    else:
        line_means = gray.mean(axis=0)  # una media por COLUMNA
        line_stds = gray.std(axis=0)

    total = line_means.shape[0]
    margin = max(1, int(total * _EDGE_MARGIN_FRACTION))
    max_thickness = max(_MIN_BAND_THICKNESS_PX, int(total * _MAX_BAND_THICKNESS_FRACTION))

    is_candidate = (line_stds < _UNIFORM_STD_THRESHOLD) & (
        (line_means > _WHITE_MEAN_THRESHOLD) | (line_means < _BLACK_MEAN_THRESHOLD)
    )
    # Solo interesan franjas del interior -- excluir el margen exterior en
    # los dos extremos (letterboxing/padding normal, ver docstring).
    is_candidate[:margin] = False
    is_candidate[total - margin:] = False

    # Agrupar líneas candidatas consecutivas en franjas, y comprobar que
    # cada franja tiene un grosor compatible con un margen de collage real
    # (ni una sola línea de ruido, ni una zona de color liso demasiado
    # ancha -- ver _MIN_BAND_THICKNESS_PX / _MAX_BAND_THICKNESS_FRACTION).
    band_start = None
    for i, candidate in enumerate(is_candidate):
        if candidate and band_start is None:
            band_start = i
        elif not candidate and band_start is not None:
            thickness = i - band_start
            if _MIN_BAND_THICKNESS_PX <= thickness <= max_thickness:
                return True
            band_start = None
    if band_start is not None:
        thickness = total - band_start
        if _MIN_BAND_THICKNESS_PX <= thickness <= max_thickness:
            return True

    return False
