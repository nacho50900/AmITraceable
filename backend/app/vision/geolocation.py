"""
Módulo 7 (nuevo): estima la provincia/ciudad más probable de una imagen
comparándola, vía embeddings DINOv2, contra un índice FAISS construido con
imágenes de calle geoetiquetadas de España (OSV-5M, ver scripts/).

Decisión de diseño (para la memoria): en vez de predecir coordenadas GPS
exactas (poco realista sin infraestructura tipo GeoSpy, ver benchmarks:
incluso el estado del arte solo acierta ~26-29% a nivel de calle/1km),
este módulo agrega el voto de los k vecinos más cercanos a nivel de
PROVINCIA, que es la granularidad que ya usa el resto del pipeline
(scoring/k_anonymity.py). Es una estimación probabilística con nivel de
confianza, no una respuesta exacta.

Requiere que ya exista el índice generado por scripts/build_faiss_index.py
(index.faiss + index_meta.csv). Si no existe, `estimate_location_from_image`
devuelve None en vez de fallar, para que el resto del análisis pueda seguir
funcionando sin este módulo (es opcional/best-effort, no crítico).

También marca como no representativa (campo `representative=False`, no
descarta) una estimación cuando la foto no tiene contenido suficientemente
distintivo como para geolocalizarla con algún sentido -- p.ej. una foto
donde solo se ve el mar y la espalda de alguien puede parecerse,
visualmente, a imágenes de referencia de medio litoral español a la vez.
Ver `_neighbor_spread_km` / `_MAX_NEIGHBOR_SPREAD_KM`: si los vecinos más
parecidos están repartidos por una zona demasiado amplia, ninguna
provincia "ganadora" sería significativa como CONCLUSIÓN de residencia
(ver `_infer_home_region` en report/generator.py, que filtra por este
campo) -- pero la estimación sigue siendo información real y se sigue
mostrando en el mapa con su confianza real, no se oculta.
"""
import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.models.schemas import InferredAttribute
from app.log.performance_log import PhotoAnalysisTiming, log_photo_analysis_run
from app.vision.scene_analysis import analyze_image_content

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "osv5m_spain"
_MODEL_NAME = "facebook/dinov2-small"

# Umbral de "foto no representativa" (ver `_neighbor_spread_km` y su uso en
# estimate_location_from_image): si los k vecinos más parecidos están
# repartidos, en MEDIA, a más de esto del centroide, la foto se descarta en
# vez de asignarle una provincia. Motivación: una foto genérica (solo mar,
# solo cielo, un primer plano sin fondo distintivo...) puede tener similitud
# visual alta con imágenes de referencia de MUCHOS sitios distintos de
# España a la vez -- no porque la foto sea "de" ninguno de ellos en
# particular, sino porque no tiene ningún rasgo que la ancle a un lugar
# concreto. En ese caso, aunque una provincia gane la votación por mayoría
# simple, esa "victoria" no significa nada: sería tan válida cualquier otra
# de las provincias representadas entre los vecinos. 300 km es un punto de
# partida razonable (aprox. el orden de magnitud de una comunidad autónoma
# grande), no una cifra derivada de ningún estudio -- ajustable si en la
# práctica resulta demasiado (o poco) permisivo.
_MAX_NEIGHBOR_SPREAD_KM = 300.0
_MIN_NEIGHBORS_WITH_COORDS_FOR_SPREAD_CHECK = 3

# Carga perezosa: el modelo/índice solo se cargan la primera vez que se usan,
# para no penalizar el arranque de la app cuando este módulo no se necesita.
_model = None
_processor = None
_index = None
_index_meta = None
# Dispositivo torch donde corre DINOv2 en ESTE proceso, decidido una sola
# vez (ver `_select_dinov2_device()`) -- normalmente "cuda" (comparte la
# GPU dedicada con Moondream2, comportamiento de siempre) o "cpu", pero
# puede ser un dispositivo DirectML (`torch_directml.device(N)`) si
# `settings.enable_igpu_offload=True` y la máquina tiene una segunda GPU
# distinta de la dedicada (ver docstring de `_select_dinov2_device`).
_device = None
# Si un forward pass en DirectML falla en tiempo de ejecución (operador no
# soportado, ver README de torch-directml -- "actively adding more
# operators", no todo está cubierto todavía), se recuerda aquí para no
# volver a intentarlo en cada foto: cae a CPU de forma permanente para el
# resto de la vida del proceso, sin reintentar contra la GPU dedicada
# (evitaría el problema de contención que se acaba de arreglar).
_dml_failed = False


@dataclass
class ImageLocationEstimate:
    province: str
    confidence: float  # proporción de los k vecinos que coinciden con `province`
    k_neighbors: int
    mean_similarity: float  # similitud coseno media de los vecinos considerados
    # Centroide (media) de las coordenadas de los vecinos que votaron por
    # `province`, usado solo para pintar un punto en el mapa -- NO es una
    # coordenada exacta de la foto, es una aproximación basada en dónde
    # están las imágenes de referencia más parecidas.
    lat: float | None = None
    lon: float | None = None
    # False cuando los k vecinos más parecidos están repartidos por una zona
    # demasiado amplia (ver _MAX_NEIGHBOR_SPREAD_KM) -- la foto NO es
    # representativa de un lugar concreto, así que esta estimación no debe
    # usarse para AFIRMAR una provincia/comunidad de residencia (ver
    # `_infer_home_region` en report/generator.py, que filtra por este
    # campo). Sigue siendo una estimación real con su confianza real, y
    # sigue mostrándose en el mapa -- lo único que no debe hacer es decidir
    # por sí sola (ni en grupo con otras igual de dispersas) la conclusión
    # de dónde vive la persona.
    representative: bool = True
    # Enlace a ESTA foto en concreto, distinto del permalink de la
    # publicación cuando la publicación tiene varias fotos (carrusel) --
    # ver `_photo_link()`. None si no se pudo determinar (no debería pasar
    # en el flujo real, pero los tests que construyen ImageLocationEstimate
    # directamente sin pasar este campo lo dejan en None; generator.py hace
    # fallback al permalink de la publicación en ese caso). Con una sola
    # foto en la publicación, coincide con el permalink normal.
    photo_link: str | None = None


def _photo_link(permalink: str, index: int, total: int) -> str:
    """Enlace a UNA foto concreta dentro de una publicación -- si la
    publicación tiene varias fotos (carrusel), el permalink normal de
    Instagram siempre apunta a la primera; para enlazar a una foto
    concreta, Instagram soporta el parámetro `?img_index=N` en la propia
    URL del post (1-indexado, igual que la numeración que ve el usuario
    dentro de la propia app).

    NOTA: no se ha podido verificar de forma 100% fiable que Instagram
    siga soportando este parámetro tal cual en la web actual -- si algún
    enlace generado no salta a la foto correcta, confirmarlo abriendo una
    publicación de carrusel real y comprobando si la URL cambia al pasar
    de una foto a otra.

    Con una sola foto en la "publicación" (total <= 1) se devuelve el
    permalink tal cual, sin tocarlo -- cubre tanto una publicación normal
    de una sola foto como la foto de perfil (ver `estimate_locations_for_posts`,
    parámetro `avatar_url`), que siempre se trata como total=1.

    Arregla dos problemas reales a la vez (ver `_collect_photo_result`):
    (1) antes, todas las fotos de un mismo carrusel compartían el MISMO
    permalink como clave en `visual_descriptions`/`general_descriptions`
    -- con varias fotos analizadas del mismo carrusel, cada nueva
    descripción SOBREESCRIBÍA a la anterior, así que solo sobrevivía la
    última en procesarse (no determinista, por la concurrencia) y esa
    MISMA descripción se aplicaba a TODAS las fotos de ese carrusel.
    (2) el frontend usa `point.permalink` como `key` de React (ver
    LocationMap.tsx) -- con permalinks repetidos entre fotos del mismo
    carrusel, React tenía colisiones de key silenciosas."""
    if total <= 1:
        return permalink
    separator = "&" if "?" in permalink else "?"
    return f"{permalink}{separator}img_index={index}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import atan2, cos, radians, sin, sqrt

    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def _neighbor_spread_km(neighbor_rows) -> float | None:
    """Distancia media de los vecinos (con coordenadas) a su centroide.
    None si hay demasiado pocos vecinos con coordenadas para que la medida
    signifique algo (en vez de arriesgarse a descartar una foto por falta
    de datos, no por falta de rasgos distintivos)."""
    valid = neighbor_rows.dropna(subset=["lat", "lon"])
    if len(valid) < _MIN_NEIGHBORS_WITH_COORDS_FOR_SPREAD_CHECK:
        return None

    lats, lons = valid["lat"].tolist(), valid["lon"].tolist()
    centroid_lat, centroid_lon = sum(lats) / len(lats), sum(lons) / len(lons)
    distances = [_haversine_km(centroid_lat, centroid_lon, lat, lon) for lat, lon in zip(lats, lons)]
    return sum(distances) / len(distances)


def _extract_exif_gps(image) -> tuple[float, float] | None:
    """Si la imagen tiene coordenadas GPS en su EXIF, las devuelve como
    (lat, lon) en grados decimales. None si no hay EXIF de GPS o no se
    puede leer.

    AVISO IMPORTANTE (para la memoria): en la práctica esto casi nunca
    encontrará nada en fotos descargadas de Instagram, porque Instagram --
    como la inmensa mayoría de redes sociales -- elimina los metadatos EXIF
    (incluido el GPS) de las imágenes al subirlas, por privacidad y para
    reducir peso del fichero. Se implementa de todos modos porque: (a) es
    gratis comprobarlo (una foto sin este EXIF simplemente sigue el camino
    normal de geolocalización por contenido visual), (b) si por algún
    motivo SÍ está presente, es la fuente más fiable posible -- GPS real
    del dispositivo en el momento de la foto, no una estimación por
    parecido visual --, y (c) deja de ser un caso muerto si en el futuro se
    analiza otra plataforma que no limpie EXIF, o el usuario sube el
    fichero original directamente en vez de la versión servida por Instagram.
    """
    try:
        exif = image.getexif()
        gps_ifd = exif.get_ifd(0x8825)  # GPSInfo
        if not gps_ifd:
            return None

        lat_dms, lat_ref = gps_ifd.get(2), gps_ifd.get(1)
        lon_dms, lon_ref = gps_ifd.get(4), gps_ifd.get(3)
        if not lat_dms or not lon_dms or not lat_ref or not lon_ref:
            return None

        def _dms_to_decimal(dms, ref) -> float:
            degrees, minutes, seconds = (float(v) for v in dms)
            decimal = degrees + minutes / 60 + seconds / 3600
            return -decimal if ref in ("S", "W") else decimal

        return _dms_to_decimal(lat_dms, lat_ref), _dms_to_decimal(lon_dms, lon_ref)
    except Exception:
        # EXIF corrupto/con forma inesperada: se trata igual que "no hay
        # GPS", nunca se rompe el análisis por esto.
        return None


def _estimate_from_exact_coordinates(lat: float, lon: float) -> ImageLocationEstimate:
    """La foto trae coordenadas GPS reales (ver _extract_exif_gps): en vez
    de gastar cómputo en el modelo DINOv2 para ADIVINAR la ubicación por
    parecido visual, se busca directamente la región conocida más cercana a
    esas coordenadas exactas en los metadatos del índice ya cargado. La
    confianza se marca al máximo porque no es una estimación -- es la
    ubicación real que la propia foto llevaba."""
    import pandas as pd

    distances = _index_meta.apply(
        lambda row: _haversine_km(lat, lon, row["lat"], row["lon"])
        if pd.notna(row["lat"]) and pd.notna(row["lon"])
        else float("inf"),
        axis=1,
    )
    nearest = _index_meta.iloc[distances.idxmin()]

    return ImageLocationEstimate(
        province=nearest["region"],
        confidence=1.0,
        k_neighbors=0,  # no aplica: no hubo votación, es la coordenada real
        mean_similarity=1.0,
        lat=round(lat, 4),
        lon=round(lon, 4),
    )


def _geolocation_available() -> bool:
    """Comprobación barata (sin cargar el modelo ni el índice) de si el
    módulo de geolocalización está realmente operativo: ficheros del índice
    presentes Y dependencias opcionales (torch/faiss/transformers)
    instaladas. Se usa para poder distinguir en el informe "el índice no
    está construido en este servidor" de "se analizaron tus fotos pero
    ninguna tuvo suficiente confianza" -- son situaciones distintas y no
    deben mostrarse con el mismo mensaje (ver report/generator.py)."""
    index_path = _INDEX_DIR / "index.faiss"
    meta_path = _INDEX_DIR / "index_meta.csv"
    if not index_path.exists() or not meta_path.exists():
        return False
    try:
        import faiss  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModel  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class GeolocationOutcome:
    # Ver _geolocation_available(). Si es False, `results` está vacío
    # siempre (no se llegó ni a intentar procesar imágenes).
    index_available: bool
    # UNA entrada por cada foto que se pudo procesar (embedding + consulta
    # al índice), con su confianza REAL, sin filtrar por ningún umbral --
    # a diferencia de versiones anteriores de este módulo, que descartaban
    # en silencio las de baja confianza. El umbral de fiabilidad (para
    # decidir si una estimación es lo bastante buena como para alimentar
    # el cálculo de k-anonimato) es responsabilidad del llamador, no de
    # este módulo -- ver MIN_CONFIDENCE_FOR_POPULATION_NARROWING en
    # report/generator.py.
    results: list[tuple[str, ImageLocationEstimate]]
    # Inferencias de CONTENIDO visual (aficiones, actividades -- ver
    # app/vision/scene_analysis.py), una entrada por foto y hallazgo, con
    # el permalink de la publicación que lo generó. Se calcula sobre la
    # MISMA imagen ya descargada para geolocalización, sin descarga aparte.
    visual_inferences: list[tuple[str, InferredAttribute]] = field(default_factory=list)
    # Permalinks de fotos donde scene_analysis.py detectó a la persona en
    # actitud romántica con otra persona (sin identificar a esa otra
    # persona en ningún sentido -- ver docstring de scene_analysis.py).
    # report/generator.py usa esto como señal de "estado_civil" con
    # source="imagen" cuando el texto no dio ninguna señal por sí solo.
    partner_signal_permalinks: set[str] = field(default_factory=set)
    # Descripción CRUDA de Moondream2 (ver scene_analysis.py) para cada foto
    # donde se pudo generar -- una por permalink, a diferencia de
    # `visual_inferences` (que puede tener varias entradas, o ninguna, por
    # foto). Se guarda tal cual porque: (a) se muestra en el frontend al
    # desplegar cada foto ("qué vio la IA"), y (b) al formar parte de
    # ImageLocationPoint, se incluye automáticamente en el JSON del informe
    # que se le manda a ai_analysis.py para las conclusiones finales -- sin
    # necesitar ningún cambio ahí, ya manda el informe completo tal cual.
    visual_descriptions: dict[str, str] = field(default_factory=dict)
    # Descripción GENERAL de la escena (campo DESCRIPCION del prompt de
    # scene_analysis.py, ya parseado -- p. ej. "4 personas comiendo pizza
    # alegremente en una terraza"), una por permalink. A diferencia de
    # `visual_descriptions` (las cuatro líneas crudas, pensadas para la
    # vista de detalle "qué vio la IA"), esta es una frase legible pensada
    # para mostrarse de forma prominente como pie de foto -- mismo límite
    # ético/legal del resto del módulo (nunca menciona raza/etnia/aspecto
    # físico, ver prompt en scene_analysis.py).
    general_descriptions: dict[str, str] = field(default_factory=dict)


def _select_dinov2_device():
    """Decide en qué dispositivo torch corre DINOv2 -- llamado UNA VEZ
    desde `_lazy_load()` y cacheado en `_device` (module-level).

    Comportamiento de siempre (sin cambios si `enable_igpu_offload` está
    desactivado, que es el valor por defecto -- ver más abajo por qué):
    "cuda" si `torch.cuda.is_available()`, si no "cpu". DINOv2 y
    Moondream2 comparten la misma GPU dedicada.

    Offload a iGPU (opt-in, `ENABLE_IGPU_OFFLOAD=true`): idea del propio
    usuario del proyecto -- Moondream2 (scene_analysis.py) satura la GPU
    dedicada durante el análisis de contenido; DINOv2 es una carga mucho
    más ligera (un embedding, no generación autorregresiva de texto) y en
    máquinas con una GPU dedicada + otra "compartida" (integrada en el
    procesador -- Intel/AMD), esa segunda GPU normalmente está ociosa
    durante el análisis. Moverlo ahí libera la dedicada por completo para
    Moondream2, en vez de competir por ella.

    OJO -- limitación importante, y por qué esto es opt-in en vez de
    intentarlo siempre por defecto: `torch.cuda` SOLO ve tarjetas NVIDIA.
    Una gráfica integrada Intel/AMD no aparece ahí en absoluto, así que no
    hay forma de usarla vía CUDA. La única vía es un backend distinto,
    DirectML (paquete opcional `torch-directml`, Windows/WSL2) -- pero
    tiene una restricción de versión de `torch` bastante estricta (fija
    una versión exacta como dependencia, incompatible con el rango
    `torch>=2.2,<3.0` + build CUDA cu121 que ya usa este proyecto para
    Moondream2, ver comentario en requirements-igpu.txt). Por eso NO se
    instala como dependencia por defecto (ver ese fichero) -- instalarlo
    sin comprobar antes en la máquina real podría reinstalar `torch` con
    otra versión/build y romper el CUDA de Moondream2, justo el problema
    de contención que se acaba de arreglar. Verificar con
    `pip install torch-directml` en un entorno de prueba ANTES de activar
    esto en un despliegue real.

    Solo se activa si TODAS estas condiciones se cumplen (todas
    comprobadas aquí, no hace falta configurar nada más allá del
    interruptor):
    - `settings.enable_igpu_offload` es True (opt-in explícito).
    - Hay una GPU NVIDIA dedicada disponible (`torch.cuda.is_available()`)
      -- si no la hay, no tiene sentido "liberarla": Moondream2 ya
      correría en CPU o no correría en absoluto, y mover DINOv2 a una
      iGPU en ese caso no libera nada.
    - El paquete `torch_directml` está instalado e importable.
    - Hay al menos un dispositivo DirectML cuyo nombre NO coincide con el
      de la GPU CUDA ya detectada -- si todos los dispositivos DirectML
      visibles son la MISMA tarjeta dedicada (algunos drivers exponen la
      GPU NVIDIA también vía DirectML), usarla ahí no libera nada, sería
      contención con otro nombre. En máquinas con una sola GPU (sin
      "más de una GPU", como pide el usuario del proyecto) esta condición
      nunca se cumple y se sigue el comportamiento de siempre.

    Cualquier fallo en esta detección (import, driver, lo que sea) se
    trata como "no hay iGPU utilizable": vuelve silenciosamente al
    comportamiento de siempre, nunca revienta el arranque.
    """
    import torch

    cuda_available = torch.cuda.is_available()

    if settings.enable_igpu_offload and cuda_available:
        try:
            import torch_directml

            cuda_name = torch.cuda.get_device_name(0)
            device_count = torch_directml.device_count()
            for i in range(device_count):
                dml_name = torch_directml.device_name(i)
                if dml_name and dml_name != cuda_name:
                    logger.info(
                        "ENABLE_IGPU_OFFLOAD=true: DINOv2 se ejecutara en la GPU "
                        "compartida '%s' (DirectML, dispositivo %d) -- la GPU "
                        "dedicada '%s' queda libre para Moondream2.",
                        dml_name,
                        i,
                        cuda_name,
                    )
                    return torch_directml.device(i)
            logger.info(
                "ENABLE_IGPU_OFFLOAD=true pero no se encontro ninguna GPU DirectML "
                "distinta de la dedicada ('%s') -- probablemente solo hay una GPU "
                "en esta maquina. DINOv2 seguira compartiendo la GPU dedicada con "
                "Moondream2, comportamiento de siempre.",
                cuda_name,
            )
        except ImportError:
            logger.info(
                "ENABLE_IGPU_OFFLOAD=true pero el paquete torch_directml no esta "
                "instalado (ver requirements-igpu.txt) -- DINOv2 seguira "
                "compartiendo la GPU dedicada con Moondream2, comportamiento de "
                "siempre."
            )
        except Exception:
            logger.exception(
                "ENABLE_IGPU_OFFLOAD=true pero fallo la deteccion de GPU DirectML "
                "-- DINOv2 seguira compartiendo la GPU dedicada con Moondream2, "
                "comportamiento de siempre. Traceback arriba para depurar."
            )

    return "cuda" if cuda_available else "cpu"


def _lazy_load():
    global _model, _processor, _index, _index_meta, _device
    if _model is not None:
        return

    import faiss
    import pandas as pd
    import torch
    from transformers import AutoImageProcessor, AutoModel

    index_path = _INDEX_DIR / "index.faiss"
    meta_path = _INDEX_DIR / "index_meta.csv"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No se encontró el índice en {_INDEX_DIR}. Ejecuta antes "
            "scripts/download_osv5m_spain.py y scripts/build_faiss_index.py."
        )

    _device = _select_dinov2_device()
    _processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
    _model = AutoModel.from_pretrained(_MODEL_NAME).to(_device).eval()
    _index = faiss.read_index(str(index_path))
    _index_meta = pd.read_csv(meta_path, dtype={"id": str})


def estimate_location_from_image(image, k: int = 15) -> ImageLocationEstimate | None:
    """
    image: objeto PIL.Image ya cargado (no una ruta ni una URL -- el
    llamador es responsable de descargar/abrir la imagen del post).
    k: número de vecinos más cercanos a considerar para la votación.

    Devuelve None si el índice no está construido (módulo opcional) o si
    la imagen no se puede procesar.
    """
    global _dml_failed, _device

    try:
        _lazy_load()
    except (FileNotFoundError, ImportError):
        # FileNotFoundError: índice no construido todavía (ver scripts/).
        # ImportError/ModuleNotFoundError: torch/faiss/transformers no
        # instalados en este entorno -- este módulo es opcional/best-effort,
        # así que se degrada devolviendo None en vez de tumbar el análisis.
        return None

    # Si la foto ya trae su ubicación real en el EXIF, se usa directamente
    # y NO se analiza con el modelo (ver _extract_exif_gps: en la práctica
    # esto rara vez ocurre con fotos de Instagram, pero cuando ocurre es
    # más fiable que cualquier estimación visual, y más barato).
    gps = _extract_exif_gps(image)
    if gps is not None:
        return _estimate_from_exact_coordinates(*gps)

    import torch

    try:
        with torch.no_grad():
            inputs = _processor(images=image.convert("RGB"), return_tensors="pt").to(_device)
            outputs = _model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    except Exception:
        # Si el dispositivo es DirectML (offload a iGPU, ver
        # _select_dinov2_device) y esto es la PRIMERA foto que falla,
        # asume que es un operador de DINOv2 sin soporte en DirectML (ver
        # su docstring: "actively adding more operators", no todo está
        # cubierto) en vez de un problema de esta imagen en concreto --
        # cae a CPU de forma PERMANENTE para el resto del proceso (nunca
        # a la GPU dedicada: sería reintroducir la contención con
        # Moondream2 que este offload existe para evitar) y reintenta
        # esta misma foto una vez en CPU antes de rendirse.
        is_dml_device = bool(_device) and not isinstance(_device, str)
        if is_dml_device and not _dml_failed:
            logger.exception(
                "Fallo al ejecutar DINOv2 en la GPU compartida (DirectML) -- "
                "probablemente un operador sin soporte. A partir de ahora DINOv2 "
                "correra en CPU para el resto del proceso (nunca en la GPU "
                "dedicada, para no competir con Moondream2). Traceback arriba "
                "para depurar cual operador ha fallado."
            )
            _dml_failed = True
            _device = "cpu"
            try:
                with torch.no_grad():
                    inputs = _processor(images=image.convert("RGB"), return_tensors="pt").to(_device)
                    outputs = _model.to(_device)(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            except Exception:
                return None
        else:
            return None

    similarities, indices = _index.search(embedding.reshape(1, -1).astype("float32"), k)
    similarities, indices = similarities[0], indices[0]

    neighbor_rows = _index_meta.iloc[indices]

    # "region" en OSV-5M es lo más parecido a provincia/comunidad autónoma
    # dentro de sus metadatos; ajusta esta columna si tu metadata.csv usa
    # otro nombre tras inspeccionar la fila de ejemplo del script de descarga.
    provinces = neighbor_rows["region"].fillna("desconocido").tolist()

    vote_counts = Counter(provinces)
    top_province, votes = vote_counts.most_common(1)[0]

    # Centroide de los vecinos que coincidieron con la provincia ganadora
    # (no de todos los k, para que el punto no se desplace hacia vecinos de
    # otras provincias que quedaron en minoría).
    matching = neighbor_rows[neighbor_rows["region"].fillna("desconocido") == top_province]
    lat = float(matching["lat"].mean()) if "lat" in matching and not matching["lat"].isna().all() else None
    lon = float(matching["lon"].mean()) if "lon" in matching and not matching["lon"].isna().all() else None

    # Foto no representativa (ver _MAX_NEIGHBOR_SPREAD_KM más arriba): sus
    # vecinos más parecidos están repartidos por medio país, así que
    # cualquier provincia "ganadora" sería arbitraria. NO se descarta la
    # estimación entera (eso la haría desaparecer también del mapa, donde
    # es información legítima: "esto es lo más parecido que encontramos,
    # aunque poco fiable") -- se marca `representative=False` para que
    # `_infer_home_region` la excluya de la conclusión de residencia, sin
    # dejar de mostrarla con su confianza real en `image_location_points`.
    spread = _neighbor_spread_km(neighbor_rows)
    representative = spread is None or spread <= _MAX_NEIGHBOR_SPREAD_KM

    return ImageLocationEstimate(
        province=top_province,
        confidence=round(votes / k, 2),
        k_neighbors=k,
        mean_similarity=round(float(np.mean(similarities)), 3),
        lat=round(lat, 4) if lat is not None else None,
        lon=round(lon, 4) if lon is not None else None,
        representative=representative,
    )


async def _download_image(client, semaphore, media_url):
    """Descarga UNA foto (I/O de red). La descarga es un recurso totalmente
    distinto de la CPU que usan los dos modelos de visión -- solaparla con
    el análisis de otras fotos reduce el tiempo total, sin competir por los
    mismos núcleos que usa torch.

    BUG REAL corregido aquí (confirmado en producción, no una precaución
    teórica): `Image.open()` es PEREZOSO -- no decodifica los píxeles al
    momento, se limita a guardar una referencia al `BytesIO` y decodifica
    bajo demanda la PRIMERA vez que algo accede de verdad a la imagen
    (`.convert()`, `.copy()`, iterar píxeles...). `_process_photo` pasa
    esta MISMA imagen a `estimate_location_from_image` y
    `analyze_image_content` EN PARALELO (`asyncio.gather`, cada una en su
    propio hilo vía `asyncio.to_thread`) -- y esa decodificación perezosa
    de PIL NO es segura frente a acceso concurrente desde dos hilos
    distintos. En producción esto se manifestaba como una exclusión mutua
    perfecta: SIEMPRE que DINOv2 conseguía una estimación, Moondream2
    fallaba con `OSError: image file is truncated`, y viceversa -- nunca
    los dos a la vez, porque el hilo que llegaba primero completaba la
    decodificación, y el segundo se encontraba el stream ya parcialmente
    consumido.

    La solución: forzar la decodificación completa AQUÍ, con `.load()`,
    mientras todavía estamos en un único hilo (el de la propia descarga) --
    así, cuando `_process_photo` reparte la imagen entre los dos modelos,
    ya no queda ninguna lectura perezosa pendiente sobre la que competir;
    ambos hilos solo LEEN píxeles ya decodificados en memoria, lo cual sí
    es seguro."""
    from PIL import Image
    import io

    async with semaphore:
        try:
            resp = await client.get(media_url)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
            image.load()  # fuerza la decodificación completa AQUÍ, en un único hilo -- ver docstring
            return image
        except Exception:
            return None  # imagen no descargable/decodificable: se omite, no se aborta el análisis


async def _maybe_analyze_content(image):
    """Envoltorio sobre `analyze_image_content` que respeta
    `settings.enable_scene_analysis` (ver docstring en config.py --
    desactivado por defecto, Moondream2 en CPU no es fiable actualmente).
    Con el interruptor desactivado, ni siquiera se intenta cargar el
    modelo -- se devuelve el mismo resultado "sin nada que aportar" que ya
    usa `analyze_image_content` cuando el módulo no está disponible en
    absoluto, así que el resto del pipeline no necesita distinguir entre
    "desactivado a propósito" y "no instalado".

    DOS bugs reales corregidos aquí (no precauciones teóricas):
    1. `analyze_image_content` es una función SÍNCRONA y puede bloquear
       mucho tiempo (carga del modelo, generación de texto, o -- visto en
       producción -- reintentos de red de hasta ~10s cada uno si
       `huggingface_hub` necesita comprobar la caché). Llamarla
       directamente aquí (sin `asyncio.to_thread`) bloquea el propio event
       loop del proceso durante ese rato, no solo esta tarea.
    2. Como esta llamada corre en `asyncio.gather` junto a la de DINOv2
       (ver `_process_photo` más abajo), `asyncio.gather` espera a que
       TERMINEN LAS DOS antes de devolver nada -- así que si Moondream2 se
       queda colgado, el resultado de DINOv2 para esa MISMA foto (que
       puede llevar rato ya calculado) se queda esperando sin necesidad.
       `asyncio.wait_for` con un tope pone coto a esto: si Moondream2 no
       termina a tiempo, esta foto se degrada a "sin descripción" (mismo
       resultado que si el módulo no estuviera disponible) y el resto del
       pipeline -- incluida la geolocalización de esa misma foto -- sigue
       sin más demora."""
    if not settings.enable_scene_analysis:
        return [], False, None, None
    timeout_seconds = settings.scene_analysis_timeout_seconds
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(analyze_image_content, image),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "Análisis de contenido visual descartado para una foto: "
            "superó los %ss (ver Settings.scene_analysis_timeout_seconds, "
            "configurable con la variable de entorno "
            "SCENE_ANALYSIS_TIMEOUT_SECONDS).",
            timeout_seconds,
        )
        return [], False, None, None


async def _process_photo(client, download_semaphore, analysis_semaphore, media_url, timing):
    """Descarga UNA foto y, si se pudo, la analiza con los dos modelos.
    Devuelve (image_or_None, estimate_or_None, scene_inferences,
    indicio_pareja, description_or_None, description_general_or_None).
    Pensada para lanzarse como tarea independiente por foto (ver
    `estimate_locations_for_posts`): así varias fotos pueden estar en
    distintas etapas (descargando / en cola para analizar / analizando) a
    la vez.

    `timing` (PhotoAnalysisTiming, ver app/log/performance_log.py) recibe UNA
    medición por foto -- solo el tramo de análisis con los modelos de
    visión (no la descarga, que es I/O de red y depende de factores ajenos
    a la CPU/concurrencia que queremos medir aquí)."""
    image = await _download_image(client, download_semaphore, media_url)
    if image is None:
        return None, None, [], False, None, None

    # `analysis_semaphore` (ver Settings.photo_analysis_concurrency) acota
    # cuántas fotos ocupan a la vez los modelos de visión -- DINOv2 es
    # rápido pero Moondream2 es lento (generación autoregresiva), así que
    # sin este límite de concurrencia el núcleo que usaría DINOv2 para la
    # siguiente foto se quedaría ocioso mientras Moondream2 termina la
    # actual. Se deja como configurable porque el punto óptimo depende de
    # cuántos núcleos tenga la máquina que lo ejecute.
    async with analysis_semaphore:
        start = time.monotonic()
        estimate, (scene_inferences, indicio_pareja, description, description_general) = await asyncio.gather(
            asyncio.to_thread(estimate_location_from_image, image),
            _maybe_analyze_content(image),
        )
        timing.record(time.monotonic() - start)
    # `image` sale de scope tras este bloque y se descarta (nunca se escribe a disco)
    return image, estimate, scene_inferences, indicio_pareja, description, description_general


def _collect_photo_result(
    permalink: str,
    photo_link: str,
    photo_result: tuple,
    results: list,
    visual_inferences: list,
    partner_signal_permalinks: set,
    visual_descriptions: dict,
    general_descriptions: dict,
) -> None:
    """Vuelca el resultado de UNA foto (ya resuelto, ver `_process_photo`)
    en las listas/diccionarios compartidos de `estimate_locations_for_posts`.
    Extraído a su propia función para que el bucle principal se limite a
    orquestar -- consumir la tarea, delegar el volcado, emitir progreso --
    en vez de acumular aquí varias ramas `if` seguidas.

    Dos claves distintas a propósito, cada una a su nivel: `permalink` es
    el de la PUBLICACIÓN (varias fotos de un carrusel comparten el mismo)
    -- se usa para `partner_signal_permalinks`, señal a nivel de
    publicación, no de foto concreta. `photo_link` (ver `_photo_link`) es
    ÚNICO por foto -- se usa como clave de `visual_descriptions`/
    `general_descriptions` (arregla el bug real de que, antes, varias
    fotos del mismo carrusel se sobrescribían entre sí al compartir la
    misma clave) y se guarda en `estimate.photo_link` para que
    `report/generator.py` pueda enlazar a la foto exacta."""
    _image, estimate, scene_inferences, indicio_pareja, description, description_general = photo_result

    if estimate is not None:
        estimate.photo_link = photo_link
        results.append((permalink, estimate))
    for inferred in scene_inferences:
        inferred.evidence.append(permalink)
        visual_inferences.append((permalink, inferred))
    if indicio_pareja:
        partner_signal_permalinks.add(permalink)
    if description:
        visual_descriptions[photo_link] = description
    if description_general:
        general_descriptions[photo_link] = description_general


async def estimate_locations_for_posts(
    posts: list, avatar_url: str | None = None, progress_callback=None
) -> GeolocationOutcome:
    """
    Orquestación de alto nivel: para CADA foto de CADA SocialPost de tipo
    imagen (todas las de un carrusel, no solo la primera -- ver
    `media_urls` en app/models/schemas.py e InstagramClient), descarga la
    imagen EN MEMORIA (nunca a disco), y sobre ESA MISMA imagen ejecuta dos
    análisis independientes en paralelo: geolocalización por similitud
    (embedding DINOv2 + consulta al índice) y análisis de CONTENIDO visual
    (aficiones/actividades/señal de pareja, ver app/vision/scene_analysis.py),
    antes de descartar la imagen. Devuelve TODAS las estimaciones de
    ubicación que se pudieron calcular (con su confianza real, sin
    filtrar), junto al permalink de la publicación que la generó -- una
    publicación con varias fotos puede aportar varias estimaciones con el
    MISMO permalink; el filtrado por umbral de fiabilidad, si hace falta,
    lo hace el llamador (ver `report/generator.py`). También devuelve las
    inferencias de contenido visual y qué fotos dieron señal de pareja.

    `avatar_url`, si se da, se analiza como UNA foto más de este mismo
    pipeline (mismo tratamiento que cualquier otra: geolocalización +
    análisis de contenido), usando la propia URL de la foto de perfil como
    "permalink" -- no hay una página de publicación a la que enlazar, así
    que el llamador (`report/generator.py`) usa esta igualdad
    (`permalink == avatar_url`) para reconocer esta entrada en concreto y
    tratarla de forma especial: se muestra en el listado del frontend como
    "Foto de perfil" en vez de "Ver publicación", y se EXCLUYE del cálculo
    de consenso de residencia (`_infer_home_region`) -- un primer plano o
    selfie de perfil no es necesariamente representativo de dónde vive la
    persona, mismo criterio ya aplicado a las fotos de viaje (ver ADR-16).

    Se ejecuta en segundo plano de forma best-effort: si el índice no
    existe (no se ha corrido scripts/build_faiss_index.py) o falla la
    descarga de una imagen concreta, simplemente se omite esa imagen sin
    interrumpir el resto del análisis. `GeolocationOutcome.index_available`
    permite al llamador distinguir "el índice no está construido" de "se
    analizaron fotos pero ninguna dio una estimación" -- son mensajes
    distintos de cara al usuario.

    `progress_callback`, si se da, se llama tras CADA foto procesada
    (llegue o no a producir una estimación válida), con el nº de fotos
    procesadas hasta ahora y el total a procesar -- para que el endpoint
    de streaming pueda mostrar "analizando foto X de Y" en tiempo real.
    """
    import httpx

    from app.progress import emit_progress

    index_available = _geolocation_available()

    results: list[tuple[str, ImageLocationEstimate]] = []
    # Una entrada por FOTO, no por publicación: un carrusel con 5 fotos
    # aporta 5 entradas aquí, todas con el mismo permalink DE PUBLICACIÓN
    # (el índice/total dentro del carrusel se llevan aparte, para poder
    # construir un enlace específico a cada foto -- ver `_photo_link`) --
    # así se analizan TODAS las fotos, no solo la primera. Ver
    # InstagramClient._extract_media_urls().
    photo_units: list[tuple[str, str, int, int]] = []  # (permalink, media_url, index, total)
    for post in posts:
        if post.type not in ("image", "carousel_album"):
            continue
        media_urls = getattr(post, "media_urls", None) or []
        post_total = len(media_urls)
        for photo_index, url in enumerate(media_urls, start=1):
            photo_units.append((post.permalink, url, photo_index, post_total))
    if avatar_url:
        # La foto de perfil siempre se trata como total=1 (nunca lleva
        # ?img_index=, ver _photo_link) -- no es parte de ningún carrusel.
        photo_units.append((avatar_url, avatar_url, 1, 1))
    total = len(photo_units)

    if total == 0 or not index_available:
        return GeolocationOutcome(index_available=index_available, results=results)

    try:
        from PIL import Image
        import io
    except ImportError:
        # Pillow no es una dependencia obligatoria del núcleo de la app
        # (requirements.txt), solo se necesita para este módulo opcional de
        # geolocalización. Si no está instalada, se degrada devolviendo
        # lista vacía en vez de romper el análisis completo -- mismo
        # principio que ya se aplica a torch/faiss/transformers en
        # _lazy_load() más arriba. Se marca index_available=False para que
        # el frontend lo trate igual que "módulo no operativo en este
        # servidor", que es justo lo que es.
        return GeolocationOutcome(index_available=False, results=results)

    visual_inferences: list[tuple[str, InferredAttribute]] = []
    partner_signal_permalinks: set[str] = set()
    visual_descriptions: dict[str, str] = {}
    general_descriptions: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Se lanza YA el pipeline completo (descarga + análisis) de TODAS
        # las fotos como tareas independientes, acotadas por dos límites
        # de concurrencia distintos: uno para las descargas (por no
        # golpear de golpe el CDN de Instagram) y otro para cuántas fotos
        # ocupan a la vez los modelos de visión (ver
        # Settings.photo_analysis_concurrency). Se sigue consumiendo el
        # resultado de cada tarea EN EL ORDEN ORIGINAL de las fotos (no en
        # el orden en que terminan) para que `results` y los eventos de
        # progreso salgan deterministas -- pero como las tareas ya están
        # todas en marcha de fondo, para cuando toca esperar a la foto i,
        # normalmente ya lleva un rato avanzando (o incluso ha terminado),
        # en vez de empezar recién en ese momento.
        _MAX_CONCURRENT_DOWNLOADS = 4
        download_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)
        # Capado a `total`: si este análisis solo tiene, p. ej., 2 fotos,
        # no tiene sentido abrir más de 2 "huecos" de concurrencia aunque
        # la máquina tenga músculo para más -- reduce el nº de huecos
        # abiertos, nunca lo aumenta, así que no hay riesgo de
        # sobre-suscripción de CPU: `torch.set_num_threads()` ya se fijó en
        # el arranque asumiendo como máximo `settings.photo_analysis_concurrency`
        # fotos a la vez (ver app/main.py), y aquí nunca se supera ese tope.
        actual_concurrency = max(1, min(settings.photo_analysis_concurrency, total))
        analysis_semaphore = asyncio.Semaphore(actual_concurrency)
        timing = PhotoAnalysisTiming()
        run_start = time.monotonic()
        photo_tasks = [
            asyncio.create_task(
                _process_photo(client, download_semaphore, analysis_semaphore, media_url, timing)
            )
            for _permalink, media_url, _index, _post_total in photo_units
        ]

        for i, (permalink, _media_url, photo_index, post_total) in enumerate(photo_units, start=1):
            photo_result = await photo_tasks[i - 1]
            photo_link = _photo_link(permalink, photo_index, post_total)
            _collect_photo_result(
                permalink,
                photo_link,
                photo_result,
                results,
                visual_inferences,
                partner_signal_permalinks,
                visual_descriptions,
                general_descriptions,
            )

            # Dos líneas de progreso independientes para dos análisis
            # distintos sobre la misma foto (ver el asyncio.gather dentro
            # de `_process_photo`): geolocalización por similitud visual
            # (DINOv2) y análisis de contenido -- aficiones, pareja --
            # (Moondream2, ver scene_analysis.py). Avanzan siempre a la vez
            # en el backend (se esperan juntas con gather), pero se
            # muestran como dos líneas separadas en el frontend porque son
            # dos modelos y dos propósitos distintos -- mezclarlas en una
            # sola línea ("Analizando fotos...") no dejaba claro que se
            # estaban haciendo dos cosas diferentes sobre cada foto.
            await emit_progress(
                progress_callback,
                "Geolocalizando fotos...",
                photos_analyzed=i,
                total_photos=total,
                track="geolocalizacion",
            )
            await emit_progress(
                progress_callback,
                "Analizando fotos...",
                photos_analyzed=i,
                total_photos=total,
                # Desde que este análisis corre en PARALELO con el resto
                # del pipeline (ver analysis_router._build_report), sus
                # eventos pueden intercalarse con los de fingerprint/
                # atributos/IA en el mismo stream SSE. `track` le permite
                # al frontend mostrarlo como su propia línea independiente
                # en vez de mezclarlo con la fase "actual" del resto del
                # análisis.
                track="fotos",
            )

    import os

    log_photo_analysis_run(
        total_photos=total,
        cpu_count=os.cpu_count() or 4,
        configured_concurrency=settings.photo_analysis_concurrency,
        actual_concurrency=actual_concurrency,
        # Igual que se calcula en app/main.py -- se repite aquí (en vez de
        # leerlo de algún sitio compartido) porque es una cuenta trivial y
        # así este módulo no depende de detalles de arranque de main.py.
        threads_per_inference=max(1, (os.cpu_count() or 4) // max(1, settings.photo_analysis_concurrency)),
        enable_scene_analysis=settings.enable_scene_analysis,
        total_wall_seconds=time.monotonic() - run_start,
        per_photo_seconds=timing.per_photo_seconds,
    )

    return GeolocationOutcome(
        index_available=index_available,
        results=results,
        visual_inferences=visual_inferences,
        partner_signal_permalinks=partner_signal_permalinks,
        visual_descriptions=visual_descriptions,
        general_descriptions=general_descriptions,
    )
