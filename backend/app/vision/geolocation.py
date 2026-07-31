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

También devuelve None (foto descartada, no analizada) cuando la foto no
tiene contenido suficientemente distintivo como para geolocalizarla con
algún sentido -- p.ej. una foto donde solo se ve el mar y la espalda de
alguien puede parecerse, visualmente, a imágenes de referencia de medio
litoral español a la vez. Ver `_neighbor_spread_km` /
`_MAX_NEIGHBOR_SPREAD_KM`: si los vecinos más parecidos están repartidos
por una zona demasiado amplia, ninguna provincia "ganadora" sería
significativa, así que se descarta en vez de mostrar una ubicación que
parecería más fiable de lo que en realidad es.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    distances = _index_meta.apply(
        lambda row: _haversine_km(lat, lon, row["lat"], row["lon"])
        if row["lat"] == row["lat"] and row["lon"] == row["lon"]  # descarta NaN
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


def _lazy_load():
    global _model, _processor, _index, _index_meta
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
    _model = AutoModel.from_pretrained(_MODEL_NAME).to(device).eval()
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with torch.no_grad():
            inputs = _processor(images=image.convert("RGB"), return_tensors="pt").to(device)
            outputs = _model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    except Exception:
        return None

    similarities, indices = _index.search(embedding.reshape(1, -1).astype("float32"), k)
    similarities, indices = similarities[0], indices[0]

    neighbor_rows = _index_meta.iloc[indices]

    # Foto no representativa (ver _MAX_NEIGHBOR_SPREAD_KM más arriba): sus
    # vecinos más parecidos están repartidos por medio país, así que
    # cualquier provincia "ganadora" sería arbitraria -- se descarta en vez
    # de dar una ubicación que parecería más segura de lo que realmente es.
    spread = _neighbor_spread_km(neighbor_rows)
    if spread is not None and spread > _MAX_NEIGHBOR_SPREAD_KM:
        return None

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

    return ImageLocationEstimate(
        province=top_province,
        confidence=round(votes / k, 2),
        k_neighbors=k,
        mean_similarity=round(float(np.mean(similarities)), 3),
        lat=round(lat, 4) if lat is not None else None,
        lon=round(lon, 4) if lon is not None else None,
    )


async def estimate_locations_for_posts(posts: list, progress_callback=None) -> GeolocationOutcome:
    """
    Orquestación de alto nivel: para cada SocialPost de tipo imagen que
    tenga `media_url`, descarga la imagen EN MEMORIA (nunca a disco),
    extrae el embedding, consulta el índice, y descarta la imagen
    inmediatamente. Devuelve TODAS las estimaciones que se pudieron
    calcular (con su confianza real, sin filtrar), junto al permalink del
    post que la generó -- el filtrado por umbral de fiabilidad, si hace
    falta, lo hace el llamador (ver `report/generator.py`).

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
    candidate_posts = [
        p for p in posts if getattr(p, "media_url", None) and p.type in ("image", "carousel_album")
    ]
    total = len(candidate_posts)

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

    async with httpx.AsyncClient(timeout=10.0) as client:
        for i, post in enumerate(candidate_posts, start=1):
            try:
                resp = await client.get(post.media_url)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content))
            except Exception:
                image = None  # imagen no descargable/decodificable: se omite, no se aborta el análisis

            if image is not None:
                estimate = estimate_location_from_image(image)
                # `image` sale de scope tras este bloque y se descarta (nunca se escribe a disco)
                if estimate is not None:
                    results.append((post.permalink, estimate))

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

    return GeolocationOutcome(index_available=index_available, results=results)
