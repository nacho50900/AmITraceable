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
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "osv5m_spain"
_MODEL_NAME = "facebook/dinov2-small"

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
    except FileNotFoundError:
        return None

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


async def estimate_locations_for_posts(
    posts: list, min_confidence: float = 0.4
) -> list[tuple[str, ImageLocationEstimate]]:
    """
    Orquestación de alto nivel: para cada SocialPost de tipo imagen que
    tenga `media_url`, descarga la imagen EN MEMORIA (nunca a disco),
    extrae el embedding, consulta el índice, y descarta la imagen
    inmediatamente. Devuelve solo las estimaciones con confianza >=
    min_confidence, junto al permalink del post que la generó (para poder
    mostrar evidencia en el informe).

    Se ejecuta en segundo plano de forma best-effort: si el índice no
    existe (no se ha corrido scripts/build_faiss_index.py) o falla la
    descarga de una imagen concreta, simplemente se omite esa imagen sin
    interrumpir el resto del análisis.
    """
    import httpx
    from PIL import Image
    import io

    results: list[tuple[str, ImageLocationEstimate]] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        for post in posts:
            media_url = getattr(post, "media_url", None)
            if not media_url or post.type not in ("image", "carousel_album"):
                continue

            try:
                resp = await client.get(media_url)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content))
            except Exception:
                continue  # imagen no descargable/decodificable: se omite, no se aborta el análisis

            estimate = estimate_location_from_image(image)
            # `image` sale de scope aquí y se descarta (nunca se escribe a disco)

            if estimate and estimate.confidence >= min_confidence:
                results.append((post.permalink, estimate))

    return results
