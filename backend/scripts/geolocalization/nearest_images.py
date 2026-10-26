"""
Dada una foto, la pasa por el mismo DINOv2 que usa la app para geolocalizar
(app/vision/geolocation.py) y muestra las imágenes más parecidas del índice
FAISS, con un enlace a cada una.

Sirve para entender por qué una foto acaba en un sitio concreto: abre los
enlaces y comprueba si los vecinos se parecen a la foto por el lugar o solo por
el aspecto general (oscuridad, nieve, niebla...). Un parecido alto entre
fotogramas casi idénticos de un mismo recorrido no es evidencia del lugar.

Uso:
    python nearest_images.py foto.jpg
    python nearest_images.py foto.jpg --k 10 --index ../../data/spain_combined

Enlaces según el origen (por el prefijo del id):
    mapillary_<id>  -> mapillary.com/app/?pKey=<id>
    commons_<id>    -> commons.wikimedia.org/?curid=<id>
    <solo números>  -> OSV-5M; su id es el de la imagen en Mapillary, así que
                       se enlaza igual que las de Mapillary

Ignora el EXIF de la foto a propósito: muestra lo que haría el modelo visual,
no las coordenadas reales (la app, si la foto trae GPS, no llega a usar DINOv2).
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

_MODEL_NAME = "facebook/dinov2-small"  # mismo modelo que _MODEL_NAME en geolocation.py
_DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "spain_combined"


def image_link(image_id: str) -> tuple[str, str]:
    """Devuelve (origen, enlace) de una imagen del índice a partir de su id."""
    if image_id.startswith("commons_"):
        return "commons", f"https://commons.wikimedia.org/?curid={image_id.removeprefix('commons_')}"
    if image_id.startswith("mapillary_"):
        return "mapillary", f"https://www.mapillary.com/app/?pKey={image_id.removeprefix('mapillary_')}"
    return "osv5m", f"https://www.mapillary.com/app/?pKey={image_id}"


def embed_image(image_path: Path) -> np.ndarray:
    """Embedding DINOv2 (token CLS, normalizado L2) igual que
    `_get_dinov2_embedding` en geolocation.py, sin el worker de iGPU."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME).to(device).eval()

    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding / (np.linalg.norm(embedding) + 1e-8)


def load_index(index_dir: Path):
    """Carga index.faiss y solo las columnas necesarias de index_meta.csv."""
    import faiss
    import pandas as pd

    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "index_meta.csv"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No se encontró el índice en {index_dir} (index.faiss + index_meta.csv).")

    index = faiss.read_index(str(index_path))
    meta = pd.read_csv(meta_path, usecols=["id", "lat", "lon", "region"], dtype={"id": str})
    if len(meta) != index.ntotal:
        raise ValueError(f"index.faiss tiene {index.ntotal} vectores pero index_meta.csv {len(meta)} filas.")
    return index, meta


def search_neighbors(index, meta, embedding: np.ndarray, k: int):
    """Los k vecinos más cercanos, como lista de dicts ordenada de más a menos parecido."""
    if embedding.shape[0] != index.d:
        raise ValueError(f"El embedding tiene dimensión {embedding.shape[0]} y el índice {index.d}.")
    similarities, indices = index.search(embedding.reshape(1, -1).astype("float32"), k)
    neighbors = []
    for similarity, row in zip(similarities[0], indices[0]):
        if row < 0:  # FAISS rellena con -1 si hay menos vectores que k
            continue
        record = meta.iloc[int(row)]
        source, link = image_link(record["id"])
        neighbors.append(
            {
                "similarity": float(similarity),
                "id": record["id"],
                "source": source,
                "link": link,
                "region": record["region"] if isinstance(record["region"], str) else "desconocido",
                "lat": float(record["lat"]),
                "lon": float(record["lon"]),
            }
        )
    return neighbors


def print_report(neighbors: list[dict]) -> None:
    for position, n in enumerate(neighbors, start=1):
        print(f"{position:>2}. similitud {n['similarity']:.3f}  {n['source']:<9} {n['region']}  ({n['lat']:.4f}, {n['lon']:.4f})")
        print(f"    imagen: {n['link']}")
        print(f"    mapa:   https://www.google.com/maps?q={n['lat']:.6f},{n['lon']:.6f}")

    if not neighbors:
        return
    votes = Counter(n["region"] for n in neighbors)
    region, count = votes.most_common(1)[0]
    mean_similarity = sum(n["similarity"] for n in neighbors) / len(neighbors)
    distinct_places = len({(round(n["lat"], 3), round(n["lon"], 3)) for n in neighbors})
    print()
    print(f"Región ganadora: {region} ({count}/{len(neighbors)} vecinos, confianza {count / len(neighbors):.2f})")
    print(f"Similitud media: {mean_similarity:.3f}")
    print(f"Lugares distintos entre los vecinos (redondeo ~100 m): {distinct_places} de {len(neighbors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Muestra las imágenes del índice FAISS más parecidas a una foto.")
    parser.add_argument("image", type=Path, help="Ruta de la foto a analizar.")
    parser.add_argument("--index", type=Path, default=_DEFAULT_INDEX_DIR, help="Carpeta con index.faiss e index_meta.csv.")
    parser.add_argument("--k", type=int, default=10, help="Número de vecinos a mostrar (por defecto 10).")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"No existe la foto: {args.image}")

    print("Cargando índice...")
    index, meta = load_index(args.index)
    print("Calculando embedding...")
    embedding = embed_image(args.image)
    print()
    print_report(search_neighbors(index, meta, embedding, args.k))


if __name__ == "__main__":
    main()
