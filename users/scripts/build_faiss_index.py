"""
Extrae embeddings DINOv2 de todas las imágenes de España descargadas por
download_osv5m_spain.py, y construye un índice FAISS para búsqueda de
vecinos más cercanos (reverse image search geográfico).

Uso:
    pip install torch transformers faiss-cpu pillow tqdm pandas numpy
    python build_faiss_index.py --images ../data/osv5m_spain

Salida:
    ../data/osv5m_spain/index.faiss       (índice FAISS)
    ../data/osv5m_spain/index_meta.csv    (mismo orden que el índice: id, lat, lon, city, region)

Nota: usa ViT-S/14 (la versión más pequeña de DINOv2, ~340MB) para que sea
viable en CPU. Si tienes GPU con CUDA, este script la detecta y usa
automáticamente (mucho más rápido).
"""
import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"  # ~340MB, 384-dim embeddings


def load_model(device: str):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return processor, model


@torch.no_grad()
def embed_image(image: Image.Image, processor, model, device: str) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # CLS token como representación global de la imagen (estándar para DINOv2)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    # Normalizar para poder usar producto interno como similitud coseno en FAISS
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding.astype("float32")


def _normalize_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """El CSV real de OSV-5M puede nombrar las columnas de forma distinta a
    lo que asumimos originalmente (p.ej. 'latitude'/'longitude' en vez de
    'lat'/'lon'). Aquí se detectan y renombran a nombres canónicos
    ('id', 'lat', 'lon', 'region') para que el resto del pipeline
    (geolocation.py) no tenga que adivinar nada."""
    rename_map = {}

    lat_col = next((c for c in metadata.columns if c.lower() in ("lat", "latitude")), None)
    lon_col = next((c for c in metadata.columns if c.lower() in ("lon", "lng", "longitude")), None)
    region_col = next((c for c in metadata.columns if c.lower() in ("region", "state", "province", "admin1")), None)
    id_col = next((c for c in metadata.columns if c.lower() == "id"), metadata.columns[0])

    if lat_col is None or lon_col is None:
        raise RuntimeError(
            f"No se encontraron columnas de latitud/longitud reconocibles. "
            f"Columnas disponibles: {list(metadata.columns)}. Ajusta _normalize_columns() manualmente."
        )
    if region_col is None:
        print(
            "⚠️  No se encontró una columna tipo 'region'/'province'; se usará 'city' como aproximación "
            "de la ubicación mostrada (menos preciso a nivel de provincia)."
        )
        region_col = next((c for c in metadata.columns if c.lower() == "city"), None)
        if region_col is None:
            raise RuntimeError(
                f"Tampoco se encontró columna 'city'. Columnas disponibles: {list(metadata.columns)}. "
                "Ajusta _normalize_columns() manualmente."
            )

    rename_map[lat_col] = "lat"
    rename_map[lon_col] = "lon"
    rename_map[region_col] = "region"
    rename_map[id_col] = "id"

    print(f"Mapeo de columnas detectado: {rename_map}")
    return metadata.rename(columns=rename_map)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../data/osv5m_spain")
    args = parser.parse_args()

    base_dir = Path(args.images)
    images_dir = base_dir / "images"
    metadata_path = base_dir / "metadata.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    metadata = pd.read_csv(metadata_path)
    metadata = _normalize_columns(metadata)
    metadata["id"] = metadata["id"].astype(str)
    print(f"{len(metadata)} imágenes en metadata.csv")

    processor, model = load_model(device)

    embeddings = []
    valid_rows = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extrayendo embeddings"):
        image_path = images_dir / f"{row['id']}.jpg"
        if not image_path.exists():
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            embedding = embed_image(image, processor, model, device)
        except Exception as e:
            print(f"Aviso: fallo procesando {image_path}: {e}")
            continue
        embeddings.append(embedding)
        valid_rows.append(row)

    embeddings_matrix = np.vstack(embeddings)
    dimension = embeddings_matrix.shape[1]

    # Índice de producto interno == similitud coseno, ya que los vectores
    # están normalizados. Para ~20k vectores un índice plano (fuerza bruta)
    # es rápido de sobra; no hace falta un índice aproximado (IVF/HNSW).
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    faiss.write_index(index, str(base_dir / "index.faiss"))
    pd.DataFrame(valid_rows).to_csv(base_dir / "index_meta.csv", index=False)

    print(f"\nÍndice FAISS construido con {index.ntotal} vectores de dimensión {dimension}")
    print(f"Guardado en {base_dir / 'index.faiss'} y {base_dir / 'index_meta.csv'}")


if __name__ == "__main__":
    main()
