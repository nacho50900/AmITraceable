"""
Construye (o amplía) un índice FAISS a partir de fotos de Flickr con
ubicación en España, tomadas a pie por personas (a diferencia de OSV-5M /
Mapillary / KartaView, que son todas fotos desde vehículo) -- pensado para
reducir el domain gap frente a las fotos de Instagram que analiza
app/vision/geolocation.py.

DIFERENCIA DE ARQUITECTURA DELIBERADA frente a
download_osv5m_spain.py + build_faiss_index.py (dos scripts, dos fases,
las imágenes quedan en disco): aquí NO se persiste ninguna imagen, ni
original ni ya blureada. Cada foto se descarga en memoria, se le blurean
las caras, se calcula su embedding, y se descarta -- solo sobreviven el
vector y sus metadatos (id, lat, lon, región). Motivo: estas son fotos de
terceros identificables que no han dado su consentimiento para este uso
más allá de la licencia CC de la foto en sí; para una herramienta que se
presenta como "de privacidad", no tiene sentido acumular en disco un
dataset de un millón de fotos de desconocidos cuando el único output que
hace falta es el índice.

Diseño de muestreo (ver conversación de diseño, resumen para la memoria):
- NO se fuerza un mínimo de fotos por celda -- si una zona apenas tiene
  fotos geolocalizadas en Flickr, se acepta tal cual: esa escasez es
  información real sobre dónde la gente fotografía (y por tanto sobre
  dónde caerán las consultas reales), no un fallo a corregir.
- SÍ se pone un techo (--cap-per-cell) por celda, para que un punto
  turístico con decenas de miles de fotos casi idénticas no entierre al
  resto del índice.
- Dentro del cap, se descartan casi-duplicados por perceptual hash
  (--phash-threshold) ANTES de aceptar una foto, para que el cap se llene
  con variedad real (ángulos/horas/estaciones distintas) y no con copias
  visuales del mismo encuadre.
- Solo se aceptan fotos con accuracy de geotag >= --min-accuracy (escala
  Flickr 1-16, 16 = nivel calle/edificio) -- fotos "geolocalizadas" a
  nivel de comunidad autónoma son ruido para un índice de similitud
  visual punto a punto.

Requiere una API key de Flickr (gratuita, sin necesidad de OAuth para
lectura de contenido público): https://www.flickr.com/services/apps/create/
    export FLICKR_API_KEY=tu_api_key

Uso:
    pip install httpx opencv-python imagehash pandas numpy faiss-cpu \
        pillow tqdm torch transformers
    python build_flickr_index.py --output ../../data/flickr_spain --cell-km 10 \
        --cap-per-cell 400 --min-accuracy 12

Salida (en --output):
    embeddings.npy      -- vectores DINOv2 normalizados, float32 (N x 384)
    index.faiss          -- mismo índice en formato FAISS (IndexFlatIP)
    index_meta.csv        -- id, lat, lon, region, source (mismo orden que embeddings.npy)
    _completed_cells.txt   -- celdas ya procesadas (para reanudar tras Ctrl+C)
    _cell_stats.csv        -- cell_id, n_vistas, n_aceptadas (para la memoria: mapa de cobertura)

Para combinar este índice con el de OSV-5M, usa después
scripts/merge_faiss_indices.py -- no lo hace este script directamente,
para poder rehacer la fusión sin volver a descargar nada si cambias qué
fuentes incluir.

Puedes interrumpir con Ctrl+C: se persiste progreso cada --flush-every-cells
celdas y al reanudar se salta lo ya hecho (lee _completed_cells.txt).
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from build_faiss_index import MODEL_NAME, embed_image, load_model  # noqa: E402
from flickr_grid import GridCell, generate_spain_grid  # noqa: E402
from image_ingest_common import (  # noqa: E402
    blur_faces,
    ensure_yunet_model,
    is_near_duplicate,
    nearest_province,
    sort_cells_by_proximity,
)

import faiss  # noqa: E402
import torch  # noqa: E402

_FLICKR_API_URL = "https://api.flickr.com/services/rest/"
# Todas las licencias CC + dominio público / "sin restricciones conocidas",
# excluyendo solo la 0 ("All Rights Reserved"). Ver
# https://www.flickr.com/services/api/flickr.photos.licenses.getInfo.html
_CC_LICENSES = "1,2,3,4,5,6,7,8,9,10"


def _flickr_request(client: httpx.Client, params: dict, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.get(_FLICKR_API_URL, params=params, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            if data.get("stat") != "ok":
                raise RuntimeError(f"Flickr API error: {data}")
            return data
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Flickr API: agotados los reintentos (posible rate limit sostenido)")


def _search_cell(client: httpx.Client, api_key: str, cell: GridCell, per_page: int, page: int) -> dict:
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "bbox": cell.bbox_str,
        "has_geo": 1,
        "license": _CC_LICENSES,
        "content_type": 1,  # solo fotos, sin vídeo/screenshot
        "media": "photos",
        "extras": "geo,url_l,url_c,url_z",
        "per_page": per_page,
        "page": page,
        "format": "json",
        "nojsoncallback": 1,
    }
    return _flickr_request(client, params)["photos"]


def _download_photo_bytes(client: httpx.Client, photo: dict) -> bytes | None:
    url = photo.get("url_l") or photo.get("url_c") or photo.get("url_z")
    if not url:
        return None
    try:
        response = client.get(url, timeout=20)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def _process_cell(
    cell: GridCell,
    client: httpx.Client,
    api_key: str,
    detector,
    processor,
    model,
    device: str,
    cap: int,
    min_accuracy: int,
    phash_threshold: int,
    max_pages: int,
) -> tuple[list[np.ndarray], list[dict], int]:
    accepted_hashes: list[imagehash.ImageHash] = []
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []
    n_seen = 0
    per_page = min(250, cap)  # 250 es el máximo por página que admite Flickr
    page = 1

    while len(meta_rows) < cap and page <= max_pages:
        try:
            photos = _search_cell(client, api_key, cell, per_page, page)
        except Exception as e:
            print(f"  Aviso: fallo consultando celda {cell.id}, página {page}: {e}")
            break

        batch = photos.get("photo", [])
        if not batch:
            break
        n_seen += len(batch)

        # El filtro de accuracy es gratis (ya viene en la respuesta de
        # búsqueda) -- se aplica ANTES de gastar ancho de banda descargando
        # nada.
        candidates = [p for p in batch if int(p.get("accuracy", 0)) >= min_accuracy]

        with ThreadPoolExecutor(max_workers=8) as pool:
            downloaded = list(pool.map(lambda p: _download_photo_bytes(client, p), candidates))

        for photo, image_bytes in zip(candidates, downloaded):
            if len(meta_rows) >= cap:
                break
            if image_bytes is None:
                continue
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                continue

            phash = imagehash.phash(pil_image)
            if is_near_duplicate(phash, accepted_hashes, phash_threshold):
                continue

            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            image_bgr = blur_faces(image_bgr, detector)
            blurred_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            try:
                embedding = embed_image(blurred_pil, processor, model, device)
            except Exception as e:
                print(f"  Aviso: fallo extrayendo embedding de {photo.get('id')}: {e}")
                continue

            lat, lon = float(photo["latitude"]), float(photo["longitude"])
            embeddings.append(embedding)
            meta_rows.append(
                {
                    "id": f"flickr_{photo['id']}",
                    "lat": lat,
                    "lon": lon,
                    "region": nearest_province(lat, lon),
                    "source": "flickr",
                }
            )
            accepted_hashes.append(phash)

        if len(batch) < per_page:
            break  # ya no hay más páginas
        page += 1
        time.sleep(0.2)  # cortesía con el rate limit de Flickr entre páginas

    return embeddings, meta_rows, n_seen


def _load_existing_state(output_dir: Path):
    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "index_meta.csv"
    completed_path = output_dir / "_completed_cells.txt"
    stats_path = output_dir / "_cell_stats.csv"

    embeddings = list(np.load(embeddings_path)) if embeddings_path.exists() else []
    meta_rows = pd.read_csv(meta_path).to_dict("records") if meta_path.exists() else []
    completed = set(completed_path.read_text().splitlines()) if completed_path.exists() else set()
    cell_stats = pd.read_csv(stats_path).to_dict("records") if stats_path.exists() else []

    if completed:
        print(f"Reanudando: {len(completed)} celdas y {len(meta_rows)} fotos ya procesadas.")

    return embeddings, meta_rows, completed, cell_stats


def _persist_state(output_dir: Path, embeddings: list, meta_rows: list, completed: set, cell_stats: list) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if embeddings:
        embeddings_matrix = np.vstack(embeddings).astype("float32")
        np.save(output_dir / "embeddings.npy", embeddings_matrix)

        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)
        faiss.write_index(index, str(output_dir / "index.faiss"))

    pd.DataFrame(meta_rows).to_csv(output_dir / "index_meta.csv", index=False)
    (output_dir / "_completed_cells.txt").write_text("\n".join(sorted(completed)))
    pd.DataFrame(cell_stats).to_csv(output_dir / "_cell_stats.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../data/flickr_spain")
    parser.add_argument("--cell-km", type=float, default=10.0)
    parser.add_argument("--cap-per-cell", type=int, default=400)
    parser.add_argument("--min-accuracy", type=int, default=12, help="Escala Flickr 1-16, 16=nivel calle")
    parser.add_argument("--phash-threshold", type=int, default=8, help="Distancia Hamming máxima para considerar casi-duplicado")
    parser.add_argument("--max-pages-per-cell", type=int, default=3)
    parser.add_argument("--flush-every-cells", type=int, default=25)
    parser.add_argument("--max-cells", type=int, default=None,
                         help="Limita el numero de celdas a procesar en esta ejecucion (para probar antes de lanzar todo)")
    parser.add_argument("--near", type=str, default=None,
                         help='"lat,lon" -- ordena las celdas por cercania a este punto en vez del orden de generacion del grid '
                              '(las primeras celdas del grid caen en el Atlantico/frontera portuguesa). Combinar con --max-cells '
                              'para un test rapido en una ciudad conocida, p.ej. --near "40.4168,-3.7038" --max-cells 1 para Madrid.')
    parser.add_argument("--api-key", default=os.environ.get("FLICKR_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("Error: falta la API key de Flickr. Pásala con --api-key o la variable FLICKR_API_KEY.")
        sys.exit(1)

    output_dir = Path(args.output)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device} (modelo {MODEL_NAME})")

    processor, model = load_model(device)
    detector = cv2.FaceDetectorYN.create(
        str(ensure_yunet_model()), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )

    embeddings, meta_rows, completed, cell_stats = _load_existing_state(output_dir)

    all_cells = generate_spain_grid(cell_km=args.cell_km)
    pending_cells = [c for c in all_cells if c.id not in completed]

    if args.near:
        near_lat, near_lon = (float(x) for x in args.near.split(","))
        pending_cells = sort_cells_by_proximity(pending_cells, near_lat, near_lon)
    if args.max_cells:
        pending_cells = pending_cells[:args.max_cells]

    print(f"{len(all_cells)} celdas en el grid ({len(pending_cells)} pendientes en esta ejecucion).")

    client = httpx.Client()
    cells_since_flush = 0

    try:
        for cell in tqdm(pending_cells, desc="Celdas"):
            cell_embeddings, cell_meta, n_seen = _process_cell(
                cell, client, args.api_key, detector, processor, model, device,
                cap=args.cap_per_cell, min_accuracy=args.min_accuracy,
                phash_threshold=args.phash_threshold, max_pages=args.max_pages_per_cell,
            )
            embeddings.extend(cell_embeddings)
            meta_rows.extend(cell_meta)
            completed.add(cell.id)
            cell_stats.append({"cell_id": cell.id, "n_vistas": n_seen, "n_aceptadas": len(cell_meta)})
            cells_since_flush += 1

            if cells_since_flush >= args.flush_every_cells:
                _persist_state(output_dir, embeddings, meta_rows, completed, cell_stats)
                cells_since_flush = 0
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Guardando progreso antes de salir...")
    finally:
        client.close()
        _persist_state(output_dir, embeddings, meta_rows, completed, cell_stats)

    print(f"\n{len(meta_rows)} fotos en el índice, guardadas en {output_dir}")
    if cell_stats:
        stats_df = pd.DataFrame(cell_stats)
        empty_cells = (stats_df["n_aceptadas"] == 0).sum()
        print(f"Celdas sin ninguna foto aceptada: {empty_cells}/{len(stats_df)} "
              f"({100 * empty_cells / len(stats_df):.1f}%) -- ver _cell_stats.csv para el detalle por celda.")
    print("\nPara fusionar con el índice de OSV-5M, usa scripts/merge_faiss_indices.py.")


if __name__ == "__main__":
    main()
