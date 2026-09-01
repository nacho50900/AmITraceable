"""
Construye (o amplía) un índice FAISS a partir de fotos de Wikimedia
Commons con coordenadas dentro de España, tomadas a pie por personas
(a diferencia de OSV-5M / Mapillary / KartaView, que son fotos desde
vehículo) -- mismo objetivo que build_flickr_index.py: reducir el domain
gap frente a las fotos de Instagram que analiza app/vision/geolocation.py.

POR QUÉ COMMONS EN VEZ DE FLICKR (decisión para la memoria del TFG):
Flickr cambió su política en 2025/2026 -- crear una API key ahora requiere
una suscripción Flickr Pro de pago (antes era gratuita). El API de
Wikimedia Commons (action=query&list=geosearch) es público, sin API key,
sin coste, y sin límite de tasa agresivo -- solo pide un User-Agent
identificable (ver _USER_AGENT más abajo, exigido por la política de la
Wikimedia Foundation: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_User-Agent_policy).

MISMO diseño de muestreo que build_flickr_index.py (ver ese docstring para
el razonamiento completo): sin mínimo forzado por celda, techo
(--cap-per-cell) para que un monumento no entierre al resto del índice,
dedup por perceptual hash dentro de cada celda, blur de caras SOLO (sin
persistir ninguna imagen a disco).

Diferencia práctica con Flickr: Commons no expone un campo de "accuracy"
del geotag como Flickr -- en cambio, casi toda foto en Commons con
coordenadas las tiene puestas a mano por quien subió el archivo (via la
plantilla {{Location}}), lo cual en la práctica suele ser más fiable que
un geotag EXIF automático de Flickr, pero no hay un número que filtrar.
Como filtro de calidad indirecto, aquí se descartan resultados que no son
fotografías reales (mapas, logos, diagramas, escaneos) por tipo MIME y
tamaño mínimo -- no es perfecto (algunos mapas/carteles se cuelan), pero
es la señal disponible sin analizar el contenido de cada imagen.

Uso:
    pip install httpx opencv-python imagehash pandas numpy faiss-cpu \
        pillow tqdm torch transformers huggingface_hub
    python build_commons_index.py --output ../data/commons_spain --cell-km 10 \
        --cap-per-cell 400

Salida (en --output): igual formato que build_flickr_index.py
    (embeddings.npy, index.faiss, index_meta.csv, _completed_cells.txt,
    _cell_stats.csv) -- combínalo con las demás fuentes usando
    scripts/merge_faiss_indices.py.

Resumible con Ctrl+C igual que build_flickr_index.py.
"""
import argparse
import io
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
from image_ingest_common import blur_faces, ensure_yunet_model, is_near_duplicate, nearest_province  # noqa: E402

import faiss  # noqa: E402
import torch  # noqa: E402

_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
# Wikimedia exige un User-Agent identificable en todas las peticiones al
# API (no es opcional -- peticiones con User-Agent genérico/por defecto de
# la librería HTTP reciben más 429 y pueden acabar bloqueadas sin aviso).
# Ver https://foundation.wikimedia.org/wiki/Policy:Wikimedia_User-Agent_policy
_USER_AGENT = "AmITraceable-TFG-ImageIngest/1.0 (https://github.com/nacho50900/AmITraceable)"

# Radio máximo que admite list=geosearch (metros). Si --cell-km implica un
# radio mayor, se recorta aquí -- ver _radius_for_cell().
_MAX_GEOSEARCH_RADIUS_M = 10000

_ACCEPTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MIN_DIMENSION_PX = 400  # descarta iconos/miniaturas/escaneos muy pequeños


def _radius_for_cell(cell: GridCell) -> int:
    """Radio en metros para cubrir la celda desde su centro -- se usa la
    semidiagonal (algo más que medio lado) para no dejar huecos sin cubrir
    en las esquinas de la celda, recortado al máximo que admite la API."""
    lat_km = (cell.max_lat - cell.min_lat) * 111.0
    lon_km = (cell.max_lon - cell.min_lon) * 111.320
    half_diagonal_km = ((lat_km ** 2 + lon_km ** 2) ** 0.5) / 2
    return min(_MAX_GEOSEARCH_RADIUS_M, int(half_diagonal_km * 1000) + 100)


def _commons_request(client: httpx.Client, params: dict, max_retries: int = 3) -> dict:
    params = {**params, "format": "json"}
    for attempt in range(max_retries):
        try:
            response = client.get(_COMMONS_API_URL, params=params, headers={"User-Agent": _USER_AGENT}, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(f"Wikimedia API error: {data['error']}")
            return data
        except httpx.HTTPError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Wikimedia API: agotados los reintentos (posible rate limit sostenido)")


def _geosearch_cell(client: httpx.Client, cell: GridCell, limit: int, gscontinue: str | None) -> tuple[list[dict], str | None]:
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{cell.center_lat}|{cell.center_lon}",
        "gsradius": _radius_for_cell(cell),
        "gsnamespace": 6,  # namespace File:
        "gslimit": limit,
    }
    if gscontinue:
        params["gscontinue"] = gscontinue

    data = _commons_request(client, params)
    results = data.get("query", {}).get("geosearch", [])
    next_continue = data.get("continue", {}).get("gscontinue")
    return results, next_continue


def _fetch_imageinfo(client: httpx.Client, titles: list[str]) -> dict[str, dict]:
    """Batch de hasta 50 títulos (límite del API sin permisos elevados).
    Devuelve {title: {url, width, height, mime, license}}."""
    if not titles:
        return {}
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
    }
    data = _commons_request(client, params)
    pages = data.get("query", {}).get("pages", {})

    result = {}
    for page in pages.values():
        title = page.get("title")
        infos = page.get("imageinfo")
        if not title or not infos:
            continue
        info = infos[0]
        license_name = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "desconocida")
        result[title] = {
            "url": info.get("url"),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "mime": info.get("mime"),
            "license": license_name,
        }
    return result


def _download_bytes(client: httpx.Client, url: str) -> bytes | None:
    try:
        response = client.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def _process_cell(
    cell: GridCell,
    client: httpx.Client,
    detector,
    processor,
    model,
    device: str,
    cap: int,
    phash_threshold: int,
    max_pages: int,
) -> tuple[list[np.ndarray], list[dict], int]:
    accepted_hashes: list[imagehash.ImageHash] = []
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []
    n_seen = 0
    per_page = min(500, cap * 2)  # 500 es el máximo admitido por gslimit sin bot flag
    gscontinue = None
    page = 0

    while len(meta_rows) < cap and page < max_pages:
        page += 1
        try:
            results, gscontinue = _geosearch_cell(client, cell, per_page, gscontinue)
        except Exception as e:
            print(f"  Aviso: fallo en geosearch celda {cell.id}, página {page}: {e}")
            break

        if not results:
            break
        n_seen += len(results)

        titles = [r["title"] for r in results]
        # imageinfo solo admite 50 títulos por petición -- se trocea.
        infos: dict[str, dict] = {}
        for i in range(0, len(titles), 50):
            infos.update(_fetch_imageinfo(client, titles[i:i + 50]))

        candidates = []
        for r in results:
            info = infos.get(r["title"])
            if not info or not info["url"]:
                continue
            if info["mime"] not in _ACCEPTED_MIME_TYPES:
                continue
            if info["width"] < _MIN_DIMENSION_PX or info["height"] < _MIN_DIMENSION_PX:
                continue
            candidates.append((r, info))

        with ThreadPoolExecutor(max_workers=8) as pool:
            downloaded = list(pool.map(lambda c: _download_bytes(client, c[1]["url"]), candidates))

        for (r, info), image_bytes in zip(candidates, downloaded):
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
                print(f"  Aviso: fallo extrayendo embedding de {r['title']}: {e}")
                continue

            lat, lon = float(r["lat"]), float(r["lon"])
            embeddings.append(embedding)
            meta_rows.append(
                {
                    "id": f"commons_{r['pageid']}",
                    "lat": lat,
                    "lon": lon,
                    "region": nearest_province(lat, lon),
                    "source": "wikimedia_commons",
                    "license": info["license"],
                }
            )
            accepted_hashes.append(phash)

        if gscontinue is None:
            break
        time.sleep(0.2)  # cortesía con el API, aunque no publique un límite estricto

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
    parser.add_argument("--output", default="../data/commons_spain")
    parser.add_argument("--cell-km", type=float, default=10.0)
    parser.add_argument("--cap-per-cell", type=int, default=400)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--max-pages-per-cell", type=int, default=3)
    parser.add_argument("--flush-every-cells", type=int, default=25)
    args = parser.parse_args()

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
    print(f"{len(all_cells)} celdas en el grid ({len(pending_cells)} pendientes).")

    client = httpx.Client()
    cells_since_flush = 0

    try:
        for cell in tqdm(pending_cells, desc="Celdas"):
            cell_embeddings, cell_meta, n_seen = _process_cell(
                cell, client, detector, processor, model, device,
                cap=args.cap_per_cell, phash_threshold=args.phash_threshold,
                max_pages=args.max_pages_per_cell,
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
              f"({100 * empty_cells / len(stats_df):.1f}%) -- ver _cell_stats.csv.")
    print("\nPara fusionar con otras fuentes, usa scripts/merge_faiss_indices.py.")


if __name__ == "__main__":
    main()
