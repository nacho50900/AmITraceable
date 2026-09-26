"""
Construye (o amplía) un índice FAISS a partir de fotos de Mapillary con
ubicación en España -- pensado para correr EN PARALELO a
build_commons_index.py mientras ese termina (usa un --output distinto),
no para sustituirlo.

CONTEXTO IMPORTANTE (para la memoria del TFG): Mapillary es imagery
mayoritariamente capturada desde vehículo, igual que OSV-5M -- el mismo
domain gap frente a fotos de Instagram que motivó añadir Commons en
primer lugar. Lo que cambia el cálculo: desde el 26 de junio de 2026,
Mapillary expone un campo `on_foot` por imagen, así que aquí SÍ se puede
filtrar por fotos tomadas a pie (--on-foot-only, activado por defecto) --
sin ese filtro, esta fuente aportaría sobre todo volumen adicional de
imagery de vehículo, no reducción de domain gap.

RESTRICCIÓN DE ESCALA (real, no un detalle menor): desde el 16 de enero
de 2026, Mapillary formalizó que toda consulta a /images debe ser
ESTRICTAMENTE menor de 0.01 grados cuadrados (~0.9-1.1km² según
latitud) -- una celda del grid de 10km (100km²) necesita ~100-130
sub-consultas para cubrirse entera (ver mapillary_grid.py). Con las
11.720 celdas del grid, son del orden de ~1,4M de sub-consultas solo
para LOCALIZAR candidatos, antes de descargar nada. Al límite de tasa
documentado (~50.000 peticiones/día), la fase de búsqueda por sí sola
para cubrir toda España son varias semanas -- confirmado con una
calibración antes de comprometerte a la ejecución completa, igual que
se hizo con Commons.

Requiere un token de acceso de Mapillary (gratuito, sin necesidad de
suscripción de pago -- a diferencia de Flickr): créalo en
https://www.mapillary.com/dashboard/developers
    export MAPILLARY_API_KEY=tu_token

Uso:
    pip install httpx opencv-python-headless imagehash pandas numpy \
        faiss-cpu pillow tqdm torch transformers huggingface_hub
    python build_mapillary_index.py --sample-cells 20 --cap-per-cell 400   # calibración
    python build_mapillary_index.py                                        # ejecución completa

Mismo formato de salida que build_commons_index.py (shards +
_completed_cells.txt + _cell_stats.csv, --compile para el índice final
usable) -- ver shard_store.py. Fusiona con las demás fuentes con
merge_faiss_indices.py cuando quieras.
"""
import argparse
import io
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import imagehash
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Carga backend/.env (dos niveles por encima de scripts/geolocalization/),
# el MISMO fichero que usa la app -- así MAPILLARY_API_KEY se pone una
# vez ahí y sirve tanto para este script como para cualquier otra cosa
# que lo necesite en el futuro, sin tener que fijar la variable de
# entorno a mano en cada sesión de terminal. Si el fichero no existe
# (p.ej. solo se ha copiado .env.example), load_dotenv() no falla, solo
# no carga nada -- --api-key/MAPILLARY_API_KEY como variable de entorno
# del sistema siguen funcionando igual como alternativa.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from build_faiss_index import MODEL_NAME, embed_image, load_model  # noqa: E402
from spain_grid import GridCell, generate_spain_grid  # noqa: E402
from image_ingest_common import blur_faces, ensure_yunet_model, is_near_duplicate, nearest_province, sort_cells_by_proximity  # noqa: E402
from mapillary_grid import generate_sub_tiles  # noqa: E402
import shard_store  # noqa: E402

import torch  # noqa: E402

_MAPILLARY_API_URL = "https://graph.mapillary.com/images"
_USER_AGENT = "AmITraceable-TFG-ImageIngest/1.0 (https://github.com/nacho50900/AmITraceable)"
_FIELDS = "id,captured_at,on_foot,geometry,computed_geometry,thumb_1024_url"

_ACCEPTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MIN_DIMENSION_PX = 400
_MAX_PIXELS = 50_000_000  # ver build_commons_index.py -- descarta escaneos/panorámicas anómalas

# Todo el contenido de Mapillary se distribuye bajo la misma licencia,
# a diferencia de Commons (donde cada fichero puede tener una distinta)
# -- ver https://www.mapillary.com/termsofuse
_MAPILLARY_LICENSE = "CC BY-SA 4.0"


def _mapillary_request(client: httpx.Client, url: str, params: dict | None, api_key: str, max_retries: int = 4) -> dict:
    headers = {"User-Agent": _USER_AGENT, "Authorization": f"OAuth {api_key}"}
    for attempt in range(max_retries):
        try:
            response = client.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else (2 ** attempt)
                time.sleep(wait_s)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Mapillary API: agotados los reintentos (posible rate limit sostenido)")


def _search_tile(client: httpx.Client, bbox: tuple[float, float, float, float], api_key: str) -> list[dict]:
    """Devuelve TODOS los resultados de una sub-tesela, siguiendo la
    paginación por cursor si hiciera falta (paging.next ya viene como URL
    completa en la respuesta del Graph API, no hace falta reconstruirla)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "access_token": api_key,
        "fields": _FIELDS,
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "limit": 100,
    }
    results = []
    url, current_params = _MAPILLARY_API_URL, params
    while url:
        data = _mapillary_request(client, url, current_params, api_key)
        results.extend(data.get("data", []))
        url = data.get("paging", {}).get("next")
        current_params = None  # 'next' ya trae todos los parámetros embebidos en la URL
    return results


def _download_bytes(client: httpx.Client, url: str, max_retries: int = 5) -> tuple[bytes | None, str | None]:
    """Igual que en build_commons_index.py -- reintentos con backoff,
    imprescindible desde que descubrimos que sin ellos el 95.8% de las
    descargas fallaban por rate limiting del CDN de la fuente."""
    for attempt in range(max_retries):
        try:
            response = client.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20)
            if response.status_code == 429:
                if attempt == max_retries - 1:
                    return None, "http_429_reintentos_agotados"
                retry_after = response.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else (2 ** attempt)
                time.sleep(wait_s)
                continue
            response.raise_for_status()
            return response.content, None
        except httpx.HTTPStatusError as e:
            return None, f"http_{e.response.status_code}"
        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                return None, "timeout"
            time.sleep(2 ** attempt)
        except httpx.ConnectError as e:
            if attempt == max_retries - 1:
                return None, f"connect_error:{e}"
            time.sleep(2 ** attempt)
        except Exception as e:
            return None, f"otro:{type(e).__name__}:{e}"
    return None, "http_429_reintentos_agotados"


def _extract_lat_lon(photo: dict) -> tuple[float, float] | None:
    # computed_geometry (corregida por el pipeline de Structure-from-Motion
    # de Mapillary) es más fiable que geometry (GPS crudo del dispositivo)
    # cuando está disponible -- se usa como preferencia, con geometry
    # como respaldo.
    geom = photo.get("computed_geometry") or photo.get("geometry")
    if not geom or geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates")
    if not coords or len(coords) != 2:
        return None
    lon, lat = coords
    return float(lat), float(lon)


def _process_cell(
    cell: GridCell,
    client: httpx.Client,
    detector,
    processor,
    model,
    device: str,
    api_key: str,
    cap: int,
    phash_threshold: int,
    on_foot_only: bool,
) -> tuple[list[np.ndarray], list[dict], dict]:
    accepted_hashes: list[imagehash.ImageHash] = []
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []
    reject_reasons = {
        "sin_geometria": 0, "no_on_foot": 0, "demasiado_pequena": 0,
        "demasiado_grande": 0, "fallo_descarga": 0, "casi_duplicada": 0,
    }
    download_error_counter: dict[str, int] = {}
    n_seen = 0
    had_unrecovered_error = False

    # FASE 1: recopilar candidatos de TODAS las sub-teselas de la celda
    # (solo metadatos, sin descargar nada todavía).
    candidates_pool: list[dict] = []
    for bbox in generate_sub_tiles(cell):
        try:
            results = _search_tile(client, bbox, api_key)
        except Exception as e:
            print(f"  Aviso: fallo consultando sub-tesela de {cell.id} ({bbox}): {e}")
            had_unrecovered_error = True
            continue  # una sub-tesela fallida no invalida el resto de la celda

        n_seen += len(results)
        for photo in results:
            if on_foot_only and not photo.get("on_foot"):
                reject_reasons["no_on_foot"] += 1
                continue
            if _extract_lat_lon(photo) is None:
                reject_reasons["sin_geometria"] += 1
                continue
            candidates_pool.append(photo)
        time.sleep(0.1)  # cortesía entre sub-teselas

    # FASE 2: barajar antes de descargar -- mismo motivo que en Commons:
    # sin esto, el cap se llenaría con las sub-teselas procesadas primero
    # (esquina de la celda) en vez de una muestra repartida por toda ella.
    random.shuffle(candidates_pool)

    # FASE 3: descargar/dedup/blur/embed hasta llenar el cap.
    _DOWNLOAD_BATCH = 3  # igual que en Commons, tras el fix del rate limiting real
    i = 0
    while len(meta_rows) < cap and i < len(candidates_pool):
        batch = candidates_pool[i:i + _DOWNLOAD_BATCH]
        i += _DOWNLOAD_BATCH

        with ThreadPoolExecutor(max_workers=3) as pool:
            downloaded = list(pool.map(lambda p: _download_bytes(client, p["thumb_1024_url"]), batch))

        for photo, (image_bytes, error) in zip(batch, downloaded):
            if len(meta_rows) >= cap:
                break
            if image_bytes is None:
                reject_reasons["fallo_descarga"] += 1
                download_error_counter[error] = download_error_counter.get(error, 0) + 1
                continue
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                reject_reasons["fallo_descarga"] += 1
                key = f"pil_no_decodifica:{type(e).__name__}"
                download_error_counter[key] = download_error_counter.get(key, 0) + 1
                continue

            w, h = pil_image.size
            if w < _MIN_DIMENSION_PX or h < _MIN_DIMENSION_PX:
                reject_reasons["demasiado_pequena"] += 1
                continue
            if w * h > _MAX_PIXELS:
                reject_reasons["demasiado_grande"] += 1
                continue

            phash = imagehash.phash(pil_image)
            if is_near_duplicate(phash, accepted_hashes, phash_threshold):
                reject_reasons["casi_duplicada"] += 1
                continue

            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            image_bgr = blur_faces(image_bgr, detector)
            blurred_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            try:
                embedding = embed_image(blurred_pil, processor, model, device)
            except Exception as e:
                print(f"  Aviso: fallo extrayendo embedding de {photo.get('id')}: {e}")
                continue

            lat, lon = _extract_lat_lon(photo)
            embeddings.append(embedding)
            meta_rows.append({
                "id": f"mapillary_{photo['id']}",
                "lat": lat,
                "lon": lon,
                "region": nearest_province(lat, lon),
                "source": "mapillary",
                "license": _MAPILLARY_LICENSE,
            })
            accepted_hashes.append(phash)

    diagnostics = {
        "n_seen": n_seen,
        "n_pool": len(candidates_pool),
        **reject_reasons,
        "download_error_counter": download_error_counter,
        "had_unrecovered_error": had_unrecovered_error,
    }
    return embeddings, meta_rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../data/mapillary_spain")
    parser.add_argument("--cell-km", type=float, default=10.0)
    parser.add_argument("--cap-per-cell", type=int, default=400)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--flush-every-cells", type=int, default=25)
    parser.add_argument("--api-key", default=os.environ.get("MAPILLARY_API_KEY"))
    parser.add_argument("--on-foot-only", action=argparse.BooleanOptionalAction, default=True,
                         help="Filtra solo fotos tomadas a pie (campo on_foot de Mapillary, desde jun-2026). "
                              "Por defecto activado -- sin esto, esta fuente aporta sobre todo imagery de "
                              "vehículo, el mismo domain gap que ya cubre OSV-5M.")
    parser.add_argument("--near", type=str, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--sample-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if args.near and args.sample_cells:
        print("Error: --near y --sample-cells son incompatibles.")
        sys.exit(1)

    output_dir = Path(args.output)

    if args.compile:
        shard_store.compile_shards(output_dir)
        return

    if not args.api_key:
        print("Error: falta el token de Mapillary. Pásalo con --api-key o la variable MAPILLARY_API_KEY.")
        print("Créalo gratis en https://www.mapillary.com/dashboard/developers")
        sys.exit(1)

    lock_path = shard_store.acquire_lock(output_dir)
    try:
        _run(args, output_dir)
    finally:
        shard_store.release_lock(lock_path)


def _run(args: argparse.Namespace, output_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device} (modelo {MODEL_NAME})")

    processor, model = load_model(device)
    detector = cv2.FaceDetectorYN.create(
        str(ensure_yunet_model()), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )

    known_ids, total_photos, completed, cell_stats = shard_store.load_existing_state(output_dir)
    photos_at_start = total_photos
    next_shard_index = shard_store.next_shard_index(output_dir / shard_store.SHARDS_SUBDIR)

    all_cells = generate_spain_grid(cell_km=args.cell_km)
    pending_cells = [c for c in all_cells if c.id not in completed]

    if args.near:
        near_lat, near_lon = (float(x) for x in args.near.split(","))
        pending_cells = sort_cells_by_proximity(pending_cells, near_lat, near_lon)
    elif args.sample_cells:
        rng = random.Random(args.seed)
        pending_cells = rng.sample(pending_cells, min(args.sample_cells, len(pending_cells)))
    if args.max_cells:
        pending_cells = pending_cells[:args.max_cells]

    print(f"{len(all_cells)} celdas en el grid ({len(pending_cells)} pendientes en esta ejecucion).")
    print(f"on_foot_only={args.on_foot_only} -- ~100-130 sub-teselas por celda (límite de bbox de Mapillary).")

    since_flush_embeddings: list = []
    since_flush_meta: list = []

    client = httpx.Client()
    cells_since_flush = 0
    cells_run_this_execution = 0
    run_start = time.monotonic()
    global_reject_totals = {"sin_geometria": 0, "no_on_foot": 0, "demasiado_pequena": 0, "demasiado_grande": 0, "fallo_descarga": 0, "casi_duplicada": 0}
    global_download_errors: dict[str, int] = {}
    n_duplicates_skipped = 0

    def _flush() -> None:
        nonlocal since_flush_embeddings, since_flush_meta, next_shard_index
        shard_store.persist_new_shard(output_dir, next_shard_index, since_flush_embeddings, since_flush_meta)
        if since_flush_meta:
            next_shard_index += 1
        shard_store.persist_progress(output_dir, completed, cell_stats)
        since_flush_embeddings = []
        since_flush_meta = []

    try:
        for cell in tqdm(pending_cells, desc="Celdas"):
            cell_embeddings, cell_meta, diag = _process_cell(
                cell, client, detector, processor, model, device, args.api_key,
                cap=args.cap_per_cell, phash_threshold=args.phash_threshold,
                on_foot_only=args.on_foot_only,
            )
            new_embeddings, new_meta = [], []
            for emb, meta in zip(cell_embeddings, cell_meta):
                if meta["id"] in known_ids:
                    n_duplicates_skipped += 1
                    continue
                known_ids.add(meta["id"])
                new_embeddings.append(emb)
                new_meta.append(meta)
            since_flush_embeddings.extend(new_embeddings)
            since_flush_meta.extend(new_meta)
            total_photos += len(new_meta)

            if not diag["had_unrecovered_error"]:
                completed.add(cell.id)
            else:
                print(f"  {cell.id} queda pendiente (alguna sub-tesela falló) -- se reintentará en la próxima ejecución.")
            cell_stats.append({
                "cell_id": cell.id, "n_vistas": diag["n_seen"], "n_aceptadas": len(cell_meta),
                "n_pool": diag["n_pool"], "sin_geometria": diag["sin_geometria"],
                "no_on_foot": diag["no_on_foot"], "demasiado_pequena": diag["demasiado_pequena"],
                "demasiado_grande": diag["demasiado_grande"], "fallo_descarga": diag["fallo_descarga"],
                "casi_duplicada": diag["casi_duplicada"], "fallo_sin_recuperar": diag["had_unrecovered_error"],
            })
            for k in global_reject_totals:
                global_reject_totals[k] += diag[k]
            for err, n in diag["download_error_counter"].items():
                global_download_errors[err] = global_download_errors.get(err, 0) + n
            cells_since_flush += 1
            cells_run_this_execution += 1

            if cells_since_flush >= args.flush_every_cells:
                _flush()
                cells_since_flush = 0
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Guardando progreso antes de salir...")
    finally:
        client.close()
        _flush()

    elapsed_s = time.monotonic() - run_start
    n_photos_run = total_photos - photos_at_start

    print(f"\n{total_photos} fotos en el índice total, guardadas en {output_dir}")
    if n_duplicates_skipped:
        print(f"({n_duplicates_skipped} fotos descartadas por ya estar en el índice.)")
    if elapsed_s > 0 and cells_run_this_execution > 0:
        s_per_cell = elapsed_s / cells_run_this_execution
        photos_per_hour = n_photos_run / (elapsed_s / 3600)
        remaining_cells = len(all_cells) - len(completed)
        eta_hours = (remaining_cells * s_per_cell) / 3600
        photos_per_cell_avg = n_photos_run / cells_run_this_execution
        print(f"\nEsta ejecución: {cells_run_this_execution} celdas, {n_photos_run} fotos, {elapsed_s/60:.1f} min "
              f"({s_per_cell:.1f}s/celda, ~{photos_per_hour:.0f} fotos/hora).")
        print(f"Extrapolado a las {remaining_cells} celdas que faltan (estimación gruesa): "
              f"~{eta_hours:.1f}h (~{eta_hours/24:.1f} días) y ~{remaining_cells * photos_per_cell_avg:.0f} fotos más.")
    n_seen_total = sum(r["n_vistas"] for r in cell_stats[-cells_run_this_execution:]) if cells_run_this_execution else 0
    if n_seen_total:
        print(f"\nDesglose de por qué se descartan candidatos (de {n_seen_total} vistos en esta ejecución):")
        for reason, n in global_reject_totals.items():
            print(f"  {reason}: {n} ({100*n/n_seen_total:.1f}%)")
        print(f"  aceptadas: {n_photos_run} ({100*n_photos_run/n_seen_total:.1f}%)")
        if global_download_errors:
            print("  Motivos de 'fallo_descarga' más comunes:")
            for err, n in sorted(global_download_errors.items(), key=lambda x: -x[1])[:8]:
                print(f"    {err}: {n}")
    if cell_stats:
        stats_df = pd.DataFrame(cell_stats)
        empty_cells = (stats_df["n_aceptadas"] == 0).sum()
        print(f"Celdas sin ninguna foto aceptada: {empty_cells}/{len(stats_df)} "
              f"({100 * empty_cells / len(stats_df):.1f}%) -- ver _cell_stats.csv.")
    print("\nPara compilar el índice usable: --compile. Para fusionar con otras fuentes: merge_faiss_indices.py.")


if __name__ == "__main__":
    main()
