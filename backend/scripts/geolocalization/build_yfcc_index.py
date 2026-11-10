"""
Construye (o amplía) un índice FAISS a partir de las fotos de España de
YFCC100M (el spain.tsv que genera yfcc_sqlite_espana.py --filtrar).

spain.tsv solo contiene METADATOS (coordenadas, URL, licencia...), no
imágenes: para tener vectores hay que descargar cada foto, difuminar las
caras, sacar su embedding DINOv2 y guardarlo. Eso hace este script, con el
MISMO formato de salida que build_commons_index.py y build_mapillary_index.py
(shards + _completed_cells.txt + _cell_stats.csv, ver shard_store.py), de
modo que luego se compila con --compile y se fusiona con las demás fuentes
con merge_faiss_indices.py.

Cómo funciona:
  1. Lee spain.tsv, descarta las fotos con geoetiqueta poco precisa
     (--min-accuracy) y las agrupa en celdas de ~10 km.
  2. Por celda: baraja y ordena priorizando precisión (--preferred-accuracy) y
     variedad de usuarios (sin descartar ninguna foto), descarga, quita duplicados con phash, difumina caras y
     calcula el embedding hasta llenar --cap-per-cell.
  3. Cada --flush-every-cells celdas escribe un shard. Si se corta, relanza el
     mismo comando: las celdas ya terminadas se saltan.

Cosas que conviene saber (para la memoria del TFG):
  - Las URLs de YFCC apuntan a Flickr y muchas fotos ya no existen: espera un
    porcentaje alto de fallo_descarga (se ve en el desglose final).
  - Flickr suele redirigir las fotos borradas a una imagen de "no disponible".
    Por eso NO se siguen redirecciones y solo se acepta HTTP 200; si no, se
    habrían indexado imágenes basura con coordenadas reales.
  - Las imágenes de YFCC son de tamaño medio (hasta ~500 px de lado largo, que
    yo sepa), así que --min-side es mucho más bajo que el 400 de Mapillary.
  - Es imagery turística/aficionada, con mucha foto de interior y de ciudades
    grandes: el cap por celda y la prioridad a usuarios distintos existen para amortiguarlo.
  - YFCC incluye licencias CC de todo tipo (también NC y ND). Aquí solo se
    guardan embeddings y coordenadas, no las imágenes; la licencia de cada
    foto queda en el campo `license` del shard.

Uso:
    pip install httpx opencv-python-headless imagehash pandas numpy \\
        faiss-cpu pillow tqdm torch transformers huggingface_hub
    python build_yfcc_index.py spain.tsv --sample-cells 20      # calibración
    python build_yfcc_index.py spain.tsv                        # ejecución completa
    python build_yfcc_index.py --compile                        # índice usable
"""
import argparse
import io
import math
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from build_faiss_index import MODEL_NAME, embed_image, load_model  # noqa: E402
from image_ingest_common import blur_faces, ensure_yunet_model, is_near_duplicate, nearest_province  # noqa: E402
import shard_store  # noqa: E402

import torch  # noqa: E402

_USER_AGENT = "AmITraceable-TFG-ImageIngest/1.0 (https://github.com/nacho50900/AmITraceable)"
_MAX_PIXELS = 50_000_000
_KM_PER_DEG_LAT = 111.32
_COLUMNS = ["photoid", "uid", "longitude", "latitude", "accuracy", "licensename", "downloadurl"]

_REJECT_KEYS = ("sin_url", "usuario_saturado", "fallo_descarga", "demasiado_pequena",
                "demasiado_grande", "casi_duplicada")


def _load_candidates(tsv_path: str, cell_km: float, min_accuracy: int) -> pd.DataFrame:
    """Lee el TSV, filtra por precisión de la geoetiqueta y asigna cada foto a
    una celda de ~cell_km x cell_km (en longitud el paso se ajusta a la latitud)."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, usecols=_COLUMNS, keep_default_na=False)
    n_total = len(df)
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["acc"] = pd.to_numeric(df["accuracy"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["downloadurl"].str.len() > 0]
    n_valid = len(df)
    df = df[df["acc"] >= min_accuracy].copy()
    print(f"{n_total:,} filas en el TSV | {n_valid:,} con coordenadas y URL | "
          f"{len(df):,} con precisión >= {min_accuracy}.")

    dlat = cell_km / _KM_PER_DEG_LAT
    ilat = np.floor(df["lat"].to_numpy() / dlat).astype(int)
    lat_centre = (ilat + 0.5) * dlat
    dlon = dlat / np.cos(np.radians(lat_centre))
    ilon = np.floor(df["lon"].to_numpy() / dlon).astype(int)
    df["cell_id"] = [f"yfcc_{a}_{b}" for a, b in zip(ilat, ilon)]
    return df


def _flickr_url(url: str) -> str:
    # Los hosts *.staticflickr.com sirven por https; el TSV trae http://.
    if url.startswith("http://") and "staticflickr.com" in url:
        return "https://" + url[len("http://"):]
    return url


def _download_bytes(client: httpx.Client, url: str, max_retries: int = 4) -> tuple[bytes | None, str | None]:
    """Nunca lanza. Solo acepta HTTP 200 y NO sigue redirecciones (ver cabecera)."""
    for attempt in range(max_retries):
        try:
            response = client.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20, follow_redirects=False)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_retries - 1:
                    return None, f"http_{response.status_code}_reintentos_agotados"
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after else 2 ** attempt
                except ValueError:
                    wait_s = 2 ** attempt
                time.sleep(wait_s)
                continue
            if response.status_code != 200:
                return None, f"http_{response.status_code}"
            return response.content, None
        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                return None, "timeout"
            time.sleep(2 ** attempt)
        except httpx.ConnectError as e:
            if attempt == max_retries - 1:
                return None, f"connect_error:{type(e).__name__}"
            time.sleep(2 ** attempt)
        except Exception as e:
            return None, f"otro:{type(e).__name__}:{e}"
    return None, "reintentos_agotados"


def _safe_download(client: httpx.Client, row: dict) -> tuple[bytes | None, str | None]:
    url = row.get("downloadurl")
    if not url:
        return None, "sin_url"
    try:
        return _download_bytes(client, _flickr_url(url))
    except Exception as e:
        return None, f"otro:{type(e).__name__}:{e}"


def _process_cell(rows: list[dict], client: httpx.Client, pool: ThreadPoolExecutor, detector,
                  processor, model, device: str, args) -> tuple[list[np.ndarray], list[dict], dict]:
    accepted_hashes: list[imagehash.ImageHash] = []
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []
    reject = {k: 0 for k in _REJECT_KEYS}
    download_errors: dict[str, int] = {}
    per_user: dict[str, int] = {}

    random.shuffle(rows)  # que el cap no se llene solo con unas pocas fotos consecutivas

    # Orden de candidatos (no se descarta nada, solo cambia el orden en que se
    # intentan hasta llenar el cap):
    #   1) nivel de precisión: primero las fotos con accuracy >= --preferred-accuracy
    #      y después las de menor precisión (hasta --min-accuracy). Así las fotos
    #      menos precisas solo entran donde faltan fotos mejores.
    #   2) dentro de cada nivel, por "rondas": la 1ª foto de cada usuario, luego
    #      la 2ª, etc. (sort estable: se mantiene el orden aleatorio).
    # --max-por-usuario > 0 añade un tope duro opcional por usuario y celda.
    per_user: dict[tuple, int] = {}
    for r in rows:
        r["_tier"] = 0 if r["acc"] >= args.preferred_accuracy else 1
        key_u = (r["uid"], r["_tier"])
        k = per_user.get(key_u, 0)
        per_user[key_u] = k + 1
        r["_ronda"] = k
    pool_rows = []
    taken: dict[str, int] = {}
    for r in sorted(rows, key=lambda r: (r["_tier"], r["_ronda"])):
        if args.max_por_usuario > 0:
            if taken.get(r["uid"], 0) >= args.max_por_usuario:
                reject["usuario_saturado"] += 1
                continue
            taken[r["uid"]] = taken.get(r["uid"], 0) + 1
        pool_rows.append(r)

    i = 0
    while len(meta_rows) < args.cap_per_cell and i < len(pool_rows):
        batch = pool_rows[i:i + args.workers]
        i += args.workers
        downloaded = list(pool.map(lambda r: _safe_download(client, r), batch))

        for row, (image_bytes, error) in zip(batch, downloaded):
            if len(meta_rows) >= args.cap_per_cell:
                break
            if image_bytes is None:
                key = "sin_url" if error == "sin_url" else "fallo_descarga"
                reject[key] += 1
                if key == "fallo_descarga":
                    download_errors[error] = download_errors.get(error, 0) + 1
                continue
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                reject["fallo_descarga"] += 1
                key = f"pil_no_decodifica:{type(e).__name__}"
                download_errors[key] = download_errors.get(key, 0) + 1
                continue

            w, h = pil_image.size
            if min(w, h) < args.min_side:
                reject["demasiado_pequena"] += 1
                continue
            if w * h > _MAX_PIXELS:
                reject["demasiado_grande"] += 1
                continue

            try:
                phash = imagehash.phash(pil_image)
                if is_near_duplicate(phash, accepted_hashes, args.phash_threshold):
                    reject["casi_duplicada"] += 1
                    continue
                image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                image_bgr = blur_faces(image_bgr, detector)
                blurred_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                embedding = embed_image(blurred_pil, processor, model, device)
            except Exception as e:
                print(f"  Aviso: fallo procesando la foto {row.get('photoid')}: {e}")
                continue

            lat, lon = float(row["lat"]), float(row["lon"])
            embeddings.append(embedding)
            meta_rows.append({
                "id": f"yfcc_{row['photoid']}",
                "lat": lat,
                "lon": lon,
                "region": nearest_province(lat, lon),
                "source": "yfcc100m",
                "accuracy": int(row["acc"]),
                "license": row.get("licensename") or "desconocida",
            })
            accepted_hashes.append(phash)

    diagnostics = {"n_seen": len(rows), "n_pool": len(pool_rows), **reject,
                   "download_error_counter": download_errors}
    return embeddings, meta_rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tsv", nargs="?", default=None, help="ruta a spain.tsv")
    parser.add_argument("--output", default="../../data/yfcc_spain")
    parser.add_argument("--cell-km", type=float, default=10.0)
    parser.add_argument("--cap-per-cell", type=int, default=400)
    parser.add_argument("--max-por-usuario", type=int, default=0,
                        help="tope duro de fotos por usuario y celda (0 = sin tope, por defecto). "
                             "Sin tope, igualmente se prioriza la variedad de usuarios al llenar el cap.")
    parser.add_argument("--min-accuracy", type=int, default=12,
                        help="precisión mínima de la geoetiqueta de Flickr (1=mundo ... 16=calle). "
                             "Por debajo de ~12 las coordenadas son el centro de una ciudad/región y "
                             "envenenarían el índice.")
    parser.add_argument("--preferred-accuracy", type=int, default=14,
                        help="al llenar el cap de cada celda se intentan primero las fotos con "
                             "accuracy >= este valor y luego las de menor precisión (>= --min-accuracy). "
                             "0 = desactivar (todas al mismo nivel).")
    parser.add_argument("--min-side", type=int, default=256, help="lado corto mínimo en px")
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8, help="descargas simultáneas")
    parser.add_argument("--flush-every-cells", type=int, default=25)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--sample-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.compile:
        shard_store.compile_shards(output_dir)
        return
    if not args.tsv:
        parser.error("falta la ruta a spain.tsv (o usa --compile)")
    if not Path(args.tsv).exists():
        sys.exit(f"No existe {args.tsv}")

    lock_path = shard_store.acquire_lock(output_dir)
    try:
        _run(args, output_dir)
    finally:
        shard_store.release_lock(lock_path)


def _run(args: argparse.Namespace, output_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device} (modelo {MODEL_NAME})")

    df = _load_candidates(args.tsv, args.cell_km, args.min_accuracy)
    processor, model = load_model(device)
    detector = cv2.FaceDetectorYN.create(
        str(ensure_yunet_model()), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000)

    known_ids, total_photos, completed, cell_stats = shard_store.load_existing_state(output_dir)
    photos_at_start = total_photos
    next_shard_index = shard_store.next_shard_index(output_dir / shard_store.SHARDS_SUBDIR)

    groups = {cid: g for cid, g in df.groupby("cell_id", sort=True) if cid not in completed}
    pending = sorted(groups)
    if args.sample_cells:
        pending = random.Random(args.seed).sample(pending, min(args.sample_cells, len(pending)))
    if args.max_cells:
        pending = pending[:args.max_cells]
    print(f"{df['cell_id'].nunique()} celdas con fotos ({len(pending)} pendientes en esta ejecución).")

    since_flush_embeddings: list = []
    since_flush_meta: list = []
    cells_since_flush = cells_run = n_duplicates_skipped = 0
    totals = {k: 0 for k in _REJECT_KEYS}
    global_errors: dict[str, int] = {}
    n_seen_total = 0
    run_start = time.monotonic()

    def _flush() -> None:
        nonlocal since_flush_embeddings, since_flush_meta, next_shard_index
        shard_store.persist_new_shard(output_dir, next_shard_index, since_flush_embeddings, since_flush_meta)
        if since_flush_meta:
            next_shard_index += 1
        shard_store.persist_progress(output_dir, completed, cell_stats)
        since_flush_embeddings = []
        since_flush_meta = []

    client = httpx.Client()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        for cell_id in tqdm(pending, desc="Celdas"):
            rows = groups[cell_id][["photoid", "uid", "lat", "lon", "acc", "licensename", "downloadurl"]].to_dict("records")
            try:
                cell_embeddings, cell_meta, diag = _process_cell(
                    rows, client, pool, detector, processor, model, device, args)
            except Exception as e:
                # Un fallo inesperado en una celda no debe tumbar horas de ejecución:
                # la celda queda pendiente y se reintenta en la próxima.
                print(f"  {cell_id} queda pendiente por error inesperado: {type(e).__name__}: {e}")
                continue

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
            completed.add(cell_id)

            cell_stats.append({"cell_id": cell_id, "n_vistas": diag["n_seen"], "n_aceptadas": len(cell_meta),
                               "n_pool": diag["n_pool"], **{k: diag[k] for k in _REJECT_KEYS}})
            n_seen_total += diag["n_seen"]
            for k in _REJECT_KEYS:
                totals[k] += diag[k]
            for err, n in diag["download_error_counter"].items():
                global_errors[err] = global_errors.get(err, 0) + n
            cells_since_flush += 1
            cells_run += 1
            if cells_since_flush >= args.flush_every_cells:
                _flush()
                cells_since_flush = 0
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Guardando progreso antes de salir...")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        client.close()
        _flush()

    elapsed = time.monotonic() - run_start
    n_run = total_photos - photos_at_start
    print(f"\n{total_photos} fotos en el índice total, guardadas en {output_dir}")
    if n_duplicates_skipped:
        print(f"({n_duplicates_skipped} fotos descartadas por ya estar en el índice.)")
    if cells_run:
        s_cell = elapsed / cells_run
        remaining = len(groups) - cells_run
        print(f"Esta ejecución: {cells_run} celdas, {n_run} fotos, {elapsed / 60:.1f} min "
              f"({s_cell:.1f}s/celda). Quedan ~{remaining} celdas: ~{remaining * s_cell / 3600:.1f} h (estimación gruesa).")
    if n_seen_total:
        print(f"\nDesglose de candidatos de esta ejecución ({n_seen_total} vistos):")
        for reason, n in totals.items():
            print(f"  {reason}: {n} ({100 * n / n_seen_total:.1f}%)")
        print(f"  aceptadas: {n_run} ({100 * n_run / n_seen_total:.1f}%)")
        if global_errors:
            print("  Motivos de 'fallo_descarga' más comunes:")
            for err, n in sorted(global_errors.items(), key=lambda x: -x[1])[:8]:
                print(f"    {err}: {n}")
    print("\nPara compilar el índice usable: --compile. Para fusionar con otras fuentes: merge_faiss_indices.py.")


if __name__ == "__main__":
    main()
