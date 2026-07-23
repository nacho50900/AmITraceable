"""
Descarga TODAS las imágenes de OpenStreetView-5M (train + test), de
CUALQUIER país, respetando un límite de disco (por defecto 35GB, para
dejar margen sobre tus 40GB libres).

Esta es la versión "mundo" de download_osv5m_spain.py: mismo mecanismo
(paralelismo por hilos, resume automático, caché local del CSV de
metadatos), pero sin el filtro de país -- se queda con TODOS los IDs de
cada shard, no solo los de España.

⚠️  AVISO DE TAMAÑO: OSV-5M completo son varios millones de imágenes.
Bajar TODO el dataset son 260 GB de red (aunque tu disco
local se mantenga acotado por --max-disk-gb, la red que se transfiere para
llegar a ese punto es mucho mayor, igual que en la versión de España pero
a una escala mucho mayor porque ahora aprovechas ~100% de cada shard en
vez de solo la fracción española). Si tu objetivo es tener una muestra
global sin bajarte el dataset entero, usa --max-shards o --max-disk-gb
generosamente bajo para parar pronto, no hace falta procesar los ~100
shards de cada split para tener un dataset grande y variado.

CÓMO FUNCIONA (disco acotado, red NO acotada):
El repositorio empaqueta las imágenes en shards .zip (images/train/00.zip,
01.zip, ...; images/test/00.zip, ...). Este script, con hasta --workers
shards en paralelo:
  1. Descarga cada shard .zip a una carpeta temporal propia del hilo.
  2. Extrae TODAS las imágenes del shard que aún no tengamos guardadas.
  3. Borra esa carpeta temporal inmediatamente.
  4. Deja de lanzar NUEVOS shards si el total guardado se acerca a
     --max-disk-gb (los que ya estén en marcha terminan).

El CSV de metadatos (train.csv / test.csv, ~3GB cada uno) se cachea
localmente tras la primera descarga (_<split>_metadata_cache.csv dentro de
--output), para no volver a bajarlo en cada relanzamiento.

Puedes interrumpir con Ctrl+C en cualquier momento sin perder lo ya
guardado -- se persiste el progreso tras cada shard que termina.

Uso:
    pip install huggingface_hub pandas pillow tqdm
    python download_osv5m_world.py --output ../data/osv5m_world --max-disk-gb 35 --workers 6
"""
import argparse
import os
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# IMPORTANTE: hay que fijar esto ANTES de importar huggingface_hub, porque la
# ruta de caché se calcula al importar la librería. Si no se hace esto,
# huggingface_hub guarda una copia de cada shard descargado en
# ~/.cache/huggingface/hub (o el equivalente en Windows) POR DEBAJO de lo que
# borra este script, así que el disco se llena igualmente aunque el script
# "borre" su copia -- ese fichero borrado era solo un symlink/copia en la
# carpeta de trabajo, no el blob real que huggingface_hub guarda en caché.
_SCRATCH_CACHE = Path(__file__).parent / "_hf_scratch_cache"
_SCRATCH_CACHE.mkdir(exist_ok=True)
os.environ["HF_HUB_CACHE"] = str(_SCRATCH_CACHE)
os.environ["HF_HOME"] = str(_SCRATCH_CACHE)

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

REPO_ID = "osv5m/osv5m"
REPO_TYPE = "dataset"


def _load_all_ids(split: str, output_dir: Path) -> tuple[set[str], pd.DataFrame, str]:
    """Descarga (o carga de caché local) el CSV de metadatos de un split y
    devuelve (todos_los_ids, filas_completas_indexadas_por_id,
    nombre_columna_id). A diferencia de la versión España, no se filtra
    por país -- se usan todas las filas del CSV."""
    local_cache_path = output_dir / f"_{split}_metadata_cache.csv"

    if local_cache_path.exists():
        print(f"Usando metadatos de {split}.csv cacheados en {local_cache_path} (no se re-descarga).")
        metadata = pd.read_csv(local_cache_path)
    else:
        print(f"Descargando metadatos de {split}.csv...")
        csv_path = hf_hub_download(repo_id=REPO_ID, filename=f"{split}.csv", repo_type=REPO_TYPE)
        metadata = pd.read_csv(csv_path)

        # Se guarda una copia local propia ANTES de tocar la caché de HF,
        # para que la próxima ejecución no tenga que volver a bajar esto.
        metadata.to_csv(local_cache_path, index=False)

        # Se borra la caché de huggingface_hub justo después (pesa ~3GB
        # por sí sola) para no dejarla ocupando disco innecesariamente;
        # nuestra propia copia en local_cache_path ya está a salvo.
        shutil.rmtree(_SCRATCH_CACHE, ignore_errors=True)
        _SCRATCH_CACHE.mkdir(exist_ok=True)

    print(f"Columnas en {split}.csv: {list(metadata.columns)}")

    id_col = next((c for c in metadata.columns if c.lower() == "id"), metadata.columns[0])

    country_col = next((c for c in metadata.columns if c.lower() in ("country", "country_code", "iso")), None)
    if country_col is not None:
        n_countries = metadata[country_col].nunique(dropna=True)
        print(f"{n_countries} países distintos detectados en la columna '{country_col}' (no se filtra ninguno).")

    all_rows = metadata.copy()
    all_rows[id_col] = all_rows[id_col].astype(str)
    print(f"{len(all_rows)} filas totales en {split}.csv.\n")

    return set(all_rows[id_col]), all_rows.set_index(id_col), id_col


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file()) / (1024 ** 3)


def _download_and_extract_shard(
    shard_path: str,
    all_ids: set[str],
    rows_by_id: pd.DataFrame,
    images_dir: Path,
    state_lock: threading.Lock,
    already_saved_ids: set[str],
    saved_rows: list,
) -> int:
    """Descarga un shard .zip y extrae todas las imágenes que aún no
    tengamos. Corre en su propio hilo, así que usa una carpeta de caché
    PROPIA (no la global) para no pisarse con otros hilos descargando en
    paralelo, y solo toca las estructuras compartidas (already_saved_ids,
    saved_rows) bajo el lock. Devuelve cuántas imágenes nuevas se guardaron."""
    worker_cache = Path(tempfile.mkdtemp(prefix="shard_", dir=_SCRATCH_CACHE))
    new_count = 0
    try:
        local_zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard_path,
            repo_type=REPO_TYPE,
            cache_dir=str(worker_cache),
        )
        with zipfile.ZipFile(local_zip_path) as zf:
            for name in zf.namelist():
                stem = Path(name).stem
                if stem not in all_ids:
                    continue

                with state_lock:
                    if stem in already_saved_ids:
                        continue
                    already_saved_ids.add(stem)

                with zf.open(name) as image_file:
                    data = image_file.read()
                (images_dir / f"{stem}.jpg").write_bytes(data)

                with state_lock:
                    saved_rows.append(rows_by_id.loc[stem])
                new_count += 1
    finally:
        # Se borra SOLO la carpeta temporal de este hilo/shard, nunca la
        # caché global -- otros hilos pueden estar usándola a la vez.
        shutil.rmtree(worker_cache, ignore_errors=True)
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../data/osv5m_world")
    parser.add_argument("--max-disk-gb", type=float, default=35.0, help="Tope de disco para las imágenes guardadas")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"], choices=["train", "test"],
        help="Qué splits procesar (por defecto ambos)",
    )
    parser.add_argument(
        "--max-shards", type=int, default=None,
        help="Límite de shards a procesar por split (útil para una prueba rápida, o para limitar "
             "cuánta red se transfiere sin depender solo de --max-disk-gb)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Shards descargados en paralelo (descarga = red, no CPU; sube esto si tienes ancho de banda "
             "de sobra). Ojo: cada worker activo puede tener un shard .zip entero en disco temporalmente.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Reanudar: si ya existe un metadata.csv de una ejecución anterior
    # (p.ej. interrumpida con Ctrl+C), se carga para no perder esas filas
    # ni volver a escribir esas imágenes -- se completa lo que falte, no se
    # empieza de cero.
    existing_metadata_path = output_dir / "metadata.csv"
    saved_rows: list = []
    already_saved_ids: set[str] = set()
    if existing_metadata_path.exists():
        previous = pd.read_csv(existing_metadata_path)
        prev_id_col = next((c for c in previous.columns if c.lower() == "id"), previous.columns[0])
        previous[prev_id_col] = previous[prev_id_col].astype(str)
        saved_rows = [row for _, row in previous.iterrows()]
        already_saved_ids = set(previous[prev_id_col])
        print(f"Reanudando: {len(already_saved_ids)} imágenes de una ejecución anterior ya en {images_dir}.\n")

    # Shards ya completados en ejecuciones anteriores (para no volver a
    # descargarlos al reanudar -- si no se hiciera esto, cada Ctrl+C y
    # relanzamiento repetiría la descarga de red de todo lo ya procesado).
    completed_shards_path = output_dir / "_completed_shards.txt"
    completed_shards: set[str] = set()
    if completed_shards_path.exists():
        completed_shards = set(completed_shards_path.read_text().splitlines())
        print(f"{len(completed_shards)} shards ya completados en ejecuciones anteriores, se omitirán.\n")

    state_lock = threading.Lock()

    def _persist_progress() -> None:
        """Debe llamarse siempre con state_lock ya adquirido."""
        completed_shards_path.write_text("\n".join(sorted(completed_shards)))
        if saved_rows:
            pd.DataFrame(saved_rows).drop_duplicates().to_csv(existing_metadata_path, index=False)

    api = HfApi()
    all_repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    stopped_early = False

    for split in args.splits:
        if stopped_early:
            break

        all_ids, rows_by_id, id_col = _load_all_ids(split, output_dir)
        if not all_ids:
            print(f"⚠️  0 filas en {split}, se omite este split.")
            continue

        shard_files = sorted(f for f in all_repo_files if f.startswith(f"images/{split}/") and f.endswith(".zip"))
        if args.max_shards:
            shard_files = shard_files[: args.max_shards]
        pending_shards = [s for s in shard_files if s not in completed_shards]
        print(f"{len(shard_files)} shards .zip en el split '{split}' ({len(pending_shards)} pendientes).")

        shard_iter = iter(pending_shards)
        pbar = tqdm(total=len(pending_shards), desc=f"Shards de {split} ({args.workers} en paralelo)")

        def _next_shard():
            return next(shard_iter, None)

        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = {}

        try:
            # Llenado inicial del pool, respetando el límite de disco desde
            # el principio.
            for _ in range(args.workers):
                if _dir_size_gb(images_dir) >= args.max_disk_gb:
                    stopped_early = True
                    break
                shard_path = _next_shard()
                if shard_path is None:
                    break
                fut = executor.submit(
                    _download_and_extract_shard,
                    shard_path, all_ids, rows_by_id, images_dir,
                    state_lock, already_saved_ids, saved_rows,
                )
                futures[fut] = shard_path

            while futures:
                done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    shard_path = futures.pop(fut)
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"Aviso: fallo procesando shard {shard_path}: {e}")
                    else:
                        with state_lock:
                            completed_shards.add(shard_path)
                            _persist_progress()
                    pbar.update(1)

                    if stopped_early:
                        continue

                    if _dir_size_gb(images_dir) >= args.max_disk_gb:
                        print(
                            f"\n🛑 Límite de disco alcanzado ({_dir_size_gb(images_dir):.1f}GB >= "
                            f"{args.max_disk_gb}GB). No se lanzan más shards nuevos "
                            f"(los ya en marcha terminan)."
                        )
                        stopped_early = True
                        continue

                    next_shard = _next_shard()
                    if next_shard is not None:
                        fut2 = executor.submit(
                            _download_and_extract_shard,
                            next_shard, all_ids, rows_by_id, images_dir,
                            state_lock, already_saved_ids, saved_rows,
                        )
                        futures[fut2] = next_shard
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario. Esperando a que terminen los shards ya en marcha "
                  "para no perder progreso a medias (puede tardar unos segundos)...")
            stopped_early = True
            wait(list(futures.keys()))
            for fut in futures:
                shard_path = futures[fut]
                try:
                    fut.result()
                except Exception:
                    continue
                with state_lock:
                    completed_shards.add(shard_path)
                    _persist_progress()
        finally:
            pbar.close()
            executor.shutdown(wait=True)

    if not saved_rows:
        print("\n⚠️  No se guardó ninguna imagen.")
        return

    with state_lock:
        final_metadata = pd.DataFrame(saved_rows).drop_duplicates()
        final_metadata.to_csv(existing_metadata_path, index=False)

    print(f"\nListo. {len(final_metadata)} imágenes guardadas en {images_dir}")
    print(f"Disco usado por las imágenes: {_dir_size_gb(images_dir):.2f} GB")
    print(f"Metadatos en {existing_metadata_path}")
    if stopped_early:
        print(
            "\nNota: el script paró antes de terminar todos los shards (límite de disco o Ctrl+C). "
            "Puedes construir el índice FAISS igualmente con lo ya guardado -- será más pequeño, "
            "pero funcional. Vuelve a lanzarlo más tarde (reanuda automáticamente) si quieres ampliarlo."
        )


if __name__ == "__main__":
    main()
