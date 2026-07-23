"""
Descarga TODAS las imágenes de España de OpenStreetView-5M (train + test),
respetando un límite de disco (por defecto 35GB, para dejar margen sobre
tus 40GB libres).

VERSIÓN CORREGIDA: la primera versión usaba
`datasets.load_dataset("osv5m/osv5m", streaming=True)`, pero las versiones
recientes de `datasets` (>=4.0) retiraron el soporte de "loading scripts"
personalizados, que es justo el mecanismo que usa este repositorio. Por
eso fallaba con "RuntimeError: Dataset scripts are no longer supported".
Esta versión no usa `datasets` en absoluto, habla directamente con
`huggingface_hub`.

VERSIÓN CON DESCARGA PARALELA: la versión anterior descargaba los shards
.zip uno detrás de otro (secuencial), lo cual es un cuello de botella de
red, no de CPU -- cada shard espera a que termine el anterior aunque tengas
ancho de banda de sobra para varias descargas simultáneas. Esta versión
descarga varios shards en paralelo con un pool de hilos (--workers, por
defecto 4). Al ser descarga (I/O), los hilos funcionan bien pese al GIL de
Python -- no hace falta multiproceso.

CÓMO FUNCIONA (disco acotado, red NO acotada):
El repositorio empaqueta las imágenes en shards .zip (images/train/00.zip,
01.zip, ...; images/test/00.zip, ...), y el CSV de metadatos no indica de
antemano qué shard contiene qué imagen -- hay que mirar dentro de cada
shard para saberlo. Por eso este script:

  1. Descarga el CSV de metadatos de train y test (ligero, sin imágenes).
  2. Filtra las filas de España.
  3. Para cada shard .zip (de train y de test), con hasta --workers
     shards en paralelo:
       a. Lo descarga a una carpeta temporal PROPIA de ese shard (para que
          los hilos no se pisen entre sí).
       b. Extrae SOLO las imágenes cuyo id esté en la lista de España.
       c. Borra esa carpeta temporal inmediatamente.
  4. Para automáticamente de lanzar NUEVOS shards si el total guardado se
     acerca a --max-disk-gb (los que ya estén en marcha terminan).

IMPORTANTE sobre el límite de disco con paralelismo: --max-disk-gb solo
mide las imágenes YA EXTRAÍDAS en `images/`, no los .zip que se están
descargando en ese momento. Con --workers 4 puede haber hasta 4 shards
completos ocupando disco temporalmente (en carpetas separadas) antes de
borrarse. Dale margen extra libre en disco proporcional a --workers.

Puedes interrumpir con Ctrl+C en cualquier momento sin perder lo ya
guardado -- se persiste el progreso tras cada shard que termina.

Uso:
    pip install huggingface_hub pandas pillow tqdm
    python download_osv5m_spain.py --output ../data/osv5m_spain --max-disk-gb 35 --workers 6
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


def _is_spain(country_value) -> bool:
    """Ajusta esto si el formato del código de país en el CSV difiere de
    lo esperado. El script imprime valores únicos de ejemplo al arrancar
    para que lo puedas verificar antes de lanzar la descarga completa."""
    return str(country_value).strip().upper() in ("ES", "ESP", "SPAIN")


def _load_spain_ids(split: str, output_dir: Path) -> tuple[set[str], pd.DataFrame, str]:
    """Descarga el CSV de metadatos de un split y devuelve (ids_españa,
    filas_completas_indexadas_por_id, nombre_columna_id).

    El CSV completo (train.csv pesa ~3GB) se cachea en local
    (_<split>_metadata_cache.csv dentro de --output) tras la primera
    descarga, para que relanzar el script tras un Ctrl+C no implique
    volver a bajar esos GB de red cada vez -- solo se re-descarga si ese
    fichero de caché no existe."""
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

    country_col = next((c for c in metadata.columns if c.lower() in ("country", "country_code", "iso")), None)
    if country_col is None:
        raise RuntimeError(
            f"No se encontró columna de país reconocible en {split}.csv. "
            f"Columnas disponibles: {list(metadata.columns)}. Ajusta _load_spain_ids() manualmente."
        )
    print(f"Valores únicos de ejemplo en '{country_col}': {metadata[country_col].dropna().unique()[:10]}")

    id_col = next((c for c in metadata.columns if c.lower() == "id"), metadata.columns[0])

    spain_rows = metadata[metadata[country_col].apply(_is_spain)].copy()
    spain_rows[id_col] = spain_rows[id_col].astype(str)
    print(f"{len(spain_rows)} filas de España en {split}.csv.\n")

    return set(spain_rows[id_col]), spain_rows.set_index(id_col), id_col


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*") if f.is_file()) / (1024 ** 3)


def _download_and_extract_shard(
    shard_path: str,
    spain_ids: set[str],
    spain_rows_by_id: pd.DataFrame,
    images_dir: Path,
    state_lock: threading.Lock,
    already_saved_ids: set[str],
    saved_rows: list,
) -> int:
    """Descarga un shard .zip y extrae las imágenes de España que aún no
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
                if stem not in spain_ids:
                    continue

                with state_lock:
                    if stem in already_saved_ids:
                        continue
                    already_saved_ids.add(stem)

                with zf.open(name) as image_file:
                    data = image_file.read()
                (images_dir / f"{stem}.jpg").write_bytes(data)

                with state_lock:
                    saved_rows.append(spain_rows_by_id.loc[stem])
                new_count += 1
    finally:
        # Se borra SOLO la carpeta temporal de este hilo/shard, nunca la
        # caché global -- otros hilos pueden estar usándola a la vez.
        shutil.rmtree(worker_cache, ignore_errors=True)
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../data/osv5m_spain")
    parser.add_argument("--max-disk-gb", type=float, default=35.0, help="Tope de disco para las imágenes guardadas")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"], choices=["train", "test"],
        help="Qué splits procesar (por defecto ambos)",
    )
    parser.add_argument(
        "--max-shards", type=int, default=None,
        help="Límite de shards a procesar por split (útil para una prueba rápida antes de lanzar todo)",
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

        spain_ids, spain_rows_by_id, id_col = _load_spain_ids(split, output_dir)
        if not spain_ids:
            print(f"⚠️  0 filas de España en {split}, se omite este split.")
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
                    shard_path, spain_ids, spain_rows_by_id, images_dir,
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
                            next_shard, spain_ids, spain_rows_by_id, images_dir,
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
        print("\n⚠️  No se guardó ninguna imagen. Revisa los mensajes de arriba (columna/valores de país).")
        return

    with state_lock:
        final_metadata = pd.DataFrame(saved_rows).drop_duplicates()
        final_metadata.to_csv(existing_metadata_path, index=False)

    print(f"\nListo. {len(final_metadata)} imágenes de España guardadas en {images_dir}")
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