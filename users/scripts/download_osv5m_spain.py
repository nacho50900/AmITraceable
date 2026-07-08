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

CÓMO FUNCIONA (disco acotado, red NO acotada):
El repositorio empaqueta las imágenes en shards .zip (images/train/00.zip,
01.zip, ...; images/test/00.zip, ...), y el CSV de metadatos no indica de
antemano qué shard contiene qué imagen -- hay que mirar dentro de cada
shard para saberlo. Por eso este script:

  1. Descarga el CSV de metadatos de train y test (ligero, sin imágenes).
  2. Filtra las filas de España.
  3. Para cada shard .zip (de train y de test):
       a. Lo descarga a un fichero temporal.
       b. Extrae SOLO las imágenes cuyo id esté en la lista de España.
       c. Borra el .zip descargado inmediatamente.
  4. Para automáticamente si el total guardado se acerca a --max-disk-gb.

Esto mantiene el DISCO acotado en todo momento (nunca hay más de un shard
"vivo" a la vez, y los descartes se borran al instante). Pero la RED sigue
teniendo que bajarse cada shard entero, coincida o no con España -- si
decides procesar train completo, puedes acabar transfiriendo varios
cientos de GB de red aunque tu disco nunca pase de --max-disk-gb. Puedes
interrumpir con Ctrl+C en cualquier momento sin perder lo ya guardado.

Uso:
    pip install huggingface_hub pandas pillow tqdm
    python download_osv5m_spain.py --output ../data/osv5m_spain --max-disk-gb 35
"""
import argparse
import os
import shutil
import zipfile
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


def _load_spain_ids(split: str) -> tuple[set[str], pd.DataFrame, str]:
    """Descarga el CSV de metadatos de un split y devuelve (ids_españa,
    filas_completas_indexadas_por_id, nombre_columna_id)."""
    print(f"Descargando metadatos de {split}.csv...")
    csv_path = hf_hub_download(repo_id=REPO_ID, filename=f"{split}.csv", repo_type=REPO_TYPE)
    metadata = pd.read_csv(csv_path)

    # Se borra la caché justo después de leer el CSV en memoria (pesa ~3GB
    # por sí solo) para no dejarlo ocupando disco innecesariamente.
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
    args = parser.parse_args()

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Reanudar: si ya existe un metadata.csv de una ejecución anterior
    # (p.ej. interrumpida con Ctrl+C), se carga para no perder esas filas
    # ni volver a escribir esas imágenes -- se completa lo que falte, no se
    # empieza de cero.
    existing_metadata_path = output_dir / "metadata.csv"
    saved_rows: list[pd.Series] = []
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

    api = HfApi()
    all_repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    stopped_early = False

    for split in args.splits:
        if stopped_early:
            break

        spain_ids, spain_rows_by_id, id_col = _load_spain_ids(split)
        if not spain_ids:
            print(f"⚠️  0 filas de España en {split}, se omite este split.")
            continue

        shard_files = sorted(f for f in all_repo_files if f.startswith(f"images/{split}/") and f.endswith(".zip"))
        if args.max_shards:
            shard_files = shard_files[: args.max_shards]
        print(f"{len(shard_files)} shards .zip en el split '{split}'.")

        for shard_path in tqdm(shard_files, desc=f"Shards de {split}"):
            if shard_path in completed_shards:
                continue

            current_gb = _dir_size_gb(images_dir)
            if current_gb >= args.max_disk_gb:
                print(f"\n🛑 Límite de disco alcanzado ({current_gb:.1f}GB >= {args.max_disk_gb}GB). Parando aquí.")
                stopped_early = True
                break

            try:
                local_zip_path = hf_hub_download(repo_id=REPO_ID, filename=shard_path, repo_type=REPO_TYPE)
            except KeyboardInterrupt:
                print("\nInterrumpido por el usuario durante la descarga de un shard.")
                stopped_early = True
                break

            try:
                with zipfile.ZipFile(local_zip_path) as zf:
                    for name in zf.namelist():
                        stem = Path(name).stem
                        if stem in spain_ids and stem not in already_saved_ids:
                            with zf.open(name) as image_file:
                                data = image_file.read()
                            (images_dir / f"{stem}.jpg").write_bytes(data)
                            saved_rows.append(spain_rows_by_id.loc[stem])
                            already_saved_ids.add(stem)
            finally:
                # Se borra TODA la carpeta de caché (no solo local_zip_path)
                # tras cada shard, para garantizar que no queda ningún blob
                # residual ocupando disco por debajo.
                shutil.rmtree(_SCRATCH_CACHE, ignore_errors=True)
                _SCRATCH_CACHE.mkdir(exist_ok=True)

            # Se persiste el progreso TRAS CADA SHARD (no solo al final), para
            # no perder nada si Ctrl+C llega durante el procesado del zip en
            # vez de durante la descarga (ese caso concreto no está
            # capturado explícitamente más arriba).
            completed_shards.add(shard_path)
            completed_shards_path.write_text("\n".join(sorted(completed_shards)))
            if saved_rows:
                pd.DataFrame(saved_rows).drop_duplicates().to_csv(existing_metadata_path, index=False)

            if stopped_early:
                break

    if not saved_rows:
        print("\n⚠️  No se guardó ninguna imagen. Revisa los mensajes de arriba (columna/valores de país).")
        return

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
