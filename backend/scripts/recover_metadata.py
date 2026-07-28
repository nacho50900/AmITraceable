"""
Reconstruye metadata.csv a partir de las imágenes YA DESCARGADAS y el CSV
completo de metadatos, para arreglar un dataset descargado con una versión
de download_osv5m_spain.py o download_osv5m_world.py que tenía el bug que
perdía la columna `id` al guardar metadata.csv (usaba el id como índice
del DataFrame y luego guardaba con to_csv(index=False), descartándolo --
ver comentarios en esos scripts, ya corregidos).

NO vuelve a descargar ninguna IMAGEN: usa las que ya tienes en --output.
Para el id/lat/lon de cada una, primero busca la caché local
(_<split>_metadata_cache.csv dentro de --output, que ambos scripts
guardan la primera vez que descargan los metadatos completos) y, si no la
encuentra, descarga el CSV de metadatos directamente de HuggingFace (train
y/o test, ~3GB en total, pero es solo texto -- nada comparable al peso de
volver a descargar las imágenes).

Uso:
    python recover_metadata.py --output ../data/osv5m_spain
    python recover_metadata.py --output ../data/osv5m_spain --splits train
"""
import argparse
from pathlib import Path

import pandas as pd

REPO_ID = "osv5m/osv5m"
REPO_TYPE = "dataset"


def _load_metadata(split: str, output_dir: Path) -> pd.DataFrame:
    """Misma lógica de caché que _load_spain_ids/_load_all_ids en los
    scripts de descarga: si ya existe una copia local, se usa esa (no hay
    red de por medio); si no, se descarga el CSV completo del split desde
    HuggingFace y se guarda una copia local para la próxima vez."""
    cache_path = output_dir / f"_{split}_metadata_cache.csv"
    if cache_path.exists():
        print(f"Usando caché local {cache_path.name}.")
        return pd.read_csv(cache_path)

    from huggingface_hub import hf_hub_download

    print(f"No hay caché local para '{split}'; descargando {split}.csv de HuggingFace...")
    csv_path = hf_hub_download(repo_id=REPO_ID, filename=f"{split}.csv", repo_type=REPO_TYPE)
    metadata = pd.read_csv(csv_path)
    metadata.to_csv(cache_path, index=False)
    print(f"Guardado en {cache_path.name} para no tener que volver a descargarlo.")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../data/osv5m_spain")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"], choices=["train", "test"],
        help="Qué splits de metadatos probar (por defecto ambos, igual que en la descarga original)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.csv"

    if not images_dir.exists():
        raise SystemExit(f"No existe {images_dir}.")

    downloaded_ids = {p.stem for p in images_dir.glob("*.jpg")}
    print(f"{len(downloaded_ids)} imágenes .jpg ya descargadas en {images_dir}.")
    if not downloaded_ids:
        raise SystemExit(f"No se encontró ninguna imagen .jpg en {images_dir}.")

    recovered_frames = []
    id_col_name = None
    for split in args.splits:
        cache = _load_metadata(split, output_dir)
        id_col = next((c for c in cache.columns if c.lower() == "id"), None)
        if id_col is None:
            print(f"  ⚠️  Los metadatos de '{split}' no tienen columna 'id' reconocible, se omite.")
            continue
        id_col_name = id_col_name or id_col
        cache[id_col] = cache[id_col].astype(str)
        matched = cache[cache[id_col].isin(downloaded_ids)]
        print(f"  {len(matched)} filas de '{split}' casan con imágenes ya descargadas.")
        recovered_frames.append(matched)

    if not recovered_frames:
        raise SystemExit("No se pudo recuperar ninguna fila. Revisa los mensajes de arriba.")

    recovered = pd.concat(recovered_frames, ignore_index=True).drop_duplicates(subset=[id_col_name])

    missing = downloaded_ids - set(recovered[id_col_name])
    if missing:
        print(
            f"\n⚠️  {len(missing)} imágenes descargadas no aparecen en los metadatos consultados "
            "(se quedarán sin fila -- build_faiss_index.py simplemente las ignorará, no rompe nada, "
            "pero no aportarán al índice)."
        )

    if metadata_path.exists():
        backup_path = metadata_path.with_suffix(".csv.roto.bak")
        metadata_path.rename(backup_path)
        print(f"\nEl metadata.csv anterior (roto) se ha guardado como {backup_path}, por si acaso.")

    recovered.to_csv(metadata_path, index=False)
    print(f"\nListo. {len(recovered)} filas recuperadas en {metadata_path}.")
    print("Ya puedes correr build_faiss_index.py con normalidad.")


if __name__ == "__main__":
    main()
