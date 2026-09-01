"""
Combina varios índices FAISS (cada uno en su propia carpeta, con
index.faiss + index_meta.csv -- p.ej. data/osv5m_spain/ y
data/flickr_spain/) en un único índice final.

Por qué un script aparte en vez de que build_flickr_index.py escriba
directamente sobre data/osv5m_spain/: mantener cada fuente en su propia
carpeta permite reconstruir el índice combinado sin volver a descargar ni
re-blurear nada si cambias qué fuentes incluir (p.ej. si más adelante
subes el umbral de accuracy de Flickr y solo quieres rehacer esa parte) --
mismo principio que ya sigue scripts/recover_metadata.py: separar "datos
crudos ya obtenidos" de "índice derivado, siempre regenerable".

Uso:
    python merge_faiss_indices.py \
        --sources ../data/osv5m_spain ../data/flickr_spain \
        --output ../data/spain_combined

Tras ejecutarlo, apunta _INDEX_DIR en app/vision/geolocation.py a
--output (o copia/symlink el resultado sobre data/osv5m_spain/ si
prefieres no tocar código).
"""
import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def _load_source(source_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    meta_path = source_dir / "index_meta.csv"
    embeddings_path = source_dir / "embeddings.npy"
    index_path = source_dir / "index.faiss"

    if not meta_path.exists():
        raise RuntimeError(f"{meta_path} no existe -- ¿carpeta de fuente correcta?")

    meta = pd.read_csv(meta_path)

    if embeddings_path.exists():
        # Fuentes como build_flickr_index.py guardan los vectores en crudo
        # aparte del índice; no hace falta reconstruirlos.
        vectors = np.load(embeddings_path)
    else:
        # Fuentes como build_faiss_index.py (OSV-5M) no guardan un .npy
        # aparte -- se reconstruyen desde el propio índice FAISS. Esto solo
        # funciona porque es un IndexFlatIP (guarda los vectores originales
        # sin pérdida); con un índice cuantizado tipo IVFPQ no sería posible
        # recuperarlos exactos.
        if not index_path.exists():
            raise RuntimeError(f"{source_dir} no tiene ni embeddings.npy ni index.faiss.")
        index = faiss.read_index(str(index_path))
        vectors = index.reconstruct_n(0, index.ntotal)

    if len(meta) != len(vectors):
        raise RuntimeError(
            f"{source_dir}: index_meta.csv tiene {len(meta)} filas pero el índice tiene "
            f"{len(vectors)} vectores -- no coinciden, revisa la fuente antes de fusionar."
        )
    return vectors, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_vectors, all_meta = [], []
    for source in args.sources:
        source_dir = Path(source)
        print(f"Cargando {source_dir}...")
        vectors, meta = _load_source(source_dir)
        print(f"  {len(meta)} vectores.")
        all_vectors.append(vectors.astype("float32"))
        all_meta.append(meta)

    combined_vectors = np.vstack(all_vectors)
    combined_meta = pd.concat(all_meta, ignore_index=True)

    # Filtra duplicados por id (p.ej. si se fusiona dos veces por error)
    # manteniendo vectors y meta alineados por posición.
    keep_mask = ~combined_meta.duplicated(subset="id")
    n_dropped = (~keep_mask).sum()
    if n_dropped:
        print(f"Aviso: {n_dropped} ids duplicados entre fuentes, se han descartado las copias repetidas.")
    combined_vectors = combined_vectors[keep_mask.to_numpy()]
    combined_meta = combined_meta[keep_mask].reset_index(drop=True)

    dimension = combined_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(combined_vectors)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    combined_meta.to_csv(output_dir / "index_meta.csv", index=False)
    np.save(output_dir / "embeddings.npy", combined_vectors)

    print(f"\nÍndice combinado: {index.ntotal} vectores de dimensión {dimension} en {output_dir}")
    if "source" in combined_meta.columns:
        print(combined_meta["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
