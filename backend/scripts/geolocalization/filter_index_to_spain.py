"""
Elimina de un índice por shards (build_mapillary_index.py / build_commons_index.py)
las fotos cuyas coordenadas caen fuera de territorio español.

Por qué existe: hasta que se añadió --spain-only, el grid base (spain_grid.py)
cubría un rectángulo que incluye Portugal, el sur de Francia, Andorra y el norte
de Marruecos, así que Mapillary pudo ingerir fotos de allí -- y build_mapillary_index
les asignó la provincia española más cercana (campo `region`).

Uso (desde backend/scripts/geolocalization):
    python filter_index_to_spain.py                  # simulacro: solo informa
    python filter_index_to_spain.py --apply          # reescribe los shards
    python filter_index_to_spain.py --output ../../data/otro_indice --apply

Tras --apply hay que recompilar: python build_mapillary_index.py --compile

Seguridad:
- Por defecto NO escribe nada (simulacro).
- Usa el mismo .lock que build_mapillary_index.py: no se puede lanzar mientras
  el script de ingestión esté corriendo sobre el mismo --output.
- Cada shard se reescribe de forma atómica (tmp + os.replace), vectores y
  metadata a la vez, manteniendo el orden de las filas que se conservan.
- Las filas eliminadas se guardan (solo metadata, sin vectores) en
  _removed_outside_spain.csv dentro de --output, para poder auditar.
- No toca el embeddings.npy/index_meta.csv/index.faiss de la raíz (salida de
  --compile): bórralos o vuelve a compilar después.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shard_store  # noqa: E402
from spain_grid import SpainLand  # noqa: E402


def _inside_mask(land: SpainLand, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """True para cada punto a menos de land.margin_deg de algún municipio.
    Consulta vectorizada: una sola llamada al STRtree por shard."""
    shapely = land._shapely
    m = land.margin_deg
    boxes = shapely.box(lons - m, lats - m, lons + m, lats + m)
    hit_boxes = land._tree.query(boxes, predicate="intersects")[0]
    mask = np.zeros(len(lats), dtype=bool)
    mask[hit_boxes] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="../../data/mapillary_spain")
    parser.add_argument("--apply", action="store_true", help="Reescribe los shards (por defecto solo simula).")
    args = parser.parse_args()

    output_dir = Path(args.output)
    shards_dir = output_dir / shard_store.SHARDS_SUBDIR
    shards = shard_store.shard_files(shards_dir)
    if not shards:
        print(f"No hay shards en {shards_dir}.")
        return
    if (output_dir / "index_meta.csv").exists():
        print("Aviso: hay index_meta.csv en la raíz (salida de --compile o fichero antiguo). Este script solo "
              "limpia los shards; recompila o bórralo después.")

    lock_path = shard_store.acquire_lock(output_dir) if args.apply else None
    try:
        land = SpainLand()
        total = removed_total = shards_touched = 0
        removed_frames = []
        for npy_path, meta_path, idx in shards:
            meta = pd.read_csv(meta_path)
            if not npy_path.exists():
                print(f"Error: falta {npy_path.name} para {meta_path.name}. Nada más se procesa.")
                sys.exit(1)
            n = len(meta)
            total += n
            mask = _inside_mask(land, meta["lat"].to_numpy(dtype=float), meta["lon"].to_numpy(dtype=float))
            n_out = int((~mask).sum())
            if n_out == 0:
                continue
            removed_total += n_out
            shards_touched += 1
            removed_frames.append(meta[~mask].assign(shard=idx))

            if not args.apply:
                continue
            vectors = np.load(npy_path)
            if len(vectors) != n:
                print(f"Error: shard {idx} tiene {len(vectors)} vectores y {n} filas de metadata. Se aborta.")
                sys.exit(1)
            kept_meta, kept_vectors = meta[mask], vectors[mask]
            pid = f".tmp{os.getpid()}"
            if len(kept_meta) == 0:
                npy_path.unlink()
                meta_path.unlink()
                continue
            npy_tmp, meta_tmp = npy_path.with_name(npy_path.name + pid), meta_path.with_name(meta_path.name + pid)
            with open(npy_tmp, "wb") as f:
                np.save(f, kept_vectors.astype("float32"))
            kept_meta.to_csv(meta_tmp, index=False)
            os.replace(npy_tmp, npy_path)
            os.replace(meta_tmp, meta_path)

        print(f"{total} fotos revisadas en {len(shards)} shards; fuera de España: {removed_total} "
              f"({100 * removed_total / max(total, 1):.2f}%), en {shards_touched} shards.")
        if removed_frames:
            removed = pd.concat(removed_frames, ignore_index=True)
            if "region" in removed.columns:
                print("Provincias a las que estaban mal asignadas (top 10):")
                print(removed["region"].value_counts().head(10).to_string())
            if args.apply:
                removed.to_csv(output_dir / "_removed_outside_spain.csv", index=False)
                print(f"Eliminadas. Metadata de lo eliminado en {output_dir / '_removed_outside_spain.csv'}.")
                print("Siguiente paso: python build_mapillary_index.py --compile")
            else:
                print("Simulacro: no se ha modificado nada. Repite con --apply para aplicar.")
    finally:
        if lock_path is not None:
            shard_store.release_lock(lock_path)


if __name__ == "__main__":
    main()
