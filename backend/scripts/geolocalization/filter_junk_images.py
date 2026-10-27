"""
Marca (y opcionalmente elimina) del índice combinado las imágenes "basura" que
no dicen nada del lugar -- parabrisas rayado, salpicadero, capó, reflejos... --
partiendo de unas pocas imágenes semilla que tú identificas a mano.

Cómo funciona: se promedian los embeddings de las semillas en un "prototipo" y
se marca cada vector del índice cuyo parecido (coseno) con ese prototipo supere
un umbral. No pasa ninguna imagen por ningún modelo: solo usa los vectores que
ya están guardados, así que sobre 1,7 M de vectores tarda segundos.

Flujo:
  1. Con nearest_images.py saca los vecinos de una foto problemática y apunta
     los `id` de los que son basura (uno por línea en un .txt; '#' = comentario).
  2. Simulación (no escribe nada): cuántas imágenes se marcarían a cada umbral
     y muestras con enlace para abrirlas y decidir el umbral:
         python filter_junk_images.py --index ../../data/spain_combined --seed-file semillas.txt
  3. Aplicar con el umbral elegido, a una carpeta NUEVA (el índice original no se toca):
         python filter_junk_images.py --index ../../data/spain_combined --seed-file semillas.txt \\
             --threshold 0.85 --apply --output ../../data/spain_combined_clean

Si vuelves a fusionar con merge_faiss_indices.py, las imágenes basura vuelven:
este script va después de la fusión.
"""
import argparse
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from nearest_images import image_link

_TABLE_STEP = 0.05
_CHUNK_ROWS = 200_000
_SAMPLE_SIZE = 8
_REGION_TABLE_ROWS = 15
_BORDERLINE_MARGIN = 0.03


def read_seed_ids(seed_file: Path) -> list[str]:
    ids = []
    for line in seed_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    if not ids:
        raise SystemExit(f"{seed_file} no contiene ningún id.")
    return ids


def load_vectors(index_dir: Path) -> np.ndarray:
    """Vectores del índice, alineados por posición con index_meta.csv. Usa
    embeddings.npy (en modo mmap, sin cargarlo entero) o, si no existe,
    los reconstruye del IndexFlatIP como hace merge_faiss_indices.py."""
    embeddings_path = index_dir / "embeddings.npy"
    if embeddings_path.exists():
        return np.load(embeddings_path, mmap_mode="r")
    index = faiss.read_index(str(index_dir / "index.faiss"))
    return index.reconstruct_n(0, index.ntotal)


def build_prototype(vectors: np.ndarray, meta: pd.DataFrame, seed_ids: list[str]) -> tuple[np.ndarray, float]:
    """Prototipo (media normalizada) de las semillas y la cohesión del grupo
    (coseno medio de cada semilla con el prototipo; si es baja, las semillas
    son de tipos distintos y conviene separarlas en varias ejecuciones)."""
    positions = meta.index[meta["id"].isin(seed_ids)].to_numpy()
    missing = set(seed_ids) - set(meta.loc[positions, "id"])
    if missing:
        print(f"Aviso: {len(missing)} semillas no están en el índice: {sorted(missing)[:5]}...")
    if len(positions) == 0:
        raise SystemExit("Ninguna semilla está en el índice.")
    seeds = np.asarray(vectors[positions], dtype="float32")
    prototype = seeds.mean(axis=0)
    prototype /= np.linalg.norm(prototype) + 1e-8
    cohesion = float(np.mean(seeds @ prototype))
    print(f"{len(positions)} semillas usadas; cohesión {cohesion:.3f}")
    return prototype, cohesion


def similarity_to_prototype(vectors: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Coseno de cada vector con el prototipo, por bloques para no cargar todo en RAM."""
    similarities = np.empty(len(vectors), dtype="float32")
    for start in range(0, len(vectors), _CHUNK_ROWS):
        block = np.asarray(vectors[start : start + _CHUNK_ROWS], dtype="float32")
        similarities[start : start + len(block)] = block @ prototype
    return similarities


def print_threshold_table(similarities: np.ndarray, threshold: float) -> None:
    """Recuento de imágenes marcadas de arriba abajo, de 0.05 en 0.05, hasta llegar al umbral dado."""
    print("\nImágenes que se marcarían según el umbral:")
    levels = [threshold + step * _TABLE_STEP for step in range(int(round((1.0 - threshold) / _TABLE_STEP)) + 1)]
    for level in reversed([level for level in levels if level < 1.0]):
        count = int((similarities >= level).sum())
        print(f"  >= {level:.2f}: {count:>8}  ({100 * count / len(similarities):.2f} %)")


def print_samples(meta: pd.DataFrame, similarities: np.ndarray, threshold: float) -> None:
    """Muestras con enlace para revisar a mano: dudosas (justo por encima del
    umbral) y claras (muy por encima). Si las dudosas ya no son basura, sube el umbral."""
    rng = random.Random(0)
    bands = (
        ("Dudosas (justo por encima del umbral)", threshold, threshold + _BORDERLINE_MARGIN),
        ("Claras (muy por encima)", threshold + _BORDERLINE_MARGIN, 1.01),
    )
    for title, low, high in bands:
        rows = np.flatnonzero((similarities >= low) & (similarities < high))
        print(f"\n{title}: {len(rows)} imágenes con similitud en [{low:.2f}, {min(high, 1.0):.2f})")
        for row in rng.sample(list(rows), min(_SAMPLE_SIZE, len(rows))):
            print(f"  {similarities[row]:.3f}  {image_link(meta.at[row, 'id'])[1]}")


def print_region_table(meta: pd.DataFrame, flagged: np.ndarray) -> None:
    """Reparto de las imágenes marcadas por región, para ver si se abre un hueco de
    cobertura en alguna zona (las regiones de OSV-5M son comunidades y las de
    Commons/Mapillary provincias, así que no se suman entre sí)."""
    regions = meta["region"].fillna("desconocido")
    table = pd.DataFrame({"marcadas": regions[flagged].value_counts(), "total": regions.value_counts()})
    table = table.dropna(subset=["marcadas"]).astype({"marcadas": int}).sort_values("marcadas", ascending=False)
    if table.empty:
        return
    table["quedan"] = table["total"] - table["marcadas"]
    table["pct"] = 100 * table["marcadas"] / table["total"]

    print(f"\nReparto de las marcadas por región ({len(table)} regiones afectadas):")
    print(f"  {'región':<28}{'marcadas':>10}{'total región':>14}{'% perdido':>11}{'quedan':>9}")
    for region, row in table.head(_REGION_TABLE_ROWS).iterrows():
        print(f"  {region:<28}{int(row.marcadas):>10}{int(row.total):>14}{row.pct:>10.1f}%{int(row.quedan):>9}")
    if len(table) > _REGION_TABLE_ROWS:
        rest = table.iloc[_REGION_TABLE_ROWS:]
        print(f"  ... y {len(rest)} regiones más ({int(rest.marcadas.sum())} marcadas en total)")


def write_filtered_index(index_dir: Path, output_dir: Path, vectors: np.ndarray, keep: np.ndarray) -> None:
    full_meta = pd.read_csv(index_dir / "index_meta.csv")
    kept_vectors = np.ascontiguousarray(np.asarray(vectors)[keep], dtype="float32")
    kept_meta = full_meta[keep].reset_index(drop=True)

    index = faiss.IndexFlatIP(kept_vectors.shape[1])
    index.add(kept_vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    kept_meta.to_csv(output_dir / "index_meta.csv", index=False)
    np.save(output_dir / "embeddings.npy", kept_vectors)
    print(f"\nÍndice filtrado: {index.ntotal} vectores en {output_dir} (se eliminaron {int((~keep).sum())}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Marca/elimina del índice las imágenes parecidas a unas semillas basura.")
    parser.add_argument("--index", type=Path, required=True, help="Carpeta con index.faiss, index_meta.csv y embeddings.npy.")
    parser.add_argument("--seed-file", type=Path, required=True, help="Texto con un id de imagen basura por línea.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Coseno mínimo con el prototipo para marcar (por defecto 0.85).")
    parser.add_argument("--apply", action="store_true", help="Escribe el índice filtrado en --output (sin esto, solo simula).")
    parser.add_argument("--output", type=Path, help="Carpeta NUEVA para el índice filtrado (obligatoria con --apply).")
    args = parser.parse_args()

    if args.apply:
        if args.output is None:
            raise SystemExit("--apply requiere --output.")
        if args.output.resolve() == args.index.resolve():
            raise SystemExit("--output no puede ser la carpeta del índice original: se perdería el índice sin filtrar.")

    meta = pd.read_csv(args.index / "index_meta.csv", usecols=["id", "region"], dtype={"id": str})
    vectors = load_vectors(args.index)
    if len(meta) != len(vectors):
        raise SystemExit(f"index_meta.csv tiene {len(meta)} filas y los vectores {len(vectors)}: no coinciden.")

    prototype, _ = build_prototype(vectors, meta, read_seed_ids(args.seed_file))
    similarities = similarity_to_prototype(vectors, prototype)

    print_threshold_table(similarities, args.threshold)
    print_samples(meta, similarities, args.threshold)

    flagged = similarities >= args.threshold
    print_region_table(meta, flagged)
    print(f"\nCon umbral {args.threshold:.2f}: {int(flagged.sum())} imágenes marcadas de {len(similarities)}.")
    if not args.apply:
        print("Simulación: no se ha escrito nada. Añade --apply --output <carpeta> para generar el índice filtrado.")
        return
    write_filtered_index(args.index, args.output, vectors, ~flagged)


if __name__ == "__main__":
    main()
