"""
Persistencia por shards + lock de ejecución + compilación final,
compartido entre todos los scripts de ingestión de fuentes externas
(build_commons_index.py, build_mapillary_index.py, y los que se añadan
después). Extraído de build_commons_index.py para no duplicar esta
lógica -- ya pasó por varias rondas de fixes reales (ver docstrings de
cada función) y conviene que todos los scripts de ingestión se
beneficien de los mismos arreglos en vez de mantener copias que puedan
divergir.

Diseño en resumen (ver docstrings individuales para el porqué de cada
pieza):
- Cada flush escribe un SHARD propio y pequeño, nunca relee ni reescribe
  el histórico acumulado -- el consumo de memoria del proceso no crece
  con el tamaño total del índice.
- Toda escritura es atómica (tmp + os.replace) -- una interrupción a
  mitad nunca deja un fichero vacío ni a medias.
- Un .lock evita que dos instancias corran a la vez sobre el mismo
  --output.
- --compile combina todo (shards + un posible fichero "antiguo" de
  antes de este diseño) en el embeddings.npy/index.faiss/index_meta.csv
  de siempre, para que lo consuma merge_faiss_indices.py o la app.
"""
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

SHARDS_SUBDIR = "_shards"


def shard_files(shards_dir: Path) -> list[tuple[Path, Path, int]]:
    """Devuelve (ruta_npy, ruta_csv, índice) de cada shard cuyo par de
    ficheros existe, ordenados por índice."""
    result = []
    if not shards_dir.exists():
        return result
    for meta_path in sorted(shards_dir.glob("shard_*_meta.csv")):
        try:
            idx = int(meta_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        result.append((shards_dir / f"shard_{idx:06d}_embeddings.npy", meta_path, idx))
    return result


def next_shard_index(shards_dir: Path) -> int:
    indices = [idx for _, _, idx in shard_files(shards_dir)]
    return (max(indices) + 1) if indices else 1


def load_existing_state(output_dir: Path):
    """Carga solo lo que hace falta mantener en memoria durante TODA la
    ejecución: known_ids (para el dedup al reabrir celdas), cuántas fotos
    hay ya en total (para los informes), qué celdas están completadas, y
    las estadísticas por celda (una fila por celda -- acotado por el
    tamaño del grid, no por el número de fotos).

    A propósito, NO devuelve los embeddings ni las filas de metadata
    completas para que el llamador las guarde en memoria durante toda la
    ejecución -- eso es justo lo que causó un
    `numpy._core._exceptions._ArrayMemoryError` real tras varias horas
    seguidas de ejecución (la lista de embeddings en memoria no paraba
    de crecer durante toda la vida del proceso). Un primer fix (vaciar
    la lista tras cada flush pero seguir fusionando con TODO el
    histórico en cada flush) ayudó pero no fue suficiente: esa
    recombinación también llegó a fallar por falta de memoria en una
    máquina con recursos muy limitados (un fallo real: no se pudo
    reservar ni 84MB para fusionar el histórico en un flush). Por eso
    las fotos se guardan en SHARDS pequeños e independientes (ver
    `persist_new_shard`) -- cargar el estado aquí solo necesita leer los
    metadatos (pequeños) de cada shard, nunca los vectores en sí ni el
    histórico completo de golpe.

    Compatibilidad: si existe un `embeddings.npy`/`index_meta.csv` "de
    toldo" en la raíz de --output (de antes de este diseño de shards),
    se lee una vez aquí como si fuera un shard más -- de solo lectura,
    nunca se vuelve a escribir -- para no perder lo ya acumulado ni
    obligar a borrar y empezar de cero otra vez.
    """
    shards_dir = output_dir / SHARDS_SUBDIR
    completed_path = output_dir / "_completed_cells.txt"
    stats_path = output_dir / "_cell_stats.csv"

    known_ids: set[str] = set()
    total_photos = 0

    legacy_meta_path = output_dir / "index_meta.csv"
    legacy_embeddings_path = output_dir / "embeddings.npy"
    if legacy_meta_path.exists():
        legacy_meta_df = pd.read_csv(legacy_meta_path)
        n_rows = len(legacy_meta_df)
        if "id" in legacy_meta_df.columns:
            known_ids.update(legacy_meta_df["id"])
        del legacy_meta_df

        n_vectors = np.load(legacy_embeddings_path, mmap_mode="r").shape[0] if legacy_embeddings_path.exists() else 0
        if n_vectors != n_rows:
            print(f"Error: embeddings.npy (fichero antiguo, pre-shards) tiene {n_vectors} vectores pero "
                  f"index_meta.csv tiene {n_rows} filas -- no coinciden.")
            print(f"Revisa manualmente {output_dir} antes de continuar.")
            sys.exit(1)
        total_photos += n_rows

    for npy_path, meta_path, idx in shard_files(shards_dir):
        meta_df = pd.read_csv(meta_path)
        n_rows = len(meta_df)
        if "id" in meta_df.columns:
            known_ids.update(meta_df["id"])
        del meta_df

        if not npy_path.exists():
            print(f"Error: el shard {idx} tiene {meta_path.name} pero falta {npy_path.name} -- par incompleto.")
            print(f"Revisa manualmente {shards_dir} antes de continuar.")
            sys.exit(1)
        n_vectors = np.load(npy_path, mmap_mode="r").shape[0]
        if n_vectors != n_rows:
            print(f"Error: el shard {idx} tiene {n_vectors} vectores pero {n_rows} filas de metadata -- no coinciden.")
            print(f"Revisa manualmente {meta_path} / {npy_path} antes de continuar.")
            sys.exit(1)
        total_photos += n_rows

    completed = set(completed_path.read_text().splitlines()) if completed_path.exists() else set()
    cell_stats = pd.read_csv(stats_path).to_dict("records") if stats_path.exists() else []

    if completed:
        n_shards = len(shard_files(shards_dir))
        print(f"Reanudando: {len(completed)} celdas y {total_photos} fotos ya procesadas ({n_shards} shards"
              f"{' + fichero antiguo' if legacy_meta_path.exists() else ''}).")

    return known_ids, total_photos, completed, cell_stats


def persist_new_shard(output_dir: Path, shard_index: int, new_embeddings: list, new_meta_rows: list) -> None:
    """Escribe un shard NUEVO -- su propio embeddings.npy + meta.csv,
    nunca toca ni relee ningún shard anterior ni el fichero antiguo. Si
    no hay fotos nuevas no crea nada (evita shards vacíos, p.ej. tras una
    celda sin resultados). Atómico por fichero (tmp + os.replace)."""
    if not new_meta_rows:
        return
    shards_dir = output_dir / SHARDS_SUBDIR
    shards_dir.mkdir(parents=True, exist_ok=True)
    pid_suffix = f".tmp{os.getpid()}"

    new_matrix = np.vstack(new_embeddings).astype("float32")
    npy_final = shards_dir / f"shard_{shard_index:06d}_embeddings.npy"
    meta_final = shards_dir / f"shard_{shard_index:06d}_meta.csv"

    npy_tmp = shards_dir / f"shard_{shard_index:06d}_embeddings.npy{pid_suffix}"
    with open(npy_tmp, "wb") as f:
        np.save(f, new_matrix)
    del new_matrix

    meta_tmp = shards_dir / f"shard_{shard_index:06d}_meta.csv{pid_suffix}"
    pd.DataFrame(new_meta_rows).to_csv(meta_tmp, index=False)

    os.replace(npy_tmp, npy_final)
    os.replace(meta_tmp, meta_final)


def persist_progress(output_dir: Path, completed: set, cell_stats: list) -> None:
    """_completed_cells.txt / _cell_stats.csv -- pequeños de verdad
    (acotados por el número de celdas del grid, no por el número de
    fotos), así que reescribirlos enteros cada vez sigue siendo seguro y
    barato. Atómico igual que el resto."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pid_suffix = f".tmp{os.getpid()}"

    completed_tmp = output_dir / f"_completed_cells.txt{pid_suffix}"
    completed_tmp.write_text("\n".join(sorted(completed)))

    stats_tmp = output_dir / f"_cell_stats.csv{pid_suffix}"
    pd.DataFrame(cell_stats).to_csv(stats_tmp, index=False)

    os.replace(completed_tmp, output_dir / "_completed_cells.txt")
    os.replace(stats_tmp, output_dir / "_cell_stats.csv")


def compile_shards(output_dir: Path) -> None:
    """Combina el fichero antiguo (si existe) + todos los shards en un
    único embeddings.npy/index_meta.csv/index.faiss en la raíz de
    --output -- el formato que espera merge_faiss_indices.py y que usa
    la app. A diferencia de un flush normal durante la ingestión, aquí SÍ
    se carga todo en memoria de golpe -- es una operación puntual que
    lanzas tú cuando quieras (--compile), no algo que tenga que sobrevivir
    sin fallos durante toda la ingestión, así que puedes cerrar otros
    programas antes de lanzarla si hace falta.
    """
    shards_dir = output_dir / SHARDS_SUBDIR
    legacy_meta_path = output_dir / "index_meta.csv"
    legacy_embeddings_path = output_dir / "embeddings.npy"

    all_matrices = []
    all_dfs = []

    if legacy_meta_path.exists():
        all_dfs.append(pd.read_csv(legacy_meta_path))
        with open(legacy_embeddings_path, "rb") as f:
            all_matrices.append(np.load(f))

    shards = shard_files(shards_dir)
    if not shards and not legacy_meta_path.exists():
        print(f"No hay nada que compilar en {output_dir} (ni fichero antiguo ni shards).")
        return

    for npy_path, meta_path, idx in shards:
        all_dfs.append(pd.read_csv(meta_path))
        with open(npy_path, "rb") as f:
            all_matrices.append(np.load(f))

    combined_matrix = np.concatenate(all_matrices, axis=0).astype("float32")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    del all_matrices, all_dfs

    if len(combined_matrix) != len(combined_df):
        print(f"Error interno: tras combinar, {len(combined_matrix)} vectores vs {len(combined_df)} filas -- no coinciden. Nada escrito.")
        sys.exit(1)

    pid_suffix = f".tmp{os.getpid()}"
    emb_tmp = output_dir / f"embeddings.npy{pid_suffix}"
    with open(emb_tmp, "wb") as f:
        np.save(f, combined_matrix)

    dimension = combined_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(combined_matrix)
    index_tmp = output_dir / f"index.faiss{pid_suffix}"
    faiss.write_index(index, str(index_tmp))

    meta_tmp = output_dir / f"index_meta.csv{pid_suffix}"
    combined_df.to_csv(meta_tmp, index=False)

    os.replace(emb_tmp, output_dir / "embeddings.npy")
    os.replace(index_tmp, output_dir / "index.faiss")
    os.replace(meta_tmp, output_dir / "index_meta.csv")

    print(f"Compilado: {len(combined_df)} fotos en {output_dir}/embeddings.npy + index.faiss + index_meta.csv")
    print(f"(los shards en {shards_dir} NO se han borrado -- bórralos a mano si ya no los necesitas.)")


def acquire_lock(output_dir: Path) -> Path:
    """Evita que dos instancias del script corran a la vez sobre el mismo
    --output. Sin esto, dos procesos escribiendo los mismos ficheros al
    mismo tiempo pueden pisarse -- visto en un caso real: una segunda
    ejecución lanzada por accidente sobre la misma carpeta leyó
    index_meta.csv justo cuando la primera lo tenía truncado a medio
    escribir, y crasheó con `pandas.errors.EmptyDataError`.

    No detecta automáticamente si el proceso dueño del lock sigue vivo
    (complicaría el script para un caso de uso de TFG en un único
    equipo) -- si el lock queda huérfano tras un corte de luz o un kill
    -9, hay que borrar el fichero .lock a mano. El mensaje de error dice
    esto explícitamente para que no haga falta adivinarlo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".lock"
    if lock_path.exists():
        owner = lock_path.read_text().strip()
        print(f"Error: ya existe {lock_path} (creado por el proceso PID {owner}).")
        print(f"Esto significa que ya hay OTRA ejecución de este script usando --output {output_dir}.")
        print("Lanzar dos instancias a la vez sobre la misma carpeta corrompe los ficheros del índice")
        print("(dos escrituras simultáneas se pisan entre sí).")
        print("Si estás seguro de que NO hay ninguna otra instancia corriendo (p.ej. el lock quedó huérfano")
        print(f"tras un cierre inesperado), borra {lock_path} a mano y vuelve a intentarlo.")
        sys.exit(1)
    lock_path.write_text(str(os.getpid()))
    return lock_path


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass
