"""Selección por novedad sobre el build de YFCC (sin tope): se descartan las fotos que son
PARECIDAS (coseno de embeddings >= tau) Y ESTÁN CERCA (<= radius-m) de otra ya elegida en su celda.
Una foto se conserva si aporta aspecto nuevo o lugar nuevo.

    # 1) calibrar tau sobre una muestra de celdas (no escribe nada)
    python select_diverse.py ..\\..\\data\\yfcc_spain_sincap --sweep 0.80,0.85,0.90,0.95 --sample-cells 200

    # 2) generar el índice reducido en una carpeta nueva
    python select_diverse.py ..\\..\\data\\yfcc_spain_sincap --output ..\\..\\data\\yfcc_spain_div --tau 0.90 --radius-m 150 --compile

Lee los shards (y embeddings.npy/index_meta.csv si ya compilaste) con mmap, procesa celda a celda y
nunca modifica la carpeta de entrada. Dentro de cada celda se prioriza la foto de mayor precisión GPS
(desempate aleatorio con --seed).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shard_store import SHARDS_SUBDIR, acquire_lock, compile_shards, persist_new_shard, release_lock, shard_files

KM_PER_DEG_LAT = 111.32  # mismo valor que build_yfcc_index.py
M_PER_DEG = KM_PER_DEG_LAT * 1000


def cell_ids(lat: np.ndarray, lon: np.ndarray, cell_km: float) -> np.ndarray:
    """Misma fórmula de celdas que build_yfcc_index._load_candidates."""
    dlat = cell_km / KM_PER_DEG_LAT
    ilat = np.floor(lat / dlat).astype(int)
    dlon = dlat / np.cos(np.radians((ilat + 0.5) * dlat))
    ilon = np.floor(lon / dlon).astype(int)
    return np.array([f"yfcc_{a}_{b}" for a, b in zip(ilat, ilon)])


def load_meta(inp: Path):
    sources = []
    legacy_npy, legacy_meta = inp / "embeddings.npy", inp / "index_meta.csv"
    if legacy_npy.exists() and legacy_meta.exists():
        sources.append((legacy_npy, legacy_meta))
    for npy, meta, _ in shard_files(inp / SHARDS_SUBDIR):
        sources.append((npy, meta))
    if not sources:
        raise SystemExit(f"No hay shards ni embeddings.npy en {inp}")
    metas = []
    for k, (_, meta) in enumerate(sources):
        m = pd.read_csv(meta)
        m["_src"] = k
        m["_row"] = np.arange(len(m))
        metas.append(m)
    meta = pd.concat(metas, ignore_index=True)
    n0 = len(meta)
    meta = meta.drop_duplicates("id", keep="first").reset_index(drop=True)  # por si hay compilado + shards
    if len(meta) < n0:
        print(f"Aviso: {n0 - len(meta):,} ids repetidos entre fichero compilado y shards; se ignoran.")
    mmaps = [np.load(npy, mmap_mode="r") for npy, _ in sources]
    return meta, mmaps


def load_embeddings(g: pd.DataFrame, mmaps) -> np.ndarray:
    dim = mmaps[0].shape[1]
    out = np.empty((len(g), dim), np.float32)
    src = g["_src"].to_numpy()
    row = g["_row"].to_numpy()
    for k in np.unique(src):
        m = src == k
        out[m] = mmaps[k][row[m]]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)  # coseno = producto escalar
    return out


def select(E, lat, lon, acc, tau, radius_m, cap, rng, block=512):
    """Devuelve los índices (de E) de las fotos elegidas. Voraz: se recorre por precisión desc.;
    una foto es redundante si existe una ya elegida con coseno >= tau y distancia <= radius_m."""
    n = len(E)
    order = np.lexsort((rng.random(n), -acc))
    E, lat, lon = E[order], lat[order], lon[order]
    x = lon * M_PER_DEG * np.cos(np.radians(lat.mean()))
    y = lat * M_PER_DEG
    r2 = np.inf if radius_m <= 0 else radius_m ** 2

    K = np.empty_like(E)
    kx = np.empty(n)
    ky = np.empty(n)
    kept = []
    nk = 0
    for s in range(0, n, block):
        if cap and nk >= cap:
            break
        e = E[s:s + block]
        b = len(e)
        redundant = np.zeros(b, bool)
        if nk:
            ii, jj = np.nonzero(e @ K[:nk].T >= tau)
            if len(ii):
                d2 = (x[s + ii] - kx[jj]) ** 2 + (y[s + ii] - ky[jj]) ** 2
                redundant[ii[d2 <= r2]] = True
        idx = np.nonzero(~redundant)[0]
        if not len(idx):
            continue
        near = (e[idx] @ e[idx].T >= tau)
        bx, by = x[s + idx], y[s + idx]
        near &= ((bx[:, None] - bx[None, :]) ** 2 + (by[:, None] - by[None, :]) ** 2) <= r2
        loc = []
        for a in range(len(idx)):
            if loc and near[a, loc].any():
                continue
            loc.append(a)
        if cap:
            loc = loc[:cap - nk]
        sel = s + idx[loc]
        K[nk:nk + len(sel)] = E[sel]
        kx[nk:nk + len(sel)] = x[sel]
        ky[nk:nk + len(sel)] = y[sel]
        nk += len(sel)
        kept.extend(sel.tolist())
    return order[np.array(kept, dtype=int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="carpeta del build sin tope (con _shards)")
    ap.add_argument("--output", default=None, help="carpeta NUEVA para el índice reducido")
    ap.add_argument("--tau", type=float, default=0.90, help="umbral de similitud coseno")
    ap.add_argument("--radius-m", type=float, default=150.0, help="radio en metros (0 = solo similitud)")
    ap.add_argument("--max-per-cell", type=int, default=0, help="tope duro tras la selección (0 = ninguno)")
    ap.add_argument("--cell-km", type=float, default=10.0)
    ap.add_argument("--sweep", default=None, help="lista de tau separados por comas: solo informa, no escribe")
    ap.add_argument("--sample-cells", type=int, default=200, help="celdas usadas en --sweep (mitad más grandes, mitad al azar)")
    ap.add_argument("--shard-every-cells", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true", help="compilar el resultado a embeddings.npy/index.faiss/index_meta.csv")
    a = ap.parse_args()

    inp = Path(a.input)
    meta, mmaps = load_meta(inp)
    meta["_cell"] = cell_ids(meta["lat"].to_numpy(), meta["lon"].to_numpy(), a.cell_km)
    groups = {c: g for c, g in meta.groupby("_cell", sort=True)}
    sizes = pd.Series({c: len(g) for c, g in groups.items()}).sort_values(ascending=False)
    print(f"{len(meta):,} fotos en {len(groups):,} celdas (mayor: {sizes.index[0]} con {sizes.iloc[0]:,}).")
    rng = np.random.default_rng(a.seed)
    acc_col = "accuracy" if "accuracy" in meta.columns else None

    def run_cell(g, tau):
        E = load_embeddings(g, mmaps)
        acc = g[acc_col].to_numpy(float) if acc_col else np.zeros(len(g))
        return select(E, g["lat"].to_numpy(float), g["lon"].to_numpy(float), acc, tau, a.radius_m, a.max_per_cell, rng), E

    if a.sweep:
        taus = [float(t) for t in a.sweep.split(",")]
        half = a.sample_cells // 2
        rest = sizes.index[half:]
        pick = list(sizes.index[:half]) + list(rng.choice(rest, size=min(half, len(rest)), replace=False))
        total = sum(len(groups[c]) for c in pick)
        print(f"\nMuestra: {len(pick)} celdas, {total:,} fotos | radio {a.radius_m:g} m")
        print(f"{'tau':>6} | {'conservadas':>11} | {'% conservado':>12}")
        for tau in taus:
            kept = 0
            for c in pick:
                idx, _ = run_cell(groups[c], tau)
                kept += len(idx)
            print(f"{tau:>6.2f} | {kept:>11,} | {kept / total:>12.1%}")
        return

    if not a.output:
        raise SystemExit("Falta --output (o usa --sweep para calibrar).")
    out = Path(a.output)
    if out.resolve() == inp.resolve():
        raise SystemExit("--output debe ser una carpeta distinta de la de entrada.")
    if shard_files(out / SHARDS_SUBDIR) or (out / "index_meta.csv").exists():
        raise SystemExit(f"{out} ya contiene un índice; usa una carpeta nueva.")

    lock = acquire_lock(out)
    try:
        t0 = time.time()
        embs, rows, stats = [], [], []
        shard_idx, done = 1, 0
        helper = ["_src", "_row", "_cell"]
        for c in sizes.index.sort_values():
            g = groups[c]
            idx, E = run_cell(g, a.tau)
            embs.append(E[idx])
            rows.extend(g.iloc[idx].drop(columns=helper).to_dict("records"))
            stats.append({"cell_id": c, "n_antes": len(g), "n_despues": len(idx)})
            done += 1
            if done % a.shard_every_cells == 0:
                persist_new_shard(out, shard_idx, embs, rows)
                shard_idx += 1
                embs, rows = [], []
                kept = sum(s["n_despues"] for s in stats)
                print(f"  {done}/{len(groups)} celdas | {kept:,} fotos conservadas | {time.time() - t0:.0f}s")
        persist_new_shard(out, shard_idx, embs, rows)
        pd.DataFrame(stats).to_csv(out / "_selection_stats.csv", index=False)
        antes = sum(s["n_antes"] for s in stats)
        despues = sum(s["n_despues"] for s in stats)
        print(f"\nListo: {despues:,} de {antes:,} fotos ({despues / antes:.1%}) | tau {a.tau:g}, radio {a.radius_m:g} m")
    finally:
        release_lock(lock)
    if a.compile:
        compile_shards(out)
    else:
        print(f"Para compilar: python select_diverse.py ... --compile, o el --compile del build sobre {out}")


if __name__ == "__main__":
    main()
