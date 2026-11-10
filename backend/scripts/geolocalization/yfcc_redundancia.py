"""Mide cuánta redundancia (ráfagas / mismo usuario) hay en las celdas densas de YFCC.

Parte 1 (solo metadatos, rápido):  usa spain.tsv (uid + datetaken).
    python yfcc_redundancia.py ..\\..\\data\\yfcc\\spain.tsv

Parte 2 (vectores):  similitud entre vecinos dentro de celdas densas del índice ya
compilado (solo mira las fotos aceptadas, que ya pasaron el filtro de phash).
    python yfcc_redundancia.py ..\\..\\data\\yfcc\\spain.tsv --indice ..\\..\\data\\yfcc_spain
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("tsv")
ap.add_argument("--min-accuracy", type=int, default=12)
ap.add_argument("--umbral-densa", type=int, default=400, help="celdas con más candidatas que esto")
ap.add_argument("--sesion-min", type=float, default=30, help="hueco (min) que separa dos sesiones de un usuario")
ap.add_argument("--tasa", type=float, default=0.72,
                help="tasa de aceptación supuesta (descargas OK, tamaño, phash) para la tabla de topes")
ap.add_argument("--indice", help="carpeta con embeddings.npy + index_meta.csv (parte 2)")
ap.add_argument("--celdas", type=int, default=5, help="nº de celdas densas a analizar en la parte 2")
a = ap.parse_args()
TOPES = (400, 800, 1000, 1500, 2000, 5000, 10000, 20000, 50000, 100000, 1000000)


def celdas(lat, lon):
    dlat = 10 / 111.32                       # mismas celdas de ~10 km que yfcc_analyze.py
    ilat = np.floor(lat / dlat).astype(int)
    dlon = dlat / np.cos(np.radians((ilat + 0.5) * dlat))
    ilon = np.floor(lon / dlon).astype(int)
    return ilat.astype(str) + "_" + ilon.astype(str)


# ------------------------------------------------------------ parte 1
df = pd.read_csv(a.tsv, sep="\t", dtype=str,
                 usecols=["photoid", "uid", "accuracy", "datetaken", "latitude", "longitude"])
df["acc"] = pd.to_numeric(df["accuracy"], errors="coerce").fillna(0).astype(int)
df = df[df["acc"] >= a.min_accuracy].copy()
df["cell"] = celdas(pd.to_numeric(df["latitude"]), pd.to_numeric(df["longitude"]))
df["t"] = pd.to_datetime(df["datetaken"], errors="coerce")

df = df.sort_values(["cell", "uid", "t"])
gap = df.groupby(["cell", "uid"])["t"].diff()
# una foto abre sesión nueva si es la primera del usuario en la celda o si el hueco es grande
df["sesion_nueva"] = gap.isna() | (gap > pd.Timedelta(minutes=a.sesion_min))

g = df.groupby("cell")
res = pd.DataFrame({
    "fotos": g.size(),
    "usuarios": g["uid"].nunique(),
    "sesiones": g["sesion_nueva"].sum(),
})
res["fotos_por_sesion"] = res["fotos"] / res["sesiones"]
densas = res[res["fotos"] > a.umbral_densa]
resto = res[res["fotos"] <= a.umbral_densa]

print(f"Fotos con acc>={a.min_accuracy}: {len(df):,} en {len(res):,} celdas")
for nombre, x in (("Celdas densas", densas), ("Resto", resto)):
    print(f"\n== {nombre}: {len(x):,} celdas ==")
    print(f"fotos {x['fotos'].sum():,} | usuarios (suma por celda) {x['usuarios'].sum():,} | "
          f"sesiones {int(x['sesiones'].sum()):,}")
    print(f"fotos por sesión: {x['fotos'].sum() / max(1, x['sesiones'].sum()):.2f} | "
          f"fotos por usuario: {x['fotos'].sum() / max(1, x['usuarios'].sum()):.2f}")
    print(f"fotos que NO abren sesión (dentro de ráfaga/visita): "
          f"{(x['fotos'].sum() - x['sesiones'].sum()) / max(1, x['fotos'].sum()):.1%}")

print("\n== Contenido distinto en celdas densas, según el tope (sesiones como unidad) ==")
print(f"{'tope':>6} | {'fotos (min(fotos,tope))':>24} | {'sesiones (min(ses,tope))':>25}")
for c in TOPES:
    print(f"{c:>6} | {np.minimum(densas['fotos'], c).sum():>24,.0f} | "
          f"{np.minimum(densas['sesiones'], c).sum():>25,.0f}")

por_usuario = df.groupby(["cell", "uid"]).size()
print(f"\n== Fotos aceptadas estimadas según --cap-per-cell y --max-por-usuario (tasa {a.tasa:.0%}) ==")
ms = [0, 100, 50, 30, 20]
print(f"{'cap':>6} | " + " | ".join(f"{('sin tope' if m == 0 else 'max ' + str(m)):>10}" for m in ms))
for c in TOPES:
    fila = []
    for m in ms:
        pool = (por_usuario if m == 0 else por_usuario.clip(upper=m)).groupby(level="cell").sum()
        fila.append(np.minimum(pool * a.tasa, c).sum())
    print(f"{c:>6} | " + " | ".join(f"{v:>10,.0f}" for v in fila))

print("\nTop 10 celdas por fotos:")
print(densas.sort_values("fotos", ascending=False).head(10).to_string())

# ------------------------------------------------------------ parte 2
if a.indice:
    import os
    emb = np.load(os.path.join(a.indice, "embeddings.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(a.indice, "index_meta.csv"), low_memory=False)
    assert len(meta) == len(emb), "embeddings y meta no coinciden en tamaño"
    meta["cell"] = celdas(meta["lat"].astype(float), meta["lon"].astype(float))
    cuentas = meta["cell"].value_counts()
    en_indice = set(cuentas.index)
    # las más densas según el tsv (no según el índice, donde todas empatan en 400)
    grandes = [c for c in densas.sort_values("fotos", ascending=False).index
               if c in en_indice][:a.celdas]
    pequena = cuentas[(cuentas > 100) & (cuentas < 300)].index[:1].tolist()  # referencia
    print("\n== Similitud coseno al vecino más cercano dentro de la celda (fotos ya aceptadas) ==")
    print("(compara las celdas densas con una celda normal; los umbrales absolutos dependen del modelo)")
    print(f"{'celda':>14} | {'n':>5} | {'p50':>5} | {'p90':>5} | {'%>0.90':>7} | {'%>0.95':>7}")
    for c in grandes + pequena:
        idx = np.where(meta["cell"].values == c)[0]
        E = np.asarray(emb[idx], dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        S = E @ E.T
        np.fill_diagonal(S, -1)
        nn = S.max(axis=1)
        print(f"{c:>14} | {len(idx):>5} | {np.percentile(nn, 50):>5.2f} | {np.percentile(nn, 90):>5.2f} | "
              f"{(nn > .90).mean():>7.1%} | {(nn > .95).mean():>7.1%}")
