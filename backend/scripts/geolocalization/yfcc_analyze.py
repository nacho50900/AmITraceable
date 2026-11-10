import sys
import numpy as np
import pandas as pd

ruta = sys.argv[1] if len(sys.argv) > 1 else r"..\..\data\yfcc\spain.tsv"
df = pd.read_csv(ruta, sep="\t", dtype=str,
                 usecols=["accuracy", "latitude", "longitude"])
lat = pd.to_numeric(df["latitude"])
lon = pd.to_numeric(df["longitude"])
df["acc"] = pd.to_numeric(df["accuracy"], errors="coerce").fillna(0).astype(int)

dlat = 10 / 111.32
ilat = np.floor(lat / dlat).astype(int)
dlon = dlat / np.cos(np.radians((ilat + 0.5) * dlat))
ilon = np.floor(lon / dlon).astype(int)
df["cell"] = ilat.astype(str) + "_" + ilon.astype(str)

print("== Concentración en coordenadas repetidas, por nivel ==")
for lvl in (11, 12, 13, 14):
    sub = df[df["acc"] == lvl]
    top = sub.groupby(["latitude", "longitude"]).size().sort_values(ascending=False)
    print(f"nivel {lvl}: {len(sub):,} fotos | {len(top):,} coordenadas distintas | "
          f"top-50 coordenadas = {top.head(50).sum():,} fotos")

print("\n== Celdas y fotos efectivas (cap 400) por umbral ==")
for t in (11, 12, 13, 14, 16):
    n = df[df["acc"] >= t].groupby("cell").size()
    print(f"acc>={t}: {len(n):,} celdas | {(n < 50).sum():,} celdas con <50 fotos | "
          f"tope de fotos indexables = {np.minimum(n, 400).sum():,}")

print("\n== Distribución completa de accuracy ==")
acc_raw = pd.to_numeric(df["accuracy"], errors="coerce")
print(acc_raw.value_counts(dropna=False).sort_index().to_string())