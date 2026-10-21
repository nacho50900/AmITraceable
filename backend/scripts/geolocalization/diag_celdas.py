"""Diagnóstico de una carpeta de ingesta (commons_spain / mapillary_spain):
progreso por fila de latitud, celdas con fallo y estado de la celda de Gijón.

Uso (desde backend/scripts/geolocalization):
    python diag_celdas.py ../../data/commons_spain
"""
import sys
from pathlib import Path

import pandas as pd

from spain_grid import generate_spain_grid

out = Path(sys.argv[1] if len(sys.argv) > 1 else "../../data/commons_spain")
cells = generate_spain_grid(cell_km=10.0)

done_path = out / "_completed_cells.txt"
completed = set(done_path.read_text().splitlines()) if done_path.exists() else set()
stats_path = out / "_cell_stats.csv"
stats = pd.read_csv(stats_path).drop_duplicates("cell_id", keep="last").set_index("cell_id") if stats_path.exists() else pd.DataFrame()

print(f"{len(cells)} celdas en el grid, {len(completed)} completadas, {len(stats)} con estadísticas")

# --- progreso por fila (cada fila ~10 km de latitud, de sur a norte) ---
filas = {}
for c in cells:
    region, row, _col = c.id.rsplit("_", 2)
    f = filas.setdefault((region, int(row)), {"lat": c.min_lat, "celdas": 0, "completadas": 0, "con_fotos": 0, "fallo": 0, "aceptadas": 0})
    f["celdas"] += 1
    if c.id in completed:
        f["completadas"] += 1
    if c.id in stats.index:
        s = stats.loc[c.id]
        f["aceptadas"] += int(s["n_aceptadas"])
        f["con_fotos"] += int(s["n_aceptadas"] > 0)
        f["fallo"] += int(bool(s.get("fallo_sin_recuperar", False)))

print("\nregion               fila  lat    celdas  completadas  con_fotos  con_fallo  fotos")
for (region, row), f in sorted(filas.items()):
    if f["completadas"] == 0 and f["fallo"] == 0:
        marca = "  <- sin procesar"
    elif f["completadas"] < f["celdas"]:
        marca = "  <- incompleta"
    else:
        marca = ""
    print(f"{region:<20} {row:>4}  {f['lat']:.2f}  {f['celdas']:>6}  {f['completadas']:>11}  {f['con_fotos']:>9}  {f['fallo']:>9}  {f['aceptadas']:>5}{marca}")

# --- celda de Gijón ---
lat, lon = 43.5322, -5.6611
for c in cells:
    if c.min_lat <= lat < c.max_lat and c.min_lon <= lon < c.max_lon:
        print(f"\nGijón -> celda {c.id}: completada={c.id in completed}")
        if c.id in stats.index:
            print(stats.loc[c.id].to_string())
        else:
            print("sin estadísticas: la celda nunca se ha procesado")
        break

# --- latitudes de las fotos realmente guardadas ---
metas = [out / "index_meta.csv"] + sorted((out / "_shards").glob("shard_*_meta.csv"))
lats = pd.concat([pd.read_csv(m, usecols=["lat"]) for m in metas if m.exists()], ignore_index=True)["lat"]
print(f"\nFotos guardadas: {len(lats)}; latitud máx: {lats.max():.2f}; al norte de 42.5: {(lats > 42.5).sum()}")
