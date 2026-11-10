"""Resumen de _cell_stats.csv del build de YFCC: enlaces rotos y fotos aceptadas por celda (cifras exactas).

    python yfcc_stats.py ..\\..\\data\\yfcc_spain\\_cell_stats.csv [--cap 400]

Sin --cap (por defecto 0) se asume que el build se hizo SIN tope, así que todas las cifras
son exactas sobre todas las fotos. Si el build se hizo con tope, pasa ese valor en --cap:
las celdas que lo alcanzaron se avisan como cota inferior (no se estima nada).
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("ruta", nargs="?", default=r"..\..\data\yfcc_spain\_cell_stats.csv")
ap.add_argument("--cap", type=int, default=0, help="tope usado en el build (0 = sin tope)")
a = ap.parse_args()

d = pd.read_csv(a.ruta)
# usuario_saturado se descarta antes de intentar descargar; el resto sí se intentó
intentos = ["sin_url", "fallo_descarga", "demasiado_pequena", "demasiado_grande", "casi_duplicada"]
d["intentadas"] = d["n_aceptadas"] + d[intentos].sum(axis=1)
d["tope"] = (d["n_aceptadas"] >= a.cap) if a.cap > 0 else False


def resumen(x, titulo):
    t = x["intentadas"].sum()
    print(f"\n== {titulo}: {len(x):,} celdas ==")
    print(f"candidatas (acc>=min): {x['n_vistas'].sum():,} | intentadas: {t:,} | aceptadas: {x['n_aceptadas'].sum():,}")
    if t:
        for k in intentos:
            print(f"  {k:18s} {x[k].sum():>10,}  ({x[k].sum() / t:6.1%} de las intentadas)")
        print(f"  tasa de aceptación: {x['n_aceptadas'].sum() / t:.1%}")


resumen(d, "Todas")
if a.cap > 0:
    resumen(d[~d["tope"]], f"Celdas SIN tope (<{a.cap} aceptadas, pool agotado)")
    resumen(d[d["tope"]], f"Celdas en tope (>= {a.cap}; cifras = cota inferior)")
    print(f"\nAVISO: el build usó tope {a.cap}; {d['tope'].sum():,} celdas lo alcanzaron y sus totales son cota inferior.")

base = d["n_aceptadas"].sum()
print(f"\nAceptadas (total real): {base:,}")

# Efecto de aplicar distintos topes por celda sobre lo realmente aceptado: sum(min(tope, aceptadas))
print("\n== Fotos resultantes con otros topes por celda ==")
print(f"{'tope':>7} | {'fotos':>10} | {'vs total':>12} | {'celdas recortadas':>17}")
for c in (400, 800, 1000, 1500, 2000, 5000, 10000, 20000, 50000, 100000):
    if a.cap > 0 and c > a.cap:
        break  # por encima del tope del build no hay datos reales
    tot = d["n_aceptadas"].clip(upper=c).sum()
    print(f"{c:>7} | {tot:>10,} | {tot - base:>12,} | {(d['n_aceptadas'] > c).sum():>17,}")
