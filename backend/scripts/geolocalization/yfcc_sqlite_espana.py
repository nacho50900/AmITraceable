#!/usr/bin/env python3
"""
Saca las fotos de España del SQLite completo de YFCC100M
(yfcc100m_dataset.sql, 61 GiB) leyéndolo en paralelo.

Reparte la tabla en rangos de `photoid` (clave primaria, así cada rango se lee
sin recorrer el resto) y los procesa con varios procesos. Cada rango escribe su
propio fichero parcial; si se corta, relanza el mismo comando y se saltan los
rangos ya terminados. Al final une todo en spain_raw.tsv.

Uso (ajusta la ruta a tu USB):
    python yfcc_sqlite_espana.py E:\\yfcc100m_dataset.sql
    python yfcc_sqlite_espana.py E:\\yfcc100m_dataset.sql --procesos 4
    python yfcc_sqlite_espana.py E:\\yfcc100m_dataset.sql --prueba    # solo 3 rangos
    python yfcc_sqlite_espana.py --filtrar    # recorta spain_raw.tsv a España -> spain.tsv

Para --filtrar hace falta: pip install pandas geopandas pyogrio

Solo usa la librería estándar de Python (3.8+). El fichero se abre en solo
lectura: no se modifica.
"""
import argparse
import csv
import os
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CAJAS = [  # (lat_min, lat_max, lon_min, lon_max): península + Baleares, Canarias
    (35.1, 43.9, -9.5, 4.5),
    (27.5, 29.5, -18.3, -13.3),
]
COLUMNAS = ["photoid", "uid", "longitude", "latitude", "accuracy", "datetaken",
            "licensename", "licenseurl", "pageurl", "downloadurl", "ext", "marker"]

CONSULTA = f"""
SELECT {', '.join(COLUMNAS)}
FROM yfcc100m_dataset
WHERE photoid BETWEEN ? AND ?
  AND marker = 0
  AND length(longitude) > 0 AND length(latitude) > 0
  AND (
    (CAST(latitude AS REAL) BETWEEN {CAJAS[0][0]} AND {CAJAS[0][1]}
       AND CAST(longitude AS REAL) BETWEEN {CAJAS[0][2]} AND {CAJAS[0][3]})
    OR
    (CAST(latitude AS REAL) BETWEEN {CAJAS[1][0]} AND {CAJAS[1][1]}
       AND CAST(longitude AS REAL) BETWEEN {CAJAS[1][2]} AND {CAJAS[1][3]})
  )
"""


def conectar(ruta):
    uri = Path(os.path.abspath(ruta)).as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=60)
    con.execute("PRAGMA query_only = ON")
    con.execute("PRAGMA cache_size = -20000")  # ~20 MB por proceso
    return con


def extremos(ruta):
    con = conectar(ruta)
    # Dos consultas separadas: así SQLite usa la clave primaria (min()+max()
    # juntos obligan a recorrer toda la tabla).
    lo = con.execute("SELECT photoid FROM yfcc100m_dataset ORDER BY photoid LIMIT 1").fetchone()[0]
    hi = con.execute("SELECT photoid FROM yfcc100m_dataset ORDER BY photoid DESC LIMIT 1").fetchone()[0]
    con.close()
    return lo, hi


def limpio(v):
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ") if v is not None else ""


def procesar_rango(ruta, n, lo, hi, carpeta):
    final = os.path.join(carpeta, f"p{n:05d}.tsv")
    if os.path.exists(final):
        return n, 0, True
    con = conectar(ruta)
    filas = 0
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for fila in con.execute(CONSULTA, (lo, hi)):
            w.writerow([limpio(v) for v in fila])
            filas += 1
    con.close()
    os.replace(tmp, final)
    return n, filas, False


NE_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"


def filtrar(args):
    """Deja solo las fotos dentro del polígono de España (+ buffer)."""
    import pandas as pd          # import tardío: solo hace falta con --filtrar
    import geopandas as gpd

    if not os.path.exists(args.salida):
        sys.exit(f"No existe {args.salida}; ejecuta primero la extracción.")
    if not os.path.exists(args.limites):
        print(f"Descargando límites a {args.limites}...", file=sys.stderr)
        urllib.request.urlretrieve(NE_URL, args.limites)

    mundo = gpd.read_file(f"zip://{args.limites}")
    espana = mundo[mundo["ADMIN"] == "Spain"].to_crs(25830)   # metros
    espana["geometry"] = espana.buffer(args.buffer_m)
    espana = espana.to_crs(4326)[["geometry"]]

    total_in = total_out = 0
    primero = True
    inicio = time.time()
    for bloque in pd.read_csv(args.salida, sep="\t", dtype=str, chunksize=500_000):
        total_in += len(bloque)
        pts = gpd.GeoDataFrame(
            bloque,
            geometry=gpd.points_from_xy(bloque["longitude"].astype(float),
                                        bloque["latitude"].astype(float)),
            crs=4326)
        dentro = gpd.sjoin(pts, espana, predicate="within", how="inner")
        dentro = dentro.drop(columns=["geometry", "index_right"])
        dentro.to_csv(args.filtrada, sep="\t", index=False,
                      mode="w" if primero else "a", header=primero,
                      lineterminator="\n")
        primero = False
        total_out += len(dentro)
        print(f"\r{total_in:,} leídas | {total_out:,} en España | "
              f"{formatear(time.time() - inicio)}", end="", file=sys.stderr, flush=True)
    print(f"\nListo: {total_out:,} de {total_in:,} fotos en {args.filtrada}",
          file=sys.stderr)


def formatear(seg):
    seg = int(seg)
    return f"{seg // 3600}h{(seg % 3600) // 60:02d}m"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite", nargs="?", help="ruta al yfcc100m_dataset.sql")
    ap.add_argument("--procesos", type=int, default=6)
    ap.add_argument("--rangos", type=int, default=1000,
                    help="en cuántos rangos de photoid se divide la tabla")
    ap.add_argument("--carpeta", default="partes_yfcc")
    ap.add_argument("--salida", default="spain_raw.tsv")
    ap.add_argument("--prueba", action="store_true",
                    help="solo procesa 3 rangos (para medir velocidad)")
    ap.add_argument("--filtrar", action="store_true",
                    help="en vez de extraer, filtra spain_raw.tsv con el polígono de España")
    ap.add_argument("--limites", default="ne_10m_admin_0_countries.zip",
                    help="zip de Natural Earth (se descarga si no existe)")
    ap.add_argument("--buffer-m", type=int, default=2000,
                    help="buffer en metros alrededor de España (costa/imprecisión GPS)")
    ap.add_argument("--filtrada", default="spain.tsv")
    args = ap.parse_args()

    if args.filtrar:
        filtrar(args)
        return
    if not args.sqlite:
        ap.error("falta la ruta al SQLite (o usa --filtrar)")

    if not os.path.exists(args.sqlite):
        sys.exit(f"No existe {args.sqlite}")
    os.makedirs(args.carpeta, exist_ok=True)

    print("Buscando extremos de photoid...", file=sys.stderr)
    lo_total, hi_total = extremos(args.sqlite)
    paso = (hi_total - lo_total) // args.rangos + 1
    rangos = [(n, lo_total + n * paso, lo_total + (n + 1) * paso - 1)
              for n in range(args.rangos)]
    if args.prueba:
        # tres rangos repartidos (principio, mitad, final) para ver la dispersión
        rangos = [rangos[0], rangos[len(rangos) // 2], rangos[-1]]
    hechos_antes = sum(os.path.exists(os.path.join(args.carpeta, f"p{n:05d}.tsv"))
                       for n, _, _ in rangos)
    print(f"{len(rangos)} rangos ({hechos_antes} ya hechos), {args.procesos} procesos.",
          file=sys.stderr)

    inicio, hechos, guardadas = time.time(), 0, 0
    pendientes = len(rangos) - hechos_antes
    with ProcessPoolExecutor(args.procesos) as ex:
        futuros = [ex.submit(procesar_rango, args.sqlite, n, lo, hi, args.carpeta)
                   for n, lo, hi in rangos]
        for fut in as_completed(futuros):
            n, filas, saltado = fut.result()
            if saltado:
                continue
            hechos += 1
            guardadas += filas
            seg = time.time() - inicio
            eta = seg / hechos * (pendientes - hechos)
            print(f"\r{hechos}/{pendientes} rangos | España: {guardadas:,} | "
                  f"{formatear(seg)} transcurridas | ~{formatear(eta)} restantes",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    if args.prueba:
        print("Prueba terminada. Mira el tiempo por rango para estimar el total.",
              file=sys.stderr)
        return

    total = 0
    with open(args.salida, "w", encoding="utf-8", newline="") as out:
        out.write("\t".join(COLUMNAS) + "\n")
        for n, _, _ in rangos:
            p = os.path.join(args.carpeta, f"p{n:05d}.tsv")
            with open(p, encoding="utf-8", newline="") as f:
                for linea in f:
                    out.write(linea)
                    total += 1
    print(f"Listo: {total:,} fotos de España en {args.salida}", file=sys.stderr)


if __name__ == "__main__":
    main()
