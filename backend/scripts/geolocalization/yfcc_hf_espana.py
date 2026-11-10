#!/usr/bin/env python3
"""
Saca las fotos de España del subconjunto YFCC100M de Hugging Face
(dalle-mini/YFCC100M_OpenAI_subset, ~14,8 M de fotos, 1,9 TB) SIN descargarlo
entero. Dos pasos, los dos reanudables:

  1) metadatos : lee los ~4.095 ficheros metadata_XXX.jsonl.gz (pequeños), filtra
                 España por coordenadas y guarda una tabla spain_meta.tsv.
  2) imagenes  : para esas filas, abre cada XXX.zip por HTTP con peticiones
                 Range (lee solo el índice del zip y los ficheros que pides) y
                 guarda cada foto en imagenes/<key[:3]>/<key>.jpg.

Uso:
    pip install requests
    python yfcc_hf_espana.py metadatos --shards 001        # prueba con 1 trozo
    python yfcc_hf_espana.py metadatos                      # todos (reanudable)
    python yfcc_hf_espana.py imagenes --max-por-shard 50    # prueba pequeña
    python yfcc_hf_espana.py imagenes                       # todas (reanudable)

Si se corta, vuelve a lanzar el MISMO comando: los trozos de metadatos ya
terminados y las imágenes ya guardadas se saltan.
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("Falta la librería requests:  pip install requests")

BASE_DEFECTO = "https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset/resolve/main"
# Mismos trozos que el script de carga del dataset: 001..fff salvo ac8.
TODOS_LOS_SHARDS = [f"{i:03x}" for i in range(1, 4096) if i != 0xAC8]

CAJAS = [  # (lat_min, lat_max, lon_min, lon_max): península + Baleares, Canarias
    (35.1, 43.9, -9.5, 4.5),
    (27.5, 29.5, -18.3, -13.3),
]
COLUMNAS = ["shard", "key", "photoid", "uid", "longitude", "latitude", "accuracy",
            "datetaken", "licensename", "licenseurl", "pageurl", "downloadurl", "ext"]
UA = "yfcc_hf_espana/1.0 (script academico; TFG)"
CODIGOS_REINTENTABLES = (429, 500, 502, 503, 504)


# ----------------------------------------------------------------- red
def nueva_sesion():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def get(sesion, url, headers=None, intentos=6):
    """GET con reintentos y espera creciente ante cortes o 429/5xx."""
    ultimo = None
    for n in range(intentos):
        try:
            r = sesion.get(url, headers=headers, timeout=90)
            if r.status_code in CODIGOS_REINTENTABLES:
                ultimo = f"HTTP {r.status_code}"
            else:
                return r
        except (requests.ConnectionError, requests.Timeout) as e:
            ultimo = f"{type(e).__name__}"
        time.sleep(min(60, 2 ** (n + 1)))
    raise RuntimeError(f"{url}: demasiados fallos ({ultimo})")


class HTTPRangeFile(io.RawIOBase):
    """Fichero remoto de solo lectura con seek(), apoyado en peticiones Range."""
    BLOQUE = 256 * 1024
    MAX_BLOQUES_EN_CACHE = 64

    def __init__(self, sesion, url):
        super().__init__()
        self.sesion, self.url, self.pos, self.cache = sesion, url, 0, {}
        r = get(sesion, url, headers={"Range": "bytes=0-0"})
        if r.status_code == 404:
            raise FileNotFoundError(url)
        if r.status_code != 206:
            raise RuntimeError(f"{url}: el servidor no admite Range (HTTP {r.status_code})")
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _bloque(self, i):
        if i not in self.cache:
            if len(self.cache) >= self.MAX_BLOQUES_EN_CACHE:
                self.cache.pop(next(iter(self.cache)))
            ini = i * self.BLOQUE
            fin = min(self.size, ini + self.BLOQUE) - 1
            r = get(self.sesion, self.url, headers={"Range": f"bytes={ini}-{fin}"})
            if r.status_code != 206 or len(r.content) != fin - ini + 1:
                raise RuntimeError(f"{self.url}: respuesta Range inesperada ({r.status_code})")
            self.cache[i] = r.content
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        partes, fin = [], self.pos + n
        while self.pos < fin:
            i, dentro = divmod(self.pos, self.BLOQUE)
            trozo = self._bloque(i)[dentro:dentro + (fin - self.pos)]
            partes.append(trozo)
            self.pos += len(trozo)
        return b"".join(partes)


# ------------------------------------------------------------- filtros
def a_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def es_foto_de_espana(d):
    marker = a_float(d.get("marker"))
    if marker is not None and marker != 0:
        return False
    lat, lon = a_float(d.get("latitude")), a_float(d.get("longitude"))
    if lat is None or lon is None:
        return False
    return any(a <= lat <= b and c <= lon <= e for a, b, c, e in CAJAS)


def limpio(v):
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")


# ---------------------------------------------------------- paso 1
def filas_de_metadatos(base, idx):
    sesion = nueva_sesion()
    r = get(sesion, f"{base}/metadata/metadata_{idx}.jsonl.gz")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    filas = []
    with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            d = json.loads(linea)
            if es_foto_de_espana(d):
                filas.append([idx] + [limpio(d.get(c, "")) for c in COLUMNAS[1:]])
    return filas


def cmd_metadatos(args):
    shards = args.shards.split(",") if args.shards else TODOS_LOS_SHARDS
    hechos = set()
    if os.path.exists(args.hechos):
        hechos = set(open(args.hechos).read().split())
    pendientes = [s for s in shards if s not in hechos]
    nuevo = not os.path.exists(args.tabla)
    out = open(args.tabla, "a", encoding="utf-8", newline="")
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    if nuevo:
        w.writerow(COLUMNAS)
    cp = open(args.hechos, "a")
    lock = threading.Lock()
    total = len(shards)
    print(f"{len(hechos)} trozos ya hechos, {len(pendientes)} pendientes.", file=sys.stderr)
    guardadas, inicio, fallos = 0, time.time(), []
    with ThreadPoolExecutor(args.hilos) as ex:
        futuros = {ex.submit(filas_de_metadatos, args.base, s): s for s in pendientes}
        for n, fut in enumerate(as_completed(futuros), 1):
            s = futuros[fut]
            try:
                filas = fut.result()
            except Exception as e:
                fallos.append(s)
                print(f"\n[fallo] {s}: {e}", file=sys.stderr)
                continue
            with lock:
                if filas is None:
                    print(f"\n[aviso] no existe metadata_{s}", file=sys.stderr)
                else:
                    w.writerows(filas)
                    guardadas += len(filas)
                out.flush()
                cp.write(s + "\n")
                cp.flush()
            seg = time.time() - inicio
            eta = seg / n * (len(pendientes) - n)
            print(f"\r{n}/{len(pendientes)} trozos | España: {guardadas:,} | "
                  f"~{int(eta // 60)} min restantes", end="", file=sys.stderr, flush=True)
    print(f"\nListo. {guardadas:,} fotos nuevas en {args.tabla}. "
          f"Trozos con fallo (relanza para reintentarlos): {fallos or 'ninguno'}",
          file=sys.stderr)


# ---------------------------------------------------------- paso 2
def extraer_de_shard(base, idx, filas, destino):
    sesion = nueva_sesion()
    guardadas = faltan = 0
    pendientes = [f for f in filas
                  if not os.path.exists(os.path.join(destino, f["key"][:3], f["key"] + ".jpg"))]
    if not pendientes:
        return idx, 0, 0, len(filas)
    rf = HTTPRangeFile(sesion, f"{base}/data/{idx}.zip")
    with zipfile.ZipFile(rf) as z:
        for f in pendientes:
            k = f["key"]
            try:
                datos = z.read(f"data/images/{k[:3]}/{k[3:6]}/{k}.jpg")
            except KeyError:
                faltan += 1
                continue
            carpeta = os.path.join(destino, k[:3])
            os.makedirs(carpeta, exist_ok=True)
            final = os.path.join(carpeta, k + ".jpg")
            with open(final + ".tmp", "wb") as o:
                o.write(datos)
            os.replace(final + ".tmp", final)
            guardadas += 1
    return idx, guardadas, faltan, len(filas) - len(pendientes)


def cmd_imagenes(args):
    por_shard = {}
    with open(args.tabla, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f, delimiter="\t"):
            acc = a_float(fila["accuracy"])
            if args.precision_min and (acc is None or acc < args.precision_min):
                continue
            if any(x.lower() in fila["licensename"].lower() for x in args.excluir_licencias):
                continue
            por_shard.setdefault(fila["shard"], []).append(fila)
    if args.shards:
        por_shard = {s: v for s, v in por_shard.items() if s in args.shards.split(",")}
    if args.max_por_shard:
        por_shard = {s: v[:args.max_por_shard] for s, v in por_shard.items()}
    total_filas = sum(len(v) for v in por_shard.values())
    print(f"{total_filas:,} fotos en {len(por_shard)} trozos.", file=sys.stderr)
    nuevas = faltan = ya = 0
    inicio = time.time()
    with ThreadPoolExecutor(args.hilos) as ex:
        futuros = {ex.submit(extraer_de_shard, args.base, s, v, args.destino): s
                   for s, v in por_shard.items()}
        for n, fut in enumerate(as_completed(futuros), 1):
            s = futuros[fut]
            try:
                _, g, fa, y = fut.result()
                nuevas += g
                faltan += fa
                ya += y
            except Exception as e:
                print(f"\n[fallo] {s}: {e}", file=sys.stderr)
            seg = time.time() - inicio
            eta = seg / n * (len(por_shard) - n)
            print(f"\r{n}/{len(por_shard)} zips | nuevas {nuevas:,} | ya estaban {ya:,} | "
                  f"no encontradas {faltan:,} | ~{int(eta // 60)} min restantes",
                  end="", file=sys.stderr, flush=True)
    print("\nListo.", file=sys.stderr)


def cmd_diagnostico(args):
    """Descarga un trozo de metadatos y cuenta qué contiene (para depurar)."""
    idx = args.shard
    url = f"{args.base}/metadata/metadata_{idx}.jsonl.gz"
    r = get(nueva_sesion(), url)
    print(f"GET {url}\n  HTTP {r.status_code}, {len(r.content):,} bytes, "
          f"primeros bytes: {r.content[:4].hex()}")
    r.raise_for_status()
    filas = []
    with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                filas.append(json.loads(linea))
    print(f"  filas: {len(filas):,}")
    if not filas:
        return
    print("  columnas del primer registro:")
    for k, v in filas[0].items():
        print(f"    {k!r}: {str(v)[:60]!r} ({type(v).__name__})")
    con_geo = [d for d in filas
               if a_float(d.get('latitude')) is not None and a_float(d.get('longitude')) is not None]
    con_marker0 = [d for d in filas if a_float(d.get('marker')) == 0]
    print(f"  con latitud y longitud numéricas: {len(con_geo):,}")
    print(f"  con marker == 0: {len(con_marker0):,}")
    print(f"  que pasan el filtro de España: {sum(es_foto_de_espana(d) for d in filas):,}")
    if con_geo:
        lats = [a_float(d['latitude']) for d in con_geo]
        lons = [a_float(d['longitude']) for d in con_geo]
        print(f"  rango latitud {min(lats):.2f}..{max(lats):.2f}, "
              f"longitud {min(lons):.2f}..{max(lons):.2f}")
        print("  3 ejemplos (lat, lon, marker):",
              [(d['latitude'], d['longitude'], d.get('marker')) for d in con_geo[:3]])
    else:
        print("  valores crudos de latitude/longitude en las 5 primeras filas:",
              [(d.get('latitude'), d.get('longitude')) for d in filas[:5]])


def main():
    ap = argparse.ArgumentParser(description="YFCC100M (HF) solo España")
    ap.add_argument("--base", default=BASE_DEFECTO)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("metadatos")
    m.add_argument("--shards", help="lista separada por comas, p. ej. 001,002 (por defecto todos)")
    m.add_argument("--tabla", default="spain_meta.tsv")
    m.add_argument("--hechos", default="shards_hechos.txt")
    m.add_argument("--hilos", type=int, default=8)
    m.set_defaults(fn=cmd_metadatos)

    i = sub.add_parser("imagenes")
    i.add_argument("--shards")
    i.add_argument("--tabla", default="spain_meta.tsv")
    i.add_argument("--destino", default="imagenes")
    i.add_argument("--hilos", type=int, default=4)
    i.add_argument("--max-por-shard", type=int, default=0)
    i.add_argument("--precision-min", type=int, default=0,
                   help="descarta fotos con 'accuracy' menor (16 = calle, 12 = ciudad...)")
    i.add_argument("--excluir-licencias", nargs="*", default=[],
                   help='p. ej. --excluir-licencias NoDerivs')
    i.set_defaults(fn=cmd_imagenes)

    d = sub.add_parser("diagnostico", help="inspecciona un trozo de metadatos")
    d.add_argument("--shard", default="001")
    d.set_defaults(fn=cmd_diagnostico)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
