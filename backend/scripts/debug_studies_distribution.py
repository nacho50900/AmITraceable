#!/usr/bin/env python3
"""
Script de DIAGNÓSTICO, aislado del proyecto AmITraceable, para investigar
si STUDIES_DISTRIBUTION (matriculados universitarios por "campo de
estudio" -- Ministerio de Ciencia, Innovación y Universidades, sistema
EDUCAbase/SIIU) tiene algún mecanismo de acceso automatizado, igual que
`update_ine_reference.py` ya tiene para las otras 6 tablas.

POR QUÉ EXISTE ESTE SCRIPT APARTE: Claude no pudo confirmar esto desde su
propio entorno de trabajo -- el portal que sirve estos datos
(estadisticas.ciencia.gob.es) bloquea el acceso automatizado desde ahí
(robots.txt), así que todo lo que hay hasta ahora sobre esta fuente es
"encontrado por búsqueda web", no "probado de verdad" como sí lo están
las otras 6 tablas (aunque sea contra datos simulados, no la API real).
Este script prueba varias hipótesis, UNA POR UNA, con log explícito de
cada intento -- para que el output completo se pueda pegar de vuelta y
decidir cuál (si alguna) funciona.

Es de SOLO LECTURA / DIAGNÓSTICO -- no escribe nada en ine_reference.py
ni en ningún otro fichero del proyecto, no depende de nada del repo.

REQUIERE: pip install httpx --break-system-packages   (o sin el flag,
según tu entorno)

USO:
    python3 debug_studies_distribution.py > output_studies.txt 2>&1

Pega TODO el contenido de output_studies.txt de vuelta, no solo la
última parte -- cada INTENTO por separado ya dice si funcionó o no, pero
los primeros intentos (aunque fallen) sirven para descartar hipótesis.
"""

from __future__ import annotations

import sys

try:
    import httpx
except ImportError:
    print("Falta el paquete 'httpx'. Instálalo con:")
    print("    pip install httpx --break-system-packages")
    sys.exit(1)

TIMEOUT = 20


def _print_header(titulo: str) -> None:
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)


def _probe(nombre: str, url: str, params: dict | None = None) -> httpx.Response | None:
    """Hace un GET y muestra status, content-type, tamaño y un fragmento
    del cuerpo -- NO asume que la petición vaya a funcionar, solo informa
    de lo que pasó de verdad. Un 404/403 aquí no es un fallo del script,
    es información real sobre si esa URL concreta existe."""
    print(f"\n--- {nombre} ---")
    print(f"URL: {url}")
    if params:
        print(f"params: {params}")
    try:
        resp = httpx.get(
            url,
            params=params,
            timeout=TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AmITraceable-TFG-research/1.0)"},
        )
    except httpx.HTTPError as e:
        print(f"FALLO DE RED (no llegó respuesta): {e!r}")
        return None
    print(f"Status: {resp.status_code}")
    print(f"URL final (tras redirecciones): {resp.url}")
    print(f"Content-Type: {resp.headers.get('content-type')}")
    print(f"Tamaño del cuerpo: {len(resp.content)} bytes")
    fragmento = resp.text[:1000] if resp.text else "(cuerpo vacío)"
    print(f"Primeros 1000 caracteres del cuerpo:\n{fragmento}")
    return resp


# ============================================================================
# INTENTO 1: preguntar a datos.gob.es, que SÍ tiene una API JSON pública
# documentada y confirmada (https://datos.gob.es/es/accessible-apidata),
# por el dataset real de "Matriculados ... campo de estudio". La idea es
# que datos.gob.es nos dé las URLs de distribución REALES (CSV/JSON/
# PC-Axis) del propio Ministerio, en vez de que nosotros las adivinemos.
# Esto es mejor punto de partida que adivinar URLs de
# estadisticas.ciencia.gob.es a ciegas, porque datos.gob.es SÍ es una API
# confirmada por su propia documentación oficial.
# ============================================================================
_print_header("INTENTO 1: buscar el dataset real en datos.gob.es (API confirmada)")

resp1 = _probe(
    "Búsqueda de datasets con 'Matriculados' en el título",
    "https://datos.gob.es/apidata/catalog/dataset",
    params={"title": "Matriculados", "_pageSize": 50},
)

candidatos_distribucion: list[tuple[str, str]] = []  # (titulo_dataset, url_distribucion)

if resp1 is not None and resp1.status_code == 200:
    try:
        data = resp1.json()
        items = data.get("result", {}).get("items", [])
        print(f"\n{len(items)} datasets encontrados con 'Matriculados' en el título (de todos los organismos).")
        for item in items:
            titulo_raw = item.get("title")
            if isinstance(titulo_raw, dict):
                titulo = titulo_raw.get("_value", "")
            elif isinstance(titulo_raw, list) and titulo_raw:
                titulo = titulo_raw[0].get("_value", "") if isinstance(titulo_raw[0], dict) else str(titulo_raw[0])
            else:
                titulo = str(titulo_raw or "")

            if "campo de estudio" not in titulo.lower():
                continue

            print(f"\n  DATASET CANDIDATO ENCONTRADO: {titulo}")
            distribuciones = item.get("distribution", [])
            if isinstance(distribuciones, dict):
                distribuciones = [distribuciones]
            for dist in distribuciones:
                url_dist = dist.get("accessURL") or dist.get("downloadURL") or ""
                if isinstance(url_dist, dict):
                    url_dist = url_dist.get("_value", "")
                formato = dist.get("format", "")
                if isinstance(formato, dict):
                    formato = formato.get("_value", formato)
                print(f"    formato={formato}  url={url_dist}")
                if url_dist:
                    candidatos_distribucion.append((titulo, url_dist))
    except Exception as e:
        print(f"\nNo se pudo interpretar la respuesta de datos.gob.es como JSON esperado: {e!r}")
        print("(la petición SÍ llegó -- ver el status/cuerpo de arriba para ver qué devolvió de verdad)")
else:
    print("\nLa búsqueda en datos.gob.es no dio 200 OK -- ver el detalle de arriba.")

if not candidatos_distribucion:
    print(
        "\nNo se encontró ningún dataset con 'Matriculados' + 'campo de "
        "estudio' en el título vía este endpoint concreto -- puede que el "
        "título real use otras palabras. Si INTENTO 1 no dio nada útil, "
        "prueba a repetir esta búsqueda a mano en "
        "https://datos.gob.es/es/catalogo/conjuntos-datos?q=matriculados"
        " (organismo: Ministerio de Ciencia) y pega aquí el título EXACTO "
        "para ajustar el filtro de este script."
    )


# ============================================================================
# INTENTO 2: si INTENTO 1 dio alguna URL de distribución candidata,
# probarlas directamente -- puede que ya sea un CSV o JSON descargable
# sin más, sin necesitar ninguna API.
# ============================================================================
_print_header("INTENTO 2: descargar directamente las distribuciones candidatas de INTENTO 1")

if not candidatos_distribucion:
    print("(nada que probar -- INTENTO 1 no dio ninguna URL de distribución)")
else:
    for titulo, url_dist in candidatos_distribucion[:5]:  # como mucho 5, para no saturar el output
        _probe(f"Distribución de: {titulo}", url_dist)


# ============================================================================
# INTENTO 3: probar si estadisticas.ciencia.gob.es acepta el MISMO
# patrón de API JSON que servicios.ine.es (mismo software "dynPx" que
# usa el propio INE en su navegador de tablas PC-Axis -- ver comentario
# de _TABLA_HOGAR_TIPO en update_ine_reference.py) -- long shot, pero
# gratis de probar: si el motor es literalmente el mismo, podría exponer
# un endpoint JSON parecido en su propio dominio.
# ============================================================================
_print_header("INTENTO 3: patrón de API JSON tipo INE (wstempus) bajo el dominio del Ministerio")

_probe(
    "wstempus-style bajo estadisticas.ciencia.gob.es",
    "https://estadisticas.ciencia.gob.es/wstempus/js/ES/DATOS_TABLA/1",
)
_probe(
    "wstempus-style bajo servicios.ciencia.gob.es",
    "https://servicios.ciencia.gob.es/wstempus/js/ES/DATOS_TABLA/1",
)


# ============================================================================
# INTENTO 4: probar el patrón de API tipo PXWeb (usado por varios
# institutos estadísticos regionales españoles, p. ej. IDESCAT/ISTAC) --
# si EDUCAbase resulta correr sobre PXWeb en vez del motor propio del
# INE, este patrón de URL sí tiene una API JSON documentada y estable.
# ============================================================================
_print_header("INTENTO 4: patrón de API JSON tipo PXWeb")

_probe(
    "PXWeb API v1 -- listado de bases de datos disponibles",
    "https://estadisticas.ciencia.gob.es/api/v1/es",
)


# ============================================================================
# INTENTO 5: pedir la página HTML del navegador de tablas PC-Axis
# directamente (la que SÍ está indexada por búsqueda web) y buscar en su
# código fuente enlaces a ficheros .px, .json o .csv incrustados -- si el
# portal bloquea la home pero no ficheros de datos sueltos referenciados
# desde ella, este es el sitio donde deberían aparecer esos enlaces.
# ============================================================================
_print_header("INTENTO 5: HTML del navegador de tablas, buscando enlaces a ficheros de datos")

resp5 = _probe(
    "Navegador PC-Axis: Matriculados por campo de estudio",
    "https://estadisticas.ciencia.gob.es/dynPx/inebase/index.htm",
    params={
        "type": "pcaxis",
        "path": "/Universitaria/Alumnado/EEU_2024/GradoCiclo/Matriculados/",
        "file": "pcaxis",
        "l": "s0",
    },
)

if resp5 is not None and resp5.status_code == 200 and resp5.text:
    import re
    enlaces = sorted(set(re.findall(r'href="([^"]+\.(?:px|json|csv|xls|xlsx))"', resp5.text, re.IGNORECASE)))
    if enlaces:
        print(f"\n{len(enlaces)} enlaces a ficheros de datos encontrados en el HTML:")
        for enlace in enlaces:
            print(f"    {enlace}")
    else:
        print("\nNo se encontraron enlaces a .px/.json/.csv/.xls en el HTML de esta página.")


print("\n\n" + "=" * 78)
print("FIN DEL DIAGNÓSTICO -- pega TODO el output de arriba, no solo el final.")
print("=" * 78)
