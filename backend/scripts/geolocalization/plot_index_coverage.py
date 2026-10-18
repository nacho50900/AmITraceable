"""
Mapa de cobertura del índice FAISS de geolocalización: cuántas imágenes del
índice caen en cada municipio de España (coropleta con degradado de color).

Pensado para la documentación / memoria del TFG, NO para la app: genera un
único HTML autocontenido (zoom, tooltips por municipio) que se abre en el
navegador. Sirve para ver dónde el geolocalizador tiene poca cobertura, lo
que se traduce en estimaciones menos fiables (sesgo geográfico del índice).

No lee index.faiss: solo necesita lat/lon de cada imagen, que ya están en
index_meta.csv (mismo orden que el índice).

Uso:
    python plot_index_coverage.py
    python plot_index_coverage.py --metric densidad --output cobertura_densidad.html
    python plot_index_coverage.py --meta ../../data/spain_combined/index_meta.csv

Dependencias (no están en pyproject.toml; solo para este script):
    pip install geopandas folium

El HTML necesita internet para cargar Leaflet (CDN) y el mapa base de Esri.

Límites municipales: TopoJSON de es-atlas (MIT; deriva de los datos del IGN),
cuyo `id` es el código INE de 5 dígitos. Se descarga una vez de jsDelivr y se
cachea en data/boundaries/; con --boundaries se puede usar una copia local.
"""
import argparse
import math
import urllib.request
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import shapely
from branca.colormap import LinearColormap

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_DEFAULT_META = _BACKEND_DIR / "data" / "osv5m_spain" / "index_meta.csv"
_BOUNDARIES_CACHE = _BACKEND_DIR / "data" / "boundaries" / "es_atlas_municipalities.json"
_BOUNDARIES_URL = "https://cdn.jsdelivr.net/npm/es-atlas@0.6.0/es/municipalities.json"

# Cajón generoso de España (península, Baleares, Canarias, Ceuta y Melilla):
# descarta coordenadas absurdas (0,0, otros países) antes del cruce espacial.
_LAT_RANGE = (27.0, 44.5)
_LON_RANGE = (-19.0, 5.0)

# CRS equiárea europeo (metros): válido para calcular superficies en km²
# también en Canarias, donde un CRS de la península daría áreas erróneas.
_CRS_AREA = "EPSG:3035"
# Foto que cae al mar/puerto/frontera por poco: se asigna al municipio más
# cercano si está a menos de esta distancia (metros).
_MAX_SNAP_M = 2000

_METRICAS = {
    "imagenes": ("Imágenes en el índice", "n_imagenes"),
    "densidad": ("Imágenes por 100 km²", "imagenes_100km2"),
}


def _cargar_limites(ruta: Path | None) -> gpd.GeoDataFrame:
    if ruta is None:
        ruta = _BOUNDARIES_CACHE
        if not ruta.exists():
            print(f"Descargando límites municipales de {_BOUNDARIES_URL} ...")
            ruta.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_BOUNDARIES_URL, ruta)
    municipios = gpd.read_file(ruta, layer="municipalities")
    municipios = municipios.rename(columns={"id": "ine", "name": "municipio"})
    municipios["ine"] = municipios["ine"].astype(str).str.zfill(5)
    return municipios.set_crs("EPSG:4326", allow_override=True)


def _cargar_puntos(meta_path: Path) -> gpd.GeoDataFrame:
    meta = pd.read_csv(meta_path)
    columnas = {c.lower(): c for c in meta.columns}
    lat_col = columnas.get("lat") or columnas.get("latitude")
    lon_col = columnas.get("lon") or columnas.get("lng") or columnas.get("longitude")
    if lat_col is None or lon_col is None:
        raise SystemExit(f"{meta_path} no tiene columnas lat/lon. Columnas: {list(meta.columns)}")

    total = len(meta)
    lat = pd.to_numeric(meta[lat_col], errors="coerce")
    lon = pd.to_numeric(meta[lon_col], errors="coerce")
    valido = (
        lat.between(*_LAT_RANGE) & lon.between(*_LON_RANGE)
    )
    descartadas = total - int(valido.sum())
    if descartadas:
        print(f"Descartadas {descartadas} de {total} filas sin coordenadas válidas en España.")
    return gpd.GeoDataFrame(
        {"lat": lat[valido], "lon": lon[valido]},
        geometry=gpd.points_from_xy(lon[valido], lat[valido]),
        crs="EPSG:4326",
    )


def _asignar_municipio(puntos: gpd.GeoDataFrame, municipios: gpd.GeoDataFrame) -> pd.Series:
    """Código INE por imagen (NaN si no se pudo asignar). Primero punto dentro
    del polígono; los que caen fuera por poco se asignan al más cercano."""
    dentro = gpd.sjoin(puntos, municipios[["ine", "geometry"]], how="left", predicate="within")
    # Un punto exactamente en un borde puede casar con dos polígonos: se queda con el primero.
    dentro = dentro[~dentro.index.duplicated(keep="first")]
    ine = dentro["ine"].reindex(puntos.index)

    sin_asignar = ine.isna()
    if sin_asignar.any():
        cercanos = gpd.sjoin_nearest(
            puntos[sin_asignar].to_crs(_CRS_AREA),
            municipios[["ine", "geometry"]].to_crs(_CRS_AREA),
            how="left",
            max_distance=_MAX_SNAP_M,
        )
        cercanos = cercanos[~cercanos.index.duplicated(keep="first")]
        ine.loc[cercanos.index] = cercanos["ine"]
    return ine


def _construir_mapa(municipios: gpd.GeoDataFrame, metrica: str, titulo_extra: str) -> folium.Map:
    etiqueta, columna = _METRICAS[metrica]
    valores = municipios[columna]
    # Escala logarítmica: sin ella Madrid/Barcelona acaparan todo el degradado
    # y el resto de España se vería casi en blanco.
    log_valores = valores.map(lambda v: math.log10(1 + v))
    vmax = max(float(log_valores.max()), 1e-9)
    colores = LinearColormap(
        ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"], vmin=0, vmax=vmax
    )
    colores.caption = f"{etiqueta} (escala logarítmica)"

    datos = municipios.assign(_log=log_valores)
    # Mapa base de Esri: no exige API key ni "Referer", así que funciona al abrir
    # el HTML desde file:// (OpenStreetMap devuelve 403 en ese caso y CartoDB
    # exige key). Mar en azul y resto de países en color normal.
    mapa = folium.Map(location=[40.0, -3.7], zoom_start=6, tiles=None, control_scale=True)
    for nombre, servicio, mostrar in (
        ("Esri calles", "World_Street_Map", True),
        ("Esri topográfico", "World_Topo_Map", False),
    ):
        folium.TileLayer(
            tiles=f"https://server.arcgisonline.com/ArcGIS/rest/services/{servicio}/MapServer/tile/{{z}}/{{y}}/{{x}}",
            attr="Tiles &copy; Esri",
            name=nombre,
            show=mostrar,
        ).add_to(mapa)

    def estilo(feature):
        valor = feature["properties"][columna]
        if not valor:
            return {"fillColor": "#e8e8e8", "color": "#bbbbbb", "weight": 0.2, "fillOpacity": 0.45}
        return {
            "fillColor": colores(feature["properties"]["_log"]),
            "color": "#666666",
            "weight": 0.2,
            "fillOpacity": 0.85,
        }

    folium.GeoJson(
        datos[["municipio", "ine", "n_imagenes", "imagenes_100km2", "_log", "geometry"]].to_json(),
        name=etiqueta,
        style_function=estilo,
        highlight_function=lambda _f: {"weight": 1.5, "color": "#000000"},
        tooltip=folium.GeoJsonTooltip(
            fields=["municipio", "ine", "n_imagenes", "imagenes_100km2"],
            aliases=["Municipio", "Código INE", "Imágenes", "Imágenes / 100 km²"],
            localize=True,
        ),
    ).add_to(mapa)
    colores.add_to(mapa)
    folium.LayerControl(collapsed=True).add_to(mapa)

    mapa.get_root().html.add_child(
        folium.Element(
            f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;'
            f'padding:6px 10px;border:1px solid #999;font:13px sans-serif;max-width:360px">'
            f"<b>Cobertura del índice FAISS por municipio</b><br>{titulo_extra}</div>"
        )
    )
    return mapa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", type=Path, default=_DEFAULT_META, help="index_meta.csv (lat, lon por imagen)")
    parser.add_argument("--boundaries", type=Path, default=None, help="TopoJSON de es-atlas local (por defecto se descarga)")
    parser.add_argument("--metric", choices=sorted(_METRICAS), default="imagenes", dest="metrica")
    parser.add_argument("--output", type=Path, default=Path("coverage_map.html"))
    args = parser.parse_args()

    municipios = _cargar_limites(args.boundaries)
    puntos = _cargar_puntos(args.meta)

    puntos["ine"] = _asignar_municipio(puntos, municipios)
    sin_municipio = int(puntos["ine"].isna().sum())
    if sin_municipio:
        print(f"{sin_municipio} imágenes quedan a más de {_MAX_SNAP_M} m de cualquier municipio y no se cuentan.")

    conteo = puntos["ine"].value_counts()
    municipios["n_imagenes"] = municipios["ine"].map(conteo).fillna(0).astype(int)
    municipios["km2"] = municipios.to_crs(_CRS_AREA).area / 1e6
    municipios["imagenes_100km2"] = (municipios["n_imagenes"] / municipios["km2"] * 100).round(2)

    con_imagenes = int((municipios["n_imagenes"] > 0).sum())
    asignadas = int(municipios["n_imagenes"].sum())
    print(f"{asignadas} imágenes asignadas a {con_imagenes} de {len(municipios)} municipios "
          f"({100 * con_imagenes / len(municipios):.1f}% con al menos una).")
    print("Top 10 municipios:")
    top = municipios.nlargest(10, "n_imagenes")[["municipio", "ine", "n_imagenes"]]
    print(top.to_string(index=False))

    # Simplificar y recortar precisión: pasa de decenas de MB a unos pocos MB sin
    # cambios apreciables a escala municipal.
    # mode="pointwise": solo redondea coordenadas, sin revalidar la topología
    # (el modo por defecto lanza TopologyException con algunos municipios).
    municipios["geometry"] = shapely.set_precision(
        municipios.geometry.simplify(0.002, preserve_topology=True).values, 1e-4, mode="pointwise"
    )

    resumen = (
        f"{asignadas:,} imágenes en {con_imagenes:,} de {len(municipios):,} municipios "
        f"({100 * con_imagenes / len(municipios):.1f}%). Gris = sin imágenes."
    ).replace(",", ".")
    mapa = _construir_mapa(municipios, args.metrica, resumen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapa.save(args.output)
    print(f"Guardado en {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
