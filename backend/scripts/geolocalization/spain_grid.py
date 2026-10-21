"""
Genera un grid de celdas rectangulares que cubre el territorio español
(península + Baleares + Canarias + Ceuta/Melilla vía el rectángulo
peninsular, que ya las incluye) -- reutilizado por todos los scripts de
ingestión de fuentes externas (build_commons_index.py,
build_mapillary_index.py) para que sus celdas sean directamente
comparables entre fuentes, aunque cada una lo use con un tamaño de
consulta distinto (Commons consulta la celda entera; Mapillary la
subdivide en sub-teselas más pequeñas, ver mapillary_grid.py, por su
límite de bbox más estricto).

(Este fichero se llamó flickr_grid.py hasta que se retiró
build_flickr_index.py -- no tiene nada específico de Flickr, siempre fue
el grid genérico de España; se renombró para que el nombre no confunda.)

Se usan dos rectángulos base (península+Baleares, y Canarias por
separado) en vez de uno solo: un único bbox que cubra ambos incluiría una
franja enorme de océano, Francia, Portugal y Marruecos en medio, gastando
cuota de API en celdas que no pueden tener ninguna foto española.

Uso como librería:
    from spain_grid import generate_spain_grid
    cells = generate_spain_grid(cell_km=10)
"""
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridCell:
    id: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def bbox_str(self) -> str:
        # Orden estándar de bbox en geo-APIs (min_lon,min_lat,max_lon,max_lat)
        # -- usado por Flickr y por buena parte de APIs georreferenciadas.
        # Sin uso actual (Commons usa gscoord+gsradius, Mapillary genera su
        # propio bbox por sub-tesela en mapillary_grid.py) -- se deja por si
        # una fuente futura lo necesita en este formato.
        return f"{self.min_lon:.6f},{self.min_lat:.6f},{self.max_lon:.6f},{self.max_lat:.6f}"

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2


# (min_lon, min_lat, max_lon, max_lat) -- rectángulos delimitadores amplios
# de cada zona. Deliberadamente generosos (incluyen algo de mar/países
# vecinos en los bordes): es más simple aceptar que algunas celdas de borde
# salgan vacías (0 resultados, coste despreciable: una sola petición) que
# mantener un polígono exacto de la frontera española.
_SPAIN_REGIONS = {
    "peninsula_baleares": (-9.5, 35.9, 4.4, 43.9),
    "canarias": (-18.2, 27.5, -13.3, 29.5),
}


def generate_spain_grid(cell_km: float = 10.0) -> list[GridCell]:
    """Genera celdas de ~cell_km x cell_km (en km reales, no en grados) sobre
    las zonas de _SPAIN_REGIONS.

    El paso de longitud se recalcula en cada fila (no una vez al principio)
    porque los grados de longitud equivalen a menos km reales cuanto más al
    norte se está (los meridianos convergen hacia los polos) -- si no se
    ajustara por cos(latitud), las celdas de Cantabria saldrían visiblemente
    más anchas en km reales que las de Andalucía para el mismo delta de
    longitud, sesgando la cobertura sin que el número de celdas lo refleje.
    """
    cells: list[GridCell] = []
    lat_step_deg = cell_km / 111.0  # 1 grado de latitud ~= 111km, ~constante en toda España

    for region_name, (min_lon, min_lat, max_lon, max_lat) in _SPAIN_REGIONS.items():
        lat = min_lat
        row = 0
        while lat < max_lat:
            lon_step_deg = cell_km / (111.320 * math.cos(math.radians(lat)))
            lon = min_lon
            col = 0
            while lon < max_lon:
                cell_max_lat = min(lat + lat_step_deg, max_lat)
                cell_max_lon = min(lon + lon_step_deg, max_lon)
                cells.append(
                    GridCell(
                        id=f"{region_name}_{row:03d}_{col:03d}",
                        min_lon=lon,
                        min_lat=lat,
                        max_lon=cell_max_lon,
                        max_lat=cell_max_lat,
                    )
                )
                lon += lon_step_deg
                col += 1
            lat += lat_step_deg
            row += 1

    return cells


if __name__ == "__main__":
    # Comprobación rápida sin red: cuántas celdas salen y con qué pinta,
    # para elegir --cell-km con criterio antes de lanzar la ingestión real.
    for km in (5, 10, 20):
        cells = generate_spain_grid(cell_km=km)
        print(f"cell_km={km}: {len(cells)} celdas")


# --- Recorte a territorio español -------------------------------------------
# Los rectángulos de _SPAIN_REGIONS incluyen Portugal, Francia, Andorra,
# Marruecos y mucho mar. Para fuentes donde cada celda cuesta ~100 peticiones
# (Mapillary) eso es cuota de API desperdiciada, y las fotos de fuera acaban
# etiquetadas con la provincia española más cercana. Se usan los mismos límites
# municipales (es-atlas, derivados del IGN) que plot_index_coverage.py.
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_BOUNDARIES_CACHE = _BACKEND_DIR / "data" / "boundaries" / "es_atlas_municipalities.json"
_BOUNDARIES_URL = "https://cdn.jsdelivr.net/npm/es-atlas@0.6.0/es/municipalities.json"


def _decode_topojson_polygons(topo: dict, object_name: str):
    """Decodifica los polígonos de un objeto TopoJSON a geometrías shapely sin
    depender de geopandas/GDAL (solo json + shapely)."""
    from shapely.geometry import Polygon

    scale, translate = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:  # coordenadas cuantizadas y codificadas por deltas
            x += dx
            y += dy
            pts.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
        arcs.append(pts)

    def ring(arc_indices):
        coords: list[tuple[float, float]] = []
        for idx in arc_indices:
            pts = arcs[~idx][::-1] if idx < 0 else arcs[idx]
            coords.extend(pts if not coords else pts[1:])  # el primer punto se comparte con el arco previo
        return coords

    def polygon(rings):
        shell = ring(rings[0])
        if len(shell) < 4:
            return None  # anillo degenerado
        holes = [h for h in (ring(r) for r in rings[1:]) if len(h) >= 4]
        return Polygon(shell, holes)

    geoms = []
    for g in topo["objects"][object_name]["geometries"]:
        polys = [g["arcs"]] if g["type"] == "Polygon" else g["arcs"] if g["type"] == "MultiPolygon" else []
        geoms.extend(poly for poly in (polygon(rings) for rings in polys) if poly is not None)
    return geoms


class SpainLand:
    """Territorio español (municipios) con consultas rápidas por STRtree.

    `margin_deg` da holgura (~1 km con 0.01) para costa, puertos y frontera:
    una foto o sub-tesela a menos de esa distancia de un municipio cuenta
    como dentro. Requiere solo `shapely` (pip install shapely)."""

    def __init__(self, boundaries_path: Path | None = None, margin_deg: float = 0.01):
        try:
            import shapely
        except ImportError as e:
            raise SystemExit("Falta shapely: pip install shapely (o usa --no-spain-only).") from e

        self._shapely = shapely
        self.margin_deg = margin_deg
        path = boundaries_path or _BOUNDARIES_CACHE
        if not path.exists():
            print(f"Descargando límites municipales de {_BOUNDARIES_URL} ...")
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_BOUNDARIES_URL, path)
        with open(path, encoding="utf-8") as f:
            topo = json.load(f)
        geoms = shapely.make_valid(_decode_topojson_polygons(topo, "municipalities"))
        self._tree = shapely.STRtree(geoms)

    def intersects_bbox(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bool:
        m = self.margin_deg
        box = self._shapely.box(min_lon - m, min_lat - m, max_lon + m, max_lat + m)
        return len(self._tree.query(box, predicate="intersects")) > 0

    def contains_point(self, lat: float, lon: float) -> bool:
        m = self.margin_deg
        return self.intersects_bbox(lon - m, lat - m, lon + m, lat + m)


def filter_cells_to_spain(cells: list[GridCell], land: SpainLand) -> list[GridCell]:
    """Conserva solo las celdas que tocan territorio español. Los ids no
    cambian, así que el estado de reanudación (_completed_cells.txt) sigue
    siendo válido."""
    return [c for c in cells if land.intersects_bbox(c.min_lon, c.min_lat, c.max_lon, c.max_lat)]
