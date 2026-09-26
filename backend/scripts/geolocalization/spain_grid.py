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
import math
from dataclasses import dataclass


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
