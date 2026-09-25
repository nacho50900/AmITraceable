"""
Genera, para una celda del grid de España (flickr_grid.GridCell, ~10km de
lado), las sub-teselas necesarias para consultar el API de Mapillary sin
violar su restricción de bbox: desde el 16 de enero de 2026, las
consultas a /images deben ser ESTRICTAMENTE menores de 0.01 grados
cuadrados (formalizado, no una recomendación) -- eso son solo
~0.9-1.1km² por consulta según la latitud, muy por debajo del tamaño de
una celda del grid (100km²). Una celda de 10km necesita del orden de
100-110 sub-teselas para cubrirse entera.

Se reutiliza el mismo grid de 10km (flickr_grid.py) como nivel EXTERIOR
en vez de generar un grid de Mapillary completamente aparte, para que la
cobertura de Mapillary sea comparable celda a celda con la de Commons
(mismos límites geográficos, misma numeración de celdas) -- útil para
comparar densidad entre fuentes más adelante.
"""
import math

from flickr_grid import GridCell

# Un poco por debajo de 0.01 para quedar con margen bajo el límite
# "estrictamente menor que" que impone Mapillary, sin apurar al límite
# exacto (evita fallos por redondeo de coma flotante en el propio
# cliente HTTP).
_MAPILLARY_MAX_BBOX_DEGREES = 0.0095


def generate_sub_tiles(cell: GridCell) -> list[tuple[float, float, float, float]]:
    """Genera sub-teselas (min_lon, min_lat, max_lon, max_lat) que cubren
    por completo la celda dada, cada una de como mucho
    _MAPILLARY_MAX_BBOX_DEGREES de lado. El número de sub-teselas por
    celda no es constante -- depende de cuánto mida la celda en grados
    (las celdas del grid están pensadas para medir ~10km reales, pero en
    grados de longitud varían con la latitud)."""
    lat_span = cell.max_lat - cell.min_lat
    lon_span = cell.max_lon - cell.min_lon

    n_lat_steps = max(1, math.ceil(lat_span / _MAPILLARY_MAX_BBOX_DEGREES))
    n_lon_steps = max(1, math.ceil(lon_span / _MAPILLARY_MAX_BBOX_DEGREES))

    lat_step = lat_span / n_lat_steps
    lon_step = lon_span / n_lon_steps

    tiles = []
    for i in range(n_lat_steps):
        tile_min_lat = cell.min_lat + i * lat_step
        tile_max_lat = cell.min_lat + (i + 1) * lat_step
        for j in range(n_lon_steps):
            tile_min_lon = cell.min_lon + j * lon_step
            tile_max_lon = cell.min_lon + (j + 1) * lon_step
            tiles.append((tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat))
    return tiles


if __name__ == "__main__":
    # Comprobación rápida sin red: cuántas sub-teselas salen por celda a
    # distinta latitud, y el total nacional aproximado.
    from flickr_grid import generate_spain_grid

    cells = generate_spain_grid(cell_km=10.0)
    sample = cells[len(cells) // 2]
    tiles = generate_sub_tiles(sample)
    print(f"Celda de ejemplo ({sample.id}): {len(tiles)} sub-teselas")
    print(f"Ancho de una sub-tesela: {tiles[0][2] - tiles[0][0]:.5f}° x {tiles[0][3] - tiles[0][1]:.5f}°")

    total_tiles = sum(len(generate_sub_tiles(c)) for c in cells[:50])
    print(f"Media de sub-teselas/celda (muestra de 50): {total_tiles / 50:.1f}")
    print(f"Estimación nacional (x{len(cells)} celdas): ~{total_tiles / 50 * len(cells):.0f} sub-teselas")
