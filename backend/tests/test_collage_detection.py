"""
Tests de app/vision/collage_detection.py -- puramente basados en numpy/PIL,
sin ningún modelo ni dependencia pesada, así que corren siempre (a
diferencia de test_geolocation.py, no necesitan mockear torch).
"""
import numpy as np
from PIL import Image

from app.vision.collage_detection import detect_collage


def _solid_gray(height: int, width: int, value: int) -> Image.Image:
    arr = np.full((height, width), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _with_band(
    height: int,
    width: int,
    base_value: int,
    band_value: int,
    orientation: str,
    position_fraction: float,
    thickness_px: int,
) -> Image.Image:
    """Imagen en gris sólido de `base_value` con una franja de
    `band_value` (grosor `thickness_px`) empezando en
    `position_fraction` de la altura (orientation='horizontal') o del
    ancho (orientation='vertical')."""
    arr = np.full((height, width), base_value, dtype=np.uint8)
    if orientation == "horizontal":
        start = int(height * position_fraction)
        arr[start : start + thickness_px, :] = band_value
    else:
        start = int(width * position_fraction)
        arr[:, start : start + thickness_px] = band_value
    return Image.fromarray(arr)


class TestDetectCollage:
    def test_imagen_solida_no_es_collage(self):
        image = _solid_gray(200, 200, 128)
        assert detect_collage(image) is False

    def test_imagen_solida_negra_no_es_collage_aunque_cumpla_el_color(self):
        # Toda la imagen es "candidata" por color (negro), pero la franja
        # resultante ocupa todo el interior -- muy por encima del grosor
        # máximo de un margen de collage real, así que se descarta (ver
        # _MAX_BAND_THICKNESS_FRACTION).
        image = _solid_gray(300, 300, 0)
        assert detect_collage(image) is False

    def test_imagen_demasiado_pequena_nunca_se_marca_como_collage(self):
        # Aunque tenga una línea perfecta de margen, por debajo de
        # _MIN_DIMENSION_PX no se intenta ni evaluar (ver docstring).
        image = _with_band(
            10, 10, base_value=100, band_value=255, orientation="horizontal", position_fraction=0.5, thickness_px=1
        )
        assert detect_collage(image) is False

    def test_banda_blanca_horizontal_interior_fina_es_collage(self):
        image = _with_band(
            300, 300, base_value=100, band_value=255, orientation="horizontal", position_fraction=0.5, thickness_px=4
        )
        assert detect_collage(image) is True

    def test_banda_negra_vertical_interior_fina_es_collage(self):
        image = _with_band(
            300, 300, base_value=200, band_value=0, orientation="vertical", position_fraction=0.5, thickness_px=4
        )
        assert detect_collage(image) is True

    def test_grid_2x2_con_cruz_blanca_es_collage(self):
        arr = np.full((300, 300), 100, dtype=np.uint8)
        arr[148:152, :] = 255  # línea horizontal
        arr[:, 148:152] = 255  # línea vertical
        image = Image.fromarray(arr)
        assert detect_collage(image) is True

    def test_banda_blanca_demasiado_ancha_no_es_collage(self):
        # Simula un cielo despejado ocupando gran parte de la foto: banda
        # legítima de color uniforme, pero mucho más gruesa que un margen
        # de collage real -- se descarta a propósito (ver
        # _MAX_BAND_THICKNESS_FRACTION y el sesgo hacia falsos negativos
        # documentado en el módulo).
        image = _with_band(
            300, 300, base_value=100, band_value=245, orientation="horizontal", position_fraction=0.3, thickness_px=90
        )
        assert detect_collage(image) is False

    def test_banda_blanca_pegada_al_borde_exterior_no_es_collage(self):
        # Letterboxing/padding normal, no un margen entre sub-fotos.
        image = _with_band(
            300, 300, base_value=100, band_value=255, orientation="horizontal", position_fraction=0.0, thickness_px=4
        )
        assert detect_collage(image) is False

    def test_ruido_aleatorio_no_es_collage(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(300, 300), dtype=np.uint8)
        image = Image.fromarray(arr)
        assert detect_collage(image) is False

    def test_banda_de_color_no_blanco_ni_negro_no_es_collage(self):
        # Un margen gris medio no cumple el criterio de color (ver
        # _WHITE_MEAN_THRESHOLD/_BLACK_MEAN_THRESHOLD) -- limitación
        # documentada a propósito: no todos los colores de margen
        # posibles se detectan.
        image = _with_band(
            300, 300, base_value=100, band_value=128, orientation="horizontal", position_fraction=0.5, thickness_px=4
        )
        assert detect_collage(image) is False

    def test_acepta_imagen_rgb_no_solo_escala_de_grises(self):
        # Las fotos reales del pipeline llegan en RGB (JPEG decodificado),
        # no en escala de grises -- confirma que la conversión interna
        # funciona igual sobre ese caso real.
        arr = np.full((300, 300, 3), 100, dtype=np.uint8)
        arr[148:152, :, :] = 255
        image = Image.fromarray(arr)
        assert detect_collage(image) is True

    def test_no_lanza_ante_un_modo_de_color_inusual(self):
        image = Image.new("P", (300, 300))
        assert detect_collage(image) is False
