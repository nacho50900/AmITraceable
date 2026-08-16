"""
Tests de app/vision/geolocation.py.

No se descarga ningún modelo real ni se necesita un índice FAISS
construido: se mockea `_lazy_load` (y las variables de módulo que rellena)
para ejercer toda la lógica de negocio (votación de vecinos, centroide,
degradación best-effort) sin dependencias externas pesadas.
"""
from collections import namedtuple
import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.vision import geolocation


class _FakeExif:
    """Sustituye al objeto Exif de Pillow lo justo para _extract_exif_gps:
    solo necesita soportar .get_ifd(tag) -> dict."""

    def __init__(self, gps_ifd=None):
        self._gps_ifd = gps_ifd or {}

    def get_ifd(self, tag):
        return self._gps_ifd


class _FakeImage:
    """Sustituye a PIL.Image: solo necesita soportar .convert('RGB') y,
    para los tests de EXIF GPS, .getexif()."""

    def __init__(self, gps_ifd=None):
        self._gps_ifd = gps_ifd

    def convert(self, mode):
        return self

    def getexif(self):
        return _FakeExif(self._gps_ifd)


class _NoGradContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _FakeTensor:
    """Sustituye a torch.Tensor lo justo para lo que geolocation.py necesita:
    indexación tipo outputs.last_hidden_state[:, 0, :] y .cpu().numpy()."""

    def __init__(self, vector):
        self._vector = vector

    def __getitem__(self, key):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([self._vector], dtype="float32")


def _make_fake_torch(output_vector):
    """torch es una dependencia OPCIONAL de este proyecto (solo la necesita
    el módulo de geolocalización por imagen), así que puede no estar
    instalada en el entorno donde corren los tests. Se inyecta un módulo
    `torch` falso en sys.modules con justo lo que geolocation.py usa
    (`torch.no_grad()`, `torch.cuda.is_available()`), sin depender del
    paquete real ni de que produzca un tensor utilizable de verdad."""
    fake_outputs = SimpleNamespace(last_hidden_state=_FakeTensor(output_vector))
    fake_torch = SimpleNamespace(
        no_grad=lambda: _NoGradContext(),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    return fake_torch, fake_outputs


@pytest.fixture(autouse=True)
def reset_module_globals(monkeypatch):
    """Cada test debe partir de _model/_processor/_index/_index_meta
    limpios, para que _lazy_load() se comporte de forma predecible."""
    monkeypatch.setattr(geolocation, "_model", None)
    monkeypatch.setattr(geolocation, "_processor", None)
    monkeypatch.setattr(geolocation, "_index", None)
    monkeypatch.setattr(geolocation, "_index_meta", None)
    yield


def _install_fake_index(monkeypatch, meta_df: pd.DataFrame, search_indices, search_similarities=None):
    """Sustituye _lazy_load para que 'cargue' un índice FAISS falso y
    metadatos controlados, sin tocar disco ni descargar ningún modelo."""
    if search_similarities is None:
        search_similarities = [0.9] * len(search_indices)

    fake_index = SimpleNamespace(
        search=lambda vector, k: (
            np.array([search_similarities[:k]], dtype="float32"),
            np.array([search_indices[:k]]),
        )
    )

    def _fake_lazy_load():
        # OJO: no se tocan _model/_processor aquí -- eso lo rellena
        # _install_fake_embedding() por separado. Si este fake también los
        # sobrescribiera, pisaría el modelo/procesador falsos que sí
        # producen una salida utilizable, y estimate_location_from_image
        # acabaría llamando a un `object()` no invocable.
        monkeypatch.setattr(geolocation, "_index", fake_index)
        monkeypatch.setattr(geolocation, "_index_meta", meta_df)

    monkeypatch.setattr(geolocation, "_lazy_load", _fake_lazy_load)


def _install_fake_embedding(monkeypatch, output_vector=None):
    """Evita depender de torch/transformers reales (no instalados en este
    entorno, dependencia opcional del módulo): inyecta un `torch` falso en
    sys.modules y sustituye _model/_processor por callables mínimos que
    producen una salida con la forma que estimate_location_from_image
    espera."""
    if output_vector is None:
        output_vector = [0.1] * 384

    fake_torch, fake_outputs = _make_fake_torch(output_vector)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(geolocation, "_model", lambda **kwargs: fake_outputs)
    monkeypatch.setattr(
        geolocation, "_processor", lambda images, return_tensors: SimpleNamespace(to=lambda d: {})
    )


class TestDownloadImage:
    """`_download_image` es compartida por DINOv2 y Moondream2, que la
    reciben en paralelo (`_process_photo`, `asyncio.gather` con
    `asyncio.to_thread` cada una) -- ver el bug real que motiva estos
    tests en el docstring de `_download_image`."""

    @pytest.mark.asyncio
    async def test_forces_full_decode_before_returning(self, monkeypatch, respx_mock):
        """Bug real corregido en producción: sin `.load()` en
        `_download_image`, `Image.open()` deja la decodificación de
        píxeles pendiente (perezosa) hasta que algo accede de verdad a la
        imagen -- y como esa misma imagen se reparte entre DOS hilos
        concurrentes (DINOv2 y Moondream2), esa decodificación perezosa
        competía entre ambos y producía `OSError: image file is
        truncated` en uno de los dos (casi siempre el que llegaba
        segundo), de forma no determinista y con una exclusión mutua casi
        perfecta observada en logs reales: cuando uno de los dos modelos
        conseguía un resultado, el otro fallaba, y viceversa. Se comprueba
        que `Image.load()` se llama aquí, ANTES de que la imagen se
        reparta entre hilos, no dejándolo para más tarde."""
        import httpx
        from PIL import Image as PILImage

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/test.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        load_calls: list[bool] = []
        original_load = PILImage.Image.load

        def _spy_load(self):
            load_calls.append(True)
            return original_load(self)

        monkeypatch.setattr(PILImage.Image, "load", _spy_load)

        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(1)
            image = await geolocation._download_image(client, semaphore, "https://cdn.fake/test.jpg")

        assert image is not None
        assert len(load_calls) >= 1

    @pytest.mark.asyncio
    async def test_returns_none_on_download_failure(self, monkeypatch, respx_mock):
        """No debe lanzar si la descarga falla -- una foto no descargable
        se omite, no aborta el análisis de las demás (ver
        `_process_photo`)."""
        import httpx

        respx_mock.get("https://cdn.fake/missing.jpg").mock(return_value=httpx.Response(404))

        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(1)
            image = await geolocation._download_image(client, semaphore, "https://cdn.fake/missing.jpg")

        assert image is None

    @pytest.mark.asyncio
    async def test_returns_none_on_undecodable_content(self, monkeypatch, respx_mock):
        """Igual que el anterior, pero con una descarga que SÍ tiene
        éxito (200 OK) pero cuyo contenido no es una imagen válida -- debe
        degradarse a None en vez de propagar la excepción de PIL."""
        import httpx

        respx_mock.get("https://cdn.fake/not-an-image.jpg").mock(
            return_value=httpx.Response(200, content=b"esto no es un jpeg valido")
        )

        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(1)
            image = await geolocation._download_image(client, semaphore, "https://cdn.fake/not-an-image.jpg")

        assert image is None


class TestEstimateLocationFromImage:
    def test_returns_none_when_index_not_built(self, monkeypatch):
        def _raise_not_found():
            raise FileNotFoundError("no index")

        monkeypatch.setattr(geolocation, "_lazy_load", _raise_not_found)

        assert geolocation.estimate_location_from_image(_FakeImage()) is None

    def test_returns_none_when_dependencies_not_installed(self, monkeypatch):
        def _raise_import_error():
            raise ModuleNotFoundError("No module named 'torch'")

        monkeypatch.setattr(geolocation, "_lazy_load", _raise_import_error)

        assert geolocation.estimate_location_from_image(_FakeImage()) is None

    def test_votes_for_majority_province_among_neighbors(self, monkeypatch):
        meta = pd.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "lat": [40.0, 40.1, 41.0, 41.5],
                "lon": [-3.7, -3.6, 2.1, 2.2],
                "region": ["Madrid", "Madrid", "Cataluna", "Cataluna"],
            }
        )
        # 3 vecinos votan Madrid (índices 0,1 repetido) y 1 vecino Cataluna
        _install_fake_index(monkeypatch, meta, search_indices=[0, 1, 0, 2])
        _install_fake_embedding(monkeypatch)

        result = geolocation.estimate_location_from_image(_FakeImage(), k=4)

        assert result is not None
        assert result.province == "Madrid"
        assert result.confidence == 0.75  # 3 de 4 vecinos
        assert result.k_neighbors == 4
        # Centroide de los vecinos que votaron Madrid (índices 0 y 1, dos veces el 0)
        assert result.lat == pytest.approx(40.033, abs=0.01)

    def test_returns_none_if_image_processing_raises(self, monkeypatch):
        meta = pd.DataFrame({"id": ["1"], "lat": [40.0], "lon": [-3.7], "region": ["Madrid"]})
        _install_fake_index(monkeypatch, meta, search_indices=[0])

        fake_torch, _ = _make_fake_torch([0.1] * 384)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        def _raise(**kwargs):
            raise RuntimeError("modelo roto")

        monkeypatch.setattr(geolocation, "_model", _raise)
        monkeypatch.setattr(
            geolocation, "_processor", lambda images, return_tensors: SimpleNamespace(to=lambda d: {})
        )

        assert geolocation.estimate_location_from_image(_FakeImage()) is None

    def test_lat_lon_none_when_metadata_missing_coordinates(self, monkeypatch):
        meta = pd.DataFrame({"id": ["1"], "lat": [None], "lon": [None], "region": ["Madrid"]})
        _install_fake_index(monkeypatch, meta, search_indices=[0])
        _install_fake_embedding(monkeypatch)

        result = geolocation.estimate_location_from_image(_FakeImage(), k=1)

        assert result is not None
        assert result.lat is None
        assert result.lon is None

    def test_marks_non_representative_instead_of_discarding_when_neighbors_are_geographically_scattered(self, monkeypatch):
        """Caso motivador: una foto de solo mar/cielo/primer plano puede
        parecerse visualmente a imágenes de referencia de puntos muy
        alejados entre sí (Galicia, Cádiz, Baleares, Barcelona...). Ninguna
        provincia "ganadora" sería significativa ahí para AFIRMAR una
        residencia -- pero sigue siendo información real que debe seguir
        apareciendo en el mapa (image_location_points) con su confianza
        real, así que NO se descarta (no devuelve None): se marca
        `representative=False` para que solo `_infer_home_region`
        (report/generator.py) la excluya de la conclusión de residencia."""
        meta = pd.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                # Galicia, Cádiz, Baleares, Barcelona: >300km de dispersión media
                "lat": [42.9, 36.5, 39.6, 41.4],
                "lon": [-8.5, -6.3, 2.9, 2.2],
                "region": ["Galicia", "Andalucia", "Baleares", "Cataluna"],
            }
        )
        _install_fake_index(monkeypatch, meta, search_indices=[0, 1, 2, 3])
        _install_fake_embedding(monkeypatch)

        result = geolocation.estimate_location_from_image(_FakeImage(), k=4)

        assert result is not None
        assert result.representative is False
        assert result.province == "Galicia"  # sigue siendo la ganadora por votos, solo que no fiable

    def test_does_not_discard_when_too_few_neighbors_have_coordinates(self, monkeypatch):
        """Con menos de _MIN_NEIGHBORS_WITH_COORDS_FOR_SPREAD_CHECK vecinos
        con coordenadas, no hay datos suficientes para juzgar dispersión --
        no debe descartarse por eso (sería un falso positivo)."""
        meta = pd.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "lat": [42.9, None, None, None],
                "lon": [-8.5, None, None, None],
                "region": ["Galicia", "Galicia", "Galicia", "Galicia"],
            }
        )
        _install_fake_index(monkeypatch, meta, search_indices=[0, 1, 2, 3])
        _install_fake_embedding(monkeypatch)

        assert geolocation.estimate_location_from_image(_FakeImage(), k=4) is not None

    def test_uses_exif_gps_directly_without_calling_the_model(self, monkeypatch):
        """Si la foto trae GPS real en el EXIF, se usa directamente (la
        región conocida más cercana a esas coordenadas) y NO se llama al
        modelo -- se comprueba sustituyendo _model por algo que revienta si
        se invoca, para asegurar que de verdad no se usa."""
        meta = pd.DataFrame(
            {
                "id": ["1", "2"],
                "lat": [40.4168, 41.3874],  # Madrid, Barcelona
                "lon": [-3.7038, 2.1686],
                "region": ["Madrid", "Cataluna"],
            }
        )
        _install_fake_index(monkeypatch, meta, search_indices=[0, 1])

        def _model_should_not_be_called(**kwargs):
            raise AssertionError("no debería llamarse al modelo cuando hay GPS en el EXIF")

        monkeypatch.setattr(geolocation, "_model", _model_should_not_be_called)
        monkeypatch.setattr(geolocation, "_processor", _model_should_not_be_called)
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=lambda: _NoGradContext()))

        # GPS EXIF: Madrid, 40°25'00"N 3°42'14"W (formato estándar EXIF: grados, minutos, segundos)
        gps_ifd = {1: "N", 2: (40.0, 25.0, 0.0), 3: "W", 4: (3.0, 42.0, 14.0)}
        image = _FakeImage(gps_ifd=gps_ifd)

        result = geolocation.estimate_location_from_image(image)

        assert result is not None
        assert result.province == "Madrid"
        assert result.confidence == 1.0
        assert result.lat == pytest.approx(40.4167, abs=0.001)
        assert result.lon == pytest.approx(-3.7039, abs=0.001)

    def test_falls_back_to_model_when_no_exif_gps(self, monkeypatch):
        meta = pd.DataFrame({"id": ["1"], "lat": [40.0], "lon": [-3.7], "region": ["Madrid"]})
        _install_fake_index(monkeypatch, meta, search_indices=[0])
        _install_fake_embedding(monkeypatch)

        result = geolocation.estimate_location_from_image(_FakeImage(gps_ifd=None), k=1)

        assert result is not None
        assert result.k_neighbors == 1  # vino del modelo (vota entre k vecinos), no del atajo EXIF (k_neighbors=0)


class TestGeolocationAvailable:
    """Cubre justo el bug real que hizo que, con el índice ya construido y
    montado, el backend siguiera diciendo "no disponible": la comprobación
    de ficheros pasaba, pero torch/faiss/transformers no estaban instaladas
    en la imagen del backend (requirements-vision.txt es opcional, ver
    Dockerfile/ARG WITH_GEOLOCATION) -- ver requirements.txt para el
    porqué esto no es solo cosa del script de construcción del índice."""

    def test_false_when_index_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(geolocation, "_INDEX_DIR", tmp_path / "no_existe")
        assert geolocation._geolocation_available() is False

    def test_false_when_only_one_of_the_two_index_files_exists(self, tmp_path, monkeypatch):
        (tmp_path / "index.faiss").write_bytes(b"x")
        # falta index_meta.csv
        monkeypatch.setattr(geolocation, "_INDEX_DIR", tmp_path)
        assert geolocation._geolocation_available() is False

    def test_false_when_index_exists_but_a_dependency_is_not_importable(self, tmp_path, monkeypatch):
        (tmp_path / "index.faiss").write_bytes(b"x")
        (tmp_path / "index_meta.csv").write_text("a,b\n")
        monkeypatch.setattr(geolocation, "_INDEX_DIR", tmp_path)
        # Poner None en sys.modules fuerza ImportError en "import faiss",
        # sin necesitar que el paquete esté realmente instalado o no.
        monkeypatch.setitem(sys.modules, "faiss", None)

        assert geolocation._geolocation_available() is False

    def test_true_when_index_exists_and_dependencies_import_correctly(self, tmp_path, monkeypatch):
        (tmp_path / "index.faiss").write_bytes(b"x")
        (tmp_path / "index_meta.csv").write_text("a,b\n")
        monkeypatch.setattr(geolocation, "_INDEX_DIR", tmp_path)
        try:
            import faiss  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            pytest.skip(
                "torch/faiss/transformers no instaladas en este entorno "
                "(requirements-vision.txt es opcional, ver Dockerfile)"
            )

        assert geolocation._geolocation_available() is True


class TestEstimateLocationsForPosts:
    @pytest.mark.asyncio
    async def test_analyzes_every_photo_of_a_multi_photo_post_not_just_the_first(self, monkeypatch, respx_mock):
        """El caso pedido: un carrusel con varias fotos debe analizarlas
        TODAS, no solo la primera -- todas comparten el mismo permalink (es
        la misma publicación), pero cada una aporta su propia estimación."""
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [
            Post(
                type="carousel_album",
                media_urls=["https://cdn.fake/c1.jpg", "https://cdn.fake/c2.jpg", "https://cdn.fake/c3.jpg"],
                permalink="https://ig/carousel",
            ),
        ]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        for i in (1, 2, 3):
            respx_mock.get(f"https://cdn.fake/c{i}.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        monkeypatch.setattr(
            geolocation,
            "estimate_location_from_image",
            lambda image, k=15: geolocation.ImageLocationEstimate(
                province="Madrid", confidence=0.7, k_neighbors=15, mean_similarity=0.6
            ),
        )

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert len(outcome.results) == 3
        assert all(permalink == "https://ig/carousel" for permalink, _ in outcome.results)

        # Regresión del bug real: antes, las 3 fotos de este carrusel
        # habrían compartido el MISMO permalink como único identificador
        # -- ahora cada estimate lleva su propio photo_link, único por
        # foto (con ?img_index=N), no solo el permalink de la publicación
        # (que sigue siendo el mismo para las 3, correctamente, es la
        # misma publicación).
        photo_links = [estimate.photo_link for _permalink, estimate in outcome.results]
        assert len(set(photo_links)) == 3, f"los photo_link deberían ser únicos por foto: {photo_links}"
        assert sorted(photo_links) == [
            "https://ig/carousel?img_index=1",
            "https://ig/carousel?img_index=2",
            "https://ig/carousel?img_index=3",
        ]

    @pytest.mark.asyncio
    async def test_single_photo_post_link_is_unchanged(self, monkeypatch, respx_mock):
        """Con una sola foto en la publicación, el photo_link es el
        permalink normal tal cual -- no tiene sentido añadir
        ?img_index=1 a una publicación que no es un carrusel."""
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))
        monkeypatch.setattr(
            geolocation,
            "estimate_location_from_image",
            lambda image, k=15: geolocation.ImageLocationEstimate(
                province="Madrid", confidence=0.7, k_neighbors=15, mean_similarity=0.6
            ),
        )

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert len(outcome.results) == 1
        assert outcome.results[0][1].photo_link == "https://ig/1"

    @pytest.mark.asyncio
    async def test_descriptions_do_not_overwrite_between_photos_of_the_same_carousel(self, monkeypatch, respx_mock):
        """Regresión del bug real: antes, visual_descriptions/
        general_descriptions usaban el permalink de la PUBLICACIÓN como
        clave -- con varias fotos del mismo carrusel dando descripción,
        cada una sobreescribía a la anterior, y solo sobrevivía una
        (la última en procesarse, no determinista) aplicada a TODAS las
        fotos del carrusel. Con photo_link como clave, cada foto conserva
        su propia descripción."""
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)
        monkeypatch.setattr(geolocation.settings, "enable_scene_analysis", True)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [
            Post(
                type="carousel_album",
                media_urls=["https://cdn.fake/c1.jpg", "https://cdn.fake/c2.jpg"],
                permalink="https://ig/carousel",
            ),
        ]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/c1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))
        respx_mock.get("https://cdn.fake/c2.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        monkeypatch.setattr(
            geolocation,
            "estimate_location_from_image",
            lambda image, k=15: geolocation.ImageLocationEstimate(
                province="Madrid", confidence=0.7, k_neighbors=15, mean_similarity=0.6
            ),
        )

        # Cada foto (identificada por su media_url) da una descripción
        # DISTINTA -- si el bug de sobrescritura sigue presente, al final
        # solo quedaría UNA de las dos en los diccionarios.
        descriptions_by_url = {
            "https://cdn.fake/c1.jpg": ("Personas en la foto: una", "a person playing guitar"),
            "https://cdn.fake/c2.jpg": ("Personas en la foto: varias", "a group of friends at the beach"),
        }
        call_count = {"n": 0}

        def _fake_analyze(image):
            # No hay forma directa de saber qué URL generó esta imagen ya
            # descargada -- se usa el orden de llamada, que coincide con
            # el orden de photo_units (mismo orden que media_urls).
            url = list(descriptions_by_url.keys())[call_count["n"]]
            call_count["n"] += 1
            raw, general = descriptions_by_url[url]
            return [], False, raw, general

        monkeypatch.setattr(geolocation, "analyze_image_content", _fake_analyze)

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert outcome.general_descriptions == {
            "https://ig/carousel?img_index=1": "a person playing guitar",
            "https://ig/carousel?img_index=2": "a group of friends at the beach",
        }
        assert outcome.visual_descriptions == {
            "https://ig/carousel?img_index=1": "Personas en la foto: una",
            "https://ig/carousel?img_index=2": "Personas en la foto: varias",
        }

    @pytest.mark.asyncio
    async def test_visual_content_analysis_runs_alongside_geolocation_on_the_same_image(self, monkeypatch, respx_mock):
        """El caso pedido: el análisis de contenido visual (aficiones,
        señal de pareja -- ver scene_analysis.py) debe ejecutarse sobre
        cada foto, con el permalink correcto en la evidencia, sin
        necesidad de una segunda descarga de la imagen."""
        import httpx

        from app.models.schemas import InferredAttribute

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)
        # El análisis de contenido visual (Moondream2) está desactivado por
        # defecto (ver Settings.enable_scene_analysis) -- este test prueba
        # justo esa integración, así que lo activa explícitamente.
        monkeypatch.setattr(geolocation.settings, "enable_scene_analysis", True)
        monkeypatch.setattr(
            geolocation,
            "estimate_location_from_image",
            lambda image, k=15: None,  # no relevante para este test
        )

        def _fake_scene_analysis(image):
            return (
                [InferredAttribute(category="aficion", value="Fan del baloncesto", confidence=0.5, evidence=[])],
                True,
                "DESCRIPCION: una persona jugando al baloncesto\nPERSONAS: una\nAFICION: Fan del baloncesto\nPAREJA: si",
                "una persona jugando al baloncesto",
            )

        monkeypatch.setattr(geolocation, "analyze_image_content", _fake_scene_analysis)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert len(outcome.visual_inferences) == 1
        permalink, inferred = outcome.visual_inferences[0]
        assert permalink == "https://ig/1"
        assert inferred.category == "aficion"
        # geolocation.py debe rellenar la evidencia con el permalink --
        # scene_analysis.py la deja vacía a propósito (no conoce el permalink).
        assert inferred.evidence == ["https://ig/1"]
        assert outcome.partner_signal_permalinks == {"https://ig/1"}
        assert outcome.visual_descriptions == {
            "https://ig/1": "DESCRIPCION: una persona jugando al baloncesto\nPERSONAS: una\nAFICION: Fan del baloncesto\nPAREJA: si"
        }
        assert outcome.general_descriptions == {"https://ig/1": "una persona jugando al baloncesto"}

    @pytest.mark.asyncio
    async def test_no_pareja_signal_leaves_partner_signal_permalinks_empty(self, monkeypatch, respx_mock):
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)
        monkeypatch.setattr(geolocation, "estimate_location_from_image", lambda image, k=15: None)
        monkeypatch.setattr(geolocation, "analyze_image_content", lambda image: ([], False, None, None))

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert outcome.visual_inferences == []
        assert outcome.partner_signal_permalinks == set()
        assert outcome.visual_descriptions == {}

    @pytest.mark.asyncio
    async def test_scene_analysis_disabled_by_default_never_calls_analyze_image_content(
        self, monkeypatch, respx_mock
    ):
        """Regresión directa del interruptor `Settings.enable_scene_analysis`
        (por defecto False, ver config.py): sin activarlo, la foto se
        descarga y se geolocaliza igual -- se verifica explícitamente que
        `estimate_location_from_image` (DINOv2) SÍ se invoca, no solo que
        no falla -- mientras que `analyze_image_content` (Moondream2) no
        debe ni llegar a invocarse. Así queda probado en ambos sentidos
        que el interruptor solo afecta a Moondream2, nunca a DINOv2."""
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)
        dinov2_calls: list[object] = []
        monkeypatch.setattr(
            geolocation,
            "estimate_location_from_image",
            lambda image, k=15: dinov2_calls.append(image) or None,
        )

        calls: list[object] = []
        monkeypatch.setattr(
            geolocation, "analyze_image_content", lambda image: calls.append(image) or ([], False, None, None)
        )

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert calls == []  # analyze_image_content (Moondream2) nunca se llamó
        assert len(dinov2_calls) == 1  # estimate_location_from_image (DINOv2) SÍ se llamó, sin verse afectado
        assert outcome.visual_inferences == []
        assert outcome.visual_descriptions == {}

    @pytest.mark.asyncio
    async def test_scene_analysis_timeout_does_not_block_geolocation_of_same_photo(self, monkeypatch, respx_mock):
        """Regresión directa del bug de acoplamiento: si Moondream2 se
        queda colgado (en producción, reintentos de red de
        huggingface_hub de varios segundos cada uno), la geolocalización
        de ESA MISMA foto (DINOv2, ya calculada) no debe quedarse
        esperando indefinidamente -- `asyncio.wait_for` debe cortar tras
        `_SCENE_ANALYSIS_TIMEOUT_SECONDS` y degradar a "sin descripción",
        dejando que el resto del resultado de la foto salga con
        normalidad."""
        import time
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)
        monkeypatch.setattr(geolocation.settings, "enable_scene_analysis", True)
        monkeypatch.setattr(geolocation, "_SCENE_ANALYSIS_TIMEOUT_SECONDS", 0.05)

        monkeypatch.setattr(
            geolocation, "estimate_location_from_image",
            lambda image, k=15: geolocation.ImageLocationEstimate(
                province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.8
            ),
        )

        def _hangs_forever(image):
            time.sleep(1)  # bastante más que el timeout de 0.05s de este test
            return [], False, "nunca debería llegar a esto", "nunca debería llegar a esto"

        monkeypatch.setattr(geolocation, "analyze_image_content", _hangs_forever)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        start = time.monotonic()
        outcome = await geolocation.estimate_locations_for_posts(posts)
        elapsed = time.monotonic() - start

        # La foto sale geolocalizada con normalidad, sin descripción de contenido
        assert len(outcome.results) == 1
        assert outcome.results[0][1].province == "Madrid"
        assert outcome.visual_descriptions == {}
        # Y sobre todo: no se esperó a los 5s del hilo colgado
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_no_description_when_scene_analysis_returns_none(self, monkeypatch, respx_mock):
        """Sin descripción (modelo no disponible, o falló para esta foto en
        concreto), el permalink simplemente no aparece en el diccionario --
        no una entrada con valor None."""
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)
        monkeypatch.setattr(geolocation, "estimate_location_from_image", lambda image, k=15: None)
        monkeypatch.setattr(geolocation, "analyze_image_content", lambda image: ([], False, None, None))

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert "https://ig/1" not in outcome.visual_descriptions

    @pytest.mark.asyncio
    async def test_no_candidate_posts_returns_empty_without_network_calls(self):
        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="text", media_urls=[], permalink="https://x/1")]

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert outcome.results == []

    @pytest.mark.asyncio
    async def test_returns_every_processed_photo_unfiltered_by_confidence(self, monkeypatch, respx_mock):
        """Ya no se descarta nada por umbral aquí dentro -- eso es
        responsabilidad de quien llama (report/generator.py), para que el
        informe pueda mostrar cada foto con su confianza real."""
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [
            Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1"),
            Post(type="image", media_urls=["https://cdn.fake/2.jpg"], permalink="https://ig/2"),
        ]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))
        respx_mock.get("https://cdn.fake/2.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        # La primera imagen "vota" con confianza alta, la segunda con
        # confianza baja -- se simula sustituyendo directamente
        # estimate_location_from_image en vez de todo el índice. Ambas
        # deben aparecer en el resultado, sin filtrar.
        call_count = {"n": 0}

        def _fake_estimate(image, k=15):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return geolocation.ImageLocationEstimate(
                    province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.8
                )
            return geolocation.ImageLocationEstimate(
                province="Sevilla", confidence=0.1, k_neighbors=15, mean_similarity=0.5
            )

        monkeypatch.setattr(geolocation, "estimate_location_from_image", _fake_estimate)

        progress_events = []

        async def on_progress(stage, counts):
            progress_events.append((stage, counts))

        outcome = await geolocation.estimate_locations_for_posts(posts, progress_callback=on_progress)

        assert outcome.index_available is True
        assert len(outcome.results) == 2
        assert outcome.results[0][0] == "https://ig/1"
        assert outcome.results[0][1].province == "Madrid"
        assert outcome.results[0][1].confidence == 0.9
        assert outcome.results[1][1].province == "Sevilla"
        assert outcome.results[1][1].confidence == 0.1

        assert progress_events == [
            ("Geolocalizando fotos...", {"photos_analyzed": 1, "total_photos": 2, "track": "geolocalizacion"}),
            ("Analizando fotos...", {"photos_analyzed": 1, "total_photos": 2, "track": "fotos"}),
            ("Geolocalizando fotos...", {"photos_analyzed": 2, "total_photos": 2, "track": "geolocalizacion"}),
            ("Analizando fotos...", {"photos_analyzed": 2, "total_photos": 2, "track": "fotos"}),
        ]

    @pytest.mark.asyncio
    async def test_model_inference_does_not_block_the_event_loop(self, monkeypatch, respx_mock):
        """Regresión: `estimate_location_from_image` es síncrona y hace
        trabajo de CPU real (la pasada del modelo) -- llamarla directamente
        bloquearía el hilo único del event loop mientras dura, impidiendo
        que CUALQUIER otra tarea (incluida la llamada a Mistral que corre
        en paralelo, ver analysis_router._build_report) avance mientras
        tanto. Se simula con un `time.sleep` (bloqueante de verdad, a
        diferencia de `asyncio.sleep`) dentro de la función "de modelo", y
        se comprueba que otra tarea concurrente sí progresa durante ese
        rato -- solo es posible si `estimate_locations_for_posts` la manda
        a un hilo aparte (`asyncio.to_thread`) en vez de llamarla en línea."""
        import time
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        def _blocking_estimate(image, k=15):
            time.sleep(0.2)  # bloqueante de verdad -- simula la pasada del modelo
            return geolocation.ImageLocationEstimate(
                province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.8
            )

        monkeypatch.setattr(geolocation, "estimate_location_from_image", _blocking_estimate)

        other_task_progress = []

        async def _other_concurrent_work():
            for _ in range(4):
                await asyncio.sleep(0.05)
                other_task_progress.append(1)

        other_task = asyncio.create_task(_other_concurrent_work())

        await geolocation.estimate_locations_for_posts(posts)
        await other_task

        # Si la inferencia hubiera bloqueado el event loop, esta tarea
        # concurrente no habría podido avanzar nada mientras tanto.
        assert sum(other_task_progress) >= 2

    @pytest.mark.asyncio
    async def test_multiple_photos_are_analyzed_concurrently_not_one_at_a_time(self, monkeypatch, respx_mock):
        """Regresión: con `Settings.photo_analysis_concurrency >= 2`, la foto
        2 debe empezar a analizarse con los modelos SIN esperar a que la
        foto 1 termine del todo -- si no, el tiempo total sería la suma de
        ambas (procesamiento estrictamente secuencial), en vez de
        aproximarse al tiempo de la más lenta (procesamiento solapado).
        Se simula con dos pasadas de modelo bloqueantes de la misma
        duración y se comprueba que el tiempo total es sensiblemente menor
        que la suma de las dos, no solo mayor que la de una sola."""
        import time
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)
        monkeypatch.setattr(geolocation.settings, "photo_analysis_concurrency", 2)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [
            Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1"),
            Post(type="image", media_urls=["https://cdn.fake/2.jpg"], permalink="https://ig/2"),
        ]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))
        respx_mock.get("https://cdn.fake/2.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        _SLEEP = 0.2

        def _blocking_estimate(image, k=15):
            time.sleep(_SLEEP)  # bloqueante de verdad -- simula la pasada del modelo
            return geolocation.ImageLocationEstimate(
                province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.8
            )

        monkeypatch.setattr(geolocation, "estimate_location_from_image", _blocking_estimate)

        start = time.monotonic()
        outcome = await geolocation.estimate_locations_for_posts(posts)
        elapsed = time.monotonic() - start

        assert len(outcome.results) == 2
        # Secuencial habría tardado >= 2 * _SLEEP; solapado, sensiblemente
        # menos -- el margen (1.5x en vez de 2x) deja hueco para el propio
        # overhead de hilos/red sin que el test sea inestable.
        assert elapsed < _SLEEP * 1.5

    @pytest.mark.asyncio
    async def test_skips_image_that_fails_to_download_without_aborting(self, monkeypatch, respx_mock):
        import httpx

        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: True)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/broken.jpg"], permalink="https://ig/1")]

        respx_mock.get("https://cdn.fake/broken.jpg").mock(return_value=httpx.Response(500))

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert outcome.results == []

    @pytest.mark.asyncio
    async def test_index_unavailable_short_circuits_without_downloading(self, monkeypatch):
        """Sin índice/dependencias, ni se intenta descargar nada -- y se
        distingue explícitamente de 'se procesaron fotos pero sin resultado'."""
        monkeypatch.setattr(geolocation, "_geolocation_available", lambda: False)

        Post = namedtuple("Post", ["type", "media_urls", "permalink"])
        posts = [Post(type="image", media_urls=["https://cdn.fake/1.jpg"], permalink="https://ig/1")]

        outcome = await geolocation.estimate_locations_for_posts(posts)

        assert outcome.index_available is False
        assert outcome.results == []
