"""
Tests de app/vision/geolocation.py.

No se descarga ningún modelo real ni se necesita un índice FAISS
construido: se mockea `_lazy_load` (y las variables de módulo que rellena)
para ejercer toda la lógica de negocio (votación de vecinos, centroide,
degradación best-effort) sin dependencias externas pesadas.
"""
from collections import namedtuple
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.vision import geolocation


class _FakeImage:
    """Sustituye a PIL.Image: solo necesita soportar .convert('RGB')."""

    def convert(self, mode):
        return self


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


class TestEstimateLocationsForPosts:
    @pytest.mark.asyncio
    async def test_no_candidate_posts_returns_empty_without_network_calls(self):
        Post = namedtuple("Post", ["type", "media_url", "permalink"])
        posts = [Post(type="text", media_url=None, permalink="https://x/1")]

        results = await geolocation.estimate_locations_for_posts(posts)

        assert results == []

    @pytest.mark.asyncio
    async def test_reports_progress_per_photo_and_filters_by_confidence(self, monkeypatch, respx_mock):
        import httpx

        Post = namedtuple("Post", ["type", "media_url", "permalink"])
        posts = [
            Post(type="image", media_url="https://cdn.fake/1.jpg", permalink="https://ig/1"),
            Post(type="image", media_url="https://cdn.fake/2.jpg", permalink="https://ig/2"),
        ]

        tiny_jpeg = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202030202"
            "020304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e"
            "0b0b1016101113141515150c0f171816141812141514ffc9000b0800010001010111"
            "00ffcc00060010100501ffda0008010100003f00d2cf20ffd9"
        )
        respx_mock.get("https://cdn.fake/1.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))
        respx_mock.get("https://cdn.fake/2.jpg").mock(return_value=httpx.Response(200, content=tiny_jpeg))

        # La primera imagen "vota" con confianza alta (pasa el filtro), la
        # segunda con confianza baja (se descarta) -- se simula sustituyendo
        # directamente estimate_location_from_image en vez de todo el índice.
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

        results = await geolocation.estimate_locations_for_posts(
            posts, min_confidence=0.4, progress_callback=on_progress
        )

        assert len(results) == 1
        assert results[0][0] == "https://ig/1"
        assert results[0][1].province == "Madrid"

        assert progress_events == [
            ("Analizando fotos...", {"photos_analyzed": 1, "total_photos": 2}),
            ("Analizando fotos...", {"photos_analyzed": 2, "total_photos": 2}),
        ]

    @pytest.mark.asyncio
    async def test_skips_image_that_fails_to_download_without_aborting(self, monkeypatch, respx_mock):
        import httpx

        Post = namedtuple("Post", ["type", "media_url", "permalink"])
        posts = [Post(type="image", media_url="https://cdn.fake/broken.jpg", permalink="https://ig/1")]

        respx_mock.get("https://cdn.fake/broken.jpg").mock(return_value=httpx.Response(500))

        results = await geolocation.estimate_locations_for_posts(posts)

        assert results == []
