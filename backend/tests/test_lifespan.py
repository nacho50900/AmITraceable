"""
Tests del lifespan de app/main.py: la precarga del modelo de
geolocalización al arrancar el contenedor, en vez de en la primera
petición de análisis (ver docstring de _lifespan en main.py).
"""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app


class TestLifespanGeolocationPreload:
    def test_preloads_model_when_geolocation_is_available(self, monkeypatch):
        fake_lazy_load = MagicMock()
        monkeypatch.setattr("app.vision.geolocation._geolocation_available", lambda: True)
        monkeypatch.setattr("app.vision.geolocation._lazy_load", fake_lazy_load)

        # Entrar en el `with` dispara el lifespan de arranque (y al salir,
        # el de apagado) -- así se ejerce _lifespan() de verdad, sin tener
        # que levantar el servidor ASGI completo.
        with TestClient(app):
            pass

        fake_lazy_load.assert_called_once()

    def test_skips_preload_when_geolocation_is_not_available(self, monkeypatch):
        fake_lazy_load = MagicMock()
        monkeypatch.setattr("app.vision.geolocation._geolocation_available", lambda: False)
        monkeypatch.setattr("app.vision.geolocation._lazy_load", fake_lazy_load)

        with TestClient(app):
            pass

        fake_lazy_load.assert_not_called()
