"""
Tests del lifespan de app/main.py: la precarga de los modelos de
geolocalización (DINOv2) y análisis de contenido (Moondream2) al
arrancar el contenedor, en vez de en la primera petición de análisis
(ver docstring de _lifespan en main.py).
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


class TestLifespanSceneAnalysisPreload:
    """Regresión directa: antes de este bloque, Moondream2 NO se
    precargaba (a diferencia de DINOv2 arriba) -- la primera foto
    analizada tras activar `enable_scene_analysis` pagaba el coste
    completo de carga del modelo (hasta ~350s en frío, medido en
    producción) DENTRO de `_SCENE_ANALYSIS_TIMEOUT_SECONDS` (30s,
    pensado para el tiempo de inferencia, no de carga), así que esa
    primera foto fallaba por timeout casi siempre."""

    def test_preloads_model_when_enabled_and_available(self, monkeypatch):
        fake_lazy_load = MagicMock()
        monkeypatch.setattr("app.config.settings.enable_scene_analysis", True)
        monkeypatch.setattr("app.vision.geolocation._geolocation_available", lambda: False)
        monkeypatch.setattr("app.vision.scene_analysis._scene_analysis_available", lambda: True)
        monkeypatch.setattr("app.vision.scene_analysis._lazy_load", fake_lazy_load)

        with TestClient(app):
            pass

        fake_lazy_load.assert_called_once()

    def test_skips_preload_when_disabled(self, monkeypatch):
        """Default: no tiene sentido pagar el coste de arranque si no se
        va a usar."""
        fake_lazy_load = MagicMock()
        monkeypatch.setattr("app.config.settings.enable_scene_analysis", False)
        monkeypatch.setattr("app.vision.geolocation._geolocation_available", lambda: False)
        monkeypatch.setattr("app.vision.scene_analysis._lazy_load", fake_lazy_load)

        with TestClient(app):
            pass

        fake_lazy_load.assert_not_called()

    def test_does_not_crash_when_enabled_but_dependencies_missing(self, monkeypatch):
        """Config inconsistente (ENABLE_SCENE_ANALYSIS=true sin
        WITH_GEOLOCATION=true en el build): debe avisar y seguir, nunca
        tumbar el arranque entero -- mismo criterio de degradación con
        gracia que ya usa analyze_image_content() por foto."""
        fake_lazy_load = MagicMock()
        monkeypatch.setattr("app.config.settings.enable_scene_analysis", True)
        monkeypatch.setattr("app.vision.geolocation._geolocation_available", lambda: False)
        monkeypatch.setattr("app.vision.scene_analysis._scene_analysis_available", lambda: False)
        monkeypatch.setattr("app.vision.scene_analysis._lazy_load", fake_lazy_load)

        with TestClient(app):
            pass  # no debe lanzar

        fake_lazy_load.assert_not_called()

