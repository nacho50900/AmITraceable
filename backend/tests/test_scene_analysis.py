"""
Tests de app/vision/scene_analysis.py.

No se descarga el modelo real (~1.8B parámetros): se mockea `_lazy_load`
(y `_model`) para ejercer toda la lógica de negocio (parseo de la
respuesta, degradación best-effort) sin dependencias pesadas.
"""
from types import SimpleNamespace

import pytest

from app.vision import scene_analysis


class _FakeModel:
    """Sustituye a Moondream2 lo justo para analyze_image_content: solo
    necesita responder a .query(image, pregunta) -> {"answer": str}."""

    def __init__(self, answer: str):
        self._answer = answer

    def query(self, image, question):
        return {"answer": self._answer}


class _RaisingModel:
    def query(self, image, question):
        raise RuntimeError("fallo simulado del modelo")


@pytest.fixture(autouse=True)
def reset_module_globals(monkeypatch):
    """Cada test debe partir de _model limpio, igual que geolocation.py."""
    monkeypatch.setattr(scene_analysis, "_model", None)
    yield


def _install_fake_model(monkeypatch, answer: str):
    monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
    monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _FakeModel(answer)))


class TestAnalyzeImageContent:
    def test_parses_aficion_and_no_pareja(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "AFICION: Posible fan de baloncesto, aparece jugando\nPAREJA: no",
        )

        inferences, indicio_pareja = scene_analysis.analyze_image_content(object())

        assert len(inferences) == 1
        assert inferences[0].category == "aficion"
        assert "baloncesto" in inferences[0].value.lower()
        assert inferences[0].confidence == 0.5
        assert inferences[0].evidence == []  # lo rellena el llamador (geolocation.py), no este módulo
        assert indicio_pareja is False

    def test_parses_pareja_true(self, monkeypatch):
        _install_fake_model(monkeypatch, "AFICION: ninguno\nPAREJA: si")

        inferences, indicio_pareja = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is True

    @pytest.mark.parametrize("negative_value", ["ninguno", "Ninguna", "none", "N/A", ""])
    def test_ninguno_variants_produce_no_inference(self, monkeypatch, negative_value):
        _install_fake_model(monkeypatch, f"AFICION: {negative_value}\nPAREJA: no")

        inferences, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    def test_missing_aficion_line_is_ignored_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "PAREJA: no")

        inferences, indicio_pareja = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_missing_pareja_line_defaults_to_false(self, monkeypatch):
        _install_fake_model(monkeypatch, "AFICION: toca la guitarra")

        inferences, indicio_pareja = scene_analysis.analyze_image_content(object())

        assert len(inferences) == 1
        assert indicio_pareja is False

    def test_completely_unexpected_format_degrades_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "esto no sigue el formato pedido en absoluto")

        inferences, indicio_pareja = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_dependencies_not_installed_returns_empty_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        inferences, indicio_pareja = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_model_exception_degrades_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel()))

        inferences, indicio_pareja = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_never_identifies_or_names_another_person(self, monkeypatch):
        """No es un test de comportamiento del modelo real (eso no se
        puede probar aquí sin descargarlo) -- es una comprobación de que
        el prompt en sí contiene la regla explícita, para que un cambio
        futuro no la borre por accidente sin darse cuenta."""
        assert "nunca" in scene_analysis._QUERY.lower()
        assert "otra persona" in scene_analysis._QUERY.lower()


class TestSceneAnalysisAvailable:
    def test_false_when_dependencies_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                raise ImportError(f"{name} no instalado")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        assert scene_analysis._scene_analysis_available() is False
