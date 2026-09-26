"""
Tests de la carga de modelo en app/vision/scene_analysis.py:
`_scene_analysis_available()`, `_lazy_load()`, `get_device()` y
`get_model_variant()`. Los tests de análisis de imagen viven en
tests/test_scene_analysis.py.

Desde ADR-49bis, este módulo YA NO carga ningún modelo propio (antes
Moondream2, con su propia cuantización cacheada en disco) -- todo eso
vive ahora en app/nlp/ai_client.py (Qwen3.5-4B compartido con el resto
del proyecto, ver tests/test_ai_client.py para la cobertura de la carga
real). Lo único que queda por probar aquí es que este módulo DELEGA
correctamente en ai_client: pide la carga, y copia sus valores."""
import pytest

from app.nlp import ai_client
from app.vision import scene_analysis


@pytest.fixture(autouse=True)
def reset_module_globals(monkeypatch):
    monkeypatch.setattr(scene_analysis, "_model", None)
    monkeypatch.setattr(scene_analysis, "_actual_device", None)
    monkeypatch.setattr(scene_analysis, "_loaded_model_name", None)


class TestSceneAnalysisAvailable:
    def test_delegates_to_ai_client_vision_available(self, monkeypatch):
        monkeypatch.setattr(ai_client, "vision_available", lambda: True)

        assert scene_analysis._scene_analysis_available() is True

        monkeypatch.setattr(ai_client, "vision_available", lambda: False)

        assert scene_analysis._scene_analysis_available() is False


class TestLazyLoad:
    def test_noop_when_already_loaded(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(scene_analysis, "_model", sentinel)

        def _fail_if_called():
            raise AssertionError("no debería llamar a ai_client.ensure_loaded() si ya hay modelo")

        monkeypatch.setattr(ai_client, "ensure_loaded", _fail_if_called)

        scene_analysis._lazy_load()

        assert scene_analysis._model is sentinel

    def test_delegates_to_ai_client_and_copies_state(self, monkeypatch):
        calls = []
        fake_model = object()
        monkeypatch.setattr(ai_client, "ensure_loaded", lambda: calls.append("ensure_loaded"))
        monkeypatch.setattr(ai_client, "get_model", lambda: fake_model)
        monkeypatch.setattr(ai_client, "get_device", lambda: "cuda")
        monkeypatch.setattr(ai_client, "get_model_variant", lambda: "Qwen3.5-4B (fake.gguf, con visión)")

        scene_analysis._lazy_load()

        assert calls == ["ensure_loaded"]
        assert scene_analysis._model is fake_model
        assert scene_analysis.get_device() == "cuda"
        assert scene_analysis.get_model_variant() == "Qwen3.5-4B (fake.gguf, con visión)"

    def test_second_call_does_not_call_ai_client_again(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ai_client, "ensure_loaded", lambda: calls.append("ensure_loaded"))
        monkeypatch.setattr(ai_client, "get_model", lambda: object())
        monkeypatch.setattr(ai_client, "get_device", lambda: "cpu")
        monkeypatch.setattr(ai_client, "get_model_variant", lambda: "Qwen3.5-4B")

        scene_analysis._lazy_load()
        scene_analysis._lazy_load()

        assert calls == ["ensure_loaded"]
