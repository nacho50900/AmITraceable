"""
Tests de la carga de Moondream2 en app/vision/scene_analysis.py:
`_scene_analysis_available()`, `_ensure_quantized_model()` (cuantización
cacheada en disco) y `_lazy_load()`. Los tests de análisis de imagen viven
en tests/test_scene_analysis.py.

Nada real: `llama_cpp`, `torch` y `huggingface_hub` se sustituyen por módulos
falsos en `sys.modules` y la caché de cuantización se redirige a `tmp_path`.
`_ensure_quantized_model` importa `shutil`/`subprocess` DENTRO de la función,
así que se parchean sobre los módulos reales.
"""
import subprocess
import sys
import types
from pathlib import Path as RealPath

import pytest

from app.vision import scene_analysis


@pytest.fixture(autouse=True)
def reset_module_globals(monkeypatch):
    monkeypatch.setattr(scene_analysis, "_model", None)
    monkeypatch.setattr(scene_analysis, "_actual_device", None)
    monkeypatch.setattr(scene_analysis, "_loaded_model_name", None)


@pytest.fixture
def redirected_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        scene_analysis,
        "Path",
        lambda p: tmp_path / RealPath(p).relative_to("/"),
    )
    return tmp_path / "root/.cache/huggingface/moondream2-quantized"


def _fake_torch(monkeypatch, cuda: bool):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", module)


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    """`llama_cpp.Llama` y `llama_cpp.llama_chat_format.MoondreamChatHandler`
    falsos, registrando con qué argumentos se les llamó."""
    calls = {"init": [], "from_pretrained": [], "handler": []}

    class FakeLlama:
        def __init__(self, **kwargs):
            calls["init"].append(kwargs)

        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["from_pretrained"].append(kwargs)
            return object.__new__(cls)

    class FakeHandler:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["handler"].append(kwargs)
            return "handler-falso"

    package = types.ModuleType("llama_cpp")
    package.Llama = FakeLlama
    chat_format = types.ModuleType("llama_cpp.llama_chat_format")
    chat_format.MoondreamChatHandler = FakeHandler
    package.llama_chat_format = chat_format
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", chat_format)
    return calls


class TestSceneAnalysisAvailable:
    def test_true_when_llama_cpp_importable(self, fake_llama_cpp):
        assert scene_analysis._scene_analysis_available() is True

    def test_false_when_llama_cpp_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "llama_cpp", None)

        assert scene_analysis._scene_analysis_available() is False


class TestEnsureQuantizedModel:
    def test_none_without_llama_quantize(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)

        assert scene_analysis._ensure_quantized_model() is None

    def test_returns_cached_file_without_downloading(self, monkeypatch, redirected_cache):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("MOONDREAM_QUANT_TYPE", raising=False)
        redirected_cache.mkdir(parents=True)
        cached = redirected_cache / "moondream2-text-model-q8_0.gguf"
        cached.write_bytes(b"gguf")
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # descargar fallaría

        assert scene_analysis._ensure_quantized_model() == (str(cached), "Q8_0")

    def test_quant_type_from_env_is_normalized(self, monkeypatch, redirected_cache):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.setenv("MOONDREAM_QUANT_TYPE", " q4_k_m ")
        redirected_cache.mkdir(parents=True)
        cached = redirected_cache / "moondream2-text-model-q4_k_m.gguf"
        cached.write_bytes(b"gguf")

        assert scene_analysis._ensure_quantized_model() == (str(cached), "Q4_K_M")

    def test_quantizes_and_renames_tmp_on_success(self, monkeypatch, redirected_cache):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("MOONDREAM_QUANT_TYPE", raising=False)
        hf_calls = []
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda **kwargs: hf_calls.append(kwargs) or "/base/f16.gguf"
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        run_args = []

        def fake_run(cmd, **kwargs):
            run_args.append(cmd)
            RealPath(cmd[2]).write_bytes(b"cuantizado")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = scene_analysis._ensure_quantized_model()

        final_path = redirected_cache / "moondream2-text-model-q8_0.gguf"
        assert result == (str(final_path), "Q8_0")
        assert final_path.read_bytes() == b"cuantizado"
        assert not (redirected_cache / "moondream2-text-model-q8_0.gguf.tmp").exists()
        assert hf_calls == [
            {"repo_id": scene_analysis._GGUF_REPO_ID, "filename": scene_analysis._GGUF_TEXT_MODEL_FILENAME_EXACT}
        ]
        assert run_args[0][0] == "/usr/bin/llama-quantize"
        assert run_args[0][1] == "/base/f16.gguf"
        assert run_args[0][3] == "Q8_0"

    def test_none_and_no_final_file_when_quantize_fails(self, monkeypatch, redirected_cache, caplog):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("MOONDREAM_QUANT_TYPE", raising=False)
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda **kwargs: "/base/f16.gguf"
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fallo de cuantización"),
        )

        with caplog.at_level("WARNING", logger=scene_analysis.logger.name):
            result = scene_analysis._ensure_quantized_model()

        assert result is None
        assert not (redirected_cache / "moondream2-text-model-q8_0.gguf").exists()
        assert "fallo de cuantización" in caplog.text

    def test_none_when_download_raises(self, monkeypatch, redirected_cache, caplog):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("MOONDREAM_QUANT_TYPE", raising=False)

        def _raise(**kwargs):
            raise ConnectionError("sin red")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = _raise
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with caplog.at_level("WARNING", logger=scene_analysis.logger.name):
            assert scene_analysis._ensure_quantized_model() is None

        assert "ConnectionError" in caplog.text


class TestLazyLoad:
    def test_noop_when_already_loaded(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(scene_analysis, "_model", sentinel)
        monkeypatch.setitem(sys.modules, "llama_cpp", None)  # si importara, fallaría

        scene_analysis._lazy_load()

        assert scene_analysis._model is sentinel

    def test_loads_quantized_text_model_on_gpu(self, monkeypatch, fake_llama_cpp):
        _fake_torch(monkeypatch, cuda=True)
        monkeypatch.setattr(scene_analysis, "_ensure_quantized_model", lambda: ("/cache/q8.gguf", "Q8_0"))

        scene_analysis._lazy_load()

        assert scene_analysis._model is not None
        assert fake_llama_cpp["from_pretrained"] == []
        kwargs = fake_llama_cpp["init"][0]
        assert kwargs["model_path"] == "/cache/q8.gguf"
        assert kwargs["n_gpu_layers"] == -1
        assert kwargs["n_ctx"] == 2048
        assert kwargs["chat_handler"] == "handler-falso"
        assert fake_llama_cpp["handler"][0]["repo_id"] == scene_analysis._GGUF_REPO_ID
        assert scene_analysis.get_device() == "cuda"
        assert scene_analysis.get_model_variant() == f"{scene_analysis._GGUF_REPO_ID} (Q8_0, texto cuantizado)"

    def test_cpu_when_no_gpu(self, monkeypatch, fake_llama_cpp):
        _fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr(scene_analysis, "_ensure_quantized_model", lambda: ("/cache/q8.gguf", "Q8_0"))

        scene_analysis._lazy_load()

        assert fake_llama_cpp["init"][0]["n_gpu_layers"] == 0
        assert scene_analysis.get_device() == "cpu"

    def test_falls_back_to_f16_from_pretrained_when_not_quantized(self, monkeypatch, fake_llama_cpp):
        _fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr(scene_analysis, "_ensure_quantized_model", lambda: None)

        scene_analysis._lazy_load()

        assert fake_llama_cpp["init"] == []
        kwargs = fake_llama_cpp["from_pretrained"][0]
        assert kwargs["repo_id"] == scene_analysis._GGUF_REPO_ID
        assert kwargs["filename"] == scene_analysis._GGUF_TEXT_MODEL_FILENAME
        assert kwargs["chat_handler"] == "handler-falso"
        assert scene_analysis.get_model_variant() == f"{scene_analysis._GGUF_REPO_ID} (F16)"
