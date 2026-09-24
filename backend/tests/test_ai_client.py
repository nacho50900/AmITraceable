"""
Tests de app/nlp/ai_client.py (cliente del modelo de IA local Qwen3.5-4B).

No se carga ningún modelo real ni se descarga nada: `llama_cpp`, `torch` y
`huggingface_hub` no son dependencias de test (ver requirements-vision.txt),
así que se sustituyen por módulos falsos en `sys.modules`, y el sistema de
ficheros de la caché de cuantización se redirige a `tmp_path`.
"""
import json
import subprocess
import sys
import types
from pathlib import Path as RealPath

import pytest

from app.config import settings
from app.nlp import ai_client
from app.nlp.ai_client import AIHTTPError, AIRequestError, call_ai_json


@pytest.fixture(autouse=True)
def reset_module_globals(monkeypatch):
    """Cada test parte sin modelo cargado y sin variante registrada."""
    monkeypatch.setattr(ai_client, "_model", None)
    monkeypatch.setattr(ai_client, "_loaded_model_name", None)


@pytest.fixture
def configured_model(monkeypatch):
    monkeypatch.setattr(settings, "qwen_gguf_repo_id", "fake/repo")
    monkeypatch.setattr(settings, "qwen_gguf_filename", "fake-model.gguf")


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    """Módulo `llama_cpp` falso: `Llama(...)` y `Llama.from_pretrained(...)`
    registran los kwargs con los que se les llamó."""
    calls = {"init": [], "from_pretrained": []}

    class FakeLlama:
        def __init__(self, **kwargs):
            calls["init"].append(kwargs)

        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["from_pretrained"].append(kwargs)
            return object.__new__(cls)  # sin pasar por __init__: no cuenta como Llama(...)

    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return calls


def _fake_torch(monkeypatch, cuda: bool):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", module)


@pytest.fixture
def redirected_cache(monkeypatch, tmp_path):
    """`_ensure_quantized_model` construye rutas absolutas bajo
    /root/.cache/...; se reencaminan a `tmp_path` para no tocar el disco
    real."""
    monkeypatch.setattr(
        ai_client,
        "Path",
        lambda p: tmp_path / RealPath(p).relative_to("/"),
    )
    return tmp_path / "root/.cache/huggingface/qwen3.5-4b-quantized"


class TestErrors:
    def test_http_error_keeps_status_and_body(self):
        err = AIHTTPError(200, "cuerpo del fallo")

        assert err.status_code == 200
        assert err.body == "cuerpo del fallo"
        assert "HTTP 200" in str(err)

    def test_http_error_truncates_long_body_in_message_only(self):
        err = AIHTTPError(200, "x" * 1000)

        assert len(err.body) == 1000
        assert len(str(err)) < 400


class TestGetModelVariant:
    def test_none_before_loading(self):
        assert ai_client.get_model_variant() is None

    def test_returns_loaded_name(self, monkeypatch):
        monkeypatch.setattr(ai_client, "_loaded_model_name", "Qwen3.5-4B (Q4_K_M)")

        assert ai_client.get_model_variant() == "Qwen3.5-4B (Q4_K_M)"


class TestQwenAvailable:
    def test_false_without_repo_id(self, monkeypatch):
        monkeypatch.setattr(settings, "qwen_gguf_repo_id", "")
        monkeypatch.setattr(settings, "qwen_gguf_filename", "fake-model.gguf")

        assert ai_client._qwen_available() is False

    def test_false_without_filename(self, monkeypatch):
        monkeypatch.setattr(settings, "qwen_gguf_repo_id", "fake/repo")
        monkeypatch.setattr(settings, "qwen_gguf_filename", "")

        assert ai_client._qwen_available() is False

    def test_false_when_llama_cpp_not_installed(self, configured_model, monkeypatch):
        # `None` en sys.modules hace que `import llama_cpp` lance ImportError
        monkeypatch.setitem(sys.modules, "llama_cpp", None)

        assert ai_client._qwen_available() is False

    def test_true_when_configured_and_installed(self, configured_model, fake_llama_cpp):
        assert ai_client._qwen_available() is True


class TestEnsureQuantizedModel:
    def test_none_without_llama_quantize(self, monkeypatch):
        monkeypatch.setattr(ai_client.shutil, "which", lambda name: None)

        assert ai_client._ensure_quantized_model() is None

    def test_returns_cached_file_without_downloading(self, monkeypatch, redirected_cache):
        monkeypatch.setattr(ai_client.shutil, "which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("QWEN_QUANT_TYPE", raising=False)
        redirected_cache.mkdir(parents=True)
        cached = redirected_cache / "qwen3.5-4b-q4_k_m.gguf"
        cached.write_bytes(b"gguf")
        # Si intentara descargar, fallaría: no hay huggingface_hub
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        assert ai_client._ensure_quantized_model() == (str(cached), "Q4_K_M")

    def test_quant_type_from_env_is_normalized(self, monkeypatch, redirected_cache):
        monkeypatch.setattr(ai_client.shutil, "which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.setenv("QWEN_QUANT_TYPE", "  q5_k_m ")
        redirected_cache.mkdir(parents=True)
        cached = redirected_cache / "qwen3.5-4b-q5_k_m.gguf"
        cached.write_bytes(b"gguf")

        assert ai_client._ensure_quantized_model() == (str(cached), "Q5_K_M")

    def test_quantizes_and_renames_tmp_on_success(self, configured_model, monkeypatch, redirected_cache):
        monkeypatch.setattr(ai_client.shutil, "which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("QWEN_QUANT_TYPE", raising=False)
        hf_calls = []
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda **kwargs: hf_calls.append(kwargs) or "/base/model.gguf"
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        run_args = []

        def fake_run(cmd, **kwargs):
            run_args.append(cmd)
            RealPath(cmd[2]).write_bytes(b"cuantizado")  # llama-quantize escribe el .tmp
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(ai_client.subprocess, "run", fake_run)

        result = ai_client._ensure_quantized_model()

        final_path = redirected_cache / "qwen3.5-4b-q4_k_m.gguf"
        assert result == (str(final_path), "Q4_K_M")
        assert final_path.read_bytes() == b"cuantizado"
        assert not (redirected_cache / "qwen3.5-4b-q4_k_m.gguf.tmp").exists()
        assert hf_calls == [{"repo_id": "fake/repo", "filename": "fake-model.gguf"}]
        assert run_args[0][0] == "/usr/bin/llama-quantize"
        assert run_args[0][1] == "/base/model.gguf"
        assert run_args[0][3] == "Q4_K_M"

    def test_none_and_no_final_file_when_quantize_fails(self, configured_model, monkeypatch, redirected_cache, caplog):
        monkeypatch.setattr(ai_client.shutil, "which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("QWEN_QUANT_TYPE", raising=False)
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda **kwargs: "/base/model.gguf"
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        monkeypatch.setattr(
            ai_client.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fallo de cuantización"),
        )

        with caplog.at_level("WARNING", logger=ai_client.logger.name):
            result = ai_client._ensure_quantized_model()

        assert result is None
        assert not (redirected_cache / "qwen3.5-4b-q4_k_m.gguf").exists()
        assert "fallo de cuantización" in caplog.text

    def test_none_when_download_raises(self, configured_model, monkeypatch, redirected_cache, caplog):
        monkeypatch.setattr(ai_client.shutil, "which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("QWEN_QUANT_TYPE", raising=False)

        def _raise(**kwargs):
            raise ConnectionError("sin red")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = _raise
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with caplog.at_level("WARNING", logger=ai_client.logger.name):
            assert ai_client._ensure_quantized_model() is None

        assert "ConnectionError" in caplog.text

    def test_none_when_huggingface_hub_missing(self, configured_model, monkeypatch, redirected_cache):
        monkeypatch.setattr(ai_client.shutil, "which", lambda name: "/usr/bin/llama-quantize")
        monkeypatch.delenv("QWEN_QUANT_TYPE", raising=False)
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        assert ai_client._ensure_quantized_model() is None


class TestLazyLoad:
    def test_noop_when_already_loaded(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(ai_client, "_model", sentinel)
        # Si intentara importar, fallaría
        monkeypatch.setitem(sys.modules, "llama_cpp", None)

        ai_client._lazy_load()

        assert ai_client._model is sentinel

    def test_loads_quantized_model_with_gpu(self, monkeypatch, fake_llama_cpp):
        _fake_torch(monkeypatch, cuda=True)
        monkeypatch.setattr(ai_client, "_ensure_quantized_model", lambda: ("/cache/q.gguf", "Q4_K_M"))

        ai_client._lazy_load()

        assert ai_client._model is not None
        assert fake_llama_cpp["from_pretrained"] == []
        kwargs = fake_llama_cpp["init"][0]
        assert kwargs["model_path"] == "/cache/q.gguf"
        assert kwargs["n_gpu_layers"] == -1
        assert kwargs["n_ctx"] == 4096
        assert "chat_handler" not in kwargs  # solo texto
        assert ai_client.get_model_variant() == "Qwen3.5-4B (Q4_K_M)"

    def test_loads_quantized_model_on_cpu_without_gpu(self, monkeypatch, fake_llama_cpp):
        _fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr(ai_client, "_ensure_quantized_model", lambda: ("/cache/q.gguf", "Q5_K_M"))

        ai_client._lazy_load()

        assert fake_llama_cpp["init"][0]["n_gpu_layers"] == 0
        assert ai_client.get_model_variant() == "Qwen3.5-4B (Q5_K_M)"

    def test_falls_back_to_from_pretrained_when_not_quantized(self, configured_model, monkeypatch, fake_llama_cpp):
        _fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr(ai_client, "_ensure_quantized_model", lambda: None)

        ai_client._lazy_load()

        assert fake_llama_cpp["init"] == []
        kwargs = fake_llama_cpp["from_pretrained"][0]
        assert kwargs["repo_id"] == "fake/repo"
        assert kwargs["filename"] == "fake-model.gguf"
        assert kwargs["n_gpu_layers"] == 0
        assert ai_client.get_model_variant() == "Qwen3.5-4B (sin cuantizar)"


class _FakeModel:
    def __init__(self, content: str | None = None, response: dict | None = None):
        self.reset_calls = 0
        self.completion_kwargs: dict | None = None
        self._response = response if response is not None else {"choices": [{"message": {"content": content}}]}

    def reset(self):
        self.reset_calls += 1

    def create_chat_completion(self, **kwargs):
        self.completion_kwargs = kwargs
        return self._response


class TestCallAiJsonSync:
    def test_resets_cache_and_parses_json(self, monkeypatch):
        fake = _FakeModel(content=json.dumps({"veredicto": "ok"}))
        monkeypatch.setattr(ai_client, "_model", fake)

        result = ai_client._call_ai_json_sync("sistema", "usuario", 123, 0.2)

        assert result == {"veredicto": "ok"}
        assert fake.reset_calls == 1
        assert fake.completion_kwargs["max_tokens"] == 123
        assert fake.completion_kwargs["temperature"] == 0.2
        assert fake.completion_kwargs["response_format"] == {"type": "json_object"}
        assert fake.completion_kwargs["messages"] == [
            {"role": "system", "content": "sistema"},
            {"role": "user", "content": "usuario"},
        ]

    def test_loads_model_lazily_when_missing(self, monkeypatch):
        fake = _FakeModel(content="{}")

        def _load():
            monkeypatch.setattr(ai_client, "_model", fake)

        monkeypatch.setattr(ai_client, "_lazy_load", _load)

        assert ai_client._call_ai_json_sync("s", "u", 10, 0.0) == {}
        assert fake.reset_calls == 1


class TestCallAiJson:
    @pytest.mark.asyncio
    async def test_raises_request_error_when_model_unavailable(self, monkeypatch):
        monkeypatch.setattr(settings, "qwen_gguf_repo_id", "")

        with pytest.raises(AIRequestError, match="no está disponible"):
            await call_ai_json("s", "u")

    @pytest.mark.asyncio
    async def test_returns_parsed_dict_and_forwards_arguments(self, monkeypatch):
        monkeypatch.setattr(ai_client, "_qwen_available", lambda: True)
        captured = {}

        def fake_sync(system_prompt, user_prompt, max_tokens, temperature):
            captured.update(
                system=system_prompt, user=user_prompt, max_tokens=max_tokens, temperature=temperature
            )
            return {"a": 1}

        monkeypatch.setattr(ai_client, "_call_ai_json_sync", fake_sync)

        result = await call_ai_json("sis", "usr", model="ignorado", max_tokens=55, temperature=0.3)

        assert result == {"a": 1}
        assert captured == {"system": "sis", "user": "usr", "max_tokens": 55, "temperature": 0.3}

    @pytest.mark.asyncio
    async def test_defaults_max_tokens_and_temperature(self, monkeypatch):
        monkeypatch.setattr(ai_client, "_qwen_available", lambda: True)
        captured = {}

        def fake_sync(system_prompt, user_prompt, max_tokens, temperature):
            captured.update(max_tokens=max_tokens, temperature=temperature)
            return {}

        monkeypatch.setattr(ai_client, "_call_ai_json_sync", fake_sync)

        await call_ai_json("s", "u")

        assert captured == {"max_tokens": 1000, "temperature": 0.0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            KeyError("choices"),
            IndexError("lista vacía"),
            TypeError("None no es subscriptable"),
            json.JSONDecodeError("JSON inválido", "no-json", 0),
        ],
    )
    async def test_malformed_response_becomes_http_error_200(self, monkeypatch, exc):
        monkeypatch.setattr(ai_client, "_qwen_available", lambda: True)

        def _raise(*args):
            raise exc

        monkeypatch.setattr(ai_client, "_call_ai_json_sync", _raise)

        with pytest.raises(AIHTTPError) as info:
            await call_ai_json("s", "u")

        assert info.value.status_code == 200
        assert "forma inesperada" in info.value.body

    @pytest.mark.asyncio
    async def test_unexpected_exception_becomes_request_error(self, monkeypatch):
        monkeypatch.setattr(ai_client, "_qwen_available", lambda: True)

        def _raise(*args):
            raise RuntimeError("OOM en la GPU")

        monkeypatch.setattr(ai_client, "_call_ai_json_sync", _raise)

        with pytest.raises(AIRequestError, match="OOM en la GPU"):
            await call_ai_json("s", "u")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", [AIHTTPError(200, "x"), AIRequestError("y")])
    async def test_own_errors_pass_through_unchanged(self, monkeypatch, exc):
        monkeypatch.setattr(ai_client, "_qwen_available", lambda: True)

        def _raise(*args):
            raise exc

        monkeypatch.setattr(ai_client, "_call_ai_json_sync", _raise)

        with pytest.raises(type(exc)) as info:
            await call_ai_json("s", "u")

        assert info.value is exc

    @pytest.mark.asyncio
    async def test_end_to_end_with_fake_model(self, configured_model, fake_llama_cpp, monkeypatch):
        """Camino completo sin mockear internos: disponibilidad -> hilo ->
        modelo falso -> JSON parseado."""
        fake = _FakeModel(content='{"reconocido": false}')
        monkeypatch.setattr(ai_client, "_model", fake)

        assert await call_ai_json("s", "u") == {"reconocido": False}
