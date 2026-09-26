"""
Tests del arranque de app/main.py que test_lifespan.py no cubre: el reparto
de hilos de PyTorch, el log explícito de disponibilidad de GPU y la
degradación cuando falla la precarga de Moondream2.

`torch` no es dependencia de test (ver requirements-vision.txt): se
sustituye por un módulo falso en `sys.modules`.
"""
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app


def _install_fake_torch(monkeypatch, *, cuda: bool, device_name: str = "NVIDIA GTX 1650", cuda_version: str = "12.1"):
    module = types.ModuleType("torch")
    module.set_num_threads = MagicMock()
    module.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_name=lambda index: device_name,
    )
    module.version = types.SimpleNamespace(cuda=cuda_version)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


class TestConfigurePytorchThreads:
    @pytest.mark.parametrize(
        "cpu_count, concurrency, expected_threads",
        [
            (8, 2, 4),  # reparto normal
            (8, 1, 8),  # una sola inferencia a la vez: todos los núcleos
            (4, 8, 1),  # más concurrencia que núcleos: nunca 0 hilos
            (8, 0, 8),  # concurrencia inválida se trata como 1
        ],
    )
    def test_splits_cores_between_concurrent_inferences(
        self, monkeypatch, cpu_count, concurrency, expected_threads
    ):
        fake_torch = _install_fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr("os.cpu_count", lambda: cpu_count)
        monkeypatch.setattr(main.settings, "photo_analysis_concurrency", concurrency)
        monkeypatch.setattr(main, "_log_gpu_availability", lambda c: None)

        main._configure_pytorch_threads_and_log_gpu()

        fake_torch.set_num_threads.assert_called_once_with(expected_threads)

    def test_falls_back_to_four_cores_when_undetectable(self, monkeypatch):
        fake_torch = _install_fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr("os.cpu_count", lambda: None)
        monkeypatch.setattr(main.settings, "photo_analysis_concurrency", 2)
        monkeypatch.setattr(main, "_log_gpu_availability", lambda c: None)

        main._configure_pytorch_threads_and_log_gpu()

        fake_torch.set_num_threads.assert_called_once_with(2)

    def test_logs_resulting_distribution(self, monkeypatch, caplog):
        _install_fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        monkeypatch.setattr(main.settings, "photo_analysis_concurrency", 2)
        monkeypatch.setattr(main, "_log_gpu_availability", lambda c: None)

        with caplog.at_level(logging.INFO, logger=main.logger.name):
            main._configure_pytorch_threads_and_log_gpu()

        assert "8 núcleos detectados" in caplog.text
        assert "hilos por inferencia=4" in caplog.text

    def test_passes_concurrency_to_gpu_log(self, monkeypatch):
        _install_fake_torch(monkeypatch, cuda=False)
        monkeypatch.setattr("os.cpu_count", lambda: 8)
        monkeypatch.setattr(main.settings, "photo_analysis_concurrency", 3)
        seen = []
        monkeypatch.setattr(main, "_log_gpu_availability", seen.append)

        main._configure_pytorch_threads_and_log_gpu()

        assert seen == [3]

    def test_silently_skips_when_torch_not_installed(self, monkeypatch):
        # `None` en sys.modules hace que `import torch` lance ImportError
        monkeypatch.setitem(sys.modules, "torch", None)

        main._configure_pytorch_threads_and_log_gpu()  # no debe lanzar


class TestLogGpuAvailability:
    def test_logs_cpu_fallback_when_no_gpu(self, monkeypatch, caplog):
        _install_fake_torch(monkeypatch, cuda=False)

        with caplog.at_level(logging.INFO, logger=main.logger.name):
            main._log_gpu_availability(concurrency=4)

        assert "GPU no detectada" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_logs_device_name_and_cuda_version_when_gpu_present(self, monkeypatch, caplog):
        _install_fake_torch(monkeypatch, cuda=True, device_name="NVIDIA GTX 1650", cuda_version="12.1")

        with caplog.at_level(logging.INFO, logger=main.logger.name):
            main._log_gpu_availability(concurrency=1)

        assert "GPU detectada: NVIDIA GTX 1650 (CUDA 12.1)" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_warns_when_gpu_present_but_concurrency_is_not_one(self, monkeypatch, caplog):
        _install_fake_torch(monkeypatch, cuda=True)

        with caplog.at_level(logging.INFO, logger=main.logger.name):
            main._log_gpu_availability(concurrency=4)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "photo_analysis_concurrency=4" in warnings[0].getMessage()


class TestLifespanSceneAnalysisLoadFailure:
    def test_startup_survives_moondream_load_error(self, monkeypatch, caplog):
        """Si la precarga de Moondream2 falla, el backend debe arrancar igual
        (Reddit/Instagram no dependen de él) y dejar el traceback en el log."""
        failing_load = MagicMock(side_effect=RuntimeError("OOM cargando Moondream2"))
        monkeypatch.setattr("app.config.settings.enable_scene_analysis", True)
        monkeypatch.setattr("app.vision.geolocation._geolocation_available", lambda: False)
        monkeypatch.setattr("app.vision.scene_analysis._scene_analysis_available", lambda: True)
        monkeypatch.setattr("app.vision.scene_analysis._lazy_load", failing_load)

        with caplog.at_level(logging.ERROR, logger=main.logger.name):
            with TestClient(app) as client:
                assert client.get("/").status_code == 200

        failing_load.assert_called_once()
        assert "Fallo al cargar el modelo de analisis de contenido" in caplog.text
        assert any(r.exc_info for r in caplog.records)  # traceback completo
