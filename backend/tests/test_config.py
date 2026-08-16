"""
Tests de app/config.py -- concretamente de _default_photo_analysis_concurrency,
la heurística GPU/CPU-aware para cuántas fotos se analizan en paralelo con
los modelos de visión (ver docstring de la función y del campo
Settings.photo_analysis_concurrency).
"""
import sys
import types

from app import config


class TestDefaultPhotoAnalysisConcurrency:
    def test_returns_1_when_cuda_available(self, monkeypatch):
        """En GPU, el recurso limitante es la propia tarjeta (VRAM y
        cómputo compartidos entre DINOv2 y Moondream2), no los núcleos de
        CPU -- varias fotos a la vez no aportan throughput real, solo
        contención (y riesgo de quedarse sin VRAM en tarjetas pequeñas).
        Un análisis a la vez es lo correcto, sea cual sea el nº de
        núcleos de CPU de la máquina."""
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(config.os, "cpu_count", lambda: 16)  # no debería importar en absoluto

        assert config._default_photo_analysis_concurrency() == 1

    def test_falls_back_to_cpu_heuristic_when_cuda_not_available(self, monkeypatch):
        """Con torch instalado pero SIN GPU (CUDA no disponible), se
        mantiene la heurística de CPU original (mitad de los núcleos)."""
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(config.os, "cpu_count", lambda: 8)

        assert config._default_photo_analysis_concurrency() == 4  # 8 // 2

    def test_falls_back_to_cpu_heuristic_when_torch_not_installed(self, monkeypatch):
        """Builds sin WITH_GEOLOCATION no tienen torch instalado en
        absoluto (ver Dockerfile) -- no debe fallar con ImportError, debe
        seguir la heurística de CPU de siempre, exactamente igual que si
        torch estuviera instalado pero sin CUDA."""
        monkeypatch.setitem(sys.modules, "torch", None)  # fuerza ImportError al hacer `import torch`
        monkeypatch.setattr(config.os, "cpu_count", lambda: 4)

        assert config._default_photo_analysis_concurrency() == 2  # 4 // 2

    def test_cpu_heuristic_never_returns_less_than_1(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setattr(config.os, "cpu_count", lambda: 1)

        assert config._default_photo_analysis_concurrency() == 1

    def test_cpu_heuristic_handles_cpu_count_none(self, monkeypatch):
        """os.cpu_count() puede devolver None en entornos donde no se
        puede determinar (documentado en la stdlib) -- no debe reventar,
        cae a un valor por defecto de 4 núcleos."""
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setattr(config.os, "cpu_count", lambda: None)

        assert config._default_photo_analysis_concurrency() == 2  # max(1, 4 // 2)
