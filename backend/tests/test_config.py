"""
Tests de app/config.py -- concretamente de _default_photo_analysis_concurrency,
la heurística GPU/CPU-aware para cuántas fotos se analizan en paralelo con
los modelos de visión (ver docstring de la función y del campo
Settings.photo_analysis_concurrency).
"""
import sys
import time
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
        mantiene la heurística de CPU original (mitad de los núcleos) --
        tras agotar los reintentos (ver siguiente test)."""
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(config.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(time, "sleep", lambda seconds: None)  # no esperar de verdad en el test

        assert config._default_photo_analysis_concurrency() == 4  # 8 // 2

    def test_retries_before_falling_back_to_cpu_heuristic(self, monkeypatch):
        """BUG REAL en producción (GTX 1650, Docker Desktop + WSL2): la
        primera llamada a `torch.cuda.is_available()` en el proceso --
        disparada por el primer `from app.config import settings`,
        normalmente muy temprano en el arranque -- puede devolver `False`
        de forma transitoria si el passthrough de GPU todavía no está
        listo, aunque la GPU sí esté disponible unos cientos de ms
        después. Debe reintentar antes de rendirse a la heurística de
        CPU, no fiarse del primer intento."""
        call_count = {"n": 0}

        def fake_is_available():
            call_count["n"] += 1
            # Falla las 2 primeras veces, disponible a la 3ª -- simula la
            # GPU apareciendo tras un pequeño retraso de arranque.
            return call_count["n"] >= 3

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=fake_is_available))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(config.os, "cpu_count", lambda: 16)  # no debería importar: se recupera antes
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

        assert config._default_photo_analysis_concurrency() == 1
        assert call_count["n"] == 3
        assert sleeps == [0.5, 0.5]  # una espera corta entre cada uno de los 2 primeros intentos fallidos

    def test_gives_up_after_three_attempts(self, monkeypatch):
        """Si la GPU sigue sin aparecer tras los reintentos, no se queda
        esperando indefinidamente -- cae a la heurística de CPU con como
        mucho 3 intentos en total."""
        call_count = {"n": 0}

        def fake_is_available():
            call_count["n"] += 1
            return False

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=fake_is_available))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(config.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(time, "sleep", lambda seconds: None)

        assert config._default_photo_analysis_concurrency() == 4  # 8 // 2
        assert call_count["n"] == 3

    def test_falls_back_to_cpu_heuristic_when_torch_not_installed(self, monkeypatch):
        """Builds sin WITH_GEOLOCATION no tienen torch instalado en
        absoluto (ver Dockerfile) -- no debe fallar con ImportError, debe
        seguir la heurística de CPU de siempre, exactamente igual que si
        torch estuviera instalado pero sin CUDA. No debe intentar
        reintentar/esperar en este caso (no hay torch que consultar)."""
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


class TestSceneAnalysisTimeoutSeconds:
    """Ver docstring de Settings.scene_analysis_timeout_seconds: 60s por
    defecto (no 30s), y configurable por variable de entorno -- visto en
    producción que 30s deja un margen demasiado ajustado en GPUs modestas
    (p. ej. ~25s de inferencia real en una GTX 1650), provocando que casi
    toda foto se descartara sin descripción."""

    def test_default_is_60_not_the_original_30(self):
        assert config.settings.scene_analysis_timeout_seconds == 60

    def test_configurable_via_env_var(self, monkeypatch):
        monkeypatch.setenv("SCENE_ANALYSIS_TIMEOUT_SECONDS", "90")

        fresh_settings = config.Settings()

        assert fresh_settings.scene_analysis_timeout_seconds == 90
