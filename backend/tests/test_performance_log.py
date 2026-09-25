"""
Tests de app/log/performance_log.py: log de rendimiento del análisis de
fotos (DINOv2 + Moondream2). Mismo patrón que tests/test_translation_log.py:
`_LOG_DIR`/`_LOG_PATH` apuntan a un directorio temporal, nunca a
`backend/data/performance/` de verdad.
"""
import json

import pytest

from app.log import performance_log
from app.log.performance_log import (
    PhotoAnalysisRunMetrics,
    PhotoAnalysisTiming,
    _average,
    _compute_device_seconds,
    _pct_of_wall_time,
    log_photo_analysis_run,
)


@pytest.fixture(autouse=True)
def isolated_log_dir(monkeypatch, tmp_path):
    log_dir = tmp_path / "performance"
    monkeypatch.setattr(performance_log, "_LOG_DIR", log_dir)
    monkeypatch.setattr(performance_log, "_LOG_PATH", log_dir / "photo_analysis_log.jsonl")
    monkeypatch.setattr(performance_log, "_warned_unwritable", False)
    return log_dir


def _read_entries(log_dir):
    path = log_dir / "photo_analysis_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_metrics(**overrides) -> PhotoAnalysisRunMetrics:
    base = dict(
        total_photos=2,
        cpu_count=8,
        configured_concurrency=1,
        actual_concurrency=1,
        threads_per_inference=8,
        enable_scene_analysis=True,
        igpu_offload_used=False,
        dinov2_local_device="cuda",
        moondream_device="cuda",
        moondream_model_variant="Moondream2 (Q8_0)",
        total_wall_seconds=10.0,
        per_photo_seconds=[4.0, 6.0],
        per_photo_dinov2_seconds=[1.0, 1.0],
        per_photo_scene_seconds=[3.0, 3.0],
    )
    base.update(overrides)
    return PhotoAnalysisRunMetrics(**base)


class TestPhotoAnalysisTiming:
    def test_records_are_rounded_to_three_decimals(self):
        timing = PhotoAnalysisTiming()

        timing.record(1.23456)
        timing.record_dinov2(0.98765)
        timing.record_scene(2.00049)

        assert timing.per_photo_seconds == [1.235]
        assert timing.dinov2_seconds == [0.988]
        assert timing.scene_seconds == [2.0]

    def test_instances_do_not_share_lists(self):
        first, second = PhotoAnalysisTiming(), PhotoAnalysisTiming()

        first.record(1.0)

        assert second.per_photo_seconds == []


class TestHelpers:
    def test_average(self):
        assert _average([1.0, 2.0, 3.0]) == 2.0
        assert _average([]) is None

    def test_pct_of_wall_time(self):
        assert _pct_of_wall_time(5.0, 10.0) == 50.0
        assert _pct_of_wall_time(5.0, 0.0) == 0.0


class TestComputeDeviceSeconds:
    def test_igpu_offload_attributes_dinov2_to_igpu(self):
        result = _compute_device_seconds(
            igpu_offload_used=True,
            dinov2_local_device="cuda",  # se ignora: manda el offload
            moondream_device="cuda",
            dinov2_total=2.0,
            scene_total=6.0,
        )

        assert result == (6.0, 2.0, 0.0)  # (cuda, igpu, cpu)

    def test_dinov2_on_cuda_shares_gpu_with_moondream(self):
        result = _compute_device_seconds(
            igpu_offload_used=False,
            dinov2_local_device="cuda",
            moondream_device="cuda",
            dinov2_total=2.0,
            scene_total=6.0,
        )

        assert result == (8.0, 0.0, 0.0)

    def test_dinov2_on_cpu(self):
        result = _compute_device_seconds(
            igpu_offload_used=False,
            dinov2_local_device="cpu",
            moondream_device="cuda",
            dinov2_total=2.0,
            scene_total=6.0,
        )

        assert result == (6.0, 0.0, 2.0)

    def test_moondream_on_cpu(self):
        result = _compute_device_seconds(
            igpu_offload_used=False,
            dinov2_local_device="cuda",
            moondream_device="cpu",
            dinov2_total=2.0,
            scene_total=6.0,
        )

        assert result == (2.0, 0.0, 6.0)

    def test_models_never_loaded_contribute_nothing(self):
        result = _compute_device_seconds(
            igpu_offload_used=False,
            dinov2_local_device=None,
            moondream_device=None,
            dinov2_total=0.0,
            scene_total=0.0,
        )

        assert result == (0.0, 0.0, 0.0)


class TestLogPhotoAnalysisRun:
    def test_writes_one_line_with_expected_fields(self, isolated_log_dir):
        log_photo_analysis_run(_run_metrics())

        entries = _read_entries(isolated_log_dir)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["total_photos"] == 2
        assert entry["moondream_model_variant"] == "Moondream2 (Q8_0)"
        assert entry["total_wall_seconds"] == 10.0
        assert entry["throughput_seconds_per_photo"] == 5.0
        assert entry["avg_seconds_per_photo"] == 5.0
        assert entry["avg_dinov2_seconds_per_photo"] == 1.0
        assert entry["avg_scene_seconds_per_photo"] == 3.0
        assert entry["per_photo_seconds"] == [4.0, 6.0]
        assert "timestamp" in entry

    def test_device_seconds_and_percentages(self, isolated_log_dir):
        log_photo_analysis_run(_run_metrics())

        entry = _read_entries(isolated_log_dir)[0]
        assert entry["cuda_gpu_seconds"] == 8.0  # 2 (DINOv2) + 6 (Moondream2)
        assert entry["igpu_seconds"] == 0.0
        assert entry["cpu_seconds"] == 0.0
        assert entry["cuda_gpu_usage_pct"] == 80.0
        assert entry["igpu_usage_pct"] == 0.0
        assert entry["cpu_usage_pct"] == 0.0

    def test_igpu_offload_is_reported_separately(self, isolated_log_dir):
        log_photo_analysis_run(_run_metrics(igpu_offload_used=True))

        entry = _read_entries(isolated_log_dir)[0]
        assert entry["igpu_seconds"] == 2.0
        assert entry["cuda_gpu_seconds"] == 6.0
        assert entry["igpu_usage_pct"] == 20.0

    def test_scene_analysis_disabled_leaves_scene_average_null(self, isolated_log_dir):
        log_photo_analysis_run(
            _run_metrics(
                enable_scene_analysis=False,
                moondream_device=None,
                moondream_model_variant=None,
                per_photo_scene_seconds=[],
            )
        )

        entry = _read_entries(isolated_log_dir)[0]
        assert entry["avg_scene_seconds_per_photo"] is None
        assert entry["moondream_device"] is None

    def test_appends_multiple_runs(self, isolated_log_dir):
        for _ in range(3):
            log_photo_analysis_run(_run_metrics())

        assert len(_read_entries(isolated_log_dir)) == 3

    def test_zero_photos_is_noop(self, isolated_log_dir):
        log_photo_analysis_run(_run_metrics(total_photos=0, per_photo_seconds=[]))

        assert not (isolated_log_dir / "photo_analysis_log.jsonl").exists()

    def test_disabled_via_settings_does_not_write(self, isolated_log_dir, monkeypatch):
        monkeypatch.setattr(performance_log.settings, "enable_performance_logging", False)

        log_photo_analysis_run(_run_metrics())

        assert not (isolated_log_dir / "photo_analysis_log.jsonl").exists()


class TestAppendEntryUnwritable:
    @staticmethod
    def _break_mkdir(monkeypatch):
        def _raise_mkdir(*args, **kwargs):
            raise OSError("sin permisos")

        monkeypatch.setattr(performance_log.Path, "mkdir", _raise_mkdir)

    def test_unwritable_dir_does_not_raise(self, monkeypatch):
        self._break_mkdir(monkeypatch)

        log_photo_analysis_run(_run_metrics())  # no debe lanzar

    def test_warns_only_once(self, monkeypatch, caplog):
        self._break_mkdir(monkeypatch)

        with caplog.at_level("WARNING", logger=performance_log.logger.name):
            log_photo_analysis_run(_run_metrics())
            log_photo_analysis_run(_run_metrics())

        warnings = [r for r in caplog.records if "No se pudo escribir el log de rendimiento" in r.getMessage()]
        assert len(warnings) == 1
        assert performance_log._warned_unwritable is True
