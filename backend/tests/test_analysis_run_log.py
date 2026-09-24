"""
Tests de app/log/analysis_run_log.py: log GENERAL de cada análisis completo.
Mismo patrón que tests/test_translation_log.py (directorio temporal).
"""
import json

import pytest

from app.analysis_timing import StageTimer
from app.log import analysis_run_log


@pytest.fixture(autouse=True)
def isolated_log_dir(monkeypatch, tmp_path):
    log_dir = tmp_path / "performance"
    monkeypatch.setattr(analysis_run_log, "_LOG_DIR", log_dir)
    monkeypatch.setattr(analysis_run_log, "_LOG_PATH", log_dir / "analysis_run_log.jsonl")
    monkeypatch.setattr(analysis_run_log, "_warned_unwritable", False)
    return log_dir


def _read_entries(log_dir):
    path = log_dir / "analysis_run_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _log(timer=None, **overrides):
    kwargs = dict(
        platform="reddit",
        n_posts=12,
        n_comments=30,
        n_media_items=4,
        n_photos=3,
        ai_enabled=True,
        scene_analysis_enabled=False,
        geolocation_available=True,
        total_seconds=3.14159,
        timer=timer if timer is not None else StageTimer(),
    )
    kwargs.update(overrides)
    analysis_run_log.log_analysis_run(**kwargs)


class TestLogAnalysisRun:
    def test_writes_one_line_with_expected_fields(self, isolated_log_dir):
        timer = StageTimer()
        timer.add("huella", 0.12345)
        timer.add("atributos", 1.5)

        _log(timer=timer)

        entries = _read_entries(isolated_log_dir)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["platform"] == "reddit"
        assert entry["n_posts"] == 12
        assert entry["n_comments"] == 30
        assert entry["n_media_items"] == 4
        assert entry["n_photos"] == 3
        assert entry["ai_enabled"] is True
        assert entry["scene_analysis_enabled"] is False
        assert entry["geolocation_available"] is True
        assert entry["total_seconds"] == 3.142
        assert entry["stages_seconds"] == {"huella": 0.123, "atributos": 1.5}
        assert "timestamp" in entry

    def test_never_stores_identifying_fields(self, isolated_log_dir):
        """Diseño RGPD (ver docstring del módulo): solo recuentos y tiempos."""
        _log()

        entry = _read_entries(isolated_log_dir)[0]
        assert not {"username", "bio", "permalink", "ip"} & set(entry)

    def test_empty_timer_gives_empty_stages(self, isolated_log_dir):
        _log()

        assert _read_entries(isolated_log_dir)[0]["stages_seconds"] == {}

    def test_appends_multiple_runs(self, isolated_log_dir):
        for platform in ("reddit", "instagram", "reddit"):
            _log(platform=platform)

        assert [e["platform"] for e in _read_entries(isolated_log_dir)] == ["reddit", "instagram", "reddit"]

    def test_disabled_via_settings_does_not_write(self, isolated_log_dir, monkeypatch):
        monkeypatch.setattr(analysis_run_log.settings, "enable_performance_logging", False)

        _log()

        assert not (isolated_log_dir / "analysis_run_log.jsonl").exists()


class TestUnwritable:
    @staticmethod
    def _break_mkdir(monkeypatch):
        def _raise_mkdir(*args, **kwargs):
            raise OSError("sin permisos")

        monkeypatch.setattr(analysis_run_log.Path, "mkdir", _raise_mkdir)

    def test_unwritable_dir_does_not_raise(self, monkeypatch):
        self._break_mkdir(monkeypatch)

        _log()  # no debe lanzar

    def test_warns_only_once(self, monkeypatch, caplog):
        self._break_mkdir(monkeypatch)

        with caplog.at_level("WARNING", logger=analysis_run_log.logger.name):
            _log()
            _log()

        warnings = [r for r in caplog.records if "No se pudo escribir el log general" in r.getMessage()]
        assert len(warnings) == 1
        assert analysis_run_log._warned_unwritable is True
