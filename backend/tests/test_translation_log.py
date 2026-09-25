"""
Tests de app/log/translation_log.py -- mismo patrón que se seguiría para
performance_log.py/analysis_run_log.py (ver sus docstrings): escribir a
disco es un efecto secundario, así que se apunta `_LOG_DIR`/`_LOG_PATH` a
un directorio temporal en vez de mockear `open()`.
"""
import json

import pytest

from app.log import translation_log


@pytest.fixture(autouse=True)
def isolated_log_dir(monkeypatch, tmp_path):
    """Cada test escribe en su propio `tmp_path`, nunca en
    `backend/data/performance/` de verdad."""
    log_dir = tmp_path / "performance"
    monkeypatch.setattr(translation_log, "_LOG_DIR", log_dir)
    monkeypatch.setattr(translation_log, "_LOG_PATH", log_dir / "translation_log.jsonl")
    monkeypatch.setattr(translation_log, "_warned_unwritable", False)
    return log_dir


def _read_entries(log_dir):
    path = log_dir / "translation_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestLogTranslationRun:
    def test_writes_one_line_with_expected_fields(self, isolated_log_dir):
        translation_log.log_translation_run(
            num_texts=3,
            source_lang="es",
            target_lang="en",
            cpu_count=8,
            total_seconds=1.5,
            translation_available=True,
        )

        entries = _read_entries(isolated_log_dir)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["num_texts"] == 3
        assert entry["source_lang"] == "es"
        assert entry["target_lang"] == "en"
        assert entry["direction"] == "es-en"
        assert entry["cpu_count"] == 8
        assert entry["device"] == "cpu"
        assert entry["translation_available"] is True
        assert entry["total_seconds"] == 1.5
        assert entry["avg_seconds_per_text"] == 0.5
        assert "timestamp" in entry

    def test_appends_multiple_calls(self, isolated_log_dir):
        for _ in range(3):
            translation_log.log_translation_run(
                num_texts=1,
                source_lang="en",
                target_lang="es",
                cpu_count=4,
                total_seconds=0.1,
                translation_available=True,
            )

        assert len(_read_entries(isolated_log_dir)) == 3

    def test_zero_texts_is_noop(self, isolated_log_dir):
        translation_log.log_translation_run(
            num_texts=0,
            source_lang="es",
            target_lang="en",
            cpu_count=4,
            total_seconds=0.0,
            translation_available=True,
        )

        assert not (isolated_log_dir / "translation_log.jsonl").exists()

    def test_disabled_via_settings_does_not_write(self, isolated_log_dir, monkeypatch):
        monkeypatch.setattr(translation_log.settings, "enable_performance_logging", False)

        translation_log.log_translation_run(
            num_texts=2,
            source_lang="es",
            target_lang="en",
            cpu_count=4,
            total_seconds=1.0,
            translation_available=True,
        )

        assert not (isolated_log_dir / "translation_log.jsonl").exists()

    def test_records_translation_available_false_without_raising(self, isolated_log_dir):
        """Ruta de degradación (modelo no convertido, ver
        translation.py) -- se registra igual, con el flag a False, para
        poder filtrarla al analizar en vez de que contamine en silencio
        la media de "traducir de verdad" (ver la sección de traducción
        dentro de scripts/analyze_performance_log.py, fusionada ahí junto
        con el análisis de fotos)."""
        translation_log.log_translation_run(
            num_texts=1,
            source_lang="es",
            target_lang="en",
            cpu_count=4,
            total_seconds=0.001,
            translation_available=False,
        )

        entries = _read_entries(isolated_log_dir)
        assert entries[0]["translation_available"] is False

    def test_unwritable_dir_does_not_raise(self, isolated_log_dir, monkeypatch):
        """Mismo criterio best-effort que performance_log.py/
        analysis_run_log.py: un fallo al escribir el log nunca debe
        propagarse hacia el llamador."""

        def _raise_mkdir(*args, **kwargs):
            raise OSError("sin permisos")

        monkeypatch.setattr(translation_log.Path, "mkdir", _raise_mkdir)

        translation_log.log_translation_run(
            num_texts=1,
            source_lang="es",
            target_lang="en",
            cpu_count=4,
            total_seconds=0.1,
            translation_available=True,
        )  # no debe lanzar
