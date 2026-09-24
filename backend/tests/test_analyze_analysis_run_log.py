"""
Tests de app/log/analyze_analysis_run_log.py: análisis offline de
`analysis_run_log.jsonl` (ver app/log/analysis_run_log.py).

Nada toca el `data/` real: los logs se generan en `tmp_path`. `matplotlib`
no es dependencia del proyecto (ni de requirements-dev.txt), así que la
rama `--plot` se ejercita con un módulo falso en `sys.modules`.
"""
import json
import sys
import types

import pandas as pd
import pytest

from app.log import analyze_analysis_run_log as analyzer


def _row(platform="reddit", n_posts=10, n_comments=5, n_photos=0, total=2.0, stages=None, timestamp=1_700_000_000.0):
    return {
        "timestamp": timestamp,
        "platform": platform,
        "n_posts": n_posts,
        "n_comments": n_comments,
        "n_media_items": n_photos,
        "n_photos": n_photos,
        "ai_enabled": True,
        "scene_analysis_enabled": False,
        "geolocation_available": False,
        "total_seconds": total,
        "stages_seconds": stages if stages is not None else {"huella": 0.5, "atributos": 1.5},
    }


def _write_log(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestLoadLog:
    def test_raises_when_log_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No existe"):
            analyzer.load_log(tmp_path / "no_existe.jsonl")

    def test_parses_rows_and_converts_timestamp(self, tmp_path):
        log = tmp_path / "log.jsonl"
        _write_log(log, [_row(), _row(platform="instagram", timestamp=1_700_000_100.0)])

        df = analyzer.load_log(log)

        assert len(df) == 2
        assert list(df["platform"]) == ["reddit", "instagram"]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].iloc[0] == pd.Timestamp("2023-11-14 22:13:20")

    def test_ignores_blank_lines(self, tmp_path):
        log = tmp_path / "log.jsonl"
        log.write_text(json.dumps(_row()) + "\n\n   \n" + json.dumps(_row()) + "\n", encoding="utf-8")

        assert len(analyzer.load_log(log)) == 2


class TestStageBreakdown:
    def test_one_row_per_analysis_and_stage(self, tmp_path):
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _row(stages={"huella": 0.5, "atributos": 1.5}),
                _row(platform="instagram", stages={"huella": 0.7}),
            ],
        )

        stages = analyzer.stage_breakdown(analyzer.load_log(log))

        assert len(stages) == 3
        assert set(stages.columns) == {"timestamp", "platform", "etapa", "segundos"}
        assert sorted(stages["etapa"]) == ["atributos", "huella", "huella"]
        instagram = stages[stages["platform"] == "instagram"]
        assert list(instagram["etapa"]) == ["huella"]
        assert list(instagram["segundos"]) == [0.7]


class TestSummarizeByStage:
    def test_aggregates_and_sorts_heaviest_first(self):
        stages = pd.DataFrame(
            [
                {"etapa": "ligera", "segundos": 0.1},
                {"etapa": "ligera", "segundos": 0.3},
                {"etapa": "pesada", "segundos": 4.0},
                {"etapa": "pesada", "segundos": 6.0},
                {"etapa": "pesada", "segundos": 20.0},
            ]
        )

        summary = analyzer.summarize_by_stage(stages)

        assert list(summary["etapa"]) == ["pesada", "ligera"]
        heavy = summary.iloc[0]
        assert heavy["analisis"] == 3
        assert heavy["media_segundos"] == pytest.approx(10.0)
        assert heavy["mediana_segundos"] == pytest.approx(6.0)
        assert heavy["max_segundos"] == pytest.approx(20.0)
        light = summary.iloc[1]
        assert light["analisis"] == 2
        assert light["media_segundos"] == pytest.approx(0.2)


class TestMain:
    @pytest.fixture
    def run_main(self, monkeypatch, tmp_path):
        """Ejecuta `main()` con un log de `tmp_path` (el valor por defecto de
        `load_log` se fija al definir la función, así que se sustituye la
        propia `load_log`) y sin argumentos de línea de comandos por defecto."""
        real_load_log = analyzer.load_log

        def _run(rows, argv=None):
            log = tmp_path / "analysis_run_log.jsonl"
            _write_log(log, rows)
            monkeypatch.setattr(analyzer, "load_log", lambda: real_load_log(log))
            monkeypatch.setattr(analyzer, "_LOG_PATH", log)
            monkeypatch.setattr(sys, "argv", ["analyze_analysis_run_log.py", *(argv or [])])
            analyzer.main()

        return _run

    def test_prints_summary_by_platform_and_heaviest_stage(self, run_main, capsys):
        run_main(
            [
                _row(platform="reddit", total=2.0, stages={"huella": 0.5, "atributos": 1.5}),
                _row(platform="instagram", total=4.0, stages={"huella": 0.5, "atributos": 3.5}),
            ]
        )

        out = capsys.readouterr().out
        assert "2 análisis completos registrados." in out
        assert "Totales por plataforma:" in out
        assert "reddit" in out and "instagram" in out
        assert "Etapa que más pesa de media: 'atributos'" in out
        assert "2.50s de media (2 análisis)" in out

    def test_prints_correlation_only_for_columns_that_vary(self, run_main, capsys):
        run_main(
            [
                _row(n_posts=10, n_comments=5, n_photos=0, total=1.0),
                _row(n_posts=20, n_comments=5, n_photos=0, total=2.0),
                _row(n_posts=30, n_comments=5, n_photos=0, total=3.0),
            ]
        )

        out = capsys.readouterr().out
        assert "Correlación entre n_posts y tiempo total: 1.00" in out
        assert "n_comments" not in out.split("Correlación")[-1]
        assert "Correlación entre n_photos" not in out

    def test_no_correlation_lines_when_nothing_varies(self, run_main, capsys):
        run_main([_row(), _row()])

        assert "Correlación" not in capsys.readouterr().out

    def test_plot_flag_saves_png_next_to_log(self, run_main, capsys, monkeypatch, tmp_path):
        saved = {}

        class _Ax:
            def barh(self, labels, values):
                saved["barh"] = (list(labels), list(values))

            def set_xlabel(self, text):
                saved["xlabel"] = text

            def set_title(self, text):
                saved["title"] = text

        class _Fig:
            def tight_layout(self):
                saved["tight"] = True

            def savefig(self, path, dpi):
                saved["path"] = path
                saved["dpi"] = dpi

        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda figsize: (_Fig(), _Ax())
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.pyplot = fake_pyplot
        monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

        run_main([_row(stages={"huella": 0.5, "atributos": 1.5})], argv=["--plot"])

        assert saved["path"] == tmp_path / "analysis_run_summary.png"
        assert saved["dpi"] == 150
        assert saved["barh"][0] == ["atributos", "huella"]
        assert saved["tight"] is True
        assert "Gráfico guardado en" in capsys.readouterr().out

    def test_without_plot_flag_never_imports_matplotlib(self, run_main, monkeypatch):
        # `None` en sys.modules haría fallar cualquier `import matplotlib...`
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

        run_main([_row()])  # no debe lanzar
