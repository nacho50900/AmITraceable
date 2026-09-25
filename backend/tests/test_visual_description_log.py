"""
Tests de app/log/visual_description_log.py: registro OPCIONAL (desactivado
por defecto) de las descripciones que genera Moondream2 por foto. Contiene
contenido personal, así que además de cubrir la escritura se comprueba que
el interruptor lo mantiene apagado y que no se guarda ningún identificador
de cuenta. Directorio temporal, nunca `backend/data/`.
"""
import json

import pytest
from PIL import Image

from app.log import visual_description_log


@pytest.fixture(autouse=True)
def isolated_log_dir(monkeypatch, tmp_path):
    log_dir = tmp_path / "visual_descriptions"
    monkeypatch.setattr(visual_description_log, "_LOG_DIR", log_dir)
    monkeypatch.setattr(visual_description_log, "_LOG_PATH", log_dir / "visual_description_log.jsonl")
    monkeypatch.setattr(visual_description_log, "_warned_unwritable", False)
    return log_dir


def _read_entries(log_dir):
    path = log_dir / "visual_description_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def logging_enabled(monkeypatch):
    monkeypatch.setattr(visual_description_log.settings, "log_visual_descriptions", True)


class TestImageContentId:
    def test_is_16_hex_chars(self):
        image = Image.new("RGB", (4, 4), (10, 20, 30))

        image_id = visual_description_log.image_content_id(image)

        assert len(image_id) == 16
        int(image_id, 16)  # es hexadecimal

    def test_same_pixels_same_id_regardless_of_object(self):
        first = Image.new("RGB", (4, 4), (10, 20, 30))
        second = Image.new("RGB", (4, 4), (10, 20, 30))

        assert visual_description_log.image_content_id(first) == visual_description_log.image_content_id(second)

    def test_different_pixels_different_id(self):
        first = Image.new("RGB", (4, 4), (10, 20, 30))
        second = Image.new("RGB", (4, 4), (10, 20, 31))

        assert visual_description_log.image_content_id(first) != visual_description_log.image_content_id(second)


class TestLogVisualDescription:
    def test_disabled_by_default_writes_nothing(self, isolated_log_dir):
        assert visual_description_log.settings.log_visual_descriptions is False

        visual_description_log.log_visual_description(
            image_id="abc", model_variant="Q8_0", caption="una playa", structured="PERSONAS: 0"
        )

        assert not isolated_log_dir.exists()

    def test_writes_entry_when_enabled(self, isolated_log_dir, logging_enabled):
        visual_description_log.log_visual_description(
            image_id="abc123", model_variant="Moondream2 (Q8_0)", caption="una playa", structured="PERSONAS: 0"
        )

        assert _read_entries(isolated_log_dir) == [
            {
                "image_id": "abc123",
                "moondream_model_variant": "Moondream2 (Q8_0)",
                "caption": "una playa",
                "structured": "PERSONAS: 0",
            }
        ]

    def test_stores_no_account_identifiers(self, isolated_log_dir, logging_enabled):
        visual_description_log.log_visual_description(image_id="abc", model_variant=None, caption=None, structured=None)

        entry = _read_entries(isolated_log_dir)[0]
        assert set(entry) == {"image_id", "moondream_model_variant", "caption", "structured"}

    def test_keeps_non_ascii_text_readable(self, isolated_log_dir, logging_enabled):
        visual_description_log.log_visual_description(
            image_id="abc", model_variant=None, caption="cartel: «Plaza Mayor» ñ", structured=None
        )

        raw = (isolated_log_dir / "visual_description_log.jsonl").read_text(encoding="utf-8")
        assert "«Plaza Mayor» ñ" in raw

    def test_appends_multiple_entries(self, isolated_log_dir, logging_enabled):
        for i in range(3):
            visual_description_log.log_visual_description(
                image_id=f"id{i}", model_variant="F16", caption="c", structured="s"
            )

        assert [e["image_id"] for e in _read_entries(isolated_log_dir)] == ["id0", "id1", "id2"]


class TestUnwritable:
    @staticmethod
    def _break_mkdir(monkeypatch):
        def _raise_mkdir(*args, **kwargs):
            raise OSError("sin permisos")

        monkeypatch.setattr(visual_description_log.Path, "mkdir", _raise_mkdir)

    def test_unwritable_dir_does_not_raise(self, monkeypatch, logging_enabled):
        self._break_mkdir(monkeypatch)

        visual_description_log.log_visual_description(
            image_id="abc", model_variant="F16", caption="c", structured="s"
        )  # no debe lanzar

    def test_warns_only_once(self, monkeypatch, logging_enabled, caplog):
        self._break_mkdir(monkeypatch)

        with caplog.at_level("WARNING", logger=visual_description_log.logger.name):
            for _ in range(2):
                visual_description_log.log_visual_description(
                    image_id="abc", model_variant="F16", caption="c", structured="s"
                )

        warnings = [r for r in caplog.records if "No se pudo escribir el log de descripciones" in r.getMessage()]
        assert len(warnings) == 1
        assert visual_description_log._warned_unwritable is True
