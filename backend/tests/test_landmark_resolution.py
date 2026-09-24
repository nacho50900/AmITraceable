"""
Tests de app/vision/landmark_resolution.py: resolución, vía el modelo local,
de las coordenadas de un edificio/monumento propuesto por Moondream2.

`call_ai_json` se sustituye por un doble que devuelve el dict ya parseado
(mismo criterio que tests/test_ai_analysis.py): no se carga ningún modelo.
"""
import pytest

from app.config import settings
from app.nlp.ai_client import AIHTTPError, AIRequestError
from app.vision import landmark_resolution
from app.vision.landmark_resolution import (
    MIN_CONFIDENCE_FOR_LANDMARK_OVERRIDE,
    LandmarkResolution,
    _valid_coordinates,
    resolve_landmark_coordinates,
)


@pytest.fixture(autouse=True)
def enable_ai(monkeypatch):
    """`ai_key_configured` exige `enable_ai_analysis=True` y un
    `qwen_gguf_repo_id` no vacío (ver app/config.py)."""
    monkeypatch.setattr(settings, "enable_ai_analysis", True)
    monkeypatch.setattr(settings, "qwen_gguf_repo_id", "fake/repo")


def _patch_ai(monkeypatch, response=None, exc: Exception | None = None):
    calls = []

    async def fake_call_ai_json(system_prompt, user_prompt, **kwargs):
        calls.append({"system": system_prompt, "user": user_prompt, **kwargs})
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(landmark_resolution, "call_ai_json", fake_call_ai_json)
    return calls


def _recognized(**overrides) -> dict:
    base = {
        "reconocido": True,
        "nombre_canonico": "Torre Eiffel",
        "lat": 48.8584,
        "lon": 2.2945,
        "confianza": 0.95,
    }
    base.update(overrides)
    return base


class TestValidCoordinates:
    def test_accepts_valid_floats_and_ints(self):
        assert _valid_coordinates(48.8, 2.3) == (48.8, 2.3)
        assert _valid_coordinates(0, 0) == (0.0, 0.0)

    def test_accepts_range_boundaries(self):
        assert _valid_coordinates(90, 180) == (90.0, 180.0)
        assert _valid_coordinates(-90, -180) == (-90.0, -180.0)

    @pytest.mark.parametrize(
        "lat, lon",
        [
            (90.1, 0),
            (-90.1, 0),
            (0, 180.1),
            (0, -180.1),
        ],
    )
    def test_rejects_out_of_range(self, lat, lon):
        assert _valid_coordinates(lat, lon) is None

    @pytest.mark.parametrize(
        "lat, lon",
        [
            ("48.8", 2.3),
            (48.8, "2.3"),
            (None, 2.3),
            (48.8, None),
            (True, 2.3),  # bool es subclase de int pero no es una coordenada
            (48.8, False),
        ],
    )
    def test_rejects_non_numeric_and_bool(self, lat, lon):
        assert _valid_coordinates(lat, lon) is None


class TestLandmarkResolution:
    def test_stores_fields(self):
        resolution = LandmarkResolution("Alhambra", 37.177, -3.588, 0.9)

        assert (resolution.canonical_name, resolution.lat, resolution.lon, resolution.confidence) == (
            "Alhambra",
            37.177,
            -3.588,
            0.9,
        )


class TestEarlyExits:
    @pytest.mark.asyncio
    async def test_none_when_ai_not_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "enable_ai_analysis", False)
        calls = _patch_ai(monkeypatch, response=_recognized())

        assert await resolve_landmark_coordinates("Torre Eiffel") is None
        assert calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["", "   ", "\n\t"])
    async def test_none_and_no_call_for_blank_name(self, monkeypatch, name):
        calls = _patch_ai(monkeypatch, response=_recognized())

        assert await resolve_landmark_coordinates(name) is None
        assert calls == []


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_contains_stripped_name_only_without_hint(self, monkeypatch):
        calls = _patch_ai(monkeypatch, response=_recognized())

        await resolve_landmark_coordinates("  Torre Eiffel  ")

        assert calls[0]["user"] == "Nombre propuesto: Torre Eiffel"
        assert calls[0]["max_tokens"] == 200
        assert calls[0]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_prompt_includes_hint_truncated_to_300_chars(self, monkeypatch):
        calls = _patch_ai(monkeypatch, response=_recognized())

        await resolve_landmark_coordinates("catedral", context_hint="  " + "a" * 500)

        prompt = calls[0]["user"]
        assert "Nombre propuesto: catedral" in prompt
        hint_line = prompt.split("\n")[1]
        assert hint_line.endswith("a" * 300)
        assert not hint_line.endswith("a" * 301)


class TestSuccess:
    @pytest.mark.asyncio
    async def test_returns_resolution(self, monkeypatch):
        _patch_ai(monkeypatch, response=_recognized(nombre_canonico="  Torre Eiffel  "))

        result = await resolve_landmark_coordinates("eiffel")

        assert isinstance(result, LandmarkResolution)
        assert result.canonical_name == "Torre Eiffel"
        assert result.lat == 48.8584
        assert result.lon == 2.2945
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_confidence_exactly_at_threshold_is_accepted(self, monkeypatch):
        _patch_ai(monkeypatch, response=_recognized(confianza=MIN_CONFIDENCE_FOR_LANDMARK_OVERRIDE))

        assert await resolve_landmark_coordinates("eiffel") is not None

    @pytest.mark.asyncio
    async def test_confidence_above_one_is_clamped(self, monkeypatch):
        _patch_ai(monkeypatch, response=_recognized(confianza=7))

        result = await resolve_landmark_coordinates("eiffel")

        assert result is not None
        assert result.confidence == 1.0


class TestRejections:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", [AIHTTPError(200, "no es JSON"), AIRequestError("modelo no disponible")])
    async def test_none_when_ai_call_fails(self, monkeypatch, exc):
        _patch_ai(monkeypatch, exc=exc)

        assert await resolve_landmark_coordinates("eiffel") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("parsed", [None, [], "texto", 42])
    async def test_none_when_response_is_not_a_dict(self, monkeypatch, parsed):
        _patch_ai(monkeypatch, response=parsed)

        assert await resolve_landmark_coordinates("eiffel") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("recognized", [False, None, 0, ""])
    async def test_none_when_not_recognized(self, monkeypatch, recognized):
        _patch_ai(monkeypatch, response=_recognized(reconocido=recognized))

        assert await resolve_landmark_coordinates("catedral") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", [None, "", "   ", 123])
    async def test_none_when_canonical_name_invalid(self, monkeypatch, name):
        _patch_ai(monkeypatch, response=_recognized(nombre_canonico=name))

        assert await resolve_landmark_coordinates("eiffel") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "overrides",
        [
            {"lat": None},
            {"lon": None},
            {"lat": "48.8"},
            {"lat": 123.0},
            {"lon": -500},
            {"lat": True},
        ],
    )
    async def test_none_when_coordinates_invalid(self, monkeypatch, overrides):
        _patch_ai(monkeypatch, response=_recognized(**overrides))

        assert await resolve_landmark_coordinates("eiffel") is None

    @pytest.mark.asyncio
    async def test_none_when_confidence_below_threshold(self, monkeypatch):
        _patch_ai(monkeypatch, response=_recognized(confianza=MIN_CONFIDENCE_FOR_LANDMARK_OVERRIDE - 0.01))

        assert await resolve_landmark_coordinates("eiffel") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("confidence", [None, "alta", True])
    async def test_none_when_confidence_missing_or_not_numeric(self, monkeypatch, confidence):
        """Sin confianza numérica válida se trata como 0.0: nunca se acepta
        un override 'a ciegas'."""
        _patch_ai(monkeypatch, response=_recognized(confianza=confidence))

        assert await resolve_landmark_coordinates("eiffel") is None

    @pytest.mark.asyncio
    async def test_negative_confidence_is_clamped_to_zero_and_rejected(self, monkeypatch):
        _patch_ai(monkeypatch, response=_recognized(confianza=-3))

        assert await resolve_landmark_coordinates("eiffel") is None
