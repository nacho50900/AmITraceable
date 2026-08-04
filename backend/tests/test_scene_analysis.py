"""
Tests de app/vision/scene_analysis.py.

No se descarga el modelo real (~1.8B parámetros): se mockea `_lazy_load`
(y `_model`) para ejercer toda la lógica de negocio (parseo de la
respuesta, degradación best-effort) sin dependencias pesadas.
"""
import pytest

from app.vision import scene_analysis


class _FakeModel:
    """Sustituye a Moondream2 lo justo para analyze_image_content: solo
    necesita responder a .query(image, pregunta) -> {"answer": str}."""

    def __init__(self, answer: str):
        self._answer = answer

    def query(self, image, question):
        return {"answer": self._answer}


class _RaisingModel:
    def query(self, image, question):
        raise RuntimeError("fallo simulado del modelo")


@pytest.fixture(autouse=True)
def reset_module_globals(monkeypatch):
    """Cada test debe partir de _model limpio, igual que geolocation.py."""
    monkeypatch.setattr(scene_analysis, "_model", None)
    yield


def _install_fake_model(monkeypatch, answer: str):
    monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
    monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _FakeModel(answer)))


class TestAnalyzeImageContent:
    def test_parses_aficion_and_no_pareja_with_one_person(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: una\nAFICION: Posible fan de baloncesto, aparece jugando\nPAREJA: no",
        )

        inferences, indicio_pareja, _ = scene_analysis.analyze_image_content(object())

        assert len(inferences) == 1
        assert inferences[0].category == "aficion"
        assert "baloncesto" in inferences[0].value.lower()
        assert inferences[0].confidence == 0.5
        assert inferences[0].evidence == []  # lo rellena el llamador (geolocation.py), no este módulo
        assert indicio_pareja is False

    def test_parses_aficion_with_no_people_at_all(self, monkeypatch):
        """Una foto sin nadie (p. ej. un vinilo sobre una mesa) es el caso
        MÁS fiable de todos: no hay ninguna ambigüedad de a quién
        atribuírselo."""
        _install_fake_model(monkeypatch, "PERSONAS: ninguna\nAFICION: vinilo de música visible\nPAREJA: no")

        inferences, _, _ = scene_analysis.analyze_image_content(object())

        assert len(inferences) == 1

    def test_discards_aficion_when_several_people_are_similarly_prominent(self, monkeypatch):
        """El caso que motivó este campo: con varias personas de
        protagonismo similar (p. ej. una pareja), no hay forma de saber si
        la afición detectada es de la cuenta analizada o de la otra
        persona -- se descarta la señal en vez de arriesgarse a
        atribuirla a quien no toca."""
        _install_fake_model(
            monkeypatch, "PERSONAS: varias\nAFICION: toca la guitarra\nPAREJA: no"
        )

        inferences, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    def test_pareja_signal_still_valid_with_several_people(self, monkeypatch):
        """A diferencia de la afición, la señal de pareja NO necesita
        resolver quién es la cuenta analizada -- de hecho 'varias'
        personas es el caso típico para esta señal."""
        _install_fake_model(monkeypatch, "PERSONAS: varias\nAFICION: ninguno\nPAREJA: si")

        inferences, indicio_pareja, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is True

    def test_unparseable_personas_value_discards_aficion_by_precaution(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: no lo sé\nAFICION: toca la guitarra\nPAREJA: no")

        inferences, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    def test_missing_personas_line_discards_aficion_by_precaution(self, monkeypatch):
        _install_fake_model(monkeypatch, "AFICION: toca la guitarra\nPAREJA: no")

        inferences, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    @pytest.mark.parametrize("negative_value", ["ninguno", "Ninguna", "none", "N/A", ""])
    def test_ninguno_variants_produce_no_inference(self, monkeypatch, negative_value):
        _install_fake_model(monkeypatch, f"PERSONAS: una\nAFICION: {negative_value}\nPAREJA: no")

        inferences, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    def test_missing_aficion_line_is_ignored_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: una\nPAREJA: no")

        inferences, indicio_pareja, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_missing_pareja_line_defaults_to_false(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: una\nAFICION: toca la guitarra")

        inferences, indicio_pareja, _ = scene_analysis.analyze_image_content(object())

        assert len(inferences) == 1
        assert indicio_pareja is False

    def test_completely_unexpected_format_degrades_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "esto no sigue el formato pedido en absoluto")

        inferences, indicio_pareja, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_dependencies_not_installed_returns_empty_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        inferences, indicio_pareja, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_model_exception_degrades_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel()))

        inferences, indicio_pareja, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_never_identifies_or_describes_any_person(self, monkeypatch):
        """No es un test de comportamiento del modelo real (eso no se
        puede probar aquí sin descargarlo) -- es una comprobación de que
        el prompt en sí contiene la regla explícita, para que un cambio
        futuro no la borre por accidente sin darse cuenta. La regla se
        aplica a CUALQUIER persona (no solo 'la otra'): el modelo no puede
        saber cuál de las personas de la foto es la cuenta analizada, ver
        docstring del módulo."""
        query_lower = scene_analysis._QUERY.lower()
        assert "no describas ni identifiques" in query_lower
        assert "ninguna persona" in query_lower

    def test_returns_the_raw_answer_as_description(self, monkeypatch):
        raw_answer = "PERSONAS: una\nAFICION: Posible fan de baloncesto\nPAREJA: no"
        _install_fake_model(monkeypatch, raw_answer)

        _, _, description = scene_analysis.analyze_image_content(object())

        assert description == raw_answer

    def test_description_is_stripped_of_surrounding_whitespace(self, monkeypatch):
        _install_fake_model(monkeypatch, "  \nPERSONAS: ninguna\nAFICION: ninguno\nPAREJA: no\n  ")

        _, _, description = scene_analysis.analyze_image_content(object())

        assert description == "PERSONAS: ninguna\nAFICION: ninguno\nPAREJA: no"

    def test_description_is_none_when_dependencies_not_installed(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        _, _, description = scene_analysis.analyze_image_content(object())

        assert description is None

    def test_description_is_none_when_model_raises(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel()))

        _, _, description = scene_analysis.analyze_image_content(object())

        assert description is None


class TestParsePersonas:
    @pytest.mark.parametrize("value", ["ninguna", "una", "varias"])
    def test_valid_values(self, value):
        assert scene_analysis._parse_personas(f"PERSONAS: {value}") == value

    def test_invalid_value_returns_none(self):
        assert scene_analysis._parse_personas("PERSONAS: muchas") is None

    def test_missing_line_returns_none(self):
        assert scene_analysis._parse_personas("AFICION: ninguno") is None


class TestSceneAnalysisAvailable:
    def test_false_when_dependencies_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                raise ImportError(f"{name} no instalado")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        assert scene_analysis._scene_analysis_available() is False
