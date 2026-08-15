"""
Tests de app/vision/scene_analysis.py.

No se descarga el modelo real (~1.8B parámetros): se mockea `_lazy_load`
(y `_model`) para ejercer toda la lógica de negocio (parseo de la
respuesta, degradación best-effort) sin dependencias pesadas.
"""
import pytest

from app.vision import scene_analysis


class _FakeModel:
    """Sustituye a Moondream2 lo justo para analyze_image_content: debe
    soportar `.encode_image(image)` (reutilizada por las DOS llamadas a
    `.query()`, ver docstring de analyze_image_content) y `.query(image,
    pregunta, settings=...)` -> {"answer": str}, distinguiendo la
    respuesta según cuál de las dos preguntas (_CAPTION_QUERY vs
    _STRUCTURED_QUERY) se le haga -- igual que hace el modelo real, que
    responde cosas distintas a cada una."""

    def __init__(self, structured_answer: str, caption_answer: str = "una escena sin detalles relevantes"):
        self._structured_answer = structured_answer
        self._caption_answer = caption_answer

    def encode_image(self, image):
        return image  # no hace falta simular una codificación real para estos tests

    def query(self, image, question, settings=None):
        if question == scene_analysis._CAPTION_QUERY:
            return {"answer": self._caption_answer}
        if question == scene_analysis._STRUCTURED_QUERY:
            return {"answer": self._structured_answer}
        raise AssertionError(f"Pregunta inesperada (no es _CAPTION_QUERY ni _STRUCTURED_QUERY): {question!r}")


class _RaisingModel:
    def encode_image(self, image):
        raise RuntimeError("fallo simulado del modelo")

    def query(self, image, question, settings=None):
        raise RuntimeError("fallo simulado del modelo")


@pytest.fixture(autouse=True)
def reset_module_globals(monkeypatch):
    """Cada test debe partir de _model limpio, igual que geolocation.py."""
    monkeypatch.setattr(scene_analysis, "_model", None)
    yield


def _install_fake_model(monkeypatch, structured_answer: str, caption_answer: str = "una escena sin detalles relevantes"):
    monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
    monkeypatch.setattr(
        scene_analysis,
        "_lazy_load",
        lambda: setattr(scene_analysis, "_model", _FakeModel(structured_answer, caption_answer)),
    )


class TestAnalyzeImageContent:
    def test_parses_aficion_and_no_pareja_with_one_person(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: una\nAFICION: Posible fan de baloncesto, aparece jugando\nPAREJA: no",
        )

        inferences, indicio_pareja, _, _ = scene_analysis.analyze_image_content(object())

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

        inferences, _, _, _ = scene_analysis.analyze_image_content(object())

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

        inferences, _, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    def test_pareja_signal_still_valid_with_several_people(self, monkeypatch):
        """A diferencia de la afición, la señal de pareja NO necesita
        resolver quién es la cuenta analizada -- de hecho 'varias'
        personas es el caso típico para esta señal."""
        _install_fake_model(monkeypatch, "PERSONAS: varias\nAFICION: ninguno\nPAREJA: si")

        inferences, indicio_pareja, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is True

    def test_unparseable_personas_value_discards_aficion_by_precaution(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: no lo sé\nAFICION: toca la guitarra\nPAREJA: no")

        inferences, _, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    def test_missing_personas_line_discards_aficion_by_precaution(self, monkeypatch):
        _install_fake_model(monkeypatch, "AFICION: toca la guitarra\nPAREJA: no")

        inferences, _, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    @pytest.mark.parametrize("negative_value", ["ninguno", "Ninguna", "none", "N/A", ""])
    def test_ninguno_variants_produce_no_inference(self, monkeypatch, negative_value):
        _install_fake_model(monkeypatch, f"PERSONAS: una\nAFICION: {negative_value}\nPAREJA: no")

        inferences, _, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []

    def test_missing_aficion_line_is_ignored_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: una\nPAREJA: no")

        inferences, indicio_pareja, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_missing_pareja_line_defaults_to_false(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: una\nAFICION: toca la guitarra")

        inferences, indicio_pareja, _, _ = scene_analysis.analyze_image_content(object())

        assert len(inferences) == 1
        assert indicio_pareja is False

    def test_completely_unexpected_format_degrades_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "esto no sigue el formato pedido en absoluto")

        inferences, indicio_pareja, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_dependencies_not_installed_returns_empty_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        inferences, indicio_pareja, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_model_exception_degrades_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel()))

        inferences, indicio_pareja, _, _ = scene_analysis.analyze_image_content(object())

        assert inferences == []
        assert indicio_pareja is False

    def test_never_identifies_or_describes_any_person(self, monkeypatch):
        """No es un test de comportamiento del modelo real (eso no se
        puede probar aquí sin descargarlo) -- es una comprobación de que
        el prompt en sí contiene la regla explícita, para que un cambio
        futuro no la borre por accidente sin darse cuenta. La regla se
        aplica a CUALQUIER persona (no solo 'la otra'): el modelo no puede
        saber cuál de las personas de la foto es la cuenta analizada, ver
        docstring del módulo. Se comprueba en AMBOS prompts (estructurado
        y de caption), no solo en uno."""
        structured_lower = scene_analysis._STRUCTURED_QUERY.lower()
        assert "no describas ni identifiques" in structured_lower
        assert "ninguna persona" in structured_lower

        caption_lower = scene_analysis._CAPTION_QUERY.lower()
        assert "race" in caption_lower
        assert "ethnicity" in caption_lower
        assert "age" in caption_lower

    def test_caption_query_forbids_race_and_physical_traits(self, monkeypatch):
        """El caption es texto libre (a diferencia de PERSONAS/PAREJA, que
        son una de tres opciones fijas), así que es el que más fácilmente
        podría colar una descripción física o racial si el prompt no lo
        prohíbe explícitamente -- ver README (exclusión de alcance
        deliberada, art. 9.1 RGPD) y docstring del módulo. EN INGLÉS
        (a diferencia del resto del módulo) porque _CAPTION_QUERY está en
        inglés a propósito: Moondream2 solo tiene datos de entrenamiento en
        inglés (confirmado por el autor, ver nota en _CAPTION_QUERY),
        pedirle generar una frase libre en español producía gramática
        rota y palabras inventadas."""
        caption_lower = scene_analysis._CAPTION_QUERY.lower()
        assert "race" in caption_lower
        assert "ethnicity" in caption_lower
        assert "skin tone" in caption_lower

    def test_caption_query_has_no_example_to_copy(self, monkeypatch):
        """Regresión del bug real: la primera versión de este campo
        incluía una línea de ejemplo ('DESCRIPCION: varias personas
        charlando alrededor de una mesa') dentro del prompt combinado, y
        Moondream2 la copiaba literalmente en vez de describir la imagen
        real -- en un caso incluso la repitió una segunda vez. La
        solución fue sacar el caption a su propia pregunta SIN ningún
        ejemplo de contenido que copiar. Se comprueba aquí que ese texto
        de ejemplo concreto no ha vuelto a colarse en el prompt."""
        assert "charlando alrededor de una mesa" not in scene_analysis._CAPTION_QUERY

    def test_caption_query_is_in_english_not_spanish(self, monkeypatch):
        """Decisión deliberada, no un descuido -- ver nota larga en
        _CAPTION_QUERY: Moondream2 solo tiene datos de entrenamiento en
        inglés (confirmado por el autor del modelo en
        huggingface.co/vikhyatk/moondream2/discussions/22), y pedirle
        generar una frase libre en español producía en producción
        gramática rota y palabras inventadas (p. ej. "comengan"). El
        resto del módulo (y del proyecto) sigue en español -- solo este
        campo, por ser generación de texto libre en vez de una de pocas
        opciones fijas, se pregunta en inglés. Este test fija esa
        decisión para que un cambio futuro no la revierta sin darse
        cuenta ni sin volver a comprobar el problema de fondo."""
        assert "describe" in scene_analysis._CAPTION_QUERY.lower()
        assert "describe en" not in scene_analysis._CAPTION_QUERY.lower()

    def test_returns_the_raw_answer_as_description(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: una\nAFICION: Posible fan de baloncesto\nPAREJA: no",
            caption_answer="a person playing basketball",
        )

        _, _, description, _ = scene_analysis.analyze_image_content(object())

        assert description == (
            "DESCRIPCION: a person playing basketball\n"
            "PERSONAS: una\nAFICION: Posible fan de baloncesto\nPAREJA: no"
        )

    def test_description_is_none_when_dependencies_not_installed(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        _, _, description, _ = scene_analysis.analyze_image_content(object())

        assert description is None

    def test_description_is_none_when_model_raises(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel()))

        _, _, description, _ = scene_analysis.analyze_image_content(object())

        assert description is None

    def test_parses_descripcion_general_from_dedicated_caption_call(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: varias\nAFICION: ninguno\nPAREJA: no",
            caption_answer="4 people happily eating pizza on a terrace",
        )

        _, _, _, descripcion_general = scene_analysis.analyze_image_content(object())

        assert descripcion_general == "4 people happily eating pizza on a terrace"

    def test_descripcion_general_is_none_when_dependencies_not_installed(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        _, _, _, descripcion_general = scene_analysis.analyze_image_content(object())

        assert descripcion_general is None

    def test_encode_image_is_called_once_and_reused_for_both_queries(self, monkeypatch):
        """Optimización real: sin esto, cada .query() re-codificaría la
        imagen desde cero -- con dos llamadas (caption + estructurada) por
        foto, eso duplicaría el coste del encoder de visión. Se comprueba
        contando las llamadas a encode_image en vez de solo confiar en que
        "debería" reutilizarse."""
        encode_calls: list[object] = []
        query_calls: list[str] = []

        class _CountingModel:
            def encode_image(self, image):
                encode_calls.append(image)
                return "encoded-sentinel"

            def query(self, image, question, settings=None):
                query_calls.append(question)
                assert image == "encoded-sentinel"  # debe usar la imagen YA codificada, no la original
                if question == scene_analysis._CAPTION_QUERY:
                    return {"answer": "una escena cualquiera"}
                return {"answer": "PERSONAS: ninguna\nAFICION: ninguno\nPAREJA: no"}

        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _CountingModel()))

        scene_analysis.analyze_image_content(object())

        assert len(encode_calls) == 1
        assert len(query_calls) == 2
        assert set(query_calls) == {scene_analysis._CAPTION_QUERY, scene_analysis._STRUCTURED_QUERY}

    def test_query_settings_cap_generation_length(self):
        """Red de seguridad frente al bug real que motivó separar los
        settings por llamada: sin límite, .query() usa max_tokens=768 por
        defecto (ver docs.moondream.ai/transformers), suficiente para que
        una respuesta confusa supere el timeout de 30s del pipeline real
        (_SCENE_ANALYSIS_TIMEOUT_SECONDS en geolocation.py). Se comprueba
        aquí que los límites siguen existiendo y siguen siendo bajos, para
        que un cambio futuro no los elimine sin darse cuenta."""
        assert scene_analysis._CAPTION_SETTINGS["max_tokens"] < 200
        assert scene_analysis._STRUCTURED_SETTINGS["max_tokens"] < 200

    def test_query_settings_include_variant_key(self):
        """Bug real descubierto en ejecución (GTX 1650, revisión pinneada
        del modelo): `encode_image()` en esta revisión hace
        settings["variant"] SIN .get(), así que cualquier `settings` que
        pasemos revienta con KeyError si no incluye esta clave. Se
        comprueba en AMBOS dicts de settings para que un cambio futuro no
        la elimine de uno de los dos sin darse cuenta."""
        assert "variant" in scene_analysis._CAPTION_SETTINGS
        assert "variant" in scene_analysis._STRUCTURED_SETTINGS


class TestParseDescripcion:
    def test_valid_value(self):
        answer = "DESCRIPCION: varias personas charlando en una terraza\nPERSONAS: varias"
        assert scene_analysis._parse_descripcion(answer) == "varias personas charlando en una terraza"

    def test_missing_line_returns_none(self):
        assert scene_analysis._parse_descripcion("PERSONAS: una\nAFICION: ninguno") is None

    @pytest.mark.parametrize("negative_value", ["ninguno", "Ninguna", "none", "N/A", ""])
    def test_ninguno_variants_return_none(self, negative_value):
        assert scene_analysis._parse_descripcion(f"DESCRIPCION: {negative_value}") is None

    def test_trailing_period_is_stripped(self):
        assert scene_analysis._parse_descripcion("DESCRIPCION: una persona leyendo un libro.") == (
            "una persona leyendo un libro"
        )


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
