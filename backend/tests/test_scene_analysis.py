"""
Tests de app/vision/scene_analysis.py.

No se descarga el modelo real (~1.8B parámetros): se mockea `_lazy_load`
(y `_model`) para ejercer toda la lógica de negocio (parseo de la
respuesta, degradación best-effort) sin dependencias pesadas.
"""
import pytest
from PIL import Image

from app.vision import scene_analysis


def _fake_image():
    """Imagen PIL real y minúscula (no un objeto genérico) -- desde que
    analyze_image_content redimensiona una COPIA de la imagen antes de
    codificarla (ver _CAPTION_MAX_DIMENSION), necesita `.copy()` y
    `.thumbnail()` reales, que `object()` no tiene."""
    return Image.new("RGB", (10, 10))


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

        inferences, indicio_pareja, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

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

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert len(inferences) == 1

    def test_parses_texto_visible_as_inferred_attribute(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: varias\nAFICION: ninguno\nPAREJA: no\nTEXTO_VISIBLE: Bar Manolo",
        )

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert len(inferences) == 1
        assert inferences[0].category == "texto_visible"
        assert "Bar Manolo" in inferences[0].value
        assert inferences[0].confidence == 0.4
        assert inferences[0].evidence == []

    def test_texto_visible_not_gated_by_personas_unlike_aficion(self, monkeypatch):
        """A diferencia de AFICION, TEXTO_VISIBLE no depende de saber
        quién es la cuenta analizada -- un cartel es verdad independientemente
        de cuántas personas salgan en la foto, así que la señal se
        mantiene incluso con 'varias' personas (caso en el que AFICION SÍ
        se descartaría, ver test_discards_aficion_when_several_people..)."""
        _install_fake_model(
            monkeypatch,
            "PERSONAS: varias\nAFICION: ninguno\nPAREJA: no\nTEXTO_VISIBLE: Ayuntamiento de Badajoz",
        )

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert len(inferences) == 1
        assert inferences[0].category == "texto_visible"

    def test_aficion_and_texto_visible_can_coexist(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: una\nAFICION: guitarra\nPAREJA: no\nTEXTO_VISIBLE: Bar Manolo",
        )

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        categories = {inferred.category for inferred in inferences}
        assert categories == {"aficion", "texto_visible"}

    def test_no_texto_visible_inference_when_ninguno(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: varias\nAFICION: ninguno\nPAREJA: no\nTEXTO_VISIBLE: ninguno",
        )

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []

    def test_discards_aficion_when_several_people_are_similarly_prominent(self, monkeypatch):
        """El caso que motivó este campo: con varias personas de
        protagonismo similar (p. ej. una pareja), no hay forma de saber si
        la afición detectada es de la cuenta analizada o de la otra
        persona -- se descarta la señal en vez de arriesgarse a
        atribuirla a quien no toca."""
        _install_fake_model(
            monkeypatch, "PERSONAS: varias\nAFICION: toca la guitarra\nPAREJA: no"
        )

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []

    def test_pareja_signal_still_valid_with_several_people(self, monkeypatch):
        """A diferencia de la afición, la señal de pareja NO necesita
        resolver quién es la cuenta analizada -- de hecho 'varias'
        personas es el caso típico para esta señal."""
        _install_fake_model(monkeypatch, "PERSONAS: varias\nAFICION: ninguno\nPAREJA: si")

        inferences, indicio_pareja, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []
        assert indicio_pareja is True

    def test_unparseable_personas_value_discards_aficion_by_precaution(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: no lo sé\nAFICION: toca la guitarra\nPAREJA: no")

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []

    def test_missing_personas_line_discards_aficion_by_precaution(self, monkeypatch):
        _install_fake_model(monkeypatch, "AFICION: toca la guitarra\nPAREJA: no")

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []

    @pytest.mark.parametrize("negative_value", ["ninguno", "Ninguna", "none", "N/A", ""])
    def test_ninguno_variants_produce_no_inference(self, monkeypatch, negative_value):
        _install_fake_model(monkeypatch, f"PERSONAS: una\nAFICION: {negative_value}\nPAREJA: no")

        inferences, _, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []

    def test_missing_aficion_line_is_ignored_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: una\nPAREJA: no")

        inferences, indicio_pareja, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []
        assert indicio_pareja is False

    def test_missing_pareja_line_defaults_to_false(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: una\nAFICION: toca la guitarra")

        inferences, indicio_pareja, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert len(inferences) == 1
        assert indicio_pareja is False

    def test_completely_unexpected_format_degrades_without_crashing(self, monkeypatch):
        _install_fake_model(monkeypatch, "esto no sigue el formato pedido en absoluto")

        inferences, indicio_pareja, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []
        assert indicio_pareja is False

    def test_dependencies_not_installed_returns_empty_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        inferences, indicio_pareja, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert inferences == []
        assert indicio_pareja is False

    def test_model_exception_degrades_without_crashing(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel()))

        inferences, indicio_pareja, _, _, _codes = scene_analysis.analyze_image_content(_fake_image())

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

    def test_texto_visible_forbids_personal_names(self, monkeypatch):
        """TEXTO_VISIBLE es el campo con más riesgo de fuga de privacidad
        de los cuatro (texto real de la foto puede incluir un nombre
        propio en una camiseta, insignia, etc.) -- se comprueba que el
        prompt se lo prohíbe EXPLÍCITAMENTE para ese campo en concreto,
        no solo con la prohibición general del final del prompt."""
        structured_lower = scene_analysis._STRUCTURED_QUERY.lower()
        assert "texto_visible" in structured_lower
        assert "nombre propio" in structured_lower

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

    def test_description_is_reconstructed_from_parsed_values_not_raw_text(self, monkeypatch):
        """descripcion_cruda ya NO es el texto crudo del modelo -- se
        reconstruye desde los valores ya parseados (ver
        _build_clean_summary). Con una afición positiva, se muestra esa
        señal; con PAREJA/TEXTO_VISIBLE negativos, esas líneas NO
        aparecen."""
        _install_fake_model(
            monkeypatch,
            "PERSONAS: una\nAFICION: Posible fan de baloncesto\nPAREJA: no\nTEXTO_VISIBLE: ninguno",
            caption_answer="a person playing basketball",
        )

        _, _, description, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert description == "Personas en la foto: una\nPosible afición o interés: Posible fan de baloncesto"
        # La descripción general (caption) NO se repite dentro de este bloque.
        assert "basketball" not in description
        # Las señales negativas por defecto no generan línea.
        assert "pareja" not in description.lower()
        assert "texto visible" not in description.lower()

    def test_description_discards_trailing_model_garbage(self, monkeypatch):
        """Regresión del bug real visto en producción: tras responder
        bien las cuatro líneas, el modelo a veces sigue generando y
        empieza a copiar fragmentos de la propia explicación del prompt
        (p. ej. 'PERSONAS solo puede val...'). Como descripcion_cruda se
        reconstruye desde valores YA PARSEADOS (no desde el texto crudo),
        esa cola queda descartada automáticamente sin importar qué
        contenga."""
        _install_fake_model(
            monkeypatch,
            "PERSONAS: varias\nAFICION: ninguno\nPAREJA: no\nTEXTO_VISIBLE: ninguno\nPERSONAS solo puede val",
            caption_answer="una escena cualquiera",
        )

        _, _, description, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert description == "Personas en la foto: varias"
        assert "solo puede" not in description

    def test_description_shows_pareja_and_texto_visible_when_positive(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: varias\nAFICION: ninguno\nPAREJA: si\nTEXTO_VISIBLE: Bar Manolo",
            caption_answer="una escena cualquiera",
        )

        _, _, description, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert description == (
            "Personas en la foto: varias\nIndicio de contexto de pareja: sí\nTexto visible: Bar Manolo"
        )

    def test_description_is_none_when_dependencies_not_installed(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        _, _, description, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert description is None

    def test_description_is_none_when_model_raises(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel()))

        _, _, description, _, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert description is None

    def test_parses_descripcion_general_from_dedicated_caption_call(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: varias\nAFICION: ninguno\nPAREJA: no",
            caption_answer="4 people happily eating pizza on a terrace",
        )

        _, _, _, descripcion_general, _codes = scene_analysis.analyze_image_content(_fake_image())

        assert descripcion_general == "4 people happily eating pizza on a terrace"

    def test_descripcion_general_is_none_when_dependencies_not_installed(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        _, _, _, descripcion_general, _codes = scene_analysis.analyze_image_content(_fake_image())

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

        scene_analysis.analyze_image_content(_fake_image())

        assert len(encode_calls) == 1
        assert len(query_calls) == 2
        assert set(query_calls) == {scene_analysis._CAPTION_QUERY, scene_analysis._STRUCTURED_QUERY}

    def test_image_is_resized_before_encoding(self, monkeypatch):
        """Optimización real (medida en producción, GTX 1650): sin
        redimensionar antes de encode_image(), Moondream2 troceaba la
        imagen en 8 crops locales + 1 global = 9 pasadas por el encoder de
        visión (~33s); redimensionando a que el lado mayor mida
        _CAPTION_MAX_DIMENSION (378, el crop_size real de esta revisión
        del modelo, ver esa constante), pasa a 2 pasadas. Se comprueba
        contando el tamaño de la imagen que de verdad llega a
        encode_image(), no solo confiando en que 'debería' redimensionarse."""
        received_sizes: list[tuple[int, int]] = []

        class _SizeCheckingModel:
            def encode_image(self, image):
                received_sizes.append(image.size)
                return "encoded-sentinel"

            def query(self, image, question, settings=None):
                if question == scene_analysis._CAPTION_QUERY:
                    return {"answer": "una escena cualquiera"}
                return {"answer": "PERSONAS: ninguna\nAFICION: ninguno\nPAREJA: no"}

        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(
            scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _SizeCheckingModel())
        )

        # Imagen grande (más que _CAPTION_MAX_DIMENSION en ambas
        # dimensiones), como llegaría de verdad ya redimensionada a los
        # 1024px de geolocation.py -- para comprobar que ESTE módulo la
        # reduce más todavía, no que ya viniera pequeña de antes.
        imagen_grande = Image.new("RGB", (1000, 700))

        scene_analysis.analyze_image_content(imagen_grande)

        assert len(received_sizes) == 1
        width, height = received_sizes[0]
        assert max(width, height) <= scene_analysis._CAPTION_MAX_DIMENSION
        # La imagen original (compartida con DINOv2 en geolocation.py, ver
        # docstring de analyze_image_content) NUNCA debe mutarse in-place:
        # solo se redimensiona una copia.
        assert imagen_grande.size == (1000, 700)

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


class TestVisualDescriptionCodes:
    """ADR-30: codes es el 5º valor de analyze_image_content(), pensado
    para que el frontend traduzca sin depender del texto ya redactado en
    español de `descripcion_cruda` (el 3er valor, que se mantiene
    intacto). Aquí solo se comprueba que `codes` refleja fielmente lo que
    el modelo devolvió -- el parseo en sí (personas/afición/texto
    visible) ya está cubierto en las clases TestParse* de abajo."""

    def test_codes_mirror_the_parsed_values(self, monkeypatch):
        _install_fake_model(
            monkeypatch,
            "PERSONAS: una\nAFICION: guitarra eléctrica\nPAREJA: no\nTEXTO_VISIBLE: Bar El Rincón",
        )

        _, _, _, _, codes = scene_analysis.analyze_image_content(_fake_image())

        assert codes.personas == "una"
        assert codes.aficion is not None and "guitarra" in codes.aficion.lower()
        assert codes.texto_visible == "Bar El Rincón"
        assert codes.indicio_pareja is False

    def test_codes_indicio_pareja_true(self, monkeypatch):
        _install_fake_model(monkeypatch, "PERSONAS: varias\nAFICION: ninguno\nPAREJA: si")

        _, _, _, _, codes = scene_analysis.analyze_image_content(_fake_image())

        assert codes.indicio_pareja is True

    def test_codes_personas_never_ninguna(self, monkeypatch):
        """Mismo filtro que _build_clean_summary aplica al texto en
        español: 'ninguna' no es señal, así que codes.personas debe salir
        None, no la cadena 'ninguna' -- para que el frontend nunca tenga
        que replicar este filtro por su cuenta ni pueda mostrarlo por
        error si algún día deja de usar _build_clean_summary."""
        _install_fake_model(monkeypatch, "PERSONAS: ninguna\nAFICION: ninguno\nPAREJA: no")

        _, _, _, _, codes = scene_analysis.analyze_image_content(_fake_image())

        assert codes.personas is None

    def test_codes_none_on_model_unavailable(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: False)

        *_, codes = scene_analysis.analyze_image_content(_fake_image())

        assert codes is None

    def test_codes_none_on_model_failure(self, monkeypatch):
        monkeypatch.setattr(scene_analysis, "_scene_analysis_available", lambda: True)
        monkeypatch.setattr(
            scene_analysis, "_lazy_load", lambda: setattr(scene_analysis, "_model", _RaisingModel())
        )

        *_, codes = scene_analysis.analyze_image_content(_fake_image())

        assert codes is None


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

    def test_prompt_example_for_personas_is_never_a_valid_category(self):
        """Regresión (segunda vuelta): el ejemplo de PERSONAS en
        `_STRUCTURED_QUERY` pasó primero por "varias" (Moondream2 lo
        copiaba de forma sistemática incluso con una sola persona) y
        luego por "ninguna" -- que sufrió EXACTAMENTE el mismo problema,
        solo que trasladado a otro valor (reportado por Comandante en
        producción: fotos con varias personas mostrando "ninguna"). El
        fallo real nunca fue "qué palabra concreta usar de ejemplo", sino
        que cualquier valor VÁLIDO puesto ahí se copia igual. Por eso el
        ejemplo ya no debe ser ninguna de las tres categorías reales
        (`ninguna`/`una`/`varias`): si el modelo lo copia por confusión,
        `_parse_personas` debe descartarlo (dato ausente) en vez de
        mostrar una categoría incorrecta pero creíble."""
        example_section = scene_analysis._STRUCTURED_QUERY.split("\n\n")[0]
        personas_line = next(line for line in example_section.splitlines() if line.startswith("PERSONAS:"))
        example_value = personas_line.split(":", 1)[1].strip()
        assert example_value not in ("ninguna", "una", "varias")
        # Y si por algún motivo se copiara igualmente, el parseo debe
        # descartarlo -- no vale con que el ejemplo "no sea una de las
        # tres", también hace falta que el propio parser lo rechace.
        assert scene_analysis._parse_personas(personas_line) is None


class TestParseTextoVisible:
    def test_valid_value(self):
        assert scene_analysis._parse_texto_visible("TEXTO_VISIBLE: Bar Manolo") == "Bar Manolo"

    def test_missing_line_returns_none(self):
        assert scene_analysis._parse_texto_visible("PERSONAS: una\nAFICION: ninguno") is None

    @pytest.mark.parametrize("negative_value", ["ninguno", "Ninguna", "none", "N/A", ""])
    def test_ninguno_variants_return_none(self, negative_value):
        assert scene_analysis._parse_texto_visible(f"TEXTO_VISIBLE: {negative_value}") is None

    def test_trailing_period_is_stripped(self):
        assert scene_analysis._parse_texto_visible("TEXTO_VISIBLE: Calle Mayor 12.") == "Calle Mayor 12"


class TestParseAficionRaw:
    """_parse_aficion_raw es la versión SIN la cautela de atribución de
    _parse_inferences (que descarta la señal con varias personas en la
    foto) -- usada tanto por _parse_inferences como por
    _build_clean_summary, cada una con sus propias reglas sobre cuándo
    usar el valor."""

    def test_valid_value_regardless_of_personas(self):
        # A diferencia de _parse_inferences, _parse_aficion_raw NO mira
        # PERSONAS en absoluto -- esa cautela vive en quien la llama.
        assert scene_analysis._parse_aficion_raw("PERSONAS: varias\nAFICION: guitarra") == "guitarra"

    @pytest.mark.parametrize("negative_value", ["ninguno", "Ninguna", "none", "N/A", ""])
    def test_ninguno_variants_return_none(self, negative_value):
        assert scene_analysis._parse_aficion_raw(f"AFICION: {negative_value}") is None

    @pytest.mark.parametrize(
        "cross_field_value",
        ["uno", "Uno", "UNO", "una", "dos", "tres", "personas", "cero"],
    )
    def test_number_words_are_discarded_as_probable_cross_field_bleed(self, cross_field_value):
        """Regresión: reportado en producción (Comandante, agosto 2026)
        "Posible afición o interés: uno" en fotos sin ninguna afición
        real visible -- un número en palabra no es una afición, y todo
        apunta a un fallo de generación cruzada con el vocabulario de
        PERSONAS en vez de una señal real. Se descarta por precaución en
        vez de mostrarse como si fuera fiable."""
        assert scene_analysis._parse_aficion_raw(f"AFICION: {cross_field_value}") is None

    def test_hobby_containing_a_number_is_still_accepted(self):
        """La guarda de arriba descarta el valor EXACTO 'uno'/'dos'/etc,
        no cualquier afición que simplemente CONTENGA un número -- una
        afición real como coleccionar cromos de un año concreto sigue
        siendo válida."""
        assert (
            scene_analysis._parse_aficion_raw("AFICION: cromos del Mundial 2010")
            == "cromos del Mundial 2010"
        )


class TestBuildCleanSummary:
    """_build_clean_summary reconstruye el bloque 'qué vio la IA' que se
    muestra en el frontend -- solo señales positivas/informativas, para
    no acumular líneas vacías tipo 'AFICION: ninguno' / 'PAREJA: no'."""

    def test_only_personas_when_nothing_else_positive(self):
        summary = scene_analysis._build_clean_summary(
            personas="varias", aficion_raw=None, indicio_pareja=False, texto_visible=None
        )
        assert summary == "Personas en la foto: varias"

    def test_includes_aficion_when_positive(self):
        summary = scene_analysis._build_clean_summary(
            personas="una", aficion_raw="guitarra", indicio_pareja=False, texto_visible=None
        )
        assert summary == "Personas en la foto: una\nPosible afición o interés: guitarra"

    def test_includes_pareja_only_when_true(self):
        summary_true = scene_analysis._build_clean_summary(
            personas="varias", aficion_raw=None, indicio_pareja=True, texto_visible=None
        )
        assert "Indicio de contexto de pareja: sí" in summary_true

        summary_false = scene_analysis._build_clean_summary(
            personas="varias", aficion_raw=None, indicio_pareja=False, texto_visible=None
        )
        assert "pareja" not in summary_false.lower()

    def test_includes_texto_visible_when_present(self):
        summary = scene_analysis._build_clean_summary(
            personas="ninguna", aficion_raw=None, indicio_pareja=False, texto_visible="Bar Manolo"
        )
        # "ninguna" ya no se muestra (ver docstring de _build_clean_summary,
        # tratado igual que el resto de valores negativos por defecto).
        assert summary == "Texto visible: Bar Manolo"

    def test_personas_ninguna_is_hidden_like_other_negative_defaults(self):
        """Regresión (Comandante, agosto 2026): 'ninguna' ya no se
        considera una excepción -- se oculta igual que 'afición: ninguno'
        o 'pareja: no', para minimizar el impacto visible si el sesgo de
        `_STRUCTURED_QUERY` documentado en scene_analysis.py (Moondream2
        copiando el valor de ejemplo del prompt) volviera a aparecer."""
        assert scene_analysis._build_clean_summary(
            personas="ninguna", aficion_raw=None, indicio_pareja=False, texto_visible=None
        ) is None

    @pytest.mark.parametrize("value", ["una", "varias"])
    def test_personas_una_o_varias_se_siguen_mostrando(self, value):
        summary = scene_analysis._build_clean_summary(
            personas=value, aficion_raw=None, indicio_pareja=False, texto_visible=None
        )
        assert summary == f"Personas en la foto: {value}"

    def test_all_four_positive_at_once(self):
        summary = scene_analysis._build_clean_summary(
            personas="varias", aficion_raw="baloncesto", indicio_pareja=True, texto_visible="Bar Manolo"
        )
        assert summary == (
            "Personas en la foto: varias\n"
            "Posible afición o interés: baloncesto\n"
            "Indicio de contexto de pareja: sí\n"
            "Texto visible: Bar Manolo"
        )

    def test_none_when_nothing_at_all(self):
        assert scene_analysis._build_clean_summary(None, None, False, None) is None


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
