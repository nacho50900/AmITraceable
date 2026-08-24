"""
Tests de app/nlp/translation.py (ADR-31).

`ctranslate2`/`sentencepiece` SÍ están instalados en este entorno de test
(pip normal, sin GPU ni modelo real necesario para importarlos), pero
ningún modelo convertido existe en disco -- así que la mayoría de tests
comprueban la degradación a "devolver los textos originales" sin mockear
nada, exactamente el comportamiento real en un checkout limpio antes de
ejecutar scripts/convert_translation_models.py. Los tests del camino
"feliz" (con modelo disponible) sustituyen `ctranslate2.Translator` y
`sentencepiece.SentencePieceProcessor` por fakes, igual que ya se hace
con `torch`/`torch_directml` en test_geolocation.py -- no hace falta un
modelo real (ni GPU) para probar la lógica de este módulo.
"""
import sys
from types import SimpleNamespace

import pytest

from app.nlp import translation


@pytest.fixture(autouse=True)
def reset_translation_caches(monkeypatch):
    """Cada test parte de las cachés de módulo vacías, para que
    `_lazy_load()` se comporte de forma predecible independientemente
    del orden de ejecución de los tests."""
    monkeypatch.setattr(translation, "_translators", {})
    monkeypatch.setattr(translation, "_source_tokenizers", {})
    monkeypatch.setattr(translation, "_target_tokenizers", {})
    yield


def _make_fake_model_dir(tmp_path, direction: str):
    """Crea `tmp_path/<direction>/{model.bin,source.spm,target.spm}`
    (ficheros vacíos, el contenido no importa -- `_direction_available()`
    solo comprueba que existan) y apunta `translation._MODELS_DIR` ahí."""
    model_dir = tmp_path / direction
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"")
    (model_dir / "source.spm").write_bytes(b"")
    (model_dir / "target.spm").write_bytes(b"")
    return model_dir


class _FakeTranslationResult:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses


class _FakeTranslator:
    """Sustituye a ctranslate2.Translator: "traduce" devolviendo los
    mismos tokens que recibió, para que el _FakeSentencePieceProcessor de
    abajo pueda dejar una marca visible (mayúsculas) al decodificar --
    así un test puede comprobar que el texto pasó por el pipeline
    completo sin necesitar un modelo real."""

    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device

    def translate_batch(self, tokenized, **kwargs):
        return [_FakeTranslationResult([tokens]) for tokens in tokenized]


class _FakeSentencePieceProcessor:
    def load(self, path):
        self.path = path

    def encode(self, text, out_type=str):
        return [text]

    def decode(self, tokens):
        # Mayúsculas como marca visible de que pasó por "traducción" --
        # ver _FakeTranslator.
        return " ".join(tokens).upper()


def _install_fake_ctranslate2(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "ctranslate2", SimpleNamespace(Translator=_FakeTranslator)
    )
    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        SimpleNamespace(SentencePieceProcessor=_FakeSentencePieceProcessor),
    )


class TestTranslateTextsLocalNoOps:
    """Casos que NUNCA deben tocar disco ni intentar cargar nada --
    comprobados con `translation._MODELS_DIR` apuntando a un directorio
    que no existe, para que cualquier acceso a disco fallase de forma
    ruidosa si se produjera."""

    def test_unsupported_lang_is_a_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path / "no-existe")

        result = translation.translate_texts_local(["texto"], lang="fr")

        assert result == ["texto"]

    def test_es_is_a_noop(self, monkeypatch, tmp_path):
        """"es" es el idioma nativo del proyecto -- nunca hace falta
        traducir HACIA español desde afición (que ya nace en español)."""
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path / "no-existe")

        result = translation.translate_texts_local(["una persona en la playa"], lang="es")

        assert result == ["una persona en la playa"]

    def test_empty_list_is_a_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path / "no-existe")

        assert translation.translate_texts_local([], lang="en") == []


class TestGracefulDegradationWithoutConvertedModels:
    """Checkout limpio: scripts/convert_translation_models.py no se ha
    ejecutado todavía -- translate_texts_local() debe devolver los
    textos originales, nunca lanzar."""

    def test_returns_originals_when_models_not_converted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path / "vacio")

        result = translation.translate_texts_local(["guitarra", "baloncesto"], lang="en")

        assert result == ["guitarra", "baloncesto"]

    def test_translation_available_false_when_models_not_converted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path / "vacio")

        assert translation.translation_available("en") is False
        assert translation.translation_available("es") is False

    def test_translation_available_false_for_unsupported_lang(self, monkeypatch, tmp_path):
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)

        assert translation.translation_available("fr") is False


class TestGracefulDegradationWithoutLibraries:
    """ctranslate2/sentencepiece no instalados (backend construido sin
    WITH_GEOLOCATION=true, ver requirements-vision.txt) -- degrada igual
    que sin modelos convertidos, nunca lanza ImportError hacia arriba."""

    def test_returns_originals_when_import_fails(self, monkeypatch, tmp_path):
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)
        # sys.modules[name] = None fuerza ImportError en `import ctranslate2`
        # -- comportamiento documentado de CPython, mismo truco que ya usa
        # el resto del proyecto para simular dependencias no instaladas.
        monkeypatch.setitem(sys.modules, "ctranslate2", None)

        result = translation.translate_texts_local(["guitarra"], lang="en")

        assert result == ["guitarra"]


class TestTranslateTextsLocalHappyPath:
    """Con el modelo "convertido" (directorio fake) y ctranslate2/
    sentencepiece sustituidos por fakes -- ver _install_fake_ctranslate2."""

    def test_calls_translate_batch_with_safeguards_against_repetition_loops(self, monkeypatch, tmp_path):
        """Regresión de un fallo real visto en producción: `beam_size=1`
        (voraz) sin ninguna salvaguarda se quedó enganchado repitiendo
        "persona" cientos de veces en vez de traducir de verdad -- ver el
        comentario junto a translate_batch(). Comprueba que se piden las
        tres salvaguardas, no que produzcan un resultado concreto (eso
        haría falta un modelo real, imposible de probar aquí)."""
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)
        received_kwargs = {}

        class _RecordingTranslator(_FakeTranslator):
            def translate_batch(self, tokenized, **kwargs):
                received_kwargs.update(kwargs)
                return super().translate_batch(tokenized, **kwargs)

        monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(Translator=_RecordingTranslator))
        monkeypatch.setitem(
            sys.modules,
            "sentencepiece",
            SimpleNamespace(SentencePieceProcessor=_FakeSentencePieceProcessor),
        )

        translation.translate_texts_local(["guitarra"], lang="en")

        assert received_kwargs["beam_size"] >= 2  # nunca voraz sin red de seguridad
        assert received_kwargs["repetition_penalty"] > 1
        assert received_kwargs["no_repeat_ngram_size"] > 0

    def test_absurdly_long_repetitive_output_falls_back_to_original(self, monkeypatch, tmp_path):
        """Reproduce el bug real tal cual se vio en producción: el modelo
        devuelve un bucle de repetición larguísimo en vez de fallar con
        una excepción -- la salvaguarda de longitud debe descartarlo y
        conservar el texto original, no colar la basura al usuario."""
        _make_fake_model_dir(tmp_path, "en-es")  # lang="es" => origen "en" => dirección "en-es"
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)

        class _LoopingTranslator(_FakeTranslator):
            def translate_batch(self, tokenized, **kwargs):
                garbage = " ".join(["persona"] * 200)
                return [_FakeTranslationResult([[garbage]]) for _ in tokenized]

        monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(Translator=_LoopingTranslator))

        class _PassthroughDecodeTokenizer(_FakeSentencePieceProcessor):
            def decode(self, tokens):
                # El _LoopingTranslator ya mete el texto final directamente
                # como ÚNICA "hipótesis" (una lista de un solo elemento,
                # ver hypotheses=[garbage] arriba) -- el decode real solo
                # tiene que devolverla tal cual, sin la marca de mayúsculas
                # de _FakeSentencePieceProcessor (que aquí solo estorbaría
                # para ver la longitud real de la basura).
                return tokens[0]

        monkeypatch.setitem(
            sys.modules,
            "sentencepiece",
            SimpleNamespace(SentencePieceProcessor=_PassthroughDecodeTokenizer),
        )

        result = translation.translate_texts_local(["a person playing guitar"], lang="es")

        assert result == ["a person playing guitar"]

    def test_translates_and_caches_the_loaded_model(self, monkeypatch, tmp_path):
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)
        _install_fake_ctranslate2(monkeypatch)

        result = translation.translate_texts_local(["guitarra", "baloncesto"], lang="en")

        assert result == ["GUITARRA", "BALONCESTO"]
        assert "es-en" in translation._translators

    def test_direction_is_derived_from_target_lang_not_passed_explicitly(self, monkeypatch, tmp_path):
        """Con solo dos idiomas soportados, pedir lang="es" implica
        origen "en" -- ver docstring del módulo."""
        _make_fake_model_dir(tmp_path, "en-es")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)
        _install_fake_ctranslate2(monkeypatch)

        result = translation.translate_texts_local(["a person at the beach"], lang="es")

        assert result == ["A PERSON AT THE BEACH"]
        assert "en-es" in translation._translators

    def test_second_call_reuses_cached_translator(self, monkeypatch, tmp_path):
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)
        _install_fake_ctranslate2(monkeypatch)

        translation.translate_texts_local(["uno"], lang="en")
        cached_translator = translation._translators["es-en"]
        translation.translate_texts_local(["dos"], lang="en")

        assert translation._translators["es-en"] is cached_translator

    def test_empty_translation_result_falls_back_to_original(self, monkeypatch, tmp_path):
        """Si el modelo devuelve una cadena vacía para una posición
        concreta, se conserva el texto ORIGINAL ahí -- mismo criterio
        defensivo que tenía la versión anterior con Mistral (ADR-30)."""
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)

        class _EmptyResultTranslator(_FakeTranslator):
            def translate_batch(self, tokenized, **kwargs):
                return [_FakeTranslationResult([""]) for _ in tokenized]

        monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(Translator=_EmptyResultTranslator))
        monkeypatch.setitem(
            sys.modules,
            "sentencepiece",
            SimpleNamespace(SentencePieceProcessor=_FakeSentencePieceProcessor),
        )

        result = translation.translate_texts_local(["guitarra"], lang="en")

        assert result == ["guitarra"]

    def test_returns_originals_if_translation_raises(self, monkeypatch, tmp_path):
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)

        class _RaisingTranslator(_FakeTranslator):
            def translate_batch(self, tokenized, **kwargs):
                raise RuntimeError("modelo corrupto")

        monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(Translator=_RaisingTranslator))
        monkeypatch.setitem(
            sys.modules,
            "sentencepiece",
            SimpleNamespace(SentencePieceProcessor=_FakeSentencePieceProcessor),
        )

        result = translation.translate_texts_local(["guitarra", "baloncesto"], lang="en")

        assert result == ["guitarra", "baloncesto"]

    def test_translation_available_true_when_model_and_files_present(self, monkeypatch, tmp_path):
        _make_fake_model_dir(tmp_path, "es-en")
        monkeypatch.setattr(translation, "_MODELS_DIR", tmp_path)

        assert translation.translation_available("en") is True
