"""
Traducción LOCAL de las descripciones de fotos (caption/afición generados
por Moondream2, ver app/vision/scene_analysis.py y ADR-30) usando modelos
MarianMT (Helsinki-NLP/opus-mt-{es-en,en-es}) convertidos a formato
CTranslate2 -- NO Mistral (ver ADR-31, que sustituye la primera versión de
esta funcionalidad): un LLM de propósito general es un martillo pilón para
traducir frases cortas, y reintroduce justo la dependencia de red/cuota
que ya se evitó una vez para el propio Moondream2 (ver la nota junto a
`_CAPTION_QUERY` en scene_analysis.py).

CTranslate2 (no `transformers`+`torch` directos) porque es un motor de
inferencia dedicado, mucho más ligero en CPU para un modelo de este
tamaño -- benchmark oficial de OpenNMT/CTranslate2 para un modelo del
tamaño de OPUS-MT: ~2.3GB de RAM con Transformers/PyTorch en CPU frente a
~516MB con CTranslate2 en int8. Tampoco hace falta `transformers` como
dependencia en tiempo de EJECUCIÓN: los modelos ya convertidos llevan sus
propios ficheros SentencePiece (`source.spm`/`target.spm`), así que la
tokenización se hace con la librería `sentencepiece` a secas -- mucho más
ligera que cargar todo `transformers`. `transformers`+`torch` solo hacen
falta UNA VEZ, para la conversión inicial (ver
scripts/convert_translation_models.py), nunca en el servidor ya en
marcha.

Solo dos idiomas soportados en todo el proyecto (ver
`ai_analysis.SUPPORTED_LANGUAGES`): "es" y "en". Como solo hay dos, el
idioma de ORIGEN de un lote de textos queda determinado por el idioma de
DESTINO pedido -- si se pide traducir a "en", el origen solo puede ser
"es" (y viceversa), nunca hace falta que el llamador lo especifique
aparte. Esto se rompe el día que se añada un tercer idioma: en ese
momento hará falta que quien llama a `translate_texts_local()` indique
también el idioma de origen en vez de inferirlo aquí.

Degradación: si los modelos convertidos no están en disco (no se ha
ejecutado `scripts/convert_translation_models.py` todavía), si
`ctranslate2`/`sentencepiece` no están instalados, o falla cualquier otra
cosa durante la traducción, se devuelven los textos ORIGINALES sin
traducir -- mismo criterio best-effort que el resto de módulos opcionales
de este proyecto (DINOv2, Moondream2, worker de iGPU): nunca se lanza una
excepción que rompa el análisis o la visualización del informe.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "translation_models"

# Traductores y tokenizadores ya cargados, uno por DIRECCIÓN ("es-en",
# "en-es") -- cacheados para no releer los ficheros del modelo en cada
# petición. A diferencia de DINOv2/Moondream2 (precargados en el arranque
# del contenedor, ver app/main.py), estos se cargan la PRIMERA VEZ que
# hacen falta: la traducción es un uso puntual (una vez por informe
# abierto o cambio de idioma), no constante durante todo el análisis, así
# que no se justifica reservar esta RAM desde el arranque si el usuario
# nunca llega a cambiar de idioma.
_translators: dict = {}
_source_tokenizers: dict = {}
_target_tokenizers: dict = {}


def _direction_available(direction: str) -> bool:
    """True si el modelo convertido de esta dirección (p. ej. "es-en")
    está presente en disco -- ver scripts/convert_translation_models.py.
    No comprueba que `ctranslate2`/`sentencepiece` estén instalados, eso
    lo hace `_lazy_load()`."""
    model_dir = _MODELS_DIR / direction
    return (
        (model_dir / "model.bin").exists()
        and (model_dir / "source.spm").exists()
        and (model_dir / "target.spm").exists()
    )


def translation_available(lang: str) -> bool:
    """Comprobación barata (solo mira el disco, no carga nada) de si
    `translate_texts_local()` podrá traducir de verdad hacia `lang`, o si
    se limitará a devolver los textos sin cambios. Pensado para que quien
    llama pueda decidir si merece la pena intentarlo, sin pagar el coste
    de cargar el modelo solo para descubrir que no está."""
    if lang not in ("es", "en"):
        return False
    source_lang = "en" if lang == "es" else "es"
    return _direction_available(f"{source_lang}-{lang}")


def _lazy_load(direction: str) -> bool:
    """Carga el traductor y los tokenizadores de `direction` (p. ej.
    "es-en") si no estaban ya cargados. Devuelve False SIN LANZAR si los
    ficheros del modelo no están (script de conversión no ejecutado
    todavía) o si `ctranslate2`/`sentencepiece` no están instalados --
    mismo patrón que `_geolocation_available()`/`_scene_analysis_available()`
    en app/vision/."""
    if direction in _translators:
        return True
    if not _direction_available(direction):
        return False

    try:
        import ctranslate2
        import sentencepiece as spm
    except ImportError:
        logger.info(
            "ctranslate2/sentencepiece no instalados -- la traducción local "
            "de descripciones no está disponible (se devolverán los textos "
            "originales sin traducir). Ver requirements-vision.txt y "
            "scripts/convert_translation_models.py."
        )
        return False

    model_dir = _MODELS_DIR / direction
    try:
        translator = ctranslate2.Translator(str(model_dir), device="cpu")
        source_tok = spm.SentencePieceProcessor()
        source_tok.load(str(model_dir / "source.spm"))
        target_tok = spm.SentencePieceProcessor()
        target_tok.load(str(model_dir / "target.spm"))
    except Exception:
        logger.exception(
            "Fallo al cargar el modelo de traducción local (%s) -- se "
            "devolverán los textos originales sin traducir.",
            direction,
        )
        return False

    _translators[direction] = translator
    _source_tokenizers[direction] = source_tok
    _target_tokenizers[direction] = target_tok
    return True


def translate_texts_local(texts: list[str], lang: str) -> list[str]:
    """Traduce `texts` a `lang` con el modelo MarianMT/CTranslate2 local
    correspondiente. Devuelve la lista en el MISMO orden y con la MISMA
    longitud que `texts` SIEMPRE -- si algo falla en cualquier punto
    (modelo no convertido, librería no instalada, error de inferencia),
    se devuelven los textos ORIGINALES sin traducir en vez de lanzar.

    `lang` no soportado, vacío, o `texts` vacía: no-op, sin tocar disco ni
    cargar nada."""
    if lang not in ("es", "en") or not texts:
        return list(texts)

    # Con solo dos idiomas soportados en todo el proyecto, el origen es
    # el otro -- ver el docstring del módulo sobre por qué esto no
    # escala a un tercer idioma sin cambios.
    source_lang = "en" if lang == "es" else "es"
    direction = f"{source_lang}-{lang}"

    if not _lazy_load(direction):
        return list(texts)

    translator = _translators[direction]
    source_tok = _source_tokenizers[direction]
    target_tok = _target_tokenizers[direction]

    try:
        tokenized = [source_tok.encode(text, out_type=str) for text in texts]
        # beam_size=1 (greedy, no beam search): estas son frases cortas y
        # descriptivas (una afición, un caption de una foto), no texto
        # donde la calidad de la búsqueda en haz marque una diferencia
        # perceptible -- prioriza velocidad/CPU sobre una mejora marginal
        # de BLEU que no se notaría en la práctica.
        results = translator.translate_batch(tokenized, beam_size=1)
        translations = [target_tok.decode(r.hypotheses[0]) for r in results]
    except Exception:
        logger.exception(
            "Fallo al traducir localmente (%s) -- se devuelven los textos originales sin traducir.",
            direction,
        )
        return list(texts)

    # Defensivo: si algún elemento sale vacío tras traducir, se conserva
    # el ORIGINAL para esa posición en vez de dejar un hueco -- mismo
    # criterio que ya tenía la versión anterior de esta función con
    # Mistral (ver ADR-31).
    return [t if t.strip() else original for t, original in zip(translations, texts)]
