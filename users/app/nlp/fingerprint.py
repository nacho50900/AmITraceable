"""
Módulo de fingerprinting de escritura.

Calcula un "perfil de estilo" a partir del texto público del usuario:
- longitud media de frase
- riqueza de vocabulario (type-token ratio)
- uso de emojis
- distribución horaria de actividad (posible zona horaria / rutina)
- subreddits más frecuentes (intereses, a veces geolocalizables)
- palabras clave recurrentes (TF-IDF)
- idioma detectado

Nota de alcance: en esta versión (Reddit-only) este fingerprint se usa
sobre todo como insumo para el motor de scoring (módulo 4), no para
correlación entre plataformas (módulo 3), que queda fuera del MVP actual.
"""
import re
from collections import Counter

import emoji
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.schemas import RedditPost, WritingFingerprint

_nlp_cache: dict[str, "spacy.language.Language"] = {}


def _get_spacy_model(lang: str):
    """Carga el modelo de spaCy de forma perezosa (lazy) y lo cachea en memoria."""
    model_name = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    if model_name not in _nlp_cache:
        _nlp_cache[model_name] = spacy.load(model_name)
    return _nlp_cache[model_name]


def _detect_language(text_sample: str) -> str:
    """Detección simple de idioma. Para el MVP: heurística por stopwords ES/EN.

    TODO (trabajo futuro): sustituir por un detector robusto (langdetect /
    fasttext) si se amplía a más idiomas.
    """
    spanish_markers = {"que", "de", "la", "el", "en", "y", "los", "se", "un", "por"}
    tokens = set(re.findall(r"\b\w+\b", text_sample.lower())[:300])
    return "es" if len(tokens & spanish_markers) >= 3 else "en"


def build_fingerprint(posts: list[RedditPost]) -> WritingFingerprint:
    if not posts:
        raise ValueError("No hay posts/comentarios para analizar")

    full_text = "\n".join(p.text for p in posts if p.text)
    language = _detect_language(full_text)
    nlp = _get_spacy_model(language)

    # Procesamos en lotes para no disparar el uso de memoria con cuentas con
    # mucho historial (principio de minimización RGPD + eficiencia)
    doc = nlp(full_text[:200_000])  # límite defensivo de caracteres

    sentences = list(doc.sents)
    avg_sentence_length = (
        sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0.0
    )

    tokens = [t.text.lower() for t in doc if t.is_alpha]
    vocabulary_richness = len(set(tokens)) / len(tokens) if tokens else 0.0

    emoji_count = sum(emoji.emoji_count(p.text) for p in posts)
    total_chars = sum(len(p.text) for p in posts) or 1
    emoji_usage_rate = emoji_count / total_chars

    hour_counter = Counter(p.created_utc.hour for p in posts)
    total = sum(hour_counter.values()) or 1
    avg_posts_per_hour = {h: hour_counter.get(h, 0) / total for h in range(24)}

    subreddit_counter = Counter(p.subreddit for p in posts)
    top_subreddits = subreddit_counter.most_common(15)

    top_keywords = _extract_keywords(posts, language)

    return WritingFingerprint(
        avg_sentence_length=round(avg_sentence_length, 2),
        vocabulary_richness=round(vocabulary_richness, 4),
        emoji_usage_rate=round(emoji_usage_rate, 5),
        avg_posts_per_hour=avg_posts_per_hour,
        top_subreddits=top_subreddits,
        top_keywords=top_keywords,
        detected_language=language,
    )


def _extract_keywords(posts: list[RedditPost], language: str, top_n: int = 20) -> list[tuple[str, float]]:
    texts = [p.text for p in posts if p.text and len(p.text) > 20]
    if len(texts) < 3:
        return []

    stop_words = "english" if language == "en" else None  # sklearn no trae stopwords ES nativas
    vectorizer = TfidfVectorizer(max_features=500, stop_words=stop_words, ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    scores = matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    ranked = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
    return [(word, round(float(score), 3)) for word, score in ranked[:top_n]]
