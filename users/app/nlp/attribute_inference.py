"""
Inferencia de atributos personales a partir del contenido público del propio
usuario autenticado.

Importante: esto NO es un motor de deanonimización de terceros. Solo se
ejecuta sobre el historial del usuario que ha dado consentimiento explícito
vía OAuth, y el resultado se le muestra únicamente a él. El objetivo es que
vea lo que YA es inferible públicamente sobre su propia cuenta.

Para el MVP se usan heurísticas simples y explicables (regex + listas de
subreddits geolocalizables) en lugar de modelos de inferencia más agresivos,
precisamente para mantener el alcance defensivo y evitar sobre-ingeniería.
"""
import re
from collections import Counter

from app.models.schemas import InferredAttribute, RedditPost

# Subreddits cuyo propio nombre ya delata ciudad/país/profesión (lista corta
# de ejemplo; en producción se externalizaría a un fichero de datos curado)
_LOCATION_SUBREDDITS = {
    "madrid", "barcelona", "spain", "es", "valencia", "sevilla",
    "argentina", "mexico", "askspain", "vzla",
}
_OCCUPATION_SUBREDDITS = {
    "programming", "cscareerquestions", "developerspain", "medlabprofessionals",
    "teachers", "nursing", "legaladvice", "accounting",
}
_ROUTINE_KEYWORDS = re.compile(r"\b(turno de noche|madrugo|salgo del trabajo|entro a las|termino a las)\b", re.I)


def infer_attributes(posts: list[RedditPost]) -> list[InferredAttribute]:
    attributes: list[InferredAttribute] = []
    attributes += _infer_location(posts)
    attributes += _infer_occupation(posts)
    attributes += _infer_routine(posts)
    return attributes


def _infer_location(posts: list[RedditPost]) -> list[InferredAttribute]:
    hits = [p for p in posts if p.subreddit.lower() in _LOCATION_SUBREDDITS]
    if not hits:
        return []

    counter = Counter(p.subreddit.lower() for p in hits)
    top_sub, count = counter.most_common(1)[0]
    confidence = min(0.4 + 0.05 * count, 0.9)

    return [
        InferredAttribute(
            category="ubicacion",
            value=f"Posible vínculo geográfico con: {top_sub}",
            confidence=round(confidence, 2),
            evidence=[p.permalink for p in hits[:5]],
        )
    ]


def _infer_occupation(posts: list[RedditPost]) -> list[InferredAttribute]:
    hits = [p for p in posts if p.subreddit.lower() in _OCCUPATION_SUBREDDITS]
    if not hits:
        return []

    counter = Counter(p.subreddit.lower() for p in hits)
    top_sub, count = counter.most_common(1)[0]
    confidence = min(0.3 + 0.05 * count, 0.8)

    return [
        InferredAttribute(
            category="ocupacion",
            value=f"Posible sector profesional relacionado con: {top_sub}",
            confidence=round(confidence, 2),
            evidence=[p.permalink for p in hits[:5]],
        )
    ]


def _infer_routine(posts: list[RedditPost]) -> list[InferredAttribute]:
    hits = [p for p in posts if _ROUTINE_KEYWORDS.search(p.text)]
    if not hits:
        return []

    return [
        InferredAttribute(
            category="rutina",
            value="Menciones explícitas de horarios/rutina diaria en el texto",
            confidence=0.6,
            evidence=[p.permalink for p in hits[:5]],
        )
    ]
