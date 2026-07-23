"""
Inferencia de atributos personales a partir del contenido público del propio
usuario autenticado.

Importante: esto NO es un motor de deanonimización de terceros. Solo se
ejecuta sobre el historial del usuario que ha dado consentimiento explícito
vía OAuth, y el resultado se le muestra únicamente a él. El objetivo es que
vea lo que YA es inferible públicamente sobre su propia cuenta.

Para el MVP se usan heurísticas simples y explicables (regex + listas de
comunidades/etiquetas geolocalizables) en lugar de modelos de inferencia
más agresivos, precisamente para mantener el alcance defensivo y evitar
sobre-ingeniería. Funciona igual sobre subreddits (Reddit) que sobre
hashtags (Instagram), ya que ambos llegan normalizados al campo `tags` de
`SocialPost` (un único subreddit en Reddit, todos los hashtags del caption
en Instagram).

Nota de diseño Reddit vs Instagram: en Reddit, el nombre de la comunidad
suele ser el nombre exacto del tema ("cscareerquestions"). En Instagram,
los hashtags suelen ser palabras compuestas sin separador
("nurselife", "amanecemosmadrid", "teachersofinstagram"), así que además
de la coincidencia exacta se comprueba si alguna palabra clave aparece
*dentro* del hashtag como subcadena. Es un poco menos preciso que la
coincidencia exacta de Reddit, pero necesario para no perder toda señal en
Instagram; se documenta aquí como decisión consciente para la memoria.
"""
import re
from collections import Counter

from app.models.schemas import InferredAttribute, SocialPost

# Palabras clave de ubicación. Sirven tanto para coincidencia exacta
# (subreddits de Reddit) como para subcadena (hashtags compuestos de
# Instagram: "amanecemosmadrid", "ig_valencia", "spainiswonderful"...).
_LOCATION_KEYWORDS = {
    "madrid", "barcelona", "spain", "espana", "valencia", "sevilla",
    "argentina", "mexico", "vzla", "malaga", "bilbao", "zaragoza",
    "galicia", "andalucia", "catalunya", "asturias",
}
# Coincidencias exactas adicionales que no tendría sentido tratar como
# subcadena (demasiado cortas / generarían falsos positivos, p. ej. "es"
# solo vale como subreddit exacto, no como subcadena de cualquier hashtag).
_LOCATION_EXACT_ONLY = {"es", "askspain"}

# Palabras clave de ocupación/sector profesional. Igual que arriba: exactas
# (subreddits) + subcadena (hashtags tipo "nurselife", "devlife",
# "teachersofinstagram", "abogadosdeinstagram").
_OCCUPATION_KEYWORDS = {
    "programming", "developer", "dev", "nurse", "enfermeria", "teacher",
    "docente", "legaladvice", "abogad", "accounting", "medlab", "medicina",
    "ingenieria", "arquitect",
}
_OCCUPATION_EXACT_ONLY = {
    "cscareerquestions", "developerspain", "medlabprofessionals",
    "teachers", "nursing", "legaladvice", "accounting",
}

_ROUTINE_KEYWORDS = re.compile(
    r"\b(turno de noche|madrugo|salgo del trabajo|entro a las|termino a las"
    r"|night shift|wfh|trabajo desde casa)\b",
    re.I,
)


def _tags_of(post: SocialPost) -> list[str]:
    """Todas las etiquetas normalizadas de un post: `tags` si está poblado
    (Instagram, o Reddit tras la migración), o `group` como fallback para
    compatibilidad con datos antiguos que no tuvieran `tags`."""
    return [t.lower() for t in post.tags] if post.tags else [post.group.lower()]


def _matches(tags: list[str], keywords: set[str], exact_only: set[str]) -> bool:
    if any(tag in exact_only for tag in tags):
        return True
    return any(keyword in tag for tag in tags for keyword in keywords)


def infer_attributes(posts: list[SocialPost]) -> list[InferredAttribute]:
    attributes: list[InferredAttribute] = []
    attributes += _infer_location(posts)
    attributes += _infer_occupation(posts)
    attributes += _infer_routine(posts)
    return attributes


def _infer_location(posts: list[SocialPost]) -> list[InferredAttribute]:
    hits = [p for p in posts if _matches(_tags_of(p), _LOCATION_KEYWORDS, _LOCATION_EXACT_ONLY)]
    if not hits:
        return []

    # Agrupamos por la primera etiqueta que disparó la coincidencia, para
    # poder mostrar un valor legible (ej. "madrid") en vez de todo el post.
    matched_tag_counter = Counter(
        next(
            (t for t in _tags_of(p) if t in _LOCATION_EXACT_ONLY or any(k in t for k in _LOCATION_KEYWORDS)),
            p.group.lower(),
        )
        for p in hits
    )
    top_tag, count = matched_tag_counter.most_common(1)[0]
    confidence = min(0.4 + 0.05 * count, 0.9)

    return [
        InferredAttribute(
            category="ubicacion",
            value=f"Posible vínculo geográfico con: {top_tag}",
            confidence=round(confidence, 2),
            evidence=[p.permalink for p in hits[:5]],
        )
    ]


def _infer_occupation(posts: list[SocialPost]) -> list[InferredAttribute]:
    hits = [p for p in posts if _matches(_tags_of(p), _OCCUPATION_KEYWORDS, _OCCUPATION_EXACT_ONLY)]
    if not hits:
        return []

    matched_tag_counter = Counter(
        next(
            (t for t in _tags_of(p) if t in _OCCUPATION_EXACT_ONLY or any(k in t for k in _OCCUPATION_KEYWORDS)),
            p.group.lower(),
        )
        for p in hits
    )
    top_tag, count = matched_tag_counter.most_common(1)[0]
    confidence = min(0.3 + 0.05 * count, 0.8)

    return [
        InferredAttribute(
            category="ocupacion",
            value=f"Posible sector profesional relacionado con: {top_tag}",
            confidence=round(confidence, 2),
            evidence=[p.permalink for p in hits[:5]],
        )
    ]


def _infer_routine(posts: list[SocialPost]) -> list[InferredAttribute]:
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
