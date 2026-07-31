"""
Detección heurística (regex) de publicaciones cuyo pie de foto indica que la
persona está DE VIAJE/VACACIONES/DE PASO en un sitio, no en su lugar de
residencia habitual.

Motivación: `app/vision/geolocation.py` estima dónde fue tomada una foto,
pero "dónde fue tomada una foto" NO es lo mismo que "dónde vive la persona"
-- una foto con el pie "De viaje en Roma 🇮🇹" con altísima confianza de
geolocalización en Roma NO debe usarse para inferir que la persona vive en
Roma. Ver report/generator.py, donde estas publicaciones se excluyen de la
votación de residencia habitual (aunque SIGUEN apareciendo en
`image_location_points`, con su confianza real, para que el frontend las
muestre igual -- solo se excluyen del cálculo de dónde vive la persona).

Esto es una red de seguridad determinista y gratuita, complementaria a la
detección por IA (ver ai_attribute_extraction.py, campo "fotos_de_viaje" del
prompt a Mistral): funciona siempre, incluso sin MISTRAL_API_KEY, y cubre
las expresiones más habituales en español. Las dos fuentes se UNEN (nunca se
sustituyen entre sí) porque aquí un falso positivo (excluir por error una
foto que sí es de casa) es mucho menos grave que un falso negativo (usar una
foto de vacaciones para "inferir" dónde vive la persona) -- mismo criterio
de cautela que ya se aplica en el resto del pipeline de geolocalización.
"""
import re

from app.models.schemas import SocialPost

_TRAVEL_RE = re.compile(
    r"\b("
    r"de viaje|de vacaciones|de vacas|de escapada|de paso por|"
    r"unos d[ií]as en|unas semanas en|"
    r"turisteando|haciendo turismo|de turismo por|visitando"
    r")\b",
    re.IGNORECASE,
)


def detect_travel_permalinks(posts: list[SocialPost]) -> set[str]:
    """Devuelve el conjunto de permalinks de `posts` cuyo texto (caption)
    contiene alguna expresión de viaje/vacaciones reconocida. No distingue
    DÓNDE dice que está de viaje (no hace falta: basta con saber que ESA
    foto concreta no es representativa de su residencia habitual)."""
    return {post.permalink for post in posts if post.text and _TRAVEL_RE.search(post.text)}
