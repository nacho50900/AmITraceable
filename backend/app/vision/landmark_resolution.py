"""
Módulo opcional, best-effort: dado el NOMBRE de un edificio o monumento
que Moondream2 dice reconocer en una foto (campo EDIFICIO_EMBLEMATICO,
ver app/vision/scene_analysis.py), le pregunta al proveedor de IA ACTIVO
(`settings.ai_provider` -- Gemini por defecto, o Mistral, ver
app/nlp/ai_client.py) si reconoce ese lugar con certeza y, si es así, sus
coordenadas reales -- para poder sobreescribir la estimación por
similitud visual de DINOv2 (app/vision/geolocation.py) en esa foto
concreta con algo mucho más preciso que "la foto se parece a otras fotos
de esta provincia".

Por qué un LLM y no Moondream2 para las coordenadas: Moondream2 es un VQA
pequeño (cuantizado localmente, ver su nota de fiabilidad en
scene_analysis.py) sin conocimiento geográfico fiable de coordenadas --
pedirle un lat/lon directamente sería inventar precisión donde no la hay.
Un LLM de propósito general (Gemini o Mistral) tiene mucho más
conocimiento del mundo real (nombres de monumentos, ciudades, países),
así que aquí se usa su conocimiento general, NO visión -- solo recibe el
NOMBRE que propuso Moondream2 (texto plano), nunca la imagen.

Nunca se usa a ciegas: se le pide explícitamente al modelo que devuelva
null si no reconoce el lugar con certeza (mismo criterio de "ante la duda,
no inventes" que el resto del proyecto usa con las regex/INE), y
`report/generator.py` solo aplica el resultado si además viene con
`confianza` alta -- ver MIN_CONFIDENCE_FOR_LANDMARK_OVERRIDE.

Nota RGPD (para la memoria, mismo razonamiento que ai_analysis.py y
ai_attribute_extraction.py): esto envía al proveedor de IA activo SOLO el
nombre de un edificio/monumento propuesto por Moondream2 y, como contexto
opcional para desambiguar nombres genéricos (p. ej. "catedral"), la
descripción general de la foto -- nunca la imagen en sí, nunca datos
personales de la cuenta analizada. Con AI_PROVIDER=gemini (el por
defecto), ese proveedor es Google (EE.UU., expuesto al Cloud Act); con
AI_PROVIDER=mistral, es Mistral AI (Francia, UE) -- ver la justificación
completa de la elección de proveedor por defecto en app/config.py.
"""
import logging

from app.config import settings
from app.nlp.ai_client import AIHTTPError, AIRequestError, call_ai_json

logger = logging.getLogger(__name__)

# Umbral de confianza para aceptar la resolución del modelo y sobreescribir
# la estimación de DINOv2 -- deliberadamente alto: un edificio mal
# resuelto sería PEOR que no resolverlo (afirmaría una ubicación exacta
# incorrecta con la misma confianza visual que un acierto real).
MIN_CONFIDENCE_FOR_LANDMARK_OVERRIDE = 0.75

_SYSTEM_PROMPT = (
    "Eres un asistente con amplio conocimiento de monumentos, edificios y lugares "
    "emblemáticos de todo el mundo. Se te da el nombre de un posible edificio/monumento "
    "(propuesto por un modelo de visión artificial a partir de una foto, así que puede "
    "estar mal escrito, ser ambiguo, o ser un nombre genérico que no basta por sí solo) y, "
    "opcionalmente, una descripción general de la foto como contexto para desambiguar.\n\n"
    "Responde EXCLUSIVAMENTE con un JSON con esta forma exacta, sin texto adicional ni "
    "backticks:\n"
    '{"reconocido": true|false, "nombre_canonico": <string>|null, "lat": <float>|null, '
    '"lon": <float>|null, "confianza": <0-1>}\n\n'
    "Usa 'reconocido': true SOLO si identificas con certeza razonable un edificio/monumento "
    "CONCRETO y REAL (no un tipo genérico de edificio) y conoces sus coordenadas "
    "aproximadas reales de memoria. Si el nombre es demasiado genérico para identificar un "
    "lugar concreto (p. ej. 'catedral', 'ayuntamiento', 'castillo' sin más contexto), si no "
    "conoces ese lugar en concreto, o si tienes cualquier duda razonable, usa "
    "'reconocido': false y deja el resto de campos en null -- NUNCA inventes coordenadas "
    "aproximadas ni 'la mejor suposición'; es preferible no responder a responder mal. "
    "'confianza' entre 0 y 1: cuánta certeza tienes de que el lugar identificado es "
    "exactamente el correcto y de que las coordenadas son razonablemente precisas -- usa "
    "valores altos (>0.8) solo para monumentos mundialmente conocidos y sin ambigüedad "
    "posible."
)


class LandmarkResolution:
    __slots__ = ("canonical_name", "lat", "lon", "confidence")

    def __init__(self, canonical_name: str, lat: float, lon: float, confidence: float):
        self.canonical_name = canonical_name
        self.lat = lat
        self.lon = lon
        self.confidence = confidence


def _valid_coordinates(lat: object, lon: object) -> tuple[float, float] | None:
    if not isinstance(lat, (int, float)) or isinstance(lat, bool):
        return None
    if not isinstance(lon, (int, float)) or isinstance(lon, bool):
        return None
    lat_f, lon_f = float(lat), float(lon)
    if not (-90.0 <= lat_f <= 90.0) or not (-180.0 <= lon_f <= 180.0):
        return None
    return lat_f, lon_f


async def resolve_landmark_coordinates(
    landmark_name: str, context_hint: str | None = None
) -> LandmarkResolution | None:
    """Punto de entrada del módulo. Devuelve None si el proveedor de IA
    activo no está configurado, no reconoce el lugar con certeza, la
    confianza queda por debajo de MIN_CONFIDENCE_FOR_LANDMARK_OVERRIDE, o
    la llamada falla por cualquier motivo -- nunca lanza excepciones,
    mismo criterio best-effort que el resto de módulos que hablan con un
    LLM en este proyecto."""
    if not settings.ai_key_configured or not landmark_name.strip():
        return None

    prompt = f"Nombre propuesto: {landmark_name.strip()}"
    if context_hint:
        prompt += f"\nDescripción general de la foto (contexto, en inglés): {context_hint.strip()[:300]}"

    try:
        parsed = await call_ai_json(
            _SYSTEM_PROMPT, prompt, max_tokens=200, temperature=0.0
        )
    except (AIHTTPError, AIRequestError) as exc:
        logger.warning("Resolución de edificio emblemático no disponible: %s", exc)
        return None

    if not isinstance(parsed, dict) or not parsed.get("reconocido"):
        return None

    canonical_name = parsed.get("nombre_canonico")
    if not isinstance(canonical_name, str) or not canonical_name.strip():
        return None

    coords = _valid_coordinates(parsed.get("lat"), parsed.get("lon"))
    if coords is None:
        return None
    lat, lon = coords

    confidence_raw = parsed.get("confianza")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) and not isinstance(confidence_raw, bool) else 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < MIN_CONFIDENCE_FOR_LANDMARK_OVERRIDE:
        return None

    return LandmarkResolution(canonical_name.strip(), lat, lon, confidence)
