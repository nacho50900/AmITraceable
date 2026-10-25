"""
Motor de scoring de exposición de privacidad.

Pondera varias señales en una puntuación 0-100 (mayor = más expuesto).
Las ponderaciones son un punto de partida razonado para el MVP; quedan
documentadas como hipótesis a validar empíricamente en la memoria del TFG
(ver sección "Plan de evaluación").

Nota de alcance Reddit-only: el componente "identity_consistency_risk"
(consistencia de identidad ENTRE plataformas) no puede calcularse en esta
versión porque solo hay una plataforma. Se deja como placeholder a 0 con
una nota explicativa, y queda para trabajo futuro cuando se añada
correlación multiplataforma (módulo 3 completo).

El score DEFINITIVO se calcula al final de `generate_report` (y de nuevo en
`POST /analyze/recalculate`), con TODO lo que el análisis ha reunido:
atributos de texto, IA y fotos, geolocalización de las fotos, población
restante tras el estrechamiento y cuentas relacionadas (ver ADR-56). Antes se
calculaba al principio del pipeline, solo con los tres atributos que salen de
hashtags y comunidades, y todo lo detectado después no contaba. Los tres
riesgos usan la misma forma de combinar fuentes independientes ("noisy-OR"):
`1 - (1 - a)(1 - b)...`; cada fuente suma, ninguna es imprescindible y el
resultado nunca pasa de 100.
"""
import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from app.data.ine_reference import TOTAL_POPULATION_ES
from app.models.schemas import (
    ImageLocationPoint,
    InferredAttribute,
    PopulationEstimate,
    PrivacyScore,
    SocialPost,
    WritingFingerprint,
)

if TYPE_CHECKING:
    from app.models.schemas import ExposureReport

# Pesos del score global (deben sumar 1.0)
_WEIGHTS = {
    "geolocation": 0.35,
    "identity_consistency": 0.0,  # no aplica en versión Reddit-only
    "inferable_data": 0.45,
    "deanonymization_ease": 0.20,
}

# Nº de posts/comentarios a partir del cual se confía plenamente en la
# concentración horaria y la riqueza de vocabulario como señales de
# fingerprinting de estilo (ver docstring de _score_deanonymization_ease).
# Deliberadamente más bajo que el umbral de saturación de `volume_factor`
# (200): estas dos estadísticas concretas dejan de ser puro ruido bastante
# antes de que haya "suficiente contenido" en términos absolutos.
_MIN_POSTS_FOR_RELIABLE_STYLE = 20

# --- Geolocalización --------------------------------------------------------

# Cuánto pesa cada nivel de una ubicación autodeclarada: saber el municipio
# expone más que saber la comunidad autónoma.
_LOCATION_LEVEL_PRECISION = {"municipio": 1.0, "provincia": 0.8, "comunidad_autonoma": 0.5}

# Categorías de atributo que cuentan como señal de ubicación a partir del texto.
_TEXT_LOCATION_CATEGORIES = ("ubicacion", "ubicacion_detallada")

# Escala de la señal de fotos: S = suma de las confianzas de las fotos que
# coinciden en la zona más repetida; riesgo = 1 - exp(-S / escala). Con 2.0:
# una foto muy segura (S=1) -> 39; tres fotos seguras (S=3) -> 78; seis -> 95.
_PHOTO_ZONE_SCALE = 2.0

# --- Datos personales inferibles -------------------------------------------

# Cuántos "datos de peso 1 y confianza 1" hacen falta para saturar la barra.
_INFERABLE_SATURATION = 8.0

# Peso de cada categoría (lo que revela sobre la persona). Las que no están
# aquí valen `_DEFAULT_CATEGORY_WEIGHT`.
_CATEGORY_WEIGHTS: dict[str, float] = {
    # Identifican o localizan directamente.
    "ubicacion": 1.0, "ubicacion_detallada": 1.0, "ocupacion": 1.0, "empresa": 1.0,
    "universidad": 1.0, "matricula": 1.0, "documento_identidad": 1.0,
    "contacto_publicado": 1.0, "dato_financiero": 1.0, "cuenta_externa": 1.0,
    # Rasgos físicos añadidos a mano en el recálculo.
    "color_ojos": 0.7, "color_pelo": 0.7, "color_piel": 0.7,
    # Revelan rutina, entorno o circunstancias personales.
    "rutina": 0.7, "lengua_materna": 0.7, "edificio_emblematico": 0.7, "texto_visible": 0.7,
    "vehiculo": 0.7, "vivienda": 0.7, "relacion": 0.7, "relacion_sentimental": 0.7,
    "salud": 0.7, "ideologia": 0.7, "ingresos": 0.7, "menor": 0.7,
    "viaje_futuro": 0.7, "identificador_tecnico": 0.7,
    # Señales blandas: hacen más reconocible, pero poco por sí solas.
    "aficion": 0.3, "mascota": 0.3, "idiomas": 0.3, "habito": 0.3, "equipo": 0.3,
}
_DEFAULT_CATEGORY_WEIGHT = 0.5

# --- Facilidad de deanonimización -------------------------------------------

# El estilo de escritura y la actividad por sí solos no identifican a nadie:
# su aportación se limita a este máximo (0-1).
_STYLE_MAX_CONTRIBUTION = 0.5

# Identificadores publicados que permiten localizar a la persona directamente.
_DIRECT_IDENTIFIER_CATEGORIES = ("documento_identidad", "contacto_publicado", "dato_financiero", "cuenta_externa")
_MIN_IDENTIFIER_CONFIDENCE = 0.5
_IDENTIFIER_CONTRIBUTION = {0: 0.0, 1: 0.7}  # 2 o más: _IDENTIFIER_CONTRIBUTION_MANY
_IDENTIFIER_CONTRIBUTION_MANY = 0.9

# Cada cuenta propia encontrada con el mismo username en otros sitios permite
# cruzar datos; se limita porque un username común puede dar coincidencias falsas.
_ACCOUNT_CONTRIBUTION_EACH = 0.3
_ACCOUNT_CONTRIBUTION_MAX = 0.7


def compute_exposure_score(
    *,
    n_posts: int,
    fingerprint: WritingFingerprint,
    inferred_attributes: Sequence[InferredAttribute],
    image_location_points: Iterable[ImageLocationPoint] = (),
    population_narrowing: Iterable[PopulationEstimate] = (),
    remaining_population: int | None = None,
    related_accounts_found: int = 0,
) -> PrivacyScore:
    """Score completo a partir de las primitivas que comparten el pipeline y
    el recálculo. Los parámetros opcionales valen "no hay esa señal" (0), que
    es lo que ocurre si no hay fotos, no se estrechó la población o la
    comprobación de cuentas está desactivada."""
    geolocation_risk = _score_geolocation(inferred_attributes, image_location_points, population_narrowing)
    inferable_data_risk = _score_inferable_data(inferred_attributes)
    deanonymization_ease = _deanonymization_ease(
        n_posts,
        fingerprint,
        remaining_population=remaining_population,
        n_identifiers=_count_direct_identifiers(inferred_attributes),
        related_accounts_found=related_accounts_found,
    )
    identity_consistency_risk = 0.0  # ver nota de alcance en el docstring

    overall = (
        geolocation_risk * _WEIGHTS["geolocation"]
        + identity_consistency_risk * _WEIGHTS["identity_consistency"]
        + inferable_data_risk * _WEIGHTS["inferable_data"]
        + deanonymization_ease * _WEIGHTS["deanonymization_ease"]
    )

    return PrivacyScore(
        overall_score=round(overall, 1),
        geolocation_risk=round(geolocation_risk, 1),
        identity_consistency_risk=round(identity_consistency_risk, 1),
        inferable_data_risk=round(inferable_data_risk, 1),
        deanonymization_ease=round(deanonymization_ease, 1),
        breakdown_explanation={
            "geolocation": "Combina la ubicación que se deduce del texto (autodeclaraciones, hashtags, "
                           "comunidades) con la de las fotos: cuantas más fotos coinciden en una misma "
                           "zona y con más confianza, mayor es el riesgo.",
            "identity_consistency": "No evaluado en esta versión (solo se analiza Reddit; "
                                     "requiere correlación multiplataforma, ver trabajo futuro).",
            "inferable_data": "Basado en todos los datos personales inferidos (ubicación, ocupación, rutina, "
                               "identificadores, aficiones...), con más peso los que identifican o localizan "
                               "y teniendo en cuenta su confianza.",
            "deanonymization_ease": "Combina el estilo de escritura y la actividad (volumen de contenido, "
                                     "vocabulario, hora habitual; con poco contenido pesan menos) con lo que "
                                     "reduce la población compatible, los identificadores publicados y las "
                                     "cuentas encontradas con el mismo usuario.",
        },
    )


def compute_score(
    posts: list[SocialPost],
    fingerprint: WritingFingerprint,
    inferred_attributes: list[InferredAttribute],
    **extra_signals,
) -> PrivacyScore:
    """Score a partir de los posts. Sin `extra_signals` solo usa texto y
    atributos (el score PROVISIONAL del principio del pipeline); con ellos
    (`image_location_points`, `population_narrowing`, `remaining_population`,
    `related_accounts_found`) es el definitivo -- ver `compute_exposure_score`."""
    return compute_exposure_score(
        n_posts=len(posts),
        fingerprint=fingerprint,
        inferred_attributes=inferred_attributes,
        **extra_signals,
    )


def compute_score_from_report(report: "ExposureReport") -> PrivacyScore:
    """Recalcula el score de un informe ya generado con exactamente la misma
    fórmula que el pipeline (lo usa `POST /analyze/recalculate` tras añadir
    rasgos manuales, para que ambos caminos no puedan divergir)."""
    return compute_exposure_score(
        n_posts=report.n_posts_analyzed,
        fingerprint=report.fingerprint,
        inferred_attributes=report.inferred_attributes,
        image_location_points=report.image_location_points,
        population_narrowing=report.population_narrowing,
        remaining_population=report.remaining_population_all_traits,
        related_accounts_found=len(report.related_accounts.matches) if report.related_accounts else 0,
    )


def _noisy_or(*probabilities: float) -> float:
    """Combina fuentes independientes de evidencia (cada una en 0-1): suma sin pasar de 1."""
    remaining = 1.0
    for p in probabilities:
        remaining *= 1.0 - max(0.0, min(p, 1.0))
    return 1.0 - remaining


def _text_location_confidence(
    attributes: Iterable[InferredAttribute], population_narrowing: Iterable[PopulationEstimate]
) -> float:
    """Mayor confianza de ubicación sacada del texto: atributos (hashtags,
    comunidades, ubicación detallada) y ubicación autodeclarada del
    estrechamiento de población (más fiable cuanto más fino el nivel)."""
    best = max((a.confidence for a in attributes if a.category in _TEXT_LOCATION_CATEGORIES), default=0.0)
    for step in population_narrowing:
        if step.category == "ubicacion" and step.confidence is not None:
            precision = _LOCATION_LEVEL_PRECISION.get(step.location_level or "", 0.8)
            best = max(best, step.confidence * precision)
    return min(best, 1.0)


def _photo_location_risk(points: Iterable[ImageLocationPoint]) -> float:
    """Riesgo (0-1) que aportan las fotos: se agrupan por provincia y cuenta la
    zona que más pesa, sumando las confianzas de sus fotos. Muchas fotos
    coherentes en una zona y con mucha confianza -> alto; una foto suelta o
    fotos repartidas por provincias distintas -> bajo. No cuentan las
    estimaciones no representativas (demasiado dispersas para fiarse) ni la
    foto de perfil (no es necesariamente representativa de dónde vive, igual
    que en el consenso de residencia de report/generator.py)."""
    per_zone: dict[str, float] = {}
    for point in points:
        if not point.representative or point.is_profile_picture:
            continue
        per_zone[point.province] = per_zone.get(point.province, 0.0) + point.confidence
    top = max(per_zone.values(), default=0.0)
    return 1.0 - math.exp(-top / _PHOTO_ZONE_SCALE)


def _score_geolocation(
    attributes: Iterable[InferredAttribute],
    image_location_points: Iterable[ImageLocationPoint] = (),
    population_narrowing: Iterable[PopulationEstimate] = (),
) -> float:
    attributes = list(attributes)
    text = _text_location_confidence(attributes, population_narrowing)
    photos = _photo_location_risk(image_location_points)
    return _noisy_or(text, photos) * 100.0


def _score_inferable_data(attributes: Iterable[InferredAttribute]) -> float:
    # Un mismo dato (categoría + valor) cuenta una vez, con su mayor confianza:
    # el regex y la IA pueden detectar lo mismo.
    best: dict[tuple[str, str], float] = {}
    for a in attributes:
        key = (a.category, a.value)
        best[key] = max(best.get(key, 0.0), a.confidence)
    weighted = sum(_CATEGORY_WEIGHTS.get(category, _DEFAULT_CATEGORY_WEIGHT) * conf for (category, _), conf in best.items())
    return min((weighted / _INFERABLE_SATURATION) * 100, 100.0)


def _count_direct_identifiers(attributes: Iterable[InferredAttribute]) -> int:
    return len(
        {
            (a.category, a.value)
            for a in attributes
            if a.category in _DIRECT_IDENTIFIER_CATEGORIES and a.confidence >= _MIN_IDENTIFIER_CONFIDENCE
        }
    )


def _population_uniqueness(remaining_population: int | None) -> float:
    """0-1: cuánto ha reducido la población compatible (en escala logarítmica,
    como los "bits" de información identificativa). 1 = una sola persona;
    0 = sin estrechamiento o no estimable."""
    if remaining_population is None or remaining_population >= TOTAL_POPULATION_ES:
        return 0.0
    if remaining_population <= 1:
        return 1.0
    return math.log(TOTAL_POPULATION_ES / remaining_population) / math.log(TOTAL_POPULATION_ES)


def _style_signal(n_posts: int, fingerprint: WritingFingerprint) -> float:
    """Señal 0-1 de reconocimiento por estilo y actividad (la fórmula original)."""
    if n_posts <= 0:
        return 0.0

    # Más contenido analizable = más fácil de re-identificar por estilo
    volume_factor = min(n_posts / 200, 1.0)  # saturado a partir de 200 posts/comentarios

    # `distinctiveness_factor` y `concentration` (más abajo) son estadísticas
    # sobre el CONJUNTO de posts, y con pocos posts son poco fiables -- no
    # inexistentes, sino RUIDOSAS: con 1 solo post, por ejemplo, el 100% de
    # "tus posts" cae SIEMPRE en una única hora (`concentration` = 1.0 de
    # forma matemáticamente segura, sin que eso indique ninguna rutina real)
    # y la riqueza de vocabulario de un texto corto tiende a salir inflada
    # simplemente porque hay poca oportunidad de repetir palabras. Se escala
    # su aportación por la cantidad de contenido disponible, saturando a
    # partir de `_MIN_POSTS_FOR_RELIABLE_STYLE` -- por debajo de ese umbral,
    # cuanto menos contenido haya, menos se confía en estas dos señales.
    sample_confidence = min(n_posts / _MIN_POSTS_FOR_RELIABLE_STYLE, 1.0)

    # Vocabulario muy distintivo (ni demasiado pobre ni genérico) facilita el
    # fingerprinting de estilo de escritura
    richness = fingerprint.vocabulary_richness
    distinctiveness_factor = 1.0 - abs(richness - 0.45) / 0.45 if richness else 0.0
    distinctiveness_factor = max(0.0, min(distinctiveness_factor, 1.0)) * sample_confidence

    # Patrón horario muy concentrado (rutina marcada) facilita inferir zona
    # horaria y hábitos
    hour_values = list(fingerprint.avg_posts_per_hour.values())
    concentration = (max(hour_values) if hour_values else 0.0) * sample_confidence  # proporción en la hora pico

    return min(volume_factor * 0.4 + distinctiveness_factor * 0.3 + concentration * 0.3, 1.0)


def _deanonymization_ease(
    n_posts: int,
    fingerprint: WritingFingerprint,
    *,
    remaining_population: int | None = None,
    n_identifiers: int = 0,
    related_accounts_found: int = 0,
) -> float:
    if n_posts <= 0:
        return 0.0
    style = _style_signal(n_posts, fingerprint) * _STYLE_MAX_CONTRIBUTION
    identifiers = _IDENTIFIER_CONTRIBUTION.get(n_identifiers, _IDENTIFIER_CONTRIBUTION_MANY)
    accounts = min(related_accounts_found * _ACCOUNT_CONTRIBUTION_EACH, _ACCOUNT_CONTRIBUTION_MAX)
    return _noisy_or(style, _population_uniqueness(remaining_population), identifiers, accounts) * 100.0


def _score_deanonymization_ease(
    posts: list[SocialPost],
    fingerprint: WritingFingerprint,
    *,
    remaining_population: int | None = None,
    n_identifiers: int = 0,
    related_accounts_found: int = 0,
) -> float:
    return _deanonymization_ease(
        len(posts),
        fingerprint,
        remaining_population=remaining_population,
        n_identifiers=n_identifiers,
        related_accounts_found=related_accounts_found,
    )
