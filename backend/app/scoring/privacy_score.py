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
"""
from datetime import datetime, timezone

from app.models.schemas import InferredAttribute, PrivacyScore, SocialPost, WritingFingerprint

# Pesos del score global (deben sumar 1.0)
_WEIGHTS = {
    "geolocation": 0.35,
    "identity_consistency": 0.0,  # no aplica en versión Reddit-only
    "inferable_data": 0.45,
    "deanonymization_ease": 0.20,
}


def compute_score(
    posts: list[SocialPost],
    fingerprint: WritingFingerprint,
    inferred_attributes: list[InferredAttribute],
) -> PrivacyScore:
    geolocation_risk = _score_geolocation(inferred_attributes)
    inferable_data_risk = _score_inferable_data(inferred_attributes)
    deanonymization_ease = _score_deanonymization_ease(posts, fingerprint)
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
            "geolocation": "Basado en menciones y subreddits geolocalizables detectados.",
            "identity_consistency": "No evaluado en esta versión (solo se analiza Reddit; "
                                     "requiere correlación multiplataforma, ver trabajo futuro).",
            "inferable_data": "Basado en número y confianza de atributos personales inferidos "
                               "(ubicación, ocupación, rutina).",
            "deanonymization_ease": "Basado en consistencia temporal de actividad, antigüedad "
                                     "de la cuenta y volumen de contenido analizable.",
        },
    )


def _score_geolocation(attributes: list[InferredAttribute]) -> float:
    geo_attrs = [a for a in attributes if a.category == "ubicacion"]
    if not geo_attrs:
        return 0.0
    max_conf = max(a.confidence for a in geo_attrs)
    return min(max_conf * 100, 100.0)


def _score_inferable_data(attributes: list[InferredAttribute]) -> float:
    if not attributes:
        return 0.0
    # Cuantos más atributos distintos con alta confianza, más expuesto
    weighted = sum(a.confidence for a in attributes)
    # Normalizamos asumiendo que >=6 atributos de confianza media-alta = saturación (100)
    return min((weighted / 6) * 100, 100.0)


def _score_deanonymization_ease(posts: list[SocialPost], fingerprint: WritingFingerprint) -> float:
    if not posts:
        return 0.0

    # Más contenido analizable = más fácil de re-identificar por estilo
    volume_factor = min(len(posts) / 200, 1.0)  # saturado a partir de 200 posts/comentarios

    # Vocabulario muy distintivo (ni demasiado pobre ni genérico) facilita el
    # fingerprinting de estilo de escritura
    richness = fingerprint.vocabulary_richness
    distinctiveness_factor = 1.0 - abs(richness - 0.45) / 0.45 if richness else 0.0
    distinctiveness_factor = max(0.0, min(distinctiveness_factor, 1.0))

    # Patrón horario muy concentrado (rutina marcada) facilita inferir zona
    # horaria y hábitos
    hour_values = list(fingerprint.avg_posts_per_hour.values())
    concentration = max(hour_values) if hour_values else 0.0  # proporción en la hora pico

    score = (volume_factor * 0.4 + distinctiveness_factor * 0.3 + concentration * 0.3) * 100
    return min(score, 100.0)
