"""
Módulo 5: ensambla el informe final y genera recomendaciones concretas
y accionables a partir del score y los atributos inferidos.
"""
from datetime import datetime, timezone

from app.models.schemas import (
    ExposureReport,
    InferredAttribute,
    PopulationEstimate,
    PrivacyScore,
    SocialPost,
    WritingFingerprint,
)
from app.nlp.demographic_extraction import extract_demographics
from app.scoring.k_anonymity import estimate_population_narrowing


def generate_report(
    platform: str,
    username: str,
    posts: list[SocialPost],
    fingerprint: WritingFingerprint,
    inferred_attributes: list[InferredAttribute],
    score: PrivacyScore,
) -> ExposureReport:
    demographic_findings = extract_demographics(posts)
    narrowing_steps = estimate_population_narrowing(demographic_findings)
    population_narrowing = [
        PopulationEstimate(
            attribute_label=step.attribute_label,
            category=step.category,
            remaining_population=step.remaining_population,
            risk_level=step.risk_level,
            evidence=step.evidence,
            note=step.note,
        )
        for step in narrowing_steps
    ]

    return ExposureReport(
        platform=platform,
        username=username,
        generated_at=datetime.now(tz=timezone.utc),
        n_posts_analyzed=len(posts),
        fingerprint=fingerprint,
        inferred_attributes=inferred_attributes,
        privacy_score=score,
        recommendations=_build_recommendations(fingerprint, inferred_attributes, score),
        population_narrowing=population_narrowing,
    )


def _build_recommendations(
    fingerprint: WritingFingerprint,
    attributes: list[InferredAttribute],
    score: PrivacyScore,
) -> list[str]:
    recs: list[str] = []

    if score.geolocation_risk > 30:
        recs.append(
            "Evita participar en comunidades o usar etiquetas muy específicas de tu "
            "ciudad/región con tu cuenta principal, o usa una cuenta separada sin "
            "vincular para ello."
        )

    if score.inferable_data_risk > 40:
        recs.append(
            "Revisa tu historial de publicaciones: hay varios datos personales "
            "(ubicación, ocupación o rutina) que se pueden inferir combinando varias "
            "publicaciones aparentemente inocuas por separado."
        )

    if score.deanonymization_ease > 50:
        recs.append(
            "Tu volumen de actividad y patrón horario son lo bastante consistentes "
            "como para servir de 'huella' de estilo. Considera variar tus horarios de "
            "publicación o reducir la frecuencia en cuentas que quieras mantener anónimas."
        )

    hour_values = fingerprint.avg_posts_per_hour
    peak_hour = max(hour_values, key=hour_values.get) if hour_values else None
    if peak_hour is not None and hour_values[peak_hour] > 0.25:
        recs.append(
            f"Más del 25% de tu actividad se concentra en torno a las {peak_hour}:00 (UTC), "
            "lo que puede ayudar a estimar tu zona horaria y rutina diaria."
        )

    if any(a.category == "ocupacion" for a in attributes):
        recs.append(
            "Tu participación en comunidades profesionales específicas puede revelar tu "
            "sector de trabajo. Si quieres mantener anonimato, evita detalles muy concretos "
            "de tu día a día laboral en esos foros."
        )

    if not recs:
        recs.append(
            "Tu nivel de exposición detectado en esta plataforma es bajo. Aun así, "
            "recuerda que este análisis es por plataforma: no tiene en cuenta lo que "
            "pueda ser inferible al combinar esta cuenta con tu actividad en otras redes."
        )

    return recs
