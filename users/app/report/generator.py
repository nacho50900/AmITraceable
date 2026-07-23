"""
Módulo 5: ensambla el informe final y genera recomendaciones concretas
y accionables a partir del score y los atributos inferidos.
"""
from datetime import datetime, timezone

from app.config import settings
from app.models.schemas import (
    ExposureReport,
    ImageLocationPoint,
    InferredAttribute,
    PopulationEstimate,
    PrivacyScore,
    SocialPost,
    WritingFingerprint,
)
from app.nlp.ai_attribute_extraction import extract_demographics_with_ai, merge_findings
from app.nlp.demographic_extraction import extract_demographics
from app.progress import ProgressCallback, emit_progress
from app.scoring.k_anonymity import estimate_population_narrowing


async def generate_report(
    platform: str,
    username: str,
    posts: list[SocialPost],
    fingerprint: WritingFingerprint,
    inferred_attributes: list[InferredAttribute],
    score: PrivacyScore,
    progress_callback: ProgressCallback | None = None,
    bio: str | None = None,
    full_name: str | None = None,
) -> ExposureReport:
    # La biografía se trata como una publicación más de cara a las regex de
    # autodeclaración (mismo criterio que un post/comentario, solo que sin
    # permalink real -- se usa "bio" como identificador de evidencia). Así
    # "estudiante de enfermería" en la bio se detecta con el mismo código
    # que si estuviera en un post, sin duplicar lógica de detección.
    posts_for_demographics = posts
    if bio:
        bio_pseudo_post = SocialPost(
            id="bio",
            platform=platform,
            type="bio",
            group="sin_etiqueta",
            tags=[],
            text=bio,
            created_utc=datetime.now(tz=timezone.utc),
            score=0,
            permalink="bio",
        )
        posts_for_demographics = [bio_pseudo_post, *posts]

    demographic_findings = extract_demographics(posts_for_demographics)

    # Extracción de autodeclaraciones con IA: complementa las regex (que
    # solo cubren un vocabulario fijo) leyendo el texto -- y también el
    # nombre público de la cuenta, que sirve como señal débil de sexo -- de
    # forma más flexible. Se ejecuta automáticamente en cada análisis, sin
    # ningún botón. Ver docstring de app/nlp/ai_attribute_extraction.py
    # para el razonamiento RGPD. Módulo opcional/best-effort: sin
    # MISTRAL_API_KEY, o si la llamada falla, esto no aporta nada y el
    # informe se sigue generando solo con lo detectado por regex.
    if settings.mistral_api_key:
        await emit_progress(progress_callback, "Buscando autodeclaraciones con IA...")
        ai_findings = await extract_demographics_with_ai(
            posts, username=username, full_name=full_name, bio=bio
        )
        demographic_findings = merge_findings(demographic_findings, ai_findings)

    # Geolocalización por imagen: solo se usa como ubicación PARA EL CÁLCULO
    # DE POBLACIÓN si el texto no dio ya una provincia/municipio explícita
    # (la autodeclaración en texto es más fiable). Pero TODAS las
    # estimaciones por imagen (no solo la mejor) se guardan igualmente en
    # `image_location_points` para pintar el mapa completo en el frontend.
    # Módulo opcional/best-effort: si el índice FAISS no está construido
    # (ver app/vision/geolocation.py), esto simplemente no aporta nada y el
    # resto del informe sigue generándose con normalidad.
    image_location_points: list[ImageLocationPoint] = []
    if platform == "instagram":
        from app.vision.geolocation import estimate_locations_for_posts

        image_estimates = await estimate_locations_for_posts(posts, progress_callback=progress_callback)
        image_location_points = [
            ImageLocationPoint(
                permalink=permalink,
                province=estimate.province,
                confidence=estimate.confidence,
                lat=estimate.lat,
                lon=estimate.lon,
            )
            for permalink, estimate in image_estimates
        ]

        if image_estimates and demographic_findings.provincia is None and demographic_findings.municipio is None:
            # Nos quedamos con la estimación de mayor confianza entre todas las imágenes
            best_permalink, best_estimate = max(image_estimates, key=lambda pair: pair[1].confidence)
            demographic_findings.provincia = best_estimate.province.lower()
            demographic_findings.evidence.setdefault("provincia", []).append(best_permalink)
            demographic_findings.source["provincia"] = "imagen"

    await emit_progress(progress_callback, "Generando el informe final...")

    narrowing_steps = estimate_population_narrowing(demographic_findings)
    population_narrowing = [
        PopulationEstimate(
            attribute_label=step.attribute_label,
            category=step.category,
            remaining_population=step.remaining_population,
            risk_level=step.risk_level,
            evidence=step.evidence,
            source=step.source,
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
        image_location_points=image_location_points,
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
