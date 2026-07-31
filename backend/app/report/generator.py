"""
Módulo 5: ensambla el informe final y genera recomendaciones concretas
y accionables a partir del score y los atributos inferidos.
"""
from datetime import datetime, timezone

import unicodedata

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
from app.data.ine_reference import (
    AUTONOMOUS_COMMUNITY_PROVINCES,
    PROVINCE_POPULATION,
    PROVINCE_TO_CCAA,
    resolve_autonomous_community,
)
from app.nlp.ai_attribute_extraction import extract_demographics_with_ai, merge_findings
from app.nlp.demographic_extraction import DemographicFindings, extract_demographics
from app.nlp.travel_detection import detect_travel_permalinks
from app.progress import ProgressCallback, emit_progress
from app.scoring.k_anonymity import estimate_population_narrowing

# Umbrales para aceptar una estimación de RESIDENCIA HABITUAL a partir de
# geolocalización de imágenes (ver `_infer_home_region`). Antes de este
# cambio bastaba UNA sola foto con >=40% de confianza para dar por buena
# una provincia/comunidad -- demasiado alegre para un dato tan sensible.
# Ahora se exige CONSENSO entre varias fotos de la MISMA comunidad
# autónoma:
#   - Al menos HIGH_CONFIDENCE_MIN_PHOTOS fotos con confianza > HIGH_CONFIDENCE, o
#   - Al menos MODERATE_CONFIDENCE_MIN_PHOTOS fotos con confianza > MODERATE_CONFIDENCE.
# Los valores son una primera propuesta razonable, no una cifra "correcta"
# derivada de algún estudio -- son ajustables si en la práctica resultan
# demasiado (o poco) exigentes. En particular MODERATE_CONFIDENCE_MIN_PHOTOS
# ("muchas fotos") es la más discutible de las cuatro: 4 es un punto de
# partida, no un número definitivo.
HIGH_CONFIDENCE = 0.8
HIGH_CONFIDENCE_MIN_PHOTOS = 2
MODERATE_CONFIDENCE = 0.6
MODERATE_CONFIDENCE_MIN_PHOTOS = 4


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _resolve_region(raw_region: str) -> tuple[str, str] | None:
    """Traduce una región cruda de geolocation.py (columna "region" de
    OSV-5M/Nominatim, en español o inglés) a `(nivel, clave)`, donde nivel
    es "provincia" o "comunidad_autonoma":

    1. Si coincide literalmente con una provincia del INE (algunos
       datasets/tests dan directamente el nombre de provincia, p.ej.
       "Sevilla"), se usa esa provincia -- es el dato más específico posible.
    2. Si resuelve a una comunidad autónoma de UNA sola provincia (Asturias,
       Madrid, Murcia, Navarra, Cantabria, La Rioja, Baleares, Ceuta,
       Melilla), tampoco hay ambigüedad: se usa esa provincia.
    3. Si resuelve a una comunidad autónoma con VARIAS provincias (p.ej.
       Canarias), no se puede elegir una sin inventar información: nivel
       "comunidad_autonoma".
    4. Si no se reconoce nada (país distinto de España, o "desconocido"),
       devuelve None.
    """
    normalized = _strip_accents(raw_region).strip().lower()

    if normalized in PROVINCE_POPULATION:
        return "provincia", normalized

    ccaa = resolve_autonomous_community(raw_region)
    if ccaa is not None:
        provinces = AUTONOMOUS_COMMUNITY_PROVINCES[ccaa]
        if len(provinces) == 1:
            return "provincia", provinces[0]
        return "comunidad_autonoma", ccaa

    return None


class _HomeRegionCandidate:
    """Resultado de `_infer_home_region`: la comunidad autónoma (o
    provincia, si todas las fotos que cuentan señalan la misma) con
    consenso suficiente, y qué fotos exactamente respaldan esa conclusión
    (para el campo `evidence` del informe)."""

    __slots__ = ("level", "key", "permalinks")

    def __init__(self, level: str, key: str, permalinks: list[str]) -> None:
        self.level = level
        self.key = key
        self.permalinks = permalinks


def _infer_home_region(
    # `object` en vez del tipo real de geolocation.py a propósito: ese
    # módulo tiene dependencias opcionales (torch/faiss) que NUNCA deben
    # importarse incondicionalmente a nivel de módulo (ver su propio
    # docstring) -- aquí solo se necesita "algo con .province/.confidence".
    results: list[tuple[str, object]],
    travel_permalinks: set[str],
) -> _HomeRegionCandidate | None:
    """Decide si hay consenso suficiente entre varias fotos para dar por
    buena una comunidad autónoma (o provincia) como residencia habitual.

    Pasos:
    1. Se descartan las fotos marcadas como "de viaje" (`travel_permalinks`,
       ver travel_detection.py y el campo "fotos_de_viaje" de la IA): una
       foto de vacaciones en Roma no debe contar como señal de dónde vive
       la persona, por muy alta que sea su confianza de geolocalización.
    2. Se descartan las fotos cuya región no se reconoce (`_resolve_region`
       devuelve None).
    3. Se agrupan las fotos restantes por COMUNIDAD AUTÓNOMA (usando
       PROVINCE_TO_CCAA cuando una foto resolvió a provincia concreta), no
       por provincia exacta -- así varias fotos de distintas provincias de
       una misma comunidad (o mezclando nombre de provincia con nombre de
       comunidad) siguen sumando señal de la misma zona.
    4. Cada grupo se acepta solo si cumple HIGH_CONFIDENCE_MIN_PHOTOS a
       HIGH_CONFIDENCE, o MODERATE_CONFIDENCE_MIN_PHOTOS a
       MODERATE_CONFIDENCE (ver constantes arriba).
    5. Si varios grupos cumplen (raro), gana el que tenga más fotos de
       señal válida; en empate, el de mayor confianza media.
    6. Dentro del grupo ganador, si TODAS las fotos que cuentan señalan la
       misma provincia concreta, se usa esa provincia (más específico);
       si no, se usa el nivel de comunidad autónoma.
    """
    resolved: list[tuple[str, str, str, float]] = []  # permalink, nivel, clave, confianza
    for permalink, estimate in results:
        if permalink in travel_permalinks:
            continue
        region = _resolve_region(estimate.province)
        if region is None:
            continue
        level, key = region
        resolved.append((permalink, level, key, estimate.confidence))

    if not resolved:
        return None

    groups: dict[str, list[tuple[str, str, str, float]]] = {}
    for item in resolved:
        _, level, key, _ = item
        ccaa = key if level == "comunidad_autonoma" else PROVINCE_TO_CCAA.get(key, key)
        groups.setdefault(ccaa, []).append(item)

    candidates: list[tuple[list[tuple[str, str, str, float]], float]] = []
    for items in groups.values():
        high = [i for i in items if i[3] > HIGH_CONFIDENCE]
        moderate = [i for i in items if i[3] > MODERATE_CONFIDENCE]  # ya incluye a `high` (superset)
        qualifies = len(high) >= HIGH_CONFIDENCE_MIN_PHOTOS or len(moderate) >= MODERATE_CONFIDENCE_MIN_PHOTOS
        if not qualifies:
            continue
        avg_confidence = sum(i[3] for i in moderate) / len(moderate)
        candidates.append((moderate, avg_confidence))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: (len(pair[0]), pair[1]), reverse=True)
    winning_items, _ = candidates[0]

    provinces_in_group = {key for _, level, key, _ in winning_items if level == "provincia"}
    if len(provinces_in_group) == 1:
        level, key = "provincia", next(iter(provinces_in_group))
    else:
        # Mezcla de provincias distintas (o alguna a nivel de comunidad
        # directamente): no se puede ser más específico que la comunidad.
        _, _, any_key, _ = winning_items[0]
        ccaa_key = any_key if any_key in AUTONOMOUS_COMMUNITY_PROVINCES else PROVINCE_TO_CCAA[any_key]
        level, key = "comunidad_autonoma", ccaa_key

    permalinks = [permalink for permalink, *_ in winning_items]
    return _HomeRegionCandidate(level=level, key=key, permalinks=permalinks)


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
    # Confirmación VISIBLE (no solo en logs) de que la biografía del perfil
    # ha llegado hasta aquí y se va a analizar -- aparece como paso más en
    # la pantalla de progreso en vivo (ver /api/analyze/{platform}/stream),
    # así que se puede comprobar a simple vista en cada análisis real, sin
    # tener que confiar a ciegas en que el dato se propagó correctamente
    # desde InstagramClient.fetch_profile().
    if bio:
        await emit_progress(progress_callback, f"Biografía del perfil recibida ({len(bio)} caracteres). Analizando...")
    else:
        await emit_progress(
            progress_callback, "No se ha recibido ninguna biografía pública de este perfil."
        )

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
    travel_permalinks = detect_travel_permalinks(posts)

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
        # La IA también señala fotos de viaje/vacaciones en el mismo pase
        # (campo "fotos_de_viaje" del prompt) -- se UNE con lo que ya haya
        # detectado la regex de travel_detection.py, nunca lo sustituye.
        travel_permalinks |= ai_findings.travel_permalinks

    # Geolocalización por imagen: solo se usa como ubicación PARA EL CÁLCULO
    # DE POBLACIÓN si el texto no dio ya una provincia/municipio explícita
    # (la autodeclaración en texto es más fiable), y solo si hay CONSENSO
    # entre varias fotos de la misma comunidad autónoma -- ver
    # `_infer_home_region` y las constantes HIGH_CONFIDENCE*/
    # MODERATE_CONFIDENCE* al principio de este módulo. Las fotos marcadas
    # como "de viaje" (travel_permalinks) nunca cuentan para esto. Pero
    # TODAS las estimaciones por imagen (de viaje, baja confianza, etc.) se
    # guardan igualmente en `image_location_points`, para que el frontend
    # pueda mostrar cada foto analizada con su confianza real -- no solo
    # las que "cuentan" para inferir dónde vive la persona.
    # Módulo opcional/best-effort: si el índice FAISS no está construido
    # (ver app/vision/geolocation.py), `geolocation_available` queda a
    # False y el resto del informe sigue generándose con normalidad.
    image_location_points: list[ImageLocationPoint] = []
    geolocation_available = False
    if platform == "instagram":
        from app.vision.geolocation import estimate_locations_for_posts

        geo_outcome = await estimate_locations_for_posts(posts, progress_callback=progress_callback)
        geolocation_available = geo_outcome.index_available
        image_location_points = [
            ImageLocationPoint(
                permalink=permalink,
                province=estimate.province,
                confidence=estimate.confidence,
                lat=estimate.lat,
                lon=estimate.lon,
            )
            for permalink, estimate in geo_outcome.results
        ]

        has_location = (
            demographic_findings.provincia is not None
            or demographic_findings.municipio is not None
            or demographic_findings.comunidad_autonoma is not None
        )
        if not has_location:
            home_candidate = _infer_home_region(geo_outcome.results, travel_permalinks)
            if home_candidate is not None:
                if home_candidate.level == "provincia":
                    demographic_findings.provincia = home_candidate.key
                    demographic_findings.evidence.setdefault("provincia", []).extend(home_candidate.permalinks)
                    demographic_findings.source["provincia"] = "imagen"
                else:
                    demographic_findings.comunidad_autonoma = home_candidate.key
                    demographic_findings.evidence.setdefault("comunidad_autonoma", []).extend(
                        home_candidate.permalinks
                    )
                    demographic_findings.source["comunidad_autonoma"] = "imagen"

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
            proportion=step.proportion,
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
        geolocation_available=geolocation_available,
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
