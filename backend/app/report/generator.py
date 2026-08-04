"""
Módulo 5: ensambla el informe final y genera recomendaciones concretas
y accionables a partir del score y los atributos inferidos.
"""
from datetime import datetime, timezone

import asyncio
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
    TOTAL_POPULATION_ES,
    resolve_autonomous_community,
)
from app.nlp.ai_attribute_extraction import extract_demographics_with_ai, merge_findings
from app.nlp.demographic_extraction import DemographicFindings, extract_demographics
from app.nlp.travel_detection import detect_travel_permalinks
from app.progress import ProgressCallback, emit_progress
from app.scoring.k_anonymity import estimate_population_narrowing, final_remaining_population

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


def _filter_and_resolve_estimates(
    results: list[tuple[str, object]],
    travel_permalinks: set[str],
) -> list[tuple[str, str, str, float]]:
    """Paso 1-2 de `_infer_home_region`: descarta fotos de viaje, no
    representativas, o cuya región no se reconoce; para el resto, devuelve
    (permalink, nivel, clave, confianza)."""
    resolved: list[tuple[str, str, str, float]] = []
    for permalink, estimate in results:
        if permalink in travel_permalinks:
            continue
        if not getattr(estimate, "representative", True):
            continue
        region = _resolve_region(estimate.province)
        if region is None:
            continue
        level, key = region
        resolved.append((permalink, level, key, estimate.confidence))
    return resolved


def _group_by_ccaa(
    resolved: list[tuple[str, str, str, float]],
) -> dict[str, list[tuple[str, str, str, float]]]:
    """Paso 3 de `_infer_home_region`: agrupa por comunidad autónoma
    (usando PROVINCE_TO_CCAA cuando una foto resolvió a provincia
    concreta), para que fotos de distintas provincias de una misma
    comunidad sumen señal juntas."""
    groups: dict[str, list[tuple[str, str, str, float]]] = {}
    for item in resolved:
        _, level, key, _ = item
        ccaa = key if level == "comunidad_autonoma" else PROVINCE_TO_CCAA.get(key, key)
        groups.setdefault(ccaa, []).append(item)
    return groups


def _qualifying_groups(
    groups: dict[str, list[tuple[str, str, str, float]]],
) -> list[tuple[list[tuple[str, str, str, float]], float]]:
    """Paso 4 de `_infer_home_region`: se queda solo con los grupos que
    cumplen el umbral de consenso (HIGH_CONFIDENCE*/MODERATE_CONFIDENCE*),
    junto a su confianza media."""
    candidates: list[tuple[list[tuple[str, str, str, float]], float]] = []
    for items in groups.values():
        high = [i for i in items if i[3] > HIGH_CONFIDENCE]
        moderate = [i for i in items if i[3] > MODERATE_CONFIDENCE]  # ya incluye a `high` (superset)
        qualifies = len(high) >= HIGH_CONFIDENCE_MIN_PHOTOS or len(moderate) >= MODERATE_CONFIDENCE_MIN_PHOTOS
        if not qualifies:
            continue
        avg_confidence = sum(i[3] for i in moderate) / len(moderate)
        candidates.append((moderate, avg_confidence))
    return candidates


def _pick_specificity(winning_items: list[tuple[str, str, str, float]]) -> tuple[str, str]:
    """Paso 6 de `_infer_home_region`: dentro del grupo ganador, si TODAS
    las fotos que cuentan señalan la misma provincia concreta, se usa esa
    provincia (más específico); si no, el nivel de comunidad autónoma."""
    provinces_in_group = {key for _, level, key, _ in winning_items if level == "provincia"}
    if len(provinces_in_group) == 1:
        return "provincia", next(iter(provinces_in_group))

    # Mezcla de provincias distintas (o alguna a nivel de comunidad
    # directamente): no se puede ser más específico que la comunidad.
    _, _, any_key, _ = winning_items[0]
    ccaa_key = any_key if any_key in AUTONOMOUS_COMMUNITY_PROVINCES else PROVINCE_TO_CCAA[any_key]
    return "comunidad_autonoma", ccaa_key


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
    Orquesta los pasos 1-6 (ver los docstrings de cada helper):

    1-2. `_filter_and_resolve_estimates`: descarta fotos de viaje, no
         representativas (dispersión geográfica excesiva entre sus vecinos
         más parecidos, ver `ImageLocationEstimate.representative` en
         app/vision/geolocation.py) o sin región reconocible -- SOLO para
         esta conclusión de residencia: siguen apareciendo en
         `image_location_points` con su confianza real.
    3.   `_group_by_ccaa`: agrupa las fotos restantes por comunidad autónoma.
    4.   `_qualifying_groups`: filtra los grupos con consenso suficiente.
    5.   Si varios grupos cumplen (raro), gana el que tenga más fotos de
         señal válida; en empate, el de mayor confianza media.
    6.   `_pick_specificity`: decide provincia vs. comunidad autónoma.
    """
    resolved = _filter_and_resolve_estimates(results, travel_permalinks)
    if not resolved:
        return None

    groups = _group_by_ccaa(resolved)
    candidates = _qualifying_groups(groups)
    if not candidates:
        return None

    candidates.sort(key=lambda pair: (len(pair[0]), pair[1]), reverse=True)
    winning_items, _ = candidates[0]

    level, key = _pick_specificity(winning_items)
    permalinks = [permalink for permalink, *_ in winning_items]
    return _HomeRegionCandidate(level=level, key=key, permalinks=permalinks)


def _posts_with_bio_pseudo_post(platform: str, posts: list[SocialPost], bio: str | None) -> list[SocialPost]:
    """La biografía se trata como una publicación más de cara a las regex
    de autodeclaración (mismo criterio que un post/comentario, solo que sin
    permalink real -- se usa "bio" como identificador de evidencia). Así
    "estudiante de enfermería" en la bio se detecta con el mismo código
    que si estuviera en un post, sin duplicar lógica de detección."""
    if not bio:
        return posts
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
    return [bio_pseudo_post, *posts]


async def _apply_ai_findings(
    demographic_findings: DemographicFindings,
    travel_permalinks: set[str],
    posts: list[SocialPost],
    username: str,
    full_name: str | None,
    bio: str | None,
    progress_callback: ProgressCallback | None,
) -> tuple[DemographicFindings, set[str], list[InferredAttribute]]:
    """Extracción de autodeclaraciones con IA: complementa las regex (que
    solo cubren un vocabulario fijo) leyendo el texto -- y también el
    nombre público de la cuenta, que sirve como señal débil de sexo -- de
    forma más flexible. Se ejecuta automáticamente en cada análisis, sin
    ningún botón. Ver docstring de app/nlp/ai_attribute_extraction.py para
    el razonamiento RGPD. Módulo opcional/best-effort: sin MISTRAL_API_KEY,
    o si la llamada falla, esto no aporta nada y el informe se sigue
    generando solo con lo detectado por regex (devuelve los mismos
    `demographic_findings`/`travel_permalinks` de entrada, sin tocar, y
    ninguna inferencia blanda)."""
    if not settings.mistral_api_key:
        return demographic_findings, travel_permalinks, []

    await emit_progress(progress_callback, "Buscando autodeclaraciones con IA...")
    ai_findings = await extract_demographics_with_ai(posts, username=username, full_name=full_name, bio=bio)
    soft_inferences = ai_findings.soft_inferences
    demographic_findings = merge_findings(demographic_findings, ai_findings)
    # La IA también señala fotos de viaje/vacaciones en el mismo pase (campo
    # "fotos_de_viaje" del prompt) -- se UNE con lo que ya haya detectado la
    # regex de travel_detection.py, nunca lo sustituye.
    return demographic_findings, travel_permalinks | ai_findings.travel_permalinks, soft_inferences


def _apply_home_candidate(demographic_findings: DemographicFindings, home_candidate: "_HomeRegionCandidate") -> None:
    """Vuelca el resultado de `_infer_home_region` en `demographic_findings`,
    al nivel (provincia o comunidad autónoma) que haya resultado."""
    if home_candidate.level == "provincia":
        demographic_findings.provincia = home_candidate.key
        demographic_findings.evidence.setdefault("provincia", []).extend(home_candidate.permalinks)
        demographic_findings.source["provincia"] = "imagen"
    else:
        demographic_findings.comunidad_autonoma = home_candidate.key
        demographic_findings.evidence.setdefault("comunidad_autonoma", []).extend(home_candidate.permalinks)
        demographic_findings.source["comunidad_autonoma"] = "imagen"


async def _apply_image_geolocation(
    platform: str,
    posts: list[SocialPost],
    demographic_findings: DemographicFindings,
    travel_permalinks: set[str],
    geolocation_task: "asyncio.Task | None",
    progress_callback: ProgressCallback | None,
) -> tuple[list[ImageLocationPoint], bool, list[InferredAttribute]]:
    """Geolocalización por imagen: solo se usa como ubicación PARA EL
    CÁLCULO DE POBLACIÓN si el texto no dio ya una provincia/municipio/
    comunidad explícita (la autodeclaración en texto es más fiable), y solo
    si hay CONSENSO entre varias fotos de la misma comunidad autónoma -- ver
    `_infer_home_region` y las constantes HIGH_CONFIDENCE*/
    MODERATE_CONFIDENCE* al principio de este módulo. Las fotos marcadas
    como "de viaje" (travel_permalinks) nunca cuentan para esto. Pero TODAS
    las estimaciones por imagen (de viaje, baja confianza, etc.) se
    devuelven igualmente para `image_location_points`, así el frontend
    puede mostrar cada foto analizada con su confianza real -- no solo las
    que "cuentan" para inferir dónde vive la persona. Muta
    `demographic_findings` in place si hay consenso de ubicación, o si el
    análisis de CONTENIDO visual (ver app/vision/scene_analysis.py)
    detectó una señal de pareja y el texto no dio ninguna ya -- misma
    lógica de "el texto explícito manda si lo hay" que con la ubicación.

    También devuelve las inferencias de contenido visual (aficiones,
    actividades) para que generate_report las añada a
    `inferred_attributes`, igual que las inferencias blandas de texto.

    Módulo opcional/best-effort: si el índice FAISS no está construido (ver
    app/vision/geolocation.py), el segundo valor devuelto (disponibilidad)
    queda a False y el resto del informe sigue generándose con normalidad.
    Si `platform` no es "instagram", no hace nada (no hay fotos que analizar)."""
    if platform != "instagram":
        return [], False, []

    if geolocation_task is not None:
        # Ya se lanzó en segundo plano al principio del pipeline (ver
        # analysis_router._build_report) -- aquí solo se recoge el
        # resultado, corriendo en paralelo con todo lo anterior
        # (fingerprint, atributos, autodeclaraciones con IA) en vez de
        # esperar a que termine todo eso para empezar con las fotos.
        geo_outcome = await geolocation_task
    else:
        from app.vision.geolocation import estimate_locations_for_posts

        geo_outcome = await estimate_locations_for_posts(posts, progress_callback=progress_callback)

    post_dates_by_permalink = {post.permalink: post.created_utc for post in posts}
    image_location_points = [
        ImageLocationPoint(
            permalink=permalink,
            province=estimate.province,
            confidence=estimate.confidence,
            lat=estimate.lat,
            lon=estimate.lon,
            representative=estimate.representative,
            created_utc=post_dates_by_permalink.get(permalink),
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
            _apply_home_candidate(demographic_findings, home_candidate)

    # Igual criterio que con la ubicación: si el TEXTO (autodeclaración o
    # razonamiento simbólico de ai_attribute_extraction.py) ya dio un
    # estado civil, manda ese -- la imagen solo rellena el hueco si el
    # texto no dijo nada. `source="imagen"` (no "ia_simbolica") para que
    # quede claro en el informe de dónde viene esta inferencia en concreto.
    if demographic_findings.estado_civil is None and geo_outcome.partner_signal_permalinks:
        demographic_findings.estado_civil = "con_pareja"
        demographic_findings.source["estado_civil"] = "imagen"
        demographic_findings.evidence["estado_civil"] = list(geo_outcome.partner_signal_permalinks)

    visual_inferences = [inferred for _, inferred in geo_outcome.visual_inferences]

    return image_location_points, geo_outcome.index_available, visual_inferences


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
    # Tarea de geolocalización de imágenes ya lanzada en segundo plano por
    # el llamador (ver analysis_router._build_report), para que el análisis
    # de fotos -- lo que más tarda de todo el pipeline -- corra en PARALELO
    # con el resto (fingerprint, atributos, IA) desde el principio, en vez
    # de esperar a que todo lo demás termine para empezar con las fotos.
    # Si no se pasa (p. ej. tests que llaman a generate_report
    # directamente), se crea aquí mismo como antes -- sin cambio de
    # comportamiento para quien no use este parámetro.
    geolocation_task: "asyncio.Task | None" = None,
) -> ExposureReport:

    posts_for_demographics = _posts_with_bio_pseudo_post(platform, posts, bio)
    demographic_findings = extract_demographics(posts_for_demographics)
    travel_permalinks = detect_travel_permalinks(posts)

    demographic_findings, travel_permalinks, soft_inferences = await _apply_ai_findings(
        demographic_findings, travel_permalinks, posts, username, full_name, bio, progress_callback
    )
    # Las inferencias blandas (emojis, fechas, señales simbólicas -- ver
    # app/nlp/ai_attribute_extraction.py) se añaden a la lista general de
    # atributos inferidos, junto a las que ya detectó infer_attributes()
    # por regex (hashtags/comunidades). Se AÑADEN, no sustituyen: el
    # parámetro `inferred_attributes` sigue siendo el que ya se calculó (y
    # ya alimentó compute_score) en analysis_router._build_report.
    inferred_attributes = [*inferred_attributes, *soft_inferences]

    image_location_points, geolocation_available, visual_inferences = await _apply_image_geolocation(
        platform, posts, demographic_findings, travel_permalinks, geolocation_task, progress_callback
    )
    # Igual que las inferencias blandas de texto: se AÑADEN a lo que ya
    # había (regex + texto por IA), nunca lo sustituyen. Ver
    # app/vision/scene_analysis.py para el prompt y el límite ético sobre
    # no identificar a otras personas que aparezcan en las fotos.
    inferred_attributes = [*inferred_attributes, *visual_inferences]

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
            reduction_percent=step.reduction_percent,
        )
        for step in narrowing_steps
    ]

    remaining_all_traits = final_remaining_population(narrowing_steps)

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
        remaining_population_all_traits=remaining_all_traits,
        remaining_population_all_traits_proportion=(
            remaining_all_traits / TOTAL_POPULATION_ES if remaining_all_traits is not None else None
        ),
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
