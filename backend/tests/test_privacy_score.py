"""
Tests de app/scoring/privacy_score.py, centrados en _score_deanonymization_ease.

Cubre en particular la regresión de un bug real: con muy pocos posts, la
concentración horaria (`max(hour_values)`) es matemáticamente segura al
100% (con 1 solo post, ESE post concentra el 100% de "tus posts" en su
hora, sin que eso indique ninguna rutina real) y la riqueza de vocabulario
de un texto corto tiende a salir inflada por pura falta de repetición --
ninguna de las dos cosas debería tratarse como una señal fiable de
"facilidad de deanonimización" con tan poco contenido.
"""
from datetime import datetime, timezone

import pytest

from app.models.schemas import SocialPost, WritingFingerprint
from app.models.schemas import ImageLocationPoint, InferredAttribute, PopulationEstimate
from app.scoring.privacy_score import (
    _STYLE_MAX_CONTRIBUTION,
    _evidence,
    _score_deanonymization_ease,
    _score_geolocation,
    _score_inferable_data,
    compute_exposure_score,
)


def _post(i: int, hour: int, text: str = "Un comentario cualquiera sobre el día a día.") -> SocialPost:
    return SocialPost(
        id=f"p{i}",
        platform="reddit",
        type="comment",
        group="test",
        text=text,
        created_utc=datetime(2025, 1, 1, hour=hour, tzinfo=timezone.utc),
        score=1,
        permalink=f"https://reddit.com/r/test/{i}",
    )


def _fingerprint(vocabulary_richness: float, avg_posts_per_hour: dict[int, float]) -> WritingFingerprint:
    return WritingFingerprint(
        avg_sentence_length=10.0,
        vocabulary_richness=vocabulary_richness,
        emoji_usage_rate=0.0,
        avg_posts_per_hour={h: avg_posts_per_hour.get(h, 0.0) for h in range(24)},
        top_groups=[],
        top_keywords=[],
        detected_language="es",
    )


def test_no_posts_scores_zero():
    fp = _fingerprint(0.0, {})
    assert _score_deanonymization_ease([], fp) == 0.0


def test_single_post_scores_low_despite_trivial_100pct_hour_concentration():
    # Con 1 solo post, `max(hour_values)` sale 1.0 SIEMPRE (100% de "tus
    # posts" cae en la única hora que hay) -- antes del fix, esto por sí
    # solo bastaba para sacar un score "Medio" (~30) con casi nada de
    # contenido real. Con el factor de confianza por muestra, debe quedar
    # claramente bajo.
    posts = [_post(0, hour=14)]
    fp = _fingerprint(vocabulary_richness=0.9, avg_posts_per_hour={14: 1.0})

    score = _score_deanonymization_ease(posts, fp)

    assert score < 10.0


def test_same_raw_signals_score_higher_with_more_posts():
    # Misma concentración horaria "cruda" (0.5) y misma riqueza de
    # vocabulario, pero con más posts detrás -- debe puntuar más alto,
    # porque la señal es más fiable con más muestra.
    fp = _fingerprint(vocabulary_richness=0.45, avg_posts_per_hour={10: 0.5, 11: 0.5})

    few_posts = [_post(i, hour=10 if i % 2 == 0 else 11) for i in range(4)]
    many_posts = [_post(i, hour=10 if i % 2 == 0 else 11) for i in range(40)]

    score_few = _score_deanonymization_ease(few_posts, fp)
    score_many = _score_deanonymization_ease(many_posts, fp)

    assert score_many > score_few


def test_score_stays_bounded_between_0_and_100():
    posts = [_post(i, hour=i % 24) for i in range(300)]
    fp = _fingerprint(vocabulary_richness=0.45, avg_posts_per_hour={h: 1 / 24 for h in range(24)})

    score = _score_deanonymization_ease(posts, fp)

    assert 0.0 <= score <= 100.0


def test_sample_confidence_saturates_at_min_posts_for_reliable_style():
    # A partir de _MIN_POSTS_FOR_RELIABLE_STYLE (20) posts, la concentración
    # horaria y la riqueza de vocabulario se ponderan al 100% -- añadir más
    # posts manteniendo la MISMA concentración/riqueza no debería cambiar
    # ya la contribución de esos dos factores (solo la de volume_factor).
    fp_20 = _fingerprint(vocabulary_richness=0.45, avg_posts_per_hour={5: 0.6, 6: 0.4})
    fp_100 = _fingerprint(vocabulary_richness=0.45, avg_posts_per_hour={5: 0.6, 6: 0.4})

    posts_20 = [_post(i, hour=5 if i < 12 else 6) for i in range(20)]
    posts_100 = [_post(i, hour=5 if i < 60 else 6) for i in range(100)]

    score_20 = _score_deanonymization_ease(posts_20, fp_20)
    score_100 = _score_deanonymization_ease(posts_100, fp_100)

    # La diferencia entre ambos debe explicarse SOLO por volume_factor
    # (20/200=0.1 vs 100/200=0.5 -> 0.4*(0.5-0.1)*100 = 16 puntos), no por
    # un salto adicional en concentration/distinctiveness (ya saturados).
    # El estilo por sí solo aporta como mucho `_STYLE_MAX_CONTRIBUTION` del
    # score (ADR-56), de ahí 16 * 0.3 = 4,8 puntos.
    assert abs((score_100 - score_20) - 16.0 * _STYLE_MAX_CONTRIBUTION) < 0.5


# --- Geolocalización: fotos coherentes en una zona -------------------------


def _photo(province: str, confidence: float, **kwargs) -> ImageLocationPoint:
    return ImageLocationPoint(
        permalink=f"https://ig/{province}/{confidence}", province=province, confidence=confidence,
        lat=40.0, lon=-3.0, **kwargs,
    )


def _attr(category: str, value: str = "x", confidence: float = 0.8) -> InferredAttribute:
    return InferredAttribute(category=category, value=value, confidence=confidence, evidence=[])


def test_geolocation_without_any_signal_is_zero():
    assert _score_geolocation([]) == 0.0


def test_one_photo_scores_less_than_many_photos_in_the_same_zone():
    one = _score_geolocation([], [_photo("Madrid", 0.8)])
    many = _score_geolocation([], [_photo("Madrid", 0.8) for _ in range(5)])

    assert 0 < one < many < 100


def test_same_photos_score_higher_with_more_confidence():
    low = _score_geolocation([], [_photo("Madrid", 0.3) for _ in range(3)])
    high = _score_geolocation([], [_photo("Madrid", 0.9) for _ in range(3)])

    assert high > low


def test_photos_spread_over_provinces_score_lower_than_the_same_photos_in_one_zone():
    together = _score_geolocation([], [_photo("Madrid", 0.8) for _ in range(4)])
    spread = _score_geolocation([], [_photo(p, 0.8) for p in ("Madrid", "Leon", "Soria", "Cadiz")])

    assert spread < together


def test_unrepresentative_and_profile_photos_do_not_count():
    points = [
        _photo("Madrid", 0.9, representative=False),
        _photo("Madrid", 0.9, is_profile_picture=True),
    ]

    assert _score_geolocation([], points) == 0.0


def test_text_location_and_photos_reinforce_each_other_without_exceeding_100():
    text_only = _score_geolocation([_attr("ubicacion", confidence=0.6)])
    photos_only = _score_geolocation([], [_photo("Madrid", 0.8) for _ in range(2)])
    both = _score_geolocation([_attr("ubicacion", confidence=0.6)], [_photo("Madrid", 0.8) for _ in range(2)])

    assert both > max(text_only, photos_only)
    assert both <= 100.0


def test_self_declared_location_counts_more_the_finer_its_level():
    def step(level: str) -> PopulationEstimate:
        return PopulationEstimate(
            attribute_label="x", category="ubicacion", remaining_population=1000, risk_level="alto",
            evidence=[], confidence=0.9, location_level=level,
        )

    municipality = _score_geolocation([], population_narrowing=[step("municipio")])
    region = _score_geolocation([], population_narrowing=[step("comunidad_autonoma")])

    assert municipality > region > 0


def test_detailed_location_published_counts_as_location_signal():
    assert _score_geolocation([_attr("ubicacion_detallada", confidence=0.9)]) == pytest.approx(_evidence(0.9) * 100, abs=0.1)


# --- Datos personales inferibles -------------------------------------------


def test_inferable_data_grows_with_the_number_of_attributes():
    few = _score_inferable_data([_attr("ocupacion", "a")])
    many = _score_inferable_data([_attr("ocupacion", "a"), _attr("empresa", "b"), _attr("universidad", "c")])

    assert 0 < few < many


def test_identifying_categories_weigh_more_than_soft_signals():
    identifying = _score_inferable_data([_attr("ocupacion", confidence=1.0)])
    soft = _score_inferable_data([_attr("aficion", confidence=1.0)])

    assert identifying > soft


def test_same_attribute_detected_twice_counts_once():
    once = _score_inferable_data([_attr("ocupacion", "medico", 0.7)])
    twice = _score_inferable_data([_attr("ocupacion", "medico", 0.7), _attr("ocupacion", "medico", 0.5)])

    assert once == twice


def test_inferable_data_saturates_at_100():
    attrs = [_attr("ocupacion", str(i), 1.0) for i in range(30)]

    assert _score_inferable_data(attrs) == 100.0


def test_three_manual_physical_traits_raise_the_score():
    base = [_attr("ocupacion", "a", 0.8), _attr("aficion", "b", 0.5)]
    manual = [_attr("color_ojos", "azules", 1.0), _attr("color_pelo", "rubio", 1.0), _attr("color_piel", "clara", 1.0)]

    assert _score_inferable_data(base + manual) > _score_inferable_data(base)


# --- Facilidad de deanonimización: señales de identidad ---------------------


def test_deanonymization_rises_when_few_people_match_the_traits():
    posts = [_post(i, hour=10) for i in range(30)]
    fp = _fingerprint(0.45, {10: 0.5, 11: 0.5})

    unknown = _score_deanonymization_ease(posts, fp)
    common = _score_deanonymization_ease(posts, fp, remaining_population=5_000_000)
    rare = _score_deanonymization_ease(posts, fp, remaining_population=50)
    unique = _score_deanonymization_ease(posts, fp, remaining_population=1)

    assert unknown < common < rare < unique == 100.0


def test_published_identifiers_and_related_accounts_raise_deanonymization():
    posts = [_post(i, hour=10) for i in range(30)]
    fp = _fingerprint(0.45, {10: 0.5, 11: 0.5})

    base = _score_deanonymization_ease(posts, fp)
    with_identifier = _score_deanonymization_ease(posts, fp, n_identifiers=1)
    with_accounts = _score_deanonymization_ease(posts, fp, related_accounts_found=2)

    assert with_identifier > base
    assert with_accounts > base
    assert _score_deanonymization_ease(posts, fp, n_identifiers=2) > with_identifier


def test_style_alone_never_exceeds_its_maximum_contribution():
    posts = [_post(i, hour=10) for i in range(300)]
    fp = _fingerprint(0.45, {10: 1.0})

    assert _score_deanonymization_ease(posts, fp) <= _STYLE_MAX_CONTRIBUTION * 100 + 1e-9


# --- Score completo ---------------------------------------------------------


def test_complete_score_uses_every_signal_and_stays_bounded():
    posts = [_post(i, hour=10) for i in range(40)]
    fp = _fingerprint(0.45, {10: 0.5, 11: 0.5})

    provisional = compute_exposure_score(n_posts=len(posts), fingerprint=fp, inferred_attributes=[])
    complete = compute_exposure_score(
        n_posts=len(posts),
        fingerprint=fp,
        inferred_attributes=[_attr("ocupacion", "a"), _attr("contacto_publicado", "x@y.es", 0.9)],
        image_location_points=[_photo("Madrid", 0.8) for _ in range(4)],
        remaining_population=200,
        related_accounts_found=1,
    )

    assert complete.overall_score > provisional.overall_score
    for value in (complete.geolocation_risk, complete.inferable_data_risk, complete.deanonymization_ease, complete.overall_score):
        assert 0.0 <= value <= 100.0


# --- Calibración: lo flojo no debe puntuar como "Medio" ---------------------


def test_confidence_at_or_below_the_noise_floor_is_no_evidence():
    assert _evidence(0.3) == 0.0
    assert _evidence(0.1) == 0.0
    assert _evidence(1.0) == 1.0
    assert 0.0 < _evidence(0.5) < _evidence(0.8) < 1.0


def test_two_weak_photo_estimates_do_not_reach_medium():
    """Caso real: una cuenta imposible de geolocalizar daba "Medio (37,8)" porque dos
    fotos con ~47 % de votos de sus vecinos contaban como casi una foto segura."""
    weak = [_photo("Madrid", 0.47), _photo("Madrid", 0.48)]

    assert _score_geolocation([], weak) < 25.0  # "Bajo" en la barra


def test_weak_text_location_does_not_reach_medium_either():
    assert _score_geolocation([_attr("ubicacion", confidence=0.45)]) < 25.0


def test_near_universal_attributes_barely_move_the_inferable_data_bar():
    attrs = [_attr("rutina", "n", 0.6), _attr("lengua_materna", "es", 0.5), _attr("aficion", "m", 0.5)]

    assert _score_inferable_data(attrs) < 10.0


def test_a_published_contact_and_workplace_still_score_clearly_higher():
    exposed = [_attr("contacto_publicado", "x@y.es", 0.9), _attr("empresa", "ACME", 0.8), _attr("ocupacion", "d", 0.8)]

    assert _score_inferable_data(exposed) > 20.0
    assert _score_inferable_data(exposed) > 3 * _score_inferable_data([_attr("aficion", "m", 0.5), _attr("rutina", "n", 0.6)])
