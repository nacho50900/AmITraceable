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

from app.models.schemas import SocialPost, WritingFingerprint
from app.scoring.privacy_score import _score_deanonymization_ease


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
    assert abs((score_100 - score_20) - 16.0) < 0.5
