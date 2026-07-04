"""
Tests del pipeline completo de análisis (módulos 2, 4 y 5 de la memoria del
TFG), usando datos simulados en lugar de llamadas reales a la API de Reddit.
"""
from datetime import datetime, timedelta, timezone

from app.models.schemas import SocialPost
from app.nlp import fingerprint as fingerprint_module
from app.nlp.attribute_inference import infer_attributes
from app.report.generator import generate_report
from app.scoring.privacy_score import compute_score


def _make_posts(n: int = 30) -> list[SocialPost]:
    return [
        SocialPost(
            id=f"p{i}",
            platform="reddit",
            type="post" if i % 2 == 0 else "comment",
            group="programming" if i % 3 else "madrid",
            title="Test post" if i % 2 == 0 else None,
            text=f"I love coding in Python. This is post number {i}. Great day today!",
            created_utc=datetime(2025, 1, 1, hour=(20 + i) % 24, tzinfo=timezone.utc)
            + timedelta(days=i),
            score=10,
            permalink=f"https://reddit.com/r/test/{i}",
        )
        for i in range(n)
    ]


def test_build_fingerprint_produces_expected_shape(patch_spacy_model):
    posts = _make_posts()
    fp = fingerprint_module.build_fingerprint(posts)

    assert fp.detected_language == "en"
    assert 0.0 <= fp.vocabulary_richness <= 1.0
    assert len(fp.top_groups) > 0
    assert set(fp.avg_posts_per_hour.keys()) == set(range(24))


def test_build_fingerprint_raises_on_empty_input(patch_spacy_model):
    import pytest

    with pytest.raises(ValueError):
        fingerprint_module.build_fingerprint([])


def test_infer_attributes_detects_location_and_occupation():
    posts = _make_posts()
    attrs = infer_attributes(posts)

    categories = {a.category for a in attrs}
    assert "ubicacion" in categories
    assert "ocupacion" in categories
    for attr in attrs:
        assert 0.0 <= attr.confidence <= 1.0
        assert len(attr.evidence) > 0


def test_compute_score_is_bounded_and_explained(patch_spacy_model):
    posts = _make_posts()
    fp = fingerprint_module.build_fingerprint(posts)
    attrs = infer_attributes(posts)

    score = compute_score(posts, fp, attrs)

    assert 0.0 <= score.overall_score <= 100.0
    assert score.identity_consistency_risk == 0.0  # fuera de alcance (sin correlación entre plataformas)
    assert "identity_consistency" in score.breakdown_explanation


def test_generate_report_includes_recommendations(patch_spacy_model):
    posts = _make_posts()
    fp = fingerprint_module.build_fingerprint(posts)
    attrs = infer_attributes(posts)
    score = compute_score(posts, fp, attrs)

    report = generate_report("reddit", "test_user", posts, fp, attrs, score)

    assert report.platform == "reddit"
    assert report.username == "test_user"
    assert report.n_posts_analyzed == len(posts)
    assert len(report.recommendations) > 0
