from datetime import datetime, timezone

import pytest

from app.models.schemas import InferredAttribute, PrivacyScore, SocialPost, WritingFingerprint
from app.report import generator
from app.report.generator import _build_recommendations, generate_report
from app.vision import geolocation


def _post(i: int = 1, platform="reddit", media_url=None, post_type="post") -> SocialPost:
    return SocialPost(
        id=str(i),
        platform=platform,
        type=post_type,
        group="madrid",
        tags=["madrid"],
        text=f"Post {i}",
        created_utc=datetime(2025, 1, 1, hour=12, tzinfo=timezone.utc),
        score=1,
        permalink=f"https://x/{i}",
        media_url=media_url,
    )


def _fingerprint(peak_hour: str | None = None, peak_value: float = 0.0) -> WritingFingerprint:
    hours = {str(h): 0.0 for h in range(24)}
    if peak_hour is not None:
        hours[peak_hour] = peak_value
    return WritingFingerprint(
        avg_sentence_length=8.0,
        vocabulary_richness=0.5,
        emoji_usage_rate=0.0,
        avg_posts_per_hour=hours,
        top_groups=[],
        top_keywords=[],
        detected_language="es",
    )


def _score(**overrides) -> PrivacyScore:
    defaults = dict(
        overall_score=10,
        geolocation_risk=0,
        identity_consistency_risk=0,
        inferable_data_risk=0,
        deanonymization_ease=0,
        breakdown_explanation={
            "geolocation": "x",
            "identity_consistency": "x",
            "inferable_data": "x",
            "deanonymization_ease": "x",
        },
    )
    defaults.update(overrides)
    return PrivacyScore(**defaults)


class TestGenerateReportPlatformBranching:
    @pytest.mark.asyncio
    async def test_reddit_never_touches_image_geolocation(self, monkeypatch):
        called = {"n": 0}

        async def _should_not_be_called(*args, **kwargs):
            called["n"] += 1
            return []

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _should_not_be_called)

        report = await generate_report(
            "reddit", "user", [_post()], _fingerprint(), [], _score()
        )

        assert called["n"] == 0
        assert report.image_location_points == []

    @pytest.mark.asyncio
    async def test_instagram_without_media_urls_produces_no_points(self, patch_spacy_model):
        # Sin índice FAISS construido en este entorno, estimate_locations_for_posts
        # real ya degrada a lista vacía -- se ejercita la rama real, no un mock.
        report = await generate_report(
            "instagram", "user", [_post(platform="instagram")], _fingerprint(), [], _score()
        )
        assert report.image_location_points == []

    @pytest.mark.asyncio
    async def test_instagram_image_estimate_fills_missing_location_with_source_imagen(self, monkeypatch):
        async def _fake_estimate(posts, progress_callback=None):
            return [
                (
                    "https://ig/1",
                    geolocation.ImageLocationEstimate(
                        province="Madrid", confidence=0.8, k_neighbors=15, mean_similarity=0.7, lat=40.4, lon=-3.7
                    ),
                )
            ]

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        assert len(report.image_location_points) == 1
        assert report.image_location_points[0].province == "Madrid"
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert location_steps[0].source == "imagen"

    @pytest.mark.asyncio
    async def test_instagram_text_location_takes_priority_over_image(self, monkeypatch):
        async def _fake_estimate(posts, progress_callback=None):
            return [
                (
                    "https://ig/1",
                    geolocation.ImageLocationEstimate(
                        province="Barcelona", confidence=0.9, k_neighbors=15, mean_similarity=0.9
                    ),
                )
            ]

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        text_post = SocialPost(
            id="text1", platform="instagram", type="image", group="sin_etiqueta", tags=[],
            text="Vivo en León y me encanta", created_utc=datetime.now(timezone.utc), score=1,
            permalink="https://ig/text", media_url="https://cdn/1.jpg",
        )

        report = await generate_report("instagram", "user", [text_post], _fingerprint(), [], _score())

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert location_steps[0].source == "texto"
        assert "eon" in location_steps[0].attribute_label  # León / Leon según normalización


class TestGenerateReportProgress:
    @pytest.mark.asyncio
    async def test_emits_final_stage_event(self, monkeypatch):
        events = []

        async def on_progress(stage, counts):
            events.append(stage)

        async def _no_images(*args, **kwargs):
            return []

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _no_images)

        await generate_report(
            "instagram", "user", [_post(platform="instagram")], _fingerprint(), [], _score(),
            progress_callback=on_progress,
        )

        assert "Generando el informe final..." in events


class TestBuildRecommendations:
    def test_high_geolocation_risk_produces_specific_recommendation(self):
        recs = _build_recommendations(_fingerprint(), [], _score(geolocation_risk=31))
        assert any("comunidades" in r for r in recs)

    def test_low_geolocation_risk_omits_that_recommendation(self):
        recs = _build_recommendations(_fingerprint(), [], _score(geolocation_risk=30))
        assert not any("comunidades" in r for r in recs)

    def test_high_inferable_data_risk_produces_specific_recommendation(self):
        recs = _build_recommendations(_fingerprint(), [], _score(inferable_data_risk=41))
        assert any("combinando varias" in r for r in recs)

    def test_high_deanonymization_ease_produces_specific_recommendation(self):
        recs = _build_recommendations(_fingerprint(), [], _score(deanonymization_ease=51))
        assert any("huella" in r for r in recs)

    def test_concentrated_peak_hour_produces_specific_recommendation(self):
        recs = _build_recommendations(_fingerprint(peak_hour="20", peak_value=0.3), [], _score())
        assert any("20:00" in r for r in recs)

    def test_low_peak_concentration_omits_that_recommendation(self):
        recs = _build_recommendations(_fingerprint(peak_hour="20", peak_value=0.25), [], _score())
        assert not any("20:00" in r for r in recs)

    def test_ocupacion_attribute_produces_specific_recommendation(self):
        attrs = [InferredAttribute(category="ocupacion", value="x", confidence=0.5, evidence=[])]
        recs = _build_recommendations(_fingerprint(), attrs, _score())
        assert any("sector de trabajo" in r for r in recs)

    def test_no_risk_conditions_produces_fallback_low_exposure_message(self):
        recs = _build_recommendations(_fingerprint(), [], _score())
        assert len(recs) == 1
        assert "bajo" in recs[0]

    def test_multiple_conditions_produce_multiple_recommendations(self):
        attrs = [InferredAttribute(category="ocupacion", value="x", confidence=0.5, evidence=[])]
        recs = _build_recommendations(
            _fingerprint(peak_hour="9", peak_value=0.3),
            attrs,
            _score(geolocation_risk=50, inferable_data_risk=60, deanonymization_ease=70),
        )
        assert len(recs) == 5  # las 4 condiciones de riesgo + la de ocupación
