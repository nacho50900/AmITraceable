from datetime import datetime, timezone

import pytest

from app.models.schemas import InferredAttribute, PrivacyScore, SocialPost, WritingFingerprint
from app.report import generator
from app.report.generator import _build_recommendations, generate_report
from app.vision import geolocation


def _post(i: int = 1, platform="reddit", media_url=None, post_type="post", permalink: str | None = None, text: str | None = None) -> SocialPost:
    return SocialPost(
        id=str(i),
        platform=platform,
        type=post_type,
        group="madrid",
        tags=["madrid"],
        text=text if text is not None else f"Post {i}",
        created_utc=datetime(2025, 1, 1, hour=12, tzinfo=timezone.utc),
        score=1,
        permalink=permalink if permalink is not None else f"https://x/{i}",
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
            return geolocation.GeolocationOutcome(index_available=False, results=[])

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
        """Con la regla de consenso actual hacen falta >=2 fotos de la
        misma comunidad con >80% de confianza (ver HIGH_CONFIDENCE* en
        report/generator.py)."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.85, k_neighbors=15, mean_similarity=0.7, lat=40.4, lon=-3.7
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.7, lat=40.4, lon=-3.7
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        assert len(report.image_location_points) == 2
        assert report.image_location_points[0].province == "Madrid"
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert location_steps[0].source == "imagen"
        assert set(location_steps[0].evidence) == {"https://ig/1", "https://ig/2"}

    @pytest.mark.asyncio
    async def test_single_high_confidence_photo_is_not_enough_on_its_own(self, monkeypatch):
        """Antes bastaba UNA foto con >=40% de confianza. Ahora, ni siquiera
        una sola foto con 95% de confianza es suficiente por sí sola: hace
        falta consenso entre varias fotos (ver report/generator.py)."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.95, k_neighbors=15, mean_similarity=0.9
                        ),
                    )
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_many_moderate_confidence_photos_of_same_ccaa_are_enough(self, monkeypatch):
        """4 fotos de la misma comunidad autónoma con >60% de confianza
        (pero ninguna llega al 80%) también deben bastar -- es la otra vía
        de consenso, pensada para "muchas fotos aunque ninguna destaque"."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        f"https://ig/{i}",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.65, k_neighbors=15, mean_similarity=0.6
                        ),
                    )
                    for i in range(4)
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert len(location_steps[0].evidence) == 4

    @pytest.mark.asyncio
    async def test_three_moderate_confidence_photos_are_not_enough(self, monkeypatch):
        """Justo por debajo del umbral de "muchas fotos" (3 en vez de 4):
        no debe asumirse la ubicación."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        f"https://ig/{i}",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.65, k_neighbors=15, mean_similarity=0.6
                        ),
                    )
                    for i in range(3)
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_travel_caption_photo_excluded_even_with_high_confidence(self, monkeypatch):
        """Una foto cuyo pie de foto indica que la persona está de viaje no
        debe contar como señal de residencia, aunque su confianza de
        geolocalización sea altísima -- ni sola ni sumando con otra."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.95, k_neighbors=15, mean_similarity=0.9
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.95, k_neighbors=15, mean_similarity=0.9
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        posts = [
            _post(platform="instagram", media_url="https://cdn/1.jpg", permalink="https://ig/1", text="De viaje en Madrid, qué pasada"),
            _post(platform="instagram", media_url="https://cdn/2.jpg", permalink="https://ig/2", text="Otra vez por Madrid"),
        ]

        report = await generate_report("instagram", "user", posts, _fingerprint(), [], _score())

        # Sigue apareciendo en los puntos del mapa (con su confianza real)...
        assert len(report.image_location_points) == 2
        # ...pero NO se usa para inferir residencia: solo queda 1 foto válida
        # (la que no menciona viaje), y una sola no basta para el consenso.
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_image_estimate_of_multi_province_ccaa_falls_back_to_comunidad_autonoma(self, monkeypatch):
        """Caso real que motivó este cambio: OSV-5M/Nominatim devuelve la
        región en inglés y a nivel de comunidad autónoma ("Canary Islands"),
        que no coincide con ninguna clave de PROVINCE_POPULATION (en
        español, a nivel de provincia) ni permite elegir entre sus dos
        provincias sin más información -- debe quedar estimable a nivel de
        comunidad autónoma en vez de "no_estimable"."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Canary Islands", confidence=0.85, k_neighbors=15, mean_similarity=0.7
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Canary Islands", confidence=0.9, k_neighbors=15, mean_similarity=0.7
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        step = location_steps[0]
        assert step.risk_level != "no_estimable"
        assert step.remaining_population is not None
        assert "comunidad autónoma" in step.attribute_label.lower()
        assert "canarias" in step.attribute_label.lower()
        assert step.source == "imagen"

    @pytest.mark.asyncio
    async def test_image_estimate_of_single_province_ccaa_resolves_to_that_province(self, monkeypatch):
        """Asturias es una comunidad autónoma de una sola provincia: no hay
        ambigüedad, así que debe resolver directamente a provincia (más
        específico que quedarse a nivel de comunidad autónoma)."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Principado de Asturias", confidence=0.85, k_neighbors=15, mean_similarity=0.7
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Principado de Asturias", confidence=0.9, k_neighbors=15, mean_similarity=0.7
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert "provincia" in location_steps[0].attribute_label.lower()
        assert "asturias" in location_steps[0].attribute_label.lower()

    @pytest.mark.asyncio
    async def test_declared_comunidad_autonoma_in_text_is_not_overridden_by_photo_consensus(self, monkeypatch):
        """Si el texto ya dice 'vivo en Canarias' (comunidad autónoma
        completa, sin provincia), la inferencia por consenso de fotos NO
        debe pisarlo, aunque hubiera consenso suficiente para otra zona."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.9
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.9
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        text_post = SocialPost(
            id="text1", platform="instagram", type="image", group="sin_etiqueta", tags=[],
            text="Vivo en Canarias, cerca del mar", created_utc=datetime.now(timezone.utc), score=1,
            permalink="https://ig/text", media_url="https://cdn/1.jpg",
        )

        report = await generate_report("instagram", "user", [text_post], _fingerprint(), [], _score())

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert location_steps[0].source == "texto"
        assert "canarias" in location_steps[0].attribute_label.lower()

    @pytest.mark.asyncio
    async def test_instagram_text_location_takes_priority_over_image(self, monkeypatch):
        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Barcelona", confidence=0.9, k_neighbors=15, mean_similarity=0.9
                        ),
                    )
                ],
            )

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


    @pytest.mark.asyncio
    async def test_low_confidence_image_shown_but_not_used_for_narrowing(self, monkeypatch):
        """Con el nuevo GeolocationOutcome sin filtrar, una estimación de baja
        confianza debe seguir apareciendo en image_location_points (para que
        el frontend la muestre), pero NO debe alimentar population_narrowing
        (no llega ni de lejos al umbral de consenso, ver HIGH_CONFIDENCE*/
        MODERATE_CONFIDENCE* en report/generator.py)."""

        async def _fake_estimate(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Sevilla", confidence=0.15, k_neighbors=15, mean_similarity=0.3
                        ),
                    )
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        assert len(report.image_location_points) == 1
        assert report.image_location_points[0].confidence == 0.15
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_geolocation_available_reflects_index_state(self, monkeypatch):
        async def _fake_unavailable(posts, progress_callback=None):
            return geolocation.GeolocationOutcome(index_available=False, results=[])

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_unavailable)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://cdn/1.jpg")],
            _fingerprint(), [], _score(),
        )

        assert report.geolocation_available is False

    @pytest.mark.asyncio
    async def test_geolocation_available_is_false_for_reddit(self, monkeypatch):
        async def _should_not_be_called(*args, **kwargs):
            raise AssertionError("no debería llamarse para Reddit")

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _should_not_be_called)

        report = await generate_report("reddit", "user", [_post()], _fingerprint(), [], _score())

        assert report.geolocation_available is False


class TestGenerateReportProgress:
    @pytest.mark.asyncio
    async def test_emits_final_stage_event(self, monkeypatch):
        events = []

        async def on_progress(stage, counts):
            events.append(stage)

        async def _no_images(*args, **kwargs):
            return geolocation.GeolocationOutcome(index_available=False, results=[])

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _no_images)

        await generate_report(
            "instagram", "user", [_post(platform="instagram")], _fingerprint(), [], _score(),
            progress_callback=on_progress,
        )

        assert "Generando el informe final..." in events

    @pytest.mark.asyncio
    async def test_uses_already_launched_geolocation_task_instead_of_calling_again(self, monkeypatch):
        """Cubre el paralelismo introducido en analysis_router._build_report:
        si ya se lanzó la geolocalización en segundo plano ANTES de llamar a
        generate_report (asyncio.create_task), este debe recoger ESE
        resultado con un simple `await`, no volver a invocar
        estimate_locations_for_posts (que sería procesar las fotos dos
        veces, o -- peor -- en serie, deshaciendo el paralelismo)."""
        import asyncio

        calls = 0

        async def _fake_estimate(*args, **kwargs):
            nonlocal calls
            calls += 1
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[("https://x/1", geolocation.ImageLocationEstimate(
                    province="madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.5,
                ))],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        task = asyncio.create_task(_fake_estimate())
        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_url="https://img/1")],
            _fingerprint(), [], _score(),
            geolocation_task=task,
        )

        assert calls == 1  # solo la llamada de la propia tarea, generate_report no repite la suya
        assert len(report.image_location_points) == 1


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
