from datetime import datetime, timezone

import pytest

from app import stages
from app.models.schemas import InferredAttribute, PrivacyScore, SocialPost, WritingFingerprint
from app.report import generator
from app.report.generator import _build_recommendations, generate_report
from app.vision import geolocation


def _post(i: int = 1, platform="reddit", media_urls=None, post_type="post", permalink: str | None = None, text: str | None = None) -> SocialPost:
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
        media_urls=media_urls if media_urls is not None else [],
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

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        assert len(report.image_location_points) == 2
        assert report.image_location_points[0].province == "Madrid"
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert location_steps[0].source == "imagen"
        assert set(location_steps[0].evidence) == {"https://ig/1", "https://ig/2"}

    @pytest.mark.asyncio
    async def test_carousel_photos_get_their_own_visual_description_by_photo_link(self, monkeypatch):
        """Regresión: dos fotos del MISMO carrusel comparten `permalink`
        (el de la publicación), pero cada una tiene su propio `photo_link`
        (ver ImageLocationEstimate.photo_link / geolocation._photo_link,
        con ?img_index=N). `_apply_image_geolocation` debe buscar
        visual_description/visual_description_general por `photo_link`,
        NO por `permalink` -- si busca por `permalink`, las dos fotos del
        carrusel comparten clave y la búsqueda falla para ambas (bug real,
        visto en producción: "Sin descripción disponible" en todas las
        fotos de un carrusel pese a que Moondream2 sí las analizó)."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/carousel",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.5, k_neighbors=15, mean_similarity=0.6,
                            photo_link="https://ig/carousel?img_index=1",
                        ),
                    ),
                    (
                        "https://ig/carousel",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.5, k_neighbors=15, mean_similarity=0.6,
                            photo_link="https://ig/carousel?img_index=2",
                        ),
                    ),
                ],
                visual_descriptions={
                    "https://ig/carousel?img_index=1": "Personas en la foto: una",
                    "https://ig/carousel?img_index=2": "Personas en la foto: varias",
                },
                general_descriptions={
                    "https://ig/carousel?img_index=1": "a person at a beach",
                    "https://ig/carousel?img_index=2": "two people at a restaurant",
                },
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user",
            [_post(platform="instagram", permalink="https://ig/carousel", media_urls=["https://cdn/1.jpg", "https://cdn/2.jpg"])],
            _fingerprint(), [], _score(),
        )

        points_by_link = {p.permalink: p for p in report.image_location_points}
        assert points_by_link["https://ig/carousel?img_index=1"].visual_description == "Personas en la foto: una"
        assert points_by_link["https://ig/carousel?img_index=1"].visual_description_general == "a person at a beach"
        assert points_by_link["https://ig/carousel?img_index=2"].visual_description == "Personas en la foto: varias"
        assert points_by_link["https://ig/carousel?img_index=2"].visual_description_general == "two people at a restaurant"

    @pytest.mark.asyncio
    async def test_image_location_points_include_the_post_publication_date(self, monkeypatch):
        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.5, k_neighbors=15, mean_similarity=0.6
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.3, k_neighbors=15, mean_similarity=0.5
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        posts = [
            SocialPost(
                id="1", platform="instagram", type="image", group="sin_etiqueta", tags=[],
                text="", created_utc=datetime(2024, 3, 5, 10, tzinfo=timezone.utc), score=1,
                permalink="https://ig/1", media_urls=["https://cdn/1.jpg"],
            ),
            SocialPost(
                id="2", platform="instagram", type="image", group="sin_etiqueta", tags=[],
                text="", created_utc=datetime(2023, 11, 20, 18, tzinfo=timezone.utc), score=1,
                permalink="https://ig/2", media_urls=["https://cdn/2.jpg"],
            ),
        ]

        report = await generate_report("instagram", "user", posts, _fingerprint(), [], _score())

        points_by_permalink = {p.permalink: p for p in report.image_location_points}
        assert points_by_permalink["https://ig/1"].created_utc == datetime(2024, 3, 5, 10, tzinfo=timezone.utc)
        assert points_by_permalink["https://ig/2"].created_utc == datetime(2023, 11, 20, 18, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_image_location_points_include_the_visual_description(self, monkeypatch):
        """La descripción cruda de Moondream2 (ver scene_analysis.py) debe
        quedar en cada ImageLocationPoint -- de ahí se incluye sola en el
        JSON del informe que ai_analysis.py le manda a la IA final, sin
        necesitar ningún cambio en ese módulo. Igual para la descripción
        general (visual_description_general), campo DESCRIPCION del mismo
        prompt ya parseado."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.5, k_neighbors=15, mean_similarity=0.6
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.3, k_neighbors=15, mean_similarity=0.5
                        ),
                    ),
                ],
                visual_descriptions={
                    "https://ig/1": "DESCRIPCION: una persona tocando la guitarra\nPERSONAS: una\nAFICION: guitarra\nPAREJA: no"
                },
                general_descriptions={"https://ig/1": "una persona tocando la guitarra"},
                # https://ig/2 no tiene entrada en ninguno de los dos dicts:
                # la foto se analizó pero Moondream2 no dio descripción
                # (modelo no disponible, o falló solo para esa foto).
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        posts = [
            _post(i=1, platform="instagram", media_urls=["https://cdn/1.jpg"], permalink="https://ig/1"),
            _post(i=2, platform="instagram", media_urls=["https://cdn/2.jpg"], permalink="https://ig/2"),
        ]

        report = await generate_report("instagram", "user", posts, _fingerprint(), [], _score())

        points_by_permalink = {p.permalink: p for p in report.image_location_points}
        assert points_by_permalink["https://ig/1"].visual_description == (
            "DESCRIPCION: una persona tocando la guitarra\nPERSONAS: una\nAFICION: guitarra\nPAREJA: no"
        )
        assert points_by_permalink["https://ig/1"].visual_description_general == "una persona tocando la guitarra"
        assert points_by_permalink["https://ig/2"].visual_description is None
        assert points_by_permalink["https://ig/2"].visual_description_general is None

    @pytest.mark.asyncio
    async def test_non_representative_photo_still_shows_on_map_but_not_used_for_residence(self, monkeypatch):
        """Una foto marcada como no representativa (dispersión geográfica
        excesiva entre sus vecinos, ver ImageLocationEstimate.representative
        en app/vision/geolocation.py) debe seguir apareciendo en
        image_location_points con su confianza real -- el usuario debe
        poder ver que esa foto se analizó, aunque sea poco fiable -- pero
        NO debe poder decidir, ni siquiera junto con otra foto igual de
        dispersa, la conclusión de dónde vive la persona."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.7,
                            lat=40.4, lon=-3.7, representative=False,
                        ),
                    ),
                    (
                        "https://ig/2",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.85, k_neighbors=15, mean_similarity=0.7,
                            lat=40.4, lon=-3.7, representative=False,
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        # Sigue en el mapa, con su confianza real, sin filtrar:
        assert len(report.image_location_points) == 2
        assert report.image_location_points[0].confidence == 0.9
        # Y el campo `representative` se propaga tal cual al informe público
        # -- es el frontend quien decide con él qué pintar en el mapa y qué
        # mostrar en el apartado de "Imágenes No Representativas" (ver
        # LocationMap.tsx).
        assert report.image_location_points[0].representative is False
        assert report.image_location_points[1].representative is False

        # Pero no cuenta para afirmar una provincia/comunidad de residencia,
        # ni siquiera con dos fotos de alta confianza que cumplirían el
        # umbral de consenso si fueran representativas:
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_representative_photo_propagates_representative_true(self, monkeypatch):
        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.9, k_neighbors=15, mean_similarity=0.7,
                            lat=40.4, lon=-3.7, representative=True,
                        ),
                    ),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        assert report.image_location_points[0].representative is True

    @pytest.mark.asyncio
    async def test_single_high_confidence_photo_is_not_enough_on_its_own(self, monkeypatch):
        """Antes bastaba UNA foto con >=40% de confianza. Ahora, ni siquiera
        una sola foto con 95% de confianza es suficiente por sí sola: hace
        falta consenso entre varias fotos (ver report/generator.py)."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_many_moderate_confidence_photos_of_same_ccaa_are_enough(self, monkeypatch):
        """4 fotos de la misma comunidad autónoma con >60% de confianza
        (pero ninguna llega al 80%) también deben bastar -- es la otra vía
        de consenso, pensada para "muchas fotos aunque ninguna destaque"."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert len(location_steps[0].evidence) == 4

    @pytest.mark.asyncio
    async def test_three_moderate_confidence_photos_are_not_enough(self, monkeypatch):
        """Justo por debajo del umbral de "muchas fotos" (3 en vez de 4):
        no debe asumirse la ubicación."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_travel_caption_photo_excluded_even_with_high_confidence(self, monkeypatch):
        """Una foto cuyo pie de foto indica que la persona está de viaje no
        debe contar como señal de residencia, aunque su confianza de
        geolocalización sea altísima -- ni sola ni sumando con otra."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            _post(platform="instagram", media_urls=["https://cdn/1.jpg"], permalink="https://ig/1", text="De viaje en Madrid, qué pasada"),
            _post(platform="instagram", media_urls=["https://cdn/2.jpg"], permalink="https://ig/2", text="Otra vez por Madrid"),
        ]

        report = await generate_report("instagram", "user", posts, _fingerprint(), [], _score())

        # Sigue apareciendo en los puntos del mapa (con su confianza real)...
        assert len(report.image_location_points) == 2
        # ...pero NO se usa para inferir residencia: solo queda 1 foto válida
        # (la que no menciona viaje), y una sola no basta para el consenso.
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_avatar_photo_excluded_from_home_region_consensus(self, monkeypatch):
        """Bug real encontrado y corregido: ImageLocationPoint.is_profile_picture
        y su docstring (schemas.py) documentaban que la foto de perfil se
        excluye del consenso de residencia 'igual que las fotos de viaje',
        pero `_infer_home_region`/`_filter_and_resolve_estimates` no tenían
        ningún mecanismo real para hacerlo -- la exclusión solo existía en
        el comentario, no en el código. Aquí: 1 foto de publicación real +
        la foto de perfil, las dos con alta confianza y la MISMA provincia
        -- si la foto de perfil contara, ya habría consenso con solo esa
        combinación; NO debe contar, así que no hay consenso (hace falta
        más de 1 foto de publicación real, ver HIGH_CONFIDENCE_MIN_PHOTOS)."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            avatar_estimate = geolocation.ImageLocationEstimate(
                province="Madrid", confidence=0.95, k_neighbors=15, mean_similarity=0.9
            )
            avatar_estimate.photo_link = "https://cdn/avatar.jpg"
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[
                    (
                        "https://ig/1",
                        geolocation.ImageLocationEstimate(
                            province="Madrid", confidence=0.95, k_neighbors=15, mean_similarity=0.9
                        ),
                    ),
                    ("https://cdn/avatar.jpg", avatar_estimate),
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        posts = [_post(platform="instagram", media_urls=["https://cdn/1.jpg"], permalink="https://ig/1")]

        report = await generate_report(
            "instagram", "user", posts, _fingerprint(), [], _score(), avatar_url="https://cdn/avatar.jpg"
        )

        # Ambas fotos siguen apareciendo en el mapa (con su confianza real)...
        assert len(report.image_location_points) == 2
        avatar_point = next(p for p in report.image_location_points if p.is_profile_picture)
        assert avatar_point.permalink == "https://cdn/avatar.jpg"
        # ...pero NO hay consenso de residencia: solo 1 foto de publicación
        # real cuenta (la del avatar se excluye), y HIGH_CONFIDENCE_MIN_PHOTOS
        # exige más de 1 para dar la ubicación por buena.
        location_steps_avatar_test = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps_avatar_test == []

    @pytest.mark.asyncio
    async def test_image_estimate_of_multi_province_ccaa_falls_back_to_comunidad_autonoma(self, monkeypatch):
        """Caso real que motivó este cambio: OSV-5M/Nominatim devuelve la
        región en inglés y a nivel de comunidad autónoma ("Canary Islands"),
        que no coincide con ninguna clave de PROVINCE_POPULATION (en
        español, a nivel de provincia) ni permite elegir entre sus dos
        provincias sin más información -- debe quedar estimable a nivel de
        comunidad autónoma en vez de "no_estimable"."""

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
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

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
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

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            permalink="https://ig/text", media_urls=["https://cdn/1.jpg"],
        )

        report = await generate_report("instagram", "user", [text_post], _fingerprint(), [], _score())

        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert len(location_steps) == 1
        assert location_steps[0].source == "texto"
        assert "canarias" in location_steps[0].attribute_label.lower()

    @pytest.mark.asyncio
    async def test_instagram_text_location_takes_priority_over_image(self, monkeypatch):
        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            permalink="https://ig/text", media_urls=["https://cdn/1.jpg"],
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

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        assert len(report.image_location_points) == 1
        assert report.image_location_points[0].confidence == 0.15
        location_steps = [s for s in report.population_narrowing if s.category == "ubicacion"]
        assert location_steps == []

    @pytest.mark.asyncio
    async def test_geolocation_available_reflects_index_state(self, monkeypatch):
        async def _fake_unavailable(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(index_available=False, results=[])

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_unavailable)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
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

    @pytest.mark.asyncio
    async def test_remaining_population_all_traits_reflects_combined_chain(self, monkeypatch):
        posts = [_post(text="Vivo en Madrid, tengo 30 años y soy hombre", platform="reddit")]

        report = await generate_report("reddit", "user", posts, _fingerprint(), [], _score())

        location_step = next(s for s in report.population_narrowing if s.category == "ubicacion")
        assert report.remaining_population_all_traits == location_step.remaining_population

    @pytest.mark.asyncio
    async def test_remaining_population_all_traits_is_none_without_chained_findings(self):
        report = await generate_report("reddit", "user", [_post()], _fingerprint(), [], _score())

        assert report.remaining_population_all_traits is None


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

        # Antes emitía el texto en español ya renderizado; desde la
        # internacionalización de la webapp (frontend/i18n) se emite un
        # código estable (ver app/stages.py) que el frontend traduce.
        assert stages.GENERATING_REPORT in events

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
            "instagram", "user", [_post(platform="instagram", media_urls=["https://img/1"])],
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


class TestSoftInferencesReachTheReport:
    @pytest.mark.asyncio
    async def test_ai_soft_inferences_are_appended_to_inferred_attributes(self, monkeypatch):
        """Cubre el caso motivador: una bio tipo '18/05/20🧡👸✨' no es una
        autodeclaración explícita (no la detectan ni las regex ni el resto
        del prompt de IA), pero sí debe poder llegar como inferencia blanda
        hasta la lista general de atributos inferidos del informe."""
        from app.config import settings
        from app.nlp.demographic_extraction import DemographicFindings

        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")

        async def _fake_ai_extraction(posts, username, full_name=None, bio=None):
            findings = DemographicFindings()
            findings.soft_inferences = [
                InferredAttribute(
                    category="relacion_sentimental",
                    value="Posible relación de pareja (fecha con emojis de corazón en la bio)",
                    confidence=0.6,
                    evidence=["bio"],
                )
            ]
            return findings

        monkeypatch.setattr(generator, "extract_demographics_with_ai", _fake_ai_extraction)

        regex_attribute = InferredAttribute(category="ubicacion", value="x", confidence=0.5, evidence=[])
        report = await generate_report(
            "instagram", "user", [_post(platform="instagram")], _fingerprint(),
            [regex_attribute], _score(), bio="18/05/20🧡👸✨",
        )

        categories = [a.category for a in report.inferred_attributes]
        assert "ubicacion" in categories  # lo de regex se conserva
        assert "relacion_sentimental" in categories  # y se añade lo de la IA
        soft = next(a for a in report.inferred_attributes if a.category == "relacion_sentimental")
        assert soft.confidence == 0.6
        assert "pareja" in soft.value.lower()

    @pytest.mark.asyncio
    async def test_without_mistral_api_key_no_soft_inferences_are_added(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "mistral_api_key", None)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram")], _fingerprint(), [], _score(),
        )

        assert report.inferred_attributes == []

    @pytest.mark.asyncio
    async def test_ai_estado_civil_reaches_population_narrowing_not_just_inferred_attributes(self, monkeypatch):
        """El caso pedido explícitamente: a diferencia de las inferencias
        blandas de categoría libre (que solo van a inferred_attributes),
        estado_civil debe aparecer en la tabla de estrechamiento del
        informe y afectar al porcentaje de población restante."""
        from app.config import settings
        from app.nlp.demographic_extraction import DemographicFindings

        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")

        async def _fake_ai_extraction(posts, username, full_name=None, bio=None):
            return DemographicFindings(
                estado_civil="casado",
                evidence={"estado_civil": ["bio"]},
                source={"estado_civil": "ia_simbolica"},
            )

        monkeypatch.setattr(generator, "extract_demographics_with_ai", _fake_ai_extraction)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram")], _fingerprint(), [], _score(),
            bio="Mi marido y yo 💍",
        )

        relacion_steps = [s for s in report.population_narrowing if s.category == "estado_civil"]
        assert len(relacion_steps) == 1
        assert relacion_steps[0].remaining_population is not None
        assert relacion_steps[0].source == "ia_simbolica"
        # Y también cuenta para el número combinado de "personas que
        # comparten tus rasgos" (ver k_anonymity.py::final_remaining_population).
        assert report.remaining_population_all_traits == relacion_steps[0].remaining_population


class TestVisualAnalysisReachesTheReport:
    """Cubre el análisis de CONTENIDO visual (aficiones/actividades/señal
    de pareja detectadas en fotos con Moondream2, ver
    app/vision/scene_analysis.py) llegando hasta el informe final."""

    @pytest.mark.asyncio
    async def test_visual_inferences_are_appended_to_inferred_attributes(self, monkeypatch):
        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[],
                visual_inferences=[
                    (
                        "https://ig/1",
                        InferredAttribute(
                            category="aficion",
                            value="Posible afición/interés detectado en una foto: baloncesto",
                            confidence=0.5,
                            evidence=["https://ig/1"],
                        ),
                    )
                ],
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        regex_attribute = InferredAttribute(category="ubicacion", value="x", confidence=0.5, evidence=[])
        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [regex_attribute], _score(),
        )

        categories = [a.category for a in report.inferred_attributes]
        assert "ubicacion" in categories  # lo anterior se conserva
        assert "aficion" in categories  # y se añade lo visual
        visual = next(a for a in report.inferred_attributes if a.category == "aficion")
        assert "baloncesto" in visual.value.lower()

    @pytest.mark.asyncio
    async def test_partner_signal_from_image_sets_estado_civil_when_text_gave_none(self, monkeypatch):
        """El caso motivador: una foto besando a alguien, sin que el texto
        dijera nada sobre pareja -- debe rellenar estado_civil con
        source='imagen', igual que la ubicación por imagen cuando el
        texto no dio ninguna."""
        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[],
                partner_signal_permalinks={"https://ig/1"},
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(),
        )

        step = next(s for s in report.population_narrowing if s.category == "estado_civil")
        assert step.attribute_label == "Tiene pareja (sin estar casado/a)"
        assert step.source == "imagen"
        assert step.evidence == ["https://ig/1"]

    @pytest.mark.asyncio
    async def test_partner_signal_from_image_does_not_override_text_estado_civil(self, monkeypatch):
        """Igual criterio que con la ubicación: si el texto ya dio un
        estado civil, la imagen NO lo pisa, aunque detecte una señal
        distinta."""
        from app.config import settings
        from app.nlp.demographic_extraction import DemographicFindings

        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")

        async def _fake_ai_extraction(posts, username, full_name=None, bio=None):
            return DemographicFindings(
                estado_civil="soltero",
                evidence={"estado_civil": ["bio"]},
                source={"estado_civil": "ia_simbolica"},
            )

        monkeypatch.setattr(generator, "extract_demographics_with_ai", _fake_ai_extraction)

        async def _fake_estimate(posts, avatar_url=None, progress_callback=None):
            return geolocation.GeolocationOutcome(
                index_available=True,
                results=[],
                partner_signal_permalinks={"https://ig/1"},  # la foto sugeriría pareja, pero el texto manda
            )

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate)

        report = await generate_report(
            "instagram", "user", [_post(platform="instagram", media_urls=["https://cdn/1.jpg"])],
            _fingerprint(), [], _score(), bio="Soltera y feliz",
        )

        step = next(s for s in report.population_narrowing if s.category == "estado_civil")
        assert step.attribute_label == "Soltero/a (sin pareja actualmente)"
        assert step.source == "ia_simbolica"
