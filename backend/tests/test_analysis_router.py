"""
Tests de app/analysis_router.py.

En vez de mockear las llamadas HTTP reales a Reddit/Instagram (ya cubierto
en test_reddit_client.py / test_instagram_client.py), aquí se sustituye
directamente la entrada correspondiente de `_PLATFORM_CLIENT_FACTORIES` por
un cliente falso controlado -- es exactamente el punto de extensión que ya
existía en el diseño (factory por plataforma), así que es la forma más
fiel de probar el endpoint sin acoplarse a los detalles de cada API externa.
"""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import analysis_router
from app.ai_analysis import AiAnalysisUnavailable
from app.main import app
from app.models.schemas import ExposureReport, PrivacyScore, SocialPost, SocialProfile, WritingFingerprint
from app.vision import geolocation

client = TestClient(app, base_url="https://testserver")


def _make_posts(n: int = 5) -> list[SocialPost]:
    return [
        SocialPost(
            id=f"p{i}",
            platform="reddit",
            type="post",
            group="madrid",
            tags=["madrid"],
            text=f"Post de prueba numero {i}",
            created_utc=datetime(2025, 1, 1, hour=i % 24, tzinfo=timezone.utc),
            score=1,
            permalink=f"https://reddit.com/r/test/{i}",
        )
        for i in range(n)
    ]


class _FakeClient:
    def __init__(self, posts, raise_exc=None):
        self._posts = posts
        self._raise_exc = raise_exc

    async def fetch_profile(self, progress_callback=None):
        if self._raise_exc:
            raise self._raise_exc
        if progress_callback:
            await progress_callback("Leyendo publicaciones...", {"posts_analyzed": len(self._posts)})
        return SocialProfile(
            platform="reddit",
            username="fake_user",
            account_created_utc=None,
            bio=None,
            posts=self._posts,
        )


@pytest.fixture
def register_fake_platform(monkeypatch):
    """Registra una plataforma de prueba ('reddit', reutilizando la ruta
    real) cuyo cliente se controla por completo desde el test, sin
    necesitar sesión ni red."""

    def _register(fake_client):
        monkeypatch.setitem(
            analysis_router._PLATFORM_CLIENT_FACTORIES, "reddit", lambda request: fake_client
        )

    return _register


class TestAnalyzeEndpoint:
    def test_unsupported_platform_returns_404(self):
        resp = client.post("/api/analyze/tiktok")
        assert resp.status_code == 404
        assert "tiktok" in resp.json()["detail"]

    def test_empty_posts_returns_422(self, register_fake_platform, patch_spacy_model):
        register_fake_platform(_FakeClient(posts=[]))

        resp = client.post("/api/analyze/reddit")

        assert resp.status_code == 422
        assert "actividad pública" in resp.json()["detail"]

    def test_success_returns_full_report(self, register_fake_platform, patch_spacy_model):
        register_fake_platform(_FakeClient(posts=_make_posts()))

        resp = client.post("/api/analyze/reddit")

        assert resp.status_code == 200
        body = resp.json()
        assert body["platform"] == "reddit"
        assert body["username"] == "fake_user"
        assert body["n_posts_analyzed"] == 5
        assert "privacy_score" in body
        assert "population_narrowing" in body
        assert "image_location_points" in body


class TestAnalyzeStreamEndpoint:
    def test_unsupported_platform_returns_404_before_opening_stream(self):
        resp = client.get("/api/analyze/tiktok/stream")
        assert resp.status_code == 404

    def test_missing_session_returns_401_before_opening_stream(self, monkeypatch):
        # Sin registrar un fake client: usa la factory real, que exige sesión.
        resp = client.get("/api/analyze/reddit/stream")
        assert resp.status_code == 401

    def test_emits_progress_events_then_final_report(self, register_fake_platform, patch_spacy_model):
        register_fake_platform(_FakeClient(posts=_make_posts()))

        resp = client.get("/api/analyze/reddit/stream")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = [
            line[len("data: "):]
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data: ")
        ]
        assert len(events) >= 2  # al menos un progreso + el evento final

        import json

        parsed = [json.loads(e) for e in events]
        # El último evento es siempre el de cierre, con el informe completo.
        assert parsed[-1]["done"] is True
        assert parsed[-1]["report"]["platform"] == "reddit"
        # Los anteriores son de progreso, con "done": false y una etapa.
        for event in parsed[:-1]:
            assert event["done"] is False
            assert "stage" in event

    def test_empty_posts_emits_error_event_not_http_500(self, register_fake_platform, patch_spacy_model):
        register_fake_platform(_FakeClient(posts=[]))

        resp = client.get("/api/analyze/reddit/stream")

        assert resp.status_code == 200  # el stream en sí se abre con éxito
        import json

        events = [
            json.loads(line[len("data: "):])
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data: ")
        ]
        assert events[-1]["done"] is True
        assert "error" in events[-1]
        assert "actividad pública" in events[-1]["error"]

    def test_unexpected_exception_emits_error_event_not_crashing(self, register_fake_platform, patch_spacy_model):
        register_fake_platform(_FakeClient(posts=[], raise_exc=RuntimeError("fallo inesperado de red")))

        resp = client.get("/api/analyze/reddit/stream")

        assert resp.status_code == 200
        import json

        events = [
            json.loads(line[len("data: "):])
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data: ")
        ]
        assert events[-1]["done"] is True
        assert "Error inesperado" in events[-1]["error"]


def _make_report() -> ExposureReport:
    return ExposureReport(
        platform="reddit",
        username="test_user",
        generated_at=datetime.now(timezone.utc),
        n_posts_analyzed=1,
        fingerprint=WritingFingerprint(
            avg_sentence_length=5.0,
            vocabulary_richness=0.5,
            emoji_usage_rate=0.0,
            avg_posts_per_hour={str(h): 0.0 for h in range(24)},
            top_groups=[],
            top_keywords=[],
            detected_language="es",
        ),
        inferred_attributes=[],
        privacy_score=PrivacyScore(
            overall_score=5,
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
        ),
        recommendations=[],
        population_narrowing=[],
        image_location_points=[],
    )


class TestAiSummaryEndpoint:
    def test_success_returns_verdict_and_conclusions(self, monkeypatch):
        async def _fake_analyze(report, lang="es"):
            return {"verdict": "Riesgo bajo en general.", "conclusions": ["Conclusión 1", "Conclusión 2"]}

        monkeypatch.setattr(analysis_router, "analyze_report_with_ai", _fake_analyze)

        resp = client.post("/api/analyze/ai-summary", json=_make_report().model_dump(mode="json"))

        assert resp.status_code == 200
        assert resp.json() == {"verdict": "Riesgo bajo en general.", "conclusions": ["Conclusión 1", "Conclusión 2"]}

    def test_unavailable_returns_503_not_500(self, monkeypatch):
        async def _fake_analyze(report, lang="es"):
            raise AiAnalysisUnavailable("no configurado")

        monkeypatch.setattr(analysis_router, "analyze_report_with_ai", _fake_analyze)

        resp = client.post("/api/analyze/ai-summary", json=_make_report().model_dump(mode="json"))

        assert resp.status_code == 503
        assert resp.json()["detail"] == "no configurado"

    def test_malformed_body_returns_422(self):
        resp = client.post("/api/analyze/ai-summary", json={"not": "a valid report"})
        assert resp.status_code == 422


class TestTranslateDescriptionsEndpoint:
    """ADR-30/ADR-31. Mismo patrón que TestAiSummaryEndpoint de arriba --
    estos tests, además de comprobar el endpoint en sí, verifican de
    forma implícita el orden de registro de rutas: si
    `/analyze/{platform}` resolviera primero, "translate-descriptions"
    haría match como nombre de plataforma (404, no 200/422).

    `translate_texts_local` (ver app/nlp/translation.py, ADR-31) es
    SÍNCRONA -- el endpoint la despacha con `asyncio.to_thread`, así que
    aquí se mockea con una función normal, no `async def`."""

    def test_success_returns_translations_in_order(self, monkeypatch):
        def _fake_translate(texts, lang="es"):
            return [t.upper() for t in texts]

        monkeypatch.setattr(analysis_router, "translate_texts_local", _fake_translate)

        resp = client.post("/api/analyze/translate-descriptions", json={"texts": ["guitarra", "baloncesto"]})

        assert resp.status_code == 200
        assert resp.json() == {"translations": ["GUITARRA", "BALONCESTO"]}

    def test_malformed_body_returns_422(self):
        resp = client.post("/api/analyze/translate-descriptions", json={"not": "the expected shape"})
        assert resp.status_code == 422

    def test_lang_query_param_is_forwarded(self, monkeypatch):
        received = {}

        def _fake_translate(texts, lang="es"):
            received["lang"] = lang
            return texts

        monkeypatch.setattr(analysis_router, "translate_texts_local", _fake_translate)

        client.post("/api/analyze/translate-descriptions?lang=en", json={"texts": ["x"]})

        assert received["lang"] == "en"

    def test_empty_texts_list_is_valid(self, monkeypatch):
        def _fake_translate(texts, lang="es"):
            return []

        monkeypatch.setattr(analysis_router, "translate_texts_local", _fake_translate)

        resp = client.post("/api/analyze/translate-descriptions", json={"texts": []})

        assert resp.status_code == 200
        assert resp.json() == {"translations": []}

    def test_no_es_translation_call_never_raises_even_if_models_missing(self):
        """Sin mockear nada: translate_texts_local() de verdad, sin
        modelos convertidos en disco (no aplica en este entorno de
        test) -- debe degradarse a devolver los textos originales, sin
        reventar el endpoint ni devolver un error."""
        resp = client.post("/api/analyze/translate-descriptions?lang=en", json={"texts": ["guitarra"]})

        assert resp.status_code == 200
        assert resp.json() == {"translations": ["guitarra"]}

    def test_logs_translation_performance_with_correct_direction(self, monkeypatch):
        """ver app/log/translation_log.py -- lang="en" pedido implica
        origen "es" (único idioma soportado que no es "en"), calculado
        con `source_language_for()` (ver TestSourceLanguageFor en
        test_translation.py)."""
        monkeypatch.setattr(analysis_router, "translate_texts_local", lambda texts, lang="es": texts)
        logged = {}

        def _fake_log(**kwargs):
            logged.update(kwargs)

        monkeypatch.setattr(analysis_router, "log_translation_run", _fake_log)

        client.post("/api/analyze/translate-descriptions?lang=en", json={"texts": ["guitarra", "baloncesto"]})

        assert logged["num_texts"] == 2
        assert logged["source_lang"] == "es"
        assert logged["target_lang"] == "en"
        assert logged["total_seconds"] >= 0

    def test_does_not_log_when_lang_unsupported(self, monkeypatch):
        monkeypatch.setattr(analysis_router, "translate_texts_local", lambda texts, lang="es": texts)
        called = []
        monkeypatch.setattr(analysis_router, "log_translation_run", lambda **kwargs: called.append(kwargs))

        resp = client.post("/api/analyze/translate-descriptions?lang=fr", json={"texts": ["guitarra"]})

        assert resp.status_code == 200
        assert called == []

    def test_does_not_log_when_texts_empty(self, monkeypatch):
        monkeypatch.setattr(analysis_router, "translate_texts_local", lambda texts, lang="es": texts)
        called = []
        monkeypatch.setattr(analysis_router, "log_translation_run", lambda **kwargs: called.append(kwargs))

        client.post("/api/analyze/translate-descriptions?lang=en", json={"texts": []})

        assert called == []

    def test_translate_endpoint_still_works_if_logging_raises(self, monkeypatch):
        """El logging de rendimiento es best-effort -- ver docstring de
        log_translation_run(). Un fallo ahí nunca debe tumbar la
        respuesta de traducción real."""
        monkeypatch.setattr(analysis_router, "translate_texts_local", lambda texts, lang="es": [t.upper() for t in texts])

        def _raising_log(**kwargs):
            raise RuntimeError("disco lleno")

        monkeypatch.setattr(analysis_router, "log_translation_run", _raising_log)

        resp = client.post("/api/analyze/translate-descriptions?lang=en", json={"texts": ["guitarra"]})

        assert resp.status_code == 200
        assert resp.json() == {"translations": ["GUITARRA"]}

    def test_lang_query_param_is_forwarded_to_analyze_report_with_ai(self, monkeypatch):
        received = {}

        async def _fake_analyze(report, lang="es"):
            received["lang"] = lang
            return {"verdict": "", "conclusions": []}

        monkeypatch.setattr(analysis_router, "analyze_report_with_ai", _fake_analyze)

        resp = client.post(
            "/api/analyze/ai-summary?lang=en", json=_make_report().model_dump(mode="json")
        )

        assert resp.status_code == 200
        assert received["lang"] == "en"

    def test_lang_defaults_to_es_when_not_given(self, monkeypatch):
        received = {}

        async def _fake_analyze(report, lang="es"):
            received["lang"] = lang
            return {"verdict": "", "conclusions": []}

        monkeypatch.setattr(analysis_router, "analyze_report_with_ai", _fake_analyze)

        resp = client.post("/api/analyze/ai-summary", json=_make_report().model_dump(mode="json"))

        assert resp.status_code == 200
        assert received["lang"] == "es"


class TestGeolocationRunsConcurrentlyFromTheStart:
    """Regresión directa: crear la tarea con `asyncio.create_task` no basta
    por sí solo -- si nunca se cede el control explícitamente, TODO el
    trabajo síncrono de _build_report (fingerprint, atributos, score) se
    ejecutaría antes de que la tarea de fotos llegara a arrancar de
    verdad (asyncio es cooperativo y de un solo hilo: una tarea "creada"
    no avanza hasta que el código que la creó cede el control), y el
    paralelismo "desde el principio" sería solo de nombre. Ver el
    `await asyncio.sleep(0)` añadido en analysis_router._build_report justo
    después de crear la tarea."""

    @pytest.mark.asyncio
    async def test_photo_task_gets_a_real_turn_before_synchronous_stages_finish(self, monkeypatch):
        order = []

        async def _fake_estimate_locations(posts, avatar_url=None, progress_callback=None):
            order.append("geo:started")
            return geolocation.GeolocationOutcome(index_available=True, results=[])

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate_locations)

        real_build_fingerprint = analysis_router.build_fingerprint

        def _tracking_build_fingerprint(posts):
            order.append("fingerprint:done")
            return WritingFingerprint(
                avg_sentence_length=0, vocabulary_richness=0, emoji_usage_rate=0,
                avg_posts_per_hour={}, top_groups=[], top_keywords=[], detected_language="es",
            )

        monkeypatch.setattr(analysis_router, "build_fingerprint", _tracking_build_fingerprint)

        posts = [
            SocialPost(
                id="p1", platform="instagram", type="image", group="viajes", tags=["viajes"],
                text="Foto de viaje", created_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
                score=1, permalink="https://ig/1", media_urls=["https://cdn.fake/1.jpg"],
            )
        ]
        profile = SocialProfile(platform="instagram", username="fake_user", posts=posts)

        await analysis_router._build_report(profile)

        # La tarea de fotos debe haber tenido su primer turno de ejecución
        # ANTES de que el trabajo síncrono del resto del pipeline termine
        # -- no solo haberse "creado" sin más.
        assert order == ["geo:started", "fingerprint:done"]

    @pytest.mark.asyncio
    async def test_profile_avatar_url_is_forwarded_to_the_background_geolocation_task(
        self, monkeypatch, patch_spacy_model
    ):
        received = {}

        async def _fake_estimate_locations(posts, avatar_url=None, progress_callback=None):
            received["avatar_url"] = avatar_url
            return geolocation.GeolocationOutcome(index_available=True, results=[])

        monkeypatch.setattr(geolocation, "estimate_locations_for_posts", _fake_estimate_locations)

        profile = SocialProfile(
            platform="instagram",
            username="fake_user",
            posts=[
                SocialPost(
                    id="p1", platform="instagram", type="image", group="viajes", tags=["viajes"],
                    text="Foto", created_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    score=1, permalink="https://ig/1", media_urls=[],
                )
            ],
            avatar_url="https://cdn.fake/avatar.jpg",
        )

        await analysis_router._build_report(profile)

        assert received["avatar_url"] == "https://cdn.fake/avatar.jpg"


class TestRecalculateEndpoint:
    """POST /api/analyze/recalculate -- rasgos fisicos manuales
    (color de ojos/pelo/piel) que no pueden inferirse automaticamente
    por proteccion de datos. Ver goal 'AmITraceable manual traits'."""

    def test_adds_manual_attribute_and_narrows_population(self):
        report = _make_report()

        resp = client.post(
            "/api/analyze/recalculate",
            json={
                "report": report.model_dump(mode="json"),
                "manual_attributes": [{"category": "color_ojos", "value": "verde"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()

        assert len(body["inferred_attributes"]) == 1
        assert body["inferred_attributes"][0]["category"] == "color_ojos"
        assert body["inferred_attributes"][0]["value"] == "verde"
        assert body["inferred_attributes"][0]["confidence"] == 1.0

        # Un unico paso nuevo de estrechamiento, marcado como manual.
        assert len(body["population_narrowing"]) == 1
        step = body["population_narrowing"][0]
        assert step["source"] == "manual"
        assert step["category"] == "color_ojos"
        assert step["remaining_population"] == pytest.approx(49_128_297 * 0.15)

        assert body["remaining_population_all_traits"] == step["remaining_population"]

    def test_multiple_manual_attributes_narrow_population_in_chain(self):
        report = _make_report()

        resp = client.post(
            "/api/analyze/recalculate",
            json={
                "report": report.model_dump(mode="json"),
                "manual_attributes": [
                    {"category": "color_ojos", "value": "verde"},
                    {"category": "color_pelo", "value": "rubio"},
                ],
            },
        )

        assert resp.status_code == 200
        body = resp.json()

        assert len(body["inferred_attributes"]) == 2
        assert len(body["population_narrowing"]) == 2

        expected = 49_128_297 * 0.15 * 0.10
        assert body["population_narrowing"][-1]["remaining_population"] == pytest.approx(expected)
        assert body["remaining_population_all_traits"] == pytest.approx(expected)

    def test_unknown_trait_value_marks_step_as_not_estimable(self):
        report = _make_report()

        resp = client.post(
            "/api/analyze/recalculate",
            json={
                "report": report.model_dump(mode="json"),
                "manual_attributes": [{"category": "color_ojos", "value": "valor_inexistente"}],
            },
        )

        assert resp.status_code == 200
        step = resp.json()["population_narrowing"][0]
        assert step["risk_level"] == "no_estimable"
        assert step["remaining_population"] is None

    def test_no_manual_attributes_returns_report_unchanged(self):
        report = _make_report()

        resp = client.post(
            "/api/analyze/recalculate",
            json={"report": report.model_dump(mode="json"), "manual_attributes": []},
        )

        assert resp.status_code == 200
        assert resp.json() == report.model_dump(mode="json")

    def test_malformed_body_returns_422(self):
        resp = client.post("/api/analyze/recalculate", json={"not": "valid"})
        assert resp.status_code == 422
