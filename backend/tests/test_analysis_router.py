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
    def test_success_returns_conclusions(self, monkeypatch):
        async def _fake_analyze(report):
            return ["Conclusión 1", "Conclusión 2"]

        monkeypatch.setattr(analysis_router, "analyze_report_with_ai", _fake_analyze)

        resp = client.post("/api/analyze/ai-summary", json=_make_report().model_dump(mode="json"))

        assert resp.status_code == 200
        assert resp.json() == {"conclusions": ["Conclusión 1", "Conclusión 2"]}

    def test_unavailable_returns_503_not_500(self, monkeypatch):
        async def _fake_analyze(report):
            raise AiAnalysisUnavailable("no configurado")

        monkeypatch.setattr(analysis_router, "analyze_report_with_ai", _fake_analyze)

        resp = client.post("/api/analyze/ai-summary", json=_make_report().model_dump(mode="json"))

        assert resp.status_code == 503
        assert resp.json()["detail"] == "no configurado"

    def test_malformed_body_returns_422(self):
        resp = client.post("/api/analyze/ai-summary", json={"not": "a valid report"})
        assert resp.status_code == 422
