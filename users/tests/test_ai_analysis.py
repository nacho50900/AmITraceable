from datetime import datetime, timezone

import httpx
import pytest

from app import ai_analysis
from app.ai_analysis import AiAnalysisUnavailable, analyze_report_with_ai
from app.config import settings
from app.models.schemas import ExposureReport, PrivacyScore, WritingFingerprint


def _make_report() -> ExposureReport:
    return ExposureReport(
        platform="instagram",
        username="test_user",
        generated_at=datetime.now(timezone.utc),
        n_posts_analyzed=3,
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
        ),
        recommendations=[],
        population_narrowing=[],
        image_location_points=[],
    )


@pytest.fixture(autouse=True)
def reset_mistral_api_key(monkeypatch):
    """Cada test parte de una API key controlada explícitamente, en vez de
    depender del .env real (que en CI puede o no tener MISTRAL_API_KEY)."""
    monkeypatch.setattr(settings, "mistral_api_key", None)
    yield


class TestAnalyzeReportWithAi:
    @pytest.mark.asyncio
    async def test_raises_when_no_api_key_configured(self):
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="no está configurado"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_success_parses_numbered_list_into_clean_conclusions(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "1. Cuidado con la ubicación.\n2. Revisa tus hashtags.\n3. Varía tus horarios."
                            }
                        }
                    ]
                },
            )
        )

        result = await analyze_report_with_ai(_make_report())

        assert result == [
            "Cuidado con la ubicación.",
            "Revisa tus hashtags.",
            "Varía tus horarios.",
        ]

    @pytest.mark.asyncio
    async def test_success_strips_bullet_and_dash_markers(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "- Primera cosa\n• Segunda cosa"}}]}
            )
        )

        result = await analyze_report_with_ai(_make_report())

        assert result == ["Primera cosa", "Segunda cosa"]

    @pytest.mark.asyncio
    async def test_falls_back_to_raw_text_when_no_lines_survive_cleanup(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        # Contenido compuesto solo por caracteres que el limpiador recorta
        # (números, puntos, guiones), sin texto real -- caso límite.
        respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "123. - •"}}]})
        )

        result = await analyze_report_with_ai(_make_report())

        assert result == [""] or result == ["123. - •"]

    @pytest.mark.asyncio
    async def test_429_raises_unavailable_without_retrying(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(429)
        )
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="límite del plan gratuito"):
            await analyze_report_with_ai(report)

        # Ni un solo reintento: exactamente una llamada HTTP.
        assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_401_raises_unavailable_with_invalid_key_message(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(return_value=httpx.Response(401))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="no es válida"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_other_4xx_5xx_raises_unavailable_with_status_code(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(return_value=httpx.Response(500))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="500"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_network_error_raises_unavailable_not_raw_exception(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("no network")
        )
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="No se pudo contactar"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_malformed_response_body_raises_unavailable(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"unexpected": "shape"})
        )
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="Respuesta inesperada"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_sends_report_json_and_system_prompt_in_payload(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Conclusión."}}]})
        )

        report = _make_report()
        await analyze_report_with_ai(report)

        sent_body = route.calls[0].request.content.decode()
        assert report.username in sent_body
        assert "<informe>" in sent_body
        assert ai_analysis._SYSTEM_PROMPT[:20] in sent_body

    @pytest.mark.asyncio
    async def test_sends_bearer_authorization_header(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "secret-123")
        route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "Conclusión."}}]})
        )

        await analyze_report_with_ai(_make_report())

        assert route.calls[0].request.headers["Authorization"] == "Bearer secret-123"
