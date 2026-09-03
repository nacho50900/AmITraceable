import json
from datetime import datetime, timezone

import httpx
import pytest

from app import ai_analysis
from app.ai_analysis import AiAnalysisUnavailable, analyze_report_with_ai
from app.config import settings
from app.models.schemas import ExposureReport, PrivacyScore, WritingFingerprint

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def _make_report(**overrides) -> ExposureReport:
    base = dict(
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
    base.update(overrides)
    return ExposureReport(**base)


def _mock_content(**fields) -> dict:
    base = {"veredicto": "", "conclusiones": []}
    base.update(fields)
    return {"choices": [{"message": {"content": json.dumps(base)}}]}


@pytest.fixture(autouse=True)
def reset_mistral_api_key(monkeypatch):
    """Cada test parte de una API key controlada explícitamente, en vez de
    depender del .env real (que en CI puede o no tener MISTRAL_API_KEY)."""
    monkeypatch.setattr(settings, "mistral_api_key", None)


class TestAnalyzeReportWithAi:
    @pytest.mark.asyncio
    async def test_raises_when_no_api_key_configured(self):
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="no está configurado"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_success_returns_verdict_and_conclusions(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    veredicto="Este perfil no comparte información que permita identificarte con facilidad.",
                    conclusiones=["Cuidado con la ubicación.", "Revisa tus hashtags."],
                ),
            )
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == "Este perfil no comparte información que permita identificarte con facilidad."
        assert result["conclusions"] == ["Cuidado con la ubicación.", "Revisa tus hashtags."]

    @pytest.mark.asyncio
    async def test_empty_conclusions_list_is_valid(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(veredicto="Riesgo bajo en general."))
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == "Riesgo bajo en general."
        assert result["conclusions"] == []

    @pytest.mark.asyncio
    async def test_non_string_items_in_conclusiones_are_filtered_out(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200, json=_mock_content(conclusiones=["Válida", 42, None, "  ", "Otra válida"])
            )
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["conclusions"] == ["Válida", "Otra válida"]

    @pytest.mark.asyncio
    async def test_missing_veredicto_defaults_to_empty_string(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": '{"conclusiones": []}'}}]})
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == ""
        assert result["conclusions"] == []

    @pytest.mark.asyncio
    async def test_429_raises_unavailable_without_retrying(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(429))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="límite del plan gratuito"):
            await analyze_report_with_ai(report)

        # Ni un solo reintento: exactamente una llamada HTTP.
        assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_401_raises_unavailable_with_invalid_key_message(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(401))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="no es válida"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_other_4xx_5xx_raises_unavailable_with_status_code(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(500))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="500"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_network_error_raises_unavailable_not_raw_exception(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(side_effect=httpx.ConnectError("no network"))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="No se pudo contactar"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_malformed_response_body_raises_unavailable(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json={"unexpected": "shape"}))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="Respuesta inesperada"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_non_json_content_raises_unavailable(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "esto no es json"}}]})
        )
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="Respuesta inesperada"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_sends_report_json_and_system_prompt_in_payload(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        report = _make_report()
        await analyze_report_with_ai(report)

        sent_body = route.calls[0].request.content.decode()
        assert report.username in sent_body
        assert "<informe>" in sent_body
        assert ai_analysis._SYSTEM_PROMPT[:20] in sent_body

    @pytest.mark.asyncio
    async def test_sends_recommendations_as_part_of_the_report_json(self, monkeypatch, respx_mock):
        """report.recommendations ya no se muestra como sección propia en el
        dashboard, pero se le sigue pasando a la IA como parte del informe
        (el prompt le pide explícitamente que las use de base)."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        report = _make_report(recommendations=["Recomendación de prueba muy concreta."])
        await analyze_report_with_ai(report)

        sent_body = route.calls[0].request.content.decode()
        assert "Recomendación de prueba muy concreta." in sent_body

    @pytest.mark.asyncio
    async def test_requests_json_object_response_format(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        await analyze_report_with_ai(_make_report())

        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_sends_bearer_authorization_header(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "secret-123")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        await analyze_report_with_ai(_make_report())

        assert route.calls[0].request.headers["Authorization"] == "Bearer secret-123"


class TestAnalyzeReportWithAiLanguage:
    """`lang` decide en qué idioma responde la IA -- se añade una instrucción
    al prompt de sistema en la MISMA llamada (ver docstring de
    _LANGUAGE_INSTRUCTIONS), no una segunda llamada de traducción."""

    @pytest.mark.asyncio
    async def test_default_lang_es_does_not_alter_the_system_prompt(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        await analyze_report_with_ai(_make_report())

        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["messages"][0]["content"] == ai_analysis._SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_lang_en_appends_english_instruction_to_system_prompt(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        await analyze_report_with_ai(_make_report(), lang="en")

        sent_payload = json.loads(route.calls[0].request.content)
        system_content = sent_payload["messages"][0]["content"]
        assert system_content.startswith(ai_analysis._SYSTEM_PROMPT)
        assert "INGLÉS" in system_content

    @pytest.mark.asyncio
    async def test_unsupported_lang_falls_back_to_spanish_silently(self, monkeypatch, respx_mock):
        """Un valor de `lang` desconocido (typo, idioma no soportado por la
        webapp) no debe romper la llamada -- es una preferencia, no un
        contrato; se sirve en español sin más."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        await analyze_report_with_ai(_make_report(), lang="fr")

        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["messages"][0]["content"] == ai_analysis._SYSTEM_PROMPT

