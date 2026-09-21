import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from app import ai_analysis
from app.ai_analysis import AiAnalysisUnavailable, analyze_report_with_ai
from app.config import settings
from app.models.schemas import ExposureReport, PrivacyScore, WritingFingerprint

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
# Debe coincidir con GEMINI_API_URL_TEMPLATE.format(model=settings.gemini_model)
# en app/nlp/ai_client.py -- si cambia el modelo por defecto ahí, cambia
# aquí también (o el mock deja de interceptar la petición real).
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


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


def _mistral_body(**fields) -> dict:
    """Forma de respuesta de la API de Mistral (chat.completions, estilo
    OpenAI): choices[0].message.content lleva el JSON como string."""
    base = {"veredicto": "", "conclusiones": []}
    base.update(fields)
    return {"choices": [{"message": {"content": json.dumps(base)}}]}


def _gemini_body(**fields) -> dict:
    """Forma de respuesta de la API de Gemini (generateContent):
    candidates[0].content.parts[0].text lleva el JSON como string --
    estructura completamente distinta a la de Mistral (ver
    _parse_gemini_content en app/nlp/ai_client.py)."""
    base = {"veredicto": "", "conclusiones": []}
    base.update(fields)
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(base)}]}}]}


class _Provider:
    """Agrupa lo que cambia entre proveedores para un test dado: la URL
    que hay que mockear con respx y cómo envolver el contenido JSON en la
    forma de respuesta real de cada uno."""

    __slots__ = ("name", "url", "wrap")

    def __init__(self, name: str, url: str, wrap):
        self.name = name
        self.url = url
        self.wrap = wrap


_PROVIDERS = {
    "mistral": _Provider("mistral", MISTRAL_URL, _mistral_body),
    "gemini": _Provider("gemini", GEMINI_URL, _gemini_body),
}


@pytest.fixture(params=["mistral", "gemini"])
def provider(request, monkeypatch) -> _Provider:
    """Parametriza el test que lo pida para correr una vez contra CADA
    backend de app/nlp/ai_client.py -- antes de este cambio, todo este
    archivo mockeaba únicamente MISTRAL_URL, así que analyze_report_with_ai()
    nunca se había validado de verdad contra la forma de respuesta real de
    Gemini (el proveedor por defecto desde AI_PROVIDER, ver app/config.py),
    solo contra la de Mistral. Deja la API key del proveedor activo puesta
    y la del otro a None, para que un test que se equivoque de URL falle
    con un mensaje claro ("no está configurado") en vez de colarse."""
    p = _PROVIDERS[request.param]
    monkeypatch.setattr(settings, "ai_provider", p.name)
    monkeypatch.setattr(settings, "mistral_api_key", "fake-key" if p.name == "mistral" else None)
    monkeypatch.setattr(settings, "gemini_api_key", "fake-key" if p.name == "gemini" else None)
    return p


@pytest.fixture(autouse=True)
def reset_ai_provider_settings(monkeypatch):
    """Ningún test parte con una key real de ningún proveedor puesta (ni
    depende del .env real, que en CI puede o no tener alguna configurada)
    -- los tests que sí necesitan una key la piden explícitamente vía el
    fixture `provider` de arriba (o, en los pocos que solo hablan de
    Mistral en concreto, la ponen ellos mismos)."""
    monkeypatch.setattr(settings, "mistral_api_key", None)
    monkeypatch.setattr(settings, "gemini_api_key", None)


@pytest.fixture(autouse=True)
def fast_retry_backoff(monkeypatch):
    """El reintento ante 429 (ver app/nlp/ai_client.py, ADR-45) espera de
    verdad _RETRY_BACKOFF_SECONDS (1.5s) en producción -- aquí se
    sustituye asyncio.sleep por un no-op para que los tests que lo
    disparan (ver test_429_raises_unavailable_after_one_retry) no tarden
    1.5s cada uno de verdad."""
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


class TestAnalyzeReportWithAi:
    """Comportamiento de analyze_report_with_ai() que debe ser IDÉNTICO
    sea cual sea el proveedor activo -- parametrizado con el fixture
    `provider` (Mistral y Gemini) en vez de mockear solo MISTRAL_URL como
    antes de este cambio."""

    @pytest.mark.asyncio
    async def test_raises_when_no_api_key_configured(self):
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="no está configurado"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_success_returns_verdict_and_conclusions(self, provider, respx_mock):
        respx_mock.post(provider.url).mock(
            return_value=httpx.Response(
                200,
                json=provider.wrap(
                    veredicto="Este perfil no comparte información que permita identificarte con facilidad.",
                    conclusiones=["Cuidado con la ubicación.", "Revisa tus hashtags."],
                ),
            )
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == "Este perfil no comparte información que permita identificarte con facilidad."
        assert result["conclusions"] == ["Cuidado con la ubicación.", "Revisa tus hashtags."]

    @pytest.mark.asyncio
    async def test_empty_conclusions_list_is_valid(self, provider, respx_mock):
        respx_mock.post(provider.url).mock(
            return_value=httpx.Response(200, json=provider.wrap(veredicto="Riesgo bajo en general."))
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == "Riesgo bajo en general."
        assert result["conclusions"] == []

    @pytest.mark.asyncio
    async def test_non_string_items_in_conclusiones_are_filtered_out(self, provider, respx_mock):
        respx_mock.post(provider.url).mock(
            return_value=httpx.Response(
                200, json=provider.wrap(conclusiones=["Válida", 42, None, "  ", "Otra válida"])
            )
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["conclusions"] == ["Válida", "Otra válida"]

    @pytest.mark.asyncio
    async def test_missing_veredicto_defaults_to_empty_string(self, provider, respx_mock):
        # Cuerpo mínimo a mano (sin veredicto) en la forma real de cada
        # proveedor -- provider.wrap() siempre rellena veredicto, así que
        # aquí no se puede reutilizar para probar justo su ausencia.
        if provider.name == "mistral":
            body = {"choices": [{"message": {"content": '{"conclusiones": []}'}}]}
        else:
            body = {"candidates": [{"content": {"parts": [{"text": '{"conclusiones": []}'}]}}]}
        respx_mock.post(provider.url).mock(return_value=httpx.Response(200, json=body))

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == ""
        assert result["conclusions"] == []

    @pytest.mark.asyncio
    async def test_429_raises_unavailable_after_one_retry(self, provider, respx_mock):
        """ADR-45: ante un 429 puntual (típico del límite de RÁFAGA del
        free tier, no de la cuota mensual), call_ai_json reintenta UNA vez
        antes de rendirse -- ya no es "cero reintentos" como antes de ese
        cambio. Si el reintento TAMBIÉN da 429 (este caso, el mock
        siempre responde 429), se rinde con el mismo mensaje de siempre,
        pero tras exactamente 2 llamadas HTTP, no 1."""
        route = respx_mock.post(provider.url).mock(return_value=httpx.Response(429))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="límite del plan gratuito"):
            await analyze_report_with_ai(report)

        # Exactamente un reintento: dos llamadas HTTP, no una ni tres.
        assert route.call_count == 2

    @pytest.mark.asyncio
    async def test_401_raises_unavailable_with_invalid_key_message(self, provider, respx_mock):
        respx_mock.post(provider.url).mock(return_value=httpx.Response(401))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="no es válida"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_other_4xx_5xx_raises_unavailable_with_status_code(self, provider, respx_mock):
        respx_mock.post(provider.url).mock(return_value=httpx.Response(500))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="500"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_network_error_raises_unavailable_not_raw_exception(self, provider, respx_mock):
        respx_mock.post(provider.url).mock(side_effect=httpx.ConnectError("no network"))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="No se pudo contactar"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_malformed_response_body_raises_unavailable(self, provider, respx_mock):
        respx_mock.post(provider.url).mock(return_value=httpx.Response(200, json={"unexpected": "shape"}))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="Respuesta inesperada"):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_non_json_content_raises_unavailable(self, provider, respx_mock):
        if provider.name == "mistral":
            body = {"choices": [{"message": {"content": "esto no es json"}}]}
        else:
            body = {"candidates": [{"content": {"parts": [{"text": "esto no es json"}]}}]}
        respx_mock.post(provider.url).mock(return_value=httpx.Response(200, json=body))
        report = _make_report()

        with pytest.raises(AiAnalysisUnavailable, match="Respuesta inesperada"):
            await analyze_report_with_ai(report)


class TestAnalyzeReportWithAiMistralRequestShape:
    """Forma exacta de la petición que se manda a Mistral (payload estilo
    OpenAI: messages/response_format, cabecera Authorization: Bearer) --
    intencionadamente específica de Mistral, no parametrizada: el
    equivalente para Gemini vive en TestAnalyzeReportWithAiGeminiRequestShape,
    con sus propias aserciones, porque la forma de la petición de cada
    proveedor es distinta de raíz (ver _post_mistral/_post_gemini en
    app/nlp/ai_client.py) -- forzar las mismas aserciones sobre las dos
    no tiene sentido."""

    @pytest.mark.asyncio
    async def test_sends_report_json_and_system_prompt_in_payload(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "ai_provider", "mistral")
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mistral_body()))

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
        monkeypatch.setattr(settings, "ai_provider", "mistral")
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mistral_body()))

        report = _make_report(recommendations=["Recomendación de prueba muy concreta."])
        await analyze_report_with_ai(report)

        sent_body = route.calls[0].request.content.decode()
        assert "Recomendación de prueba muy concreta." in sent_body

    @pytest.mark.asyncio
    async def test_requests_json_object_response_format(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "ai_provider", "mistral")
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mistral_body()))

        await analyze_report_with_ai(_make_report())

        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_sends_bearer_authorization_header(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "ai_provider", "mistral")
        monkeypatch.setattr(settings, "mistral_api_key", "secret-123")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mistral_body()))

        await analyze_report_with_ai(_make_report())

        assert route.calls[0].request.headers["Authorization"] == "Bearer secret-123"


class TestAnalyzeReportWithAiGeminiRequestShape:
    """Equivalente Gemini de TestAnalyzeReportWithAiMistralRequestShape --
    misma cobertura de intención (informe/prompt en el payload, modo JSON
    pedido, la key llega en la petición), con la forma real de la
    petición de Gemini: system_instruction/contents en vez de messages,
    generationConfig.responseMimeType en vez de response_format, la key
    en la cabecera x-goog-api-key (no Authorization: Bearer) -- ver
    _post_gemini en app/nlp/ai_client.py."""

    @pytest.mark.asyncio
    async def test_sends_report_json_and_system_prompt_in_payload(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "ai_provider", "gemini")
        monkeypatch.setattr(settings, "gemini_api_key", "fake-key")
        route = respx_mock.post(GEMINI_URL).mock(return_value=httpx.Response(200, json=_gemini_body()))

        report = _make_report()
        await analyze_report_with_ai(report)

        sent_body = route.calls[0].request.content.decode()
        assert report.username in sent_body
        assert "<informe>" in sent_body
        assert ai_analysis._SYSTEM_PROMPT[:20] in sent_body

    @pytest.mark.asyncio
    async def test_requests_json_response_mime_type(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "ai_provider", "gemini")
        monkeypatch.setattr(settings, "gemini_api_key", "fake-key")
        route = respx_mock.post(GEMINI_URL).mock(return_value=httpx.Response(200, json=_gemini_body()))

        await analyze_report_with_ai(_make_report())

        sent_payload = json.loads(route.calls[0].request.content)
        assert sent_payload["generationConfig"]["responseMimeType"] == "application/json"

    @pytest.mark.asyncio
    async def test_sends_api_key_in_goog_header_not_authorization(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "ai_provider", "gemini")
        monkeypatch.setattr(settings, "gemini_api_key", "secret-123")
        route = respx_mock.post(GEMINI_URL).mock(return_value=httpx.Response(200, json=_gemini_body()))

        await analyze_report_with_ai(_make_report())

        assert route.calls[0].request.headers["x-goog-api-key"] == "secret-123"
        # La key nunca debe acabar también en la URL (evita que quede en
        # logs de proxy que registren la URL completa de la petición) --
        # ver el comentario de _post_gemini en app/nlp/ai_client.py.
        assert "secret-123" not in str(route.calls[0].request.url)


class TestAnalyzeReportWithAiLanguage:
    """`lang` decide en qué idioma responde la IA -- se añade una instrucción
    al prompt de sistema en la MISMA llamada (ver docstring de
    _LANGUAGE_INSTRUCTIONS), no una segunda llamada de traducción.
    Parametrizado por proveedor: la instrucción de idioma se añade al
    mismo `_SYSTEM_PROMPT` independientemente de quién reciba la
    petición después."""

    @pytest.mark.asyncio
    async def test_default_lang_es_does_not_alter_the_system_prompt(self, provider, respx_mock):
        route = respx_mock.post(provider.url).mock(return_value=httpx.Response(200, json=provider.wrap()))

        await analyze_report_with_ai(_make_report())

        sent_payload = json.loads(route.calls[0].request.content)
        system_content = (
            sent_payload["messages"][0]["content"]
            if provider.name == "mistral"
            else sent_payload["system_instruction"]["parts"][0]["text"]
        )
        assert system_content == ai_analysis._SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_lang_en_appends_english_instruction_to_system_prompt(self, provider, respx_mock):
        route = respx_mock.post(provider.url).mock(return_value=httpx.Response(200, json=provider.wrap()))

        await analyze_report_with_ai(_make_report(), lang="en")

        sent_payload = json.loads(route.calls[0].request.content)
        system_content = (
            sent_payload["messages"][0]["content"]
            if provider.name == "mistral"
            else sent_payload["system_instruction"]["parts"][0]["text"]
        )
        assert system_content.startswith(ai_analysis._SYSTEM_PROMPT)
        assert "INGLÉS" in system_content

    @pytest.mark.asyncio
    async def test_unsupported_lang_falls_back_to_spanish_silently(self, provider, respx_mock):
        """Un valor de `lang` desconocido (typo, idioma no soportado por la
        webapp) no debe romper la llamada -- es una preferencia, no un
        contrato; se sirve en español sin más."""
        route = respx_mock.post(provider.url).mock(return_value=httpx.Response(200, json=provider.wrap()))

        await analyze_report_with_ai(_make_report(), lang="fr")

        sent_payload = json.loads(route.calls[0].request.content)
        system_content = (
            sent_payload["messages"][0]["content"]
            if provider.name == "mistral"
            else sent_payload["system_instruction"]["parts"][0]["text"]
        )
        assert system_content == ai_analysis._SYSTEM_PROMPT
