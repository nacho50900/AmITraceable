import json
from datetime import datetime, timezone

import pytest

from app import ai_analysis
from app.ai_analysis import AiAnalysisUnavailable, analyze_report_with_ai
from app.config import settings
from app.models.schemas import ExposureReport, PrivacyScore, WritingFingerprint
from app.nlp.ai_client import AIHTTPError, AIRequestError


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


def _verdict_body(**fields) -> dict:
    """Contenido ya parseado que `call_ai_json` devolvería tras el cambio
    a Qwen3.5-4B local -- antes este helper envolvía el contenido en la
    forma HTTP real de Mistral o Gemini (choices[...]/candidates[...])
    porque el mock operaba a nivel de transporte (respx); ahora
    `call_ai_json` ya hace ese des-envolvido internamente (ver
    app/nlp/ai_client.py), así que el mock solo necesita el dict final."""
    base = {"veredicto": "", "conclusiones": []}
    base.update(fields)
    return base


def _ai_returns(value: dict):
    async def fake(*args, **kwargs):
        return value

    return fake


def _ai_raises(exc: Exception):
    async def fake(*args, **kwargs):
        raise exc

    return fake


def _ai_spy(value: dict):
    """Como `_ai_returns`, pero acumula en `calls` los argumentos
    (system_prompt, user_prompt) de cada invocación -- para los tests que
    comprueban qué se le manda al modelo (antes inspeccionaban el cuerpo
    HTTP real capturado por respx)."""
    calls: list[tuple[str, str]] = []

    async def fake(system_prompt, user_prompt, *args, **kwargs):
        calls.append((system_prompt, user_prompt))
        return value

    return fake, calls


@pytest.fixture(autouse=True)
def enable_ai_analysis(monkeypatch):
    """`ai_key_configured` (ver app/config.py) exige tanto
    `enable_ai_analysis=True` como un `qwen_gguf_repo_id` no vacío -- ver
    el mismo fixture en tests/test_ai_attribute_extraction.py."""
    monkeypatch.setattr(settings, "enable_ai_analysis", True)
    monkeypatch.setattr(settings, "qwen_gguf_repo_id", "fake/repo")


class TestNoModelConfigured:
    @pytest.mark.asyncio
    async def test_raises_unavailable_when_ai_analysis_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "enable_ai_analysis", False)

        report = _make_report()
        with pytest.raises(AiAnalysisUnavailable):
            await analyze_report_with_ai(report)


class TestSuccessfulAnalysis:
    @pytest.mark.asyncio
    async def test_success_returns_verdict_and_conclusions(self, monkeypatch):
        monkeypatch.setattr(
            ai_analysis, "call_ai_json", _ai_returns(_verdict_body(veredicto="riesgo moderado", conclusiones=["a", "b"]))
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == "riesgo moderado"
        assert result["conclusions"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_empty_conclusions_list_is_valid(self, monkeypatch):
        monkeypatch.setattr(ai_analysis, "call_ai_json", _ai_returns(_verdict_body(conclusiones=[])))

        result = await analyze_report_with_ai(_make_report())

        assert result["conclusions"] == []

    @pytest.mark.asyncio
    async def test_non_string_items_in_conclusiones_are_filtered_out(self, monkeypatch):
        monkeypatch.setattr(
            ai_analysis, "call_ai_json", _ai_returns(_verdict_body(conclusiones=["válida", 42, None, "otra"]))
        )

        result = await analyze_report_with_ai(_make_report())

        assert result["conclusions"] == ["válida", "otra"]

    @pytest.mark.asyncio
    async def test_missing_veredicto_defaults_to_empty_string(self, monkeypatch):
        body = _verdict_body(conclusiones=["x"])
        del body["veredicto"]
        monkeypatch.setattr(ai_analysis, "call_ai_json", _ai_returns(body))

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == ""


class TestErrorHandling:
    """Ya no hay proveedor/URL que parametrizar (un único backend local,
    ver app/nlp/ai_client.py): las dos formas de fallo posibles ahora son
    que el modelo no pueda ejecutarse en absoluto (AIRequestError) o que
    responda con algo que no se pudo interpretar como JSON (AIHTTPError).
    Los antiguos casos 429 (cuota agotada del proveedor) y 401 (key
    inválida) ya no existen: no hay proveedor de terceros ni API key que
    pueda fallar de esas formas."""

    @pytest.mark.asyncio
    async def test_model_unavailable_raises_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            ai_analysis, "call_ai_json", _ai_raises(AIRequestError("llama_cpp no disponible"))
        )

        report = _make_report()
        with pytest.raises(AiAnalysisUnavailable):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_inference_failure_raises_unavailable_not_raw_exception(self, monkeypatch):
        monkeypatch.setattr(ai_analysis, "call_ai_json", _ai_raises(AIRequestError("fallo de inferencia")))

        report = _make_report()
        with pytest.raises(AiAnalysisUnavailable):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_malformed_response_raises_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            ai_analysis, "call_ai_json", _ai_raises(AIHTTPError(200, "respuesta con forma inesperada"))
        )

        report = _make_report()
        with pytest.raises(AiAnalysisUnavailable):
            await analyze_report_with_ai(report)

    @pytest.mark.asyncio
    async def test_unexpected_response_shape_defaults_gracefully(self, monkeypatch):
        """Un JSON válido pero sin los campos esperados no es un fallo de
        ai_client (el JSON SÍ se parseó bien) -- es la capa de parseo de
        analyze_report_with_ai la que debe degradarse con calma."""
        monkeypatch.setattr(ai_analysis, "call_ai_json", _ai_returns({"unexpected": "shape"}))

        result = await analyze_report_with_ai(_make_report())

        assert result["verdict"] == ""
        assert result["conclusions"] == []


class TestPromptContent:
    """Antes parametrizada por proveedor para comprobar el CUERPO/HEADERS
    HTTP reales de cada uno (forma de payload de Mistral vs. Gemini, modo
    JSON, autenticación) -- eso ahora vive dentro de app/nlp/ai_client.py.
    Aquí solo queda comprobar que analyze_report_with_ai() construye el
    PROMPT correctamente, algo independiente del backend que lo procese."""

    @pytest.mark.asyncio
    async def test_sends_report_json_and_system_prompt(self, monkeypatch):
        fake, calls = _ai_spy(_verdict_body())
        monkeypatch.setattr(ai_analysis, "call_ai_json", fake)

        await analyze_report_with_ai(_make_report(username="ana_gz"))

        system_prompt, user_prompt = calls[0]
        assert "ana_gz" in user_prompt
        assert system_prompt

    @pytest.mark.asyncio
    async def test_default_lang_es_does_not_alter_the_system_prompt(self, monkeypatch):
        fake, calls = _ai_spy(_verdict_body())
        monkeypatch.setattr(ai_analysis, "call_ai_json", fake)

        await analyze_report_with_ai(_make_report(), lang="es")

        assert "inglés" not in calls[0][0].lower()

    @pytest.mark.asyncio
    async def test_lang_en_appends_english_instruction_to_system_prompt(self, monkeypatch):
        fake, calls = _ai_spy(_verdict_body())
        monkeypatch.setattr(ai_analysis, "call_ai_json", fake)

        await analyze_report_with_ai(_make_report(), lang="en")

        assert "inglés" in calls[0][0].lower()

    @pytest.mark.asyncio
    async def test_unsupported_lang_falls_back_to_spanish_silently(self, monkeypatch):
        fake, calls = _ai_spy(_verdict_body())
        monkeypatch.setattr(ai_analysis, "call_ai_json", fake)

        await analyze_report_with_ai(_make_report(), lang="fr")

        assert "inglés" not in calls[0][0].lower()


class TestReportJsonForAi:
    """`_report_json_for_ai` -- el prompt del análisis final NO debe llevar la
    lista completa de coincidencias de Maigret (cientos de sitios dejaban sin
    contexto al modelo local y el análisis final no llegaba a generarse)."""

    @staticmethod
    def _report_with_matches(n: int):
        from app.models.schemas import UsernameCorrelationSummary, UsernameSiteMatch

        matches = [
            UsernameSiteMatch(site=f"forum.site{i}.com", url=f"https://forum.site{i}.com/u/nick", exists=True)
            for i in range(n)
        ]
        return _make_report(related_accounts=UsernameCorrelationSummary(total_sites_checked=5185, matches=matches))

    def test_related_accounts_are_summarised_not_listed(self):
        payload = json.loads(ai_analysis._report_json_for_ai(self._report_with_matches(300)))

        related = payload["related_accounts"]
        assert related["total_sites_checked"] == 5185
        assert related["accounts_found"] == 300  # el recuento real se conserva
        assert len(related["sites_sample"]) == ai_analysis._AI_MAX_RELATED_ACCOUNTS_SAMPLE
        assert "matches" not in related

    def test_urls_of_matches_are_not_sent(self):
        text = ai_analysis._report_json_for_ai(self._report_with_matches(5))

        assert "https://forum.site0.com/u/nick" not in text

    def test_prompt_size_barely_grows_with_the_number_of_matches(self):
        small = len(ai_analysis._report_json_for_ai(self._report_with_matches(5)))
        huge = len(ai_analysis._report_json_for_ai(self._report_with_matches(2000)))

        assert huge - small < 1000

    def test_json_is_compact(self):
        text = ai_analysis._report_json_for_ai(_make_report())

        assert "\n" not in text

    def test_report_without_related_accounts_is_unchanged(self):
        payload = json.loads(ai_analysis._report_json_for_ai(_make_report()))

        assert payload["related_accounts"] is None
        assert payload["username"] == "test_user"
