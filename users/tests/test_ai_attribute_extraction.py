from datetime import datetime, timezone

import httpx
import pytest

from app.config import settings
from app.models.schemas import SocialPost
from app.nlp.ai_attribute_extraction import extract_demographics_with_ai, merge_findings
from app.nlp.demographic_extraction import DemographicFindings, extract_demographics

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def _post(text: str, permalink: str = "https://x/1", i: str = "1") -> SocialPost:
    return SocialPost(
        id=i,
        platform="instagram",
        type="image",
        group="sin_etiqueta",
        tags=[],
        text=text,
        created_utc=datetime.now(timezone.utc),
        score=1,
        permalink=permalink,
    )


def _mock_content(**fields) -> dict:
    base = {
        "sexo": None,
        "edad": None,
        "provincia": None,
        "municipio": None,
        "estudios": None,
        "ocupacion": None,
        "universidad": None,
        "empresa": None,
        "sexo_por_nombre": None,
        "evidence": {},
    }
    base.update(fields)
    return {"choices": [{"message": {"content": __import__("json").dumps(base)}}]}


@pytest.fixture(autouse=True)
def reset_mistral_api_key(monkeypatch):
    monkeypatch.setattr(settings, "mistral_api_key", None)
    yield


class TestNoApiKeyOrEmptyInput:
    @pytest.mark.asyncio
    async def test_returns_empty_findings_without_api_key(self):
        findings = await extract_demographics_with_ai([_post("estudiante de enfermeria")], username="ana")
        assert findings == DemographicFindings()

    @pytest.mark.asyncio
    async def test_returns_empty_findings_when_nothing_to_send(self, monkeypatch):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        findings = await extract_demographics_with_ai([], username="")
        assert findings == DemographicFindings()


class TestSuccessfulExtraction:
    @pytest.mark.asyncio
    async def test_detects_estudios_missed_by_regex_vocabulary(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(estudios="enfermeria", evidence={"estudios": "https://x/1"}),
            )
        )

        findings = await extract_demographics_with_ai(
            [_post("Voy a 2o de Enfermeria y no doy abasto", permalink="https://x/1")],
            username="ana_gz",
        )

        assert findings.estudios == "enfermeria"
        assert findings.source["estudios"] == "ia"
        assert findings.evidence["estudios"] == ["https://x/1"]

    @pytest.mark.asyncio
    async def test_unrecognized_studies_value_is_not_estimated(self, monkeypatch, respx_mock):
        """El LLM propone un valor libre; si no coincide con ninguna clave del INE,
        no se acepta -- nunca se inventa una categoría no auditable."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(estudios="clarinete avanzado"))
        )

        findings = await extract_demographics_with_ai([_post("toco el clarinete")], username="x")

        assert findings.estudios is None

    @pytest.mark.asyncio
    async def test_detects_municipio_over_provincia(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(municipio="Leon", provincia="Leon", evidence={"municipio": "https://x/1"}),
            )
        )

        findings = await extract_demographics_with_ai([_post("vivo por leon")], username="x")

        assert findings.municipio == "leon"
        assert findings.provincia is None
        assert findings.source["municipio"] == "ia"

    @pytest.mark.asyncio
    async def test_edad_out_of_range_is_discarded(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content(edad=200)))

        findings = await extract_demographics_with_ai([_post("este puente tiene 200 años")], username="x")

        assert findings.edad is None

    @pytest.mark.asyncio
    async def test_free_text_fields_universidad_empresa(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    universidad="Oviedo",
                    empresa="Indra",
                    evidence={"universidad": "https://x/1", "empresa": "https://x/1"},
                ),
            )
        )

        findings = await extract_demographics_with_ai([_post("estudio en la universidad de Oviedo")], username="x")

        assert findings.universidad == "Oviedo"
        assert findings.empresa == "Indra"
        assert findings.source["universidad"] == "ia"


class TestSexoPorNombre:
    @pytest.mark.asyncio
    async def test_explicit_sexo_wins_over_name_guess(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(sexo="hombre", sexo_por_nombre="mujer", evidence={"sexo": "https://x/1"}),
            )
        )

        findings = await extract_demographics_with_ai([_post("soy hombre")], username="ana", full_name="Ana")

        assert findings.sexo == "hombre"
        assert findings.source["sexo"] == "ia"

    @pytest.mark.asyncio
    async def test_falls_back_to_name_guess_marked_with_distinct_source(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(sexo_por_nombre="mujer"))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="ana_gz", full_name="Ana García")

        assert findings.sexo == "mujer"
        assert findings.source["sexo"] == "ia_nombre"
        assert findings.evidence["sexo"] == ["nombre público de la cuenta"]

    @pytest.mark.asyncio
    async def test_profile_name_and_bio_are_sent_in_prompt(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        route = respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        await extract_demographics_with_ai(
            [_post("hola")], username="ana_gz", full_name="Ana García", bio="Enfermera en León"
        )

        sent_body = route.calls[0].request.content.decode()
        assert "Ana García" in sent_body
        assert "Enfermera en León" in sent_body


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_network_error_returns_empty_findings(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(side_effect=httpx.ConnectError("no network"))

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings == DemographicFindings()

    @pytest.mark.asyncio
    async def test_429_returns_empty_findings_without_raising(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(429))

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings == DemographicFindings()

    @pytest.mark.asyncio
    async def test_malformed_json_content_returns_empty_findings(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "no es json"}}]})
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings == DemographicFindings()

    @pytest.mark.asyncio
    async def test_unexpected_response_shape_returns_empty_findings(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json={"unexpected": "shape"}))

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings == DemographicFindings()


class TestMergeFindings:
    def test_regex_wins_when_both_detect_same_field(self):
        regex_findings = extract_demographics([_post("Tengo 24 años")])
        ai_findings = DemographicFindings(edad=99)
        ai_findings.source["edad"] = "ia"

        merged = merge_findings(regex_findings, ai_findings)

        assert merged.edad == 24
        assert merged.source["edad"] == "texto"

    def test_ai_fills_gap_regex_did_not_find(self):
        regex_findings = extract_demographics([_post("Hola a todos")])
        ai_findings = DemographicFindings(estudios="enfermeria")
        ai_findings.evidence["estudios"] = ["https://x/1"]
        ai_findings.source["estudios"] = "ia"

        merged = merge_findings(regex_findings, ai_findings)

        assert merged.estudios == "enfermeria"
        assert merged.source["estudios"] == "ia"
        assert merged.evidence["estudios"] == ["https://x/1"]

    def test_preserves_ia_nombre_source_tag_through_merge(self):
        regex_findings = extract_demographics([_post("Hola a todos")])
        ai_findings = DemographicFindings(sexo="mujer")
        ai_findings.evidence["sexo"] = ["nombre público de la cuenta"]
        ai_findings.source["sexo"] = "ia_nombre"

        merged = merge_findings(regex_findings, ai_findings)

        assert merged.sexo == "mujer"
        assert merged.source["sexo"] == "ia_nombre"
