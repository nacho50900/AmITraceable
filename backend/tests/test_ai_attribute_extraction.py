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
        "comunidad_autonoma": None,
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
    async def test_detects_multi_province_comunidad_autonoma(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(comunidad_autonoma="Canarias", evidence={"comunidad_autonoma": "bio"}),
            )
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="ana", bio="Vivo en Canarias")

        assert findings.comunidad_autonoma == "canarias"
        assert findings.provincia is None
        assert findings.source["comunidad_autonoma"] == "ia"
        assert findings.evidence["comunidad_autonoma"] == ["bio"]

    @pytest.mark.asyncio
    async def test_single_province_comunidad_autonoma_resolves_to_province(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(comunidad_autonoma="Región de Murcia"))
        )

        findings = await extract_demographics_with_ai([_post("Vivo en Murcia")], username="ana")

        assert findings.provincia == "murcia"
        assert findings.comunidad_autonoma is None

    @pytest.mark.asyncio
    async def test_provincia_wins_over_comunidad_autonoma_when_both_present(self, monkeypatch, respx_mock):
        """Si el modelo (por error o porque el texto lo permitía) devuelve
        ambos campos, la provincia concreta es más específica y gana."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200, json=_mock_content(provincia="Las Palmas", comunidad_autonoma="Canarias")
            )
        )

        findings = await extract_demographics_with_ai([_post("Vivo en Las Palmas, Canarias")], username="ana")

        assert findings.provincia == "las palmas"
        assert findings.comunidad_autonoma is None

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
    async def test_detects_travel_photos(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200, json=_mock_content(fotos_de_viaje=["https://x/1", "https://x/2"])
            )
        )

        findings = await extract_demographics_with_ai(
            [_post("De viaje por Italia", permalink="https://x/1"), _post("En casa", permalink="https://x/2")],
            username="ana",
        )

        assert findings.travel_permalinks == {"https://x/1", "https://x/2"}

    @pytest.mark.asyncio
    async def test_missing_fotos_de_viaje_field_defaults_to_empty_set(self, monkeypatch, respx_mock):
        """Si el LLM no devuelve el campo (o devuelve algo con forma
        inesperada), no debe romper -- se queda como conjunto vacío."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        findings = await extract_demographics_with_ai([_post("hola")], username="ana")

        assert findings.travel_permalinks == set()

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


class TestSoftInferences:
    """Inferencias BLANDAS (emojis, fechas, señales simbólicas) -- ver el
    ejemplo del aniversario en _SYSTEM_PROMPT. Van en su propia lista
    (`DemographicFindings.soft_inferences`), no en los campos de
    autodeclaración explícita."""

    @pytest.mark.asyncio
    async def test_parses_a_valid_soft_inference(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    inferencias_blandas=[
                        {
                            "categoria": "relacion_sentimental",
                            "valor": "Posible relación de pareja: la bio es solo una fecha con emojis de corazón",
                            "confianza": 0.6,
                            "evidencia": "bio",
                        }
                    ]
                ),
            )
        )

        findings = await extract_demographics_with_ai(
            [_post("hola")], username="ana", bio="18/05/20🧡👸✨"
        )

        assert len(findings.soft_inferences) == 1
        inferred = findings.soft_inferences[0]
        assert inferred.category == "relacion_sentimental"
        assert "pareja" in inferred.value.lower()
        assert inferred.confidence == 0.6
        assert inferred.evidence == ["bio"]

    @pytest.mark.asyncio
    async def test_missing_field_is_empty_list(self, monkeypatch, respx_mock):
        """Compatibilidad con respuestas de antes de este cambio (o un
        modelo que omita el campo): no debe romper, simplemente no hay
        inferencias blandas."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.soft_inferences == []

    @pytest.mark.asyncio
    async def test_entry_without_categoria_or_valor_is_skipped(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    inferencias_blandas=[
                        {"categoria": "", "valor": "algo", "confianza": 0.5},
                        {"categoria": "algo", "valor": "", "confianza": 0.5},
                        {"valor": "sin categoria", "confianza": 0.5},
                        "no es ni siquiera un dict",
                    ]
                ),
            )
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.soft_inferences == []

    @pytest.mark.asyncio
    async def test_confidence_out_of_range_is_clamped(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    inferencias_blandas=[
                        {"categoria": "a", "valor": "va sobre 1.5", "confianza": 1.5},
                        {"categoria": "b", "valor": "va bajo 0", "confianza": -3},
                    ]
                ),
            )
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.soft_inferences[0].confidence == 1.0
        assert findings.soft_inferences[1].confidence == 0.0

    @pytest.mark.asyncio
    async def test_missing_or_invalid_confidence_defaults_to_moderate_value(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    inferencias_blandas=[{"categoria": "a", "valor": "sin numero de confianza"}]
                ),
            )
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.soft_inferences[0].confidence == 0.5

    @pytest.mark.asyncio
    async def test_missing_evidence_defaults_to_empty_list_not_a_crash(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(inferencias_blandas=[{"categoria": "a", "valor": "sin evidencia"}]),
            )
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.soft_inferences[0].evidence == []

    @pytest.mark.asyncio
    async def test_caps_at_five_even_if_model_returns_more(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        many = [{"categoria": f"cat{i}", "valor": f"valor{i}", "confianza": 0.5} for i in range(9)]
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(inferencias_blandas=many))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert len(findings.soft_inferences) == 5

    @pytest.mark.asyncio
    async def test_not_a_list_is_ignored_without_crashing(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(inferencias_blandas="no es una lista"))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.soft_inferences == []


class TestEstadoCivil:
    """A diferencia de 'inferencias_blandas' (lista libre, solo va a
    inferred_attributes), 'estado_civil' es un campo dedicado con 3
    categorías (soltero/con_pareja/casado) que SÍ participa en el
    estimador de k-anonimato -- ver k_anonymity.py."""

    @pytest.mark.asyncio
    async def test_casado_is_parsed_with_ia_simbolica_source(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(estado_civil="casado", evidence={"estado_civil": "bio"}),
            )
        )

        findings = await extract_demographics_with_ai(
            [_post("hola")], username="x", bio="Mi marido y yo 💍"
        )

        assert findings.estado_civil == "casado"
        assert findings.evidence["estado_civil"] == ["bio"]
        # Nunca "ia" a secas: es una inferencia simbólica, no una
        # autodeclaración explícita -- la distinción importa para el
        # informe (ver k_anonymity.py -> nota de fiabilidad menor).
        assert findings.source["estado_civil"] == "ia_simbolica"

    @pytest.mark.asyncio
    async def test_con_pareja_and_soltero_and_viudo_are_parsed_too_not_just_casado(self, monkeypatch, respx_mock):
        for value in ("con_pareja", "soltero", "viudo"):
            respx_mock.post(MISTRAL_URL).mock(
                return_value=httpx.Response(200, json=_mock_content(estado_civil=value))
            )
            monkeypatch.setattr(settings, "mistral_api_key", "fake-key")

            findings = await extract_demographics_with_ai([_post("hola")], username="x")

            assert findings.estado_civil == value

    @pytest.mark.asyncio
    async def test_missing_field_stays_none(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.estado_civil is None
        assert "estado_civil" not in findings.source

    @pytest.mark.asyncio
    async def test_value_outside_the_three_categories_is_ignored_without_crashing(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(estado_civil="tal vez"))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.estado_civil is None

    def test_merge_findings_propagates_it_from_ai_to_regex_findings(self):
        regex_findings = DemographicFindings()
        ai_findings = DemographicFindings(
            estado_civil="casado",
            evidence={"estado_civil": ["bio"]},
            source={"estado_civil": "ia_simbolica"},
        )

        merged = merge_findings(regex_findings, ai_findings)

        assert merged.estado_civil == "casado"
        assert merged.source["estado_civil"] == "ia_simbolica"
