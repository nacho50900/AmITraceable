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


class TestEdadEstimadaPorTramo:
    """Estimación INDIRECTA de tramo de edad ('edad_estimada' en el prompt),
    distinta de una autodeclaración explícita -- ver
    ai_attribute_extraction.py::_set_edad_rango."""

    @pytest.mark.asyncio
    async def test_confianza_suficiente_se_convierte_en_tramo_ine(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(edad_estimada={"edad_aproximada": 27, "confianza": 0.8}),
            )
        )

        findings = await extract_demographics_with_ai(
            [_post("me gradue en 2019 y llevo un par de años currando")], username="x"
        )

        assert findings.edad is None
        assert findings.edad_rango == "25-29"
        assert findings.source["edad_rango"] == "ia_estimada"
        assert findings.confidence["edad_rango"] == 0.8

    @pytest.mark.asyncio
    async def test_confianza_moderada_ya_no_es_suficiente(self, monkeypatch, respx_mock):
        """Regresión (Comandante, agosto 2026): con el umbral antiguo (0.5)
        se colaban estimaciones claramente erróneas con confianza
        "moderada" (p. ej. 30 años estimados para alguien de 21 real). El
        umbral subió de 0.5 a 0.7 -- 0.6 (dentro del techo "conservador"
        que el propio prompt le pide al modelo para pistas no del todo
        claras) ya no debe ser suficiente."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(edad_estimada={"edad_aproximada": 27, "confianza": 0.6}),
            )
        )

        findings = await extract_demographics_with_ai([_post("alguna pista ambigua")], username="x")

        assert findings.edad_rango is None
        assert "edad_rango" not in findings.confidence

    @pytest.mark.asyncio
    async def test_confianza_baja_no_anade_nada(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(edad_estimada={"edad_aproximada": 27, "confianza": 0.3}),
            )
        )

        findings = await extract_demographics_with_ai([_post("texto ambiguo cualquiera")], username="x")

        assert findings.edad is None
        assert findings.edad_rango is None
        assert "edad_rango" not in findings.confidence

    @pytest.mark.asyncio
    async def test_edad_exacta_declarada_gana_sobre_el_tramo(self, monkeypatch, respx_mock):
        """Si el modelo devuelve AMBOS campos (edad exacta Y una estimación
        indirecta), la edad exacta es más precisa y se queda sola -- nunca
        conviven `edad` y `edad_rango` a la vez."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    edad=24,
                    edad_estimada={"edad_aproximada": 40, "confianza": 0.9},
                    evidence={"edad": "https://x/1"},
                ),
            )
        )

        findings = await extract_demographics_with_ai([_post("tengo 24 años", permalink="https://x/1")], username="x")

        assert findings.edad == 24
        assert findings.edad_rango is None

    @pytest.mark.asyncio
    async def test_edad_aproximada_fuera_de_rango_se_descarta(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(edad_estimada={"edad_aproximada": 500, "confianza": 0.9}),
            )
        )

        findings = await extract_demographics_with_ai([_post("texto cualquiera")], username="x")

        assert findings.edad_rango is None

    @pytest.mark.asyncio
    async def test_edad_estimada_null_no_anade_nada(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(edad_estimada=None))
        )

        findings = await extract_demographics_with_ai([_post("texto cualquiera")], username="x")

        assert findings.edad_rango is None

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
    async def test_con_pareja_and_soltero_and_viudo_and_divorciado_are_parsed_too_not_just_casado(
        self, monkeypatch, respx_mock
    ):
        for value in ("con_pareja", "soltero", "viudo", "divorciado"):
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


class TestGroupBFieldsFromAI:
    """nacionalidad, situacion_laboral, tipo_hogar y lengua_materna: a
    diferencia de estado_civil (razonamiento simbólico), el prompt le pide
    a la IA un valor EXACTO de un enum cerrado -- mismo patrón que sexo/
    edad, solo que aquí es la IA quien detecta la autodeclaración en vez de
    la regex (p.ej. porque está formulada de una manera que la regex no
    cubre). Ver `_set_exact_enum` en ai_attribute_extraction.py."""

    @pytest.mark.asyncio
    async def test_nacionalidad_is_parsed(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(nacionalidad="extranjera"))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.nacionalidad == "extranjera"
        assert findings.source["nacionalidad"] == "ia"

    @pytest.mark.asyncio
    async def test_situacion_laboral_is_parsed(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(situacion_laboral="jubilado"))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.situacion_laboral == "jubilado"

    @pytest.mark.asyncio
    async def test_tipo_hogar_is_parsed(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(tipo_hogar="monoparental"))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.tipo_hogar == "monoparental"

    @pytest.mark.asyncio
    async def test_lengua_materna_is_parsed(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(lengua_materna="euskera"))
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.lengua_materna == "euskera"

    @pytest.mark.asyncio
    async def test_value_outside_enum_is_ignored_without_crashing(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_mock_content(
                    nacionalidad="marciana", situacion_laboral="pirata", tipo_hogar="castillo", lengua_materna="klingon"
                ),
            )
        )

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.nacionalidad is None
        assert findings.situacion_laboral is None
        assert findings.tipo_hogar is None
        assert findings.lengua_materna is None

    @pytest.mark.asyncio
    async def test_missing_fields_stay_none(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mock_content()))

        findings = await extract_demographics_with_ai([_post("hola")], username="x")

        assert findings.nacionalidad is None
        assert findings.situacion_laboral is None
        assert findings.tipo_hogar is None
        assert findings.lengua_materna is None


class TestOrientacionSexualReligionSignoZodiacal:
    """Regresión: estos tres campos se guardaban tal cual devolviera el
    modelo, sin validar contra ningún vocabulario cerrado -- un valor
    inventado o mal formado se colaba en DemographicFindings y luego
    k_anonymity.py simplemente no encontraba proporción (silencioso, sin
    avisar de que el dato era basura). Ahora se validan igual que
    nacionalidad/situacion_laboral/etc (ver _SEXUAL_ORIENTATION_VALUES /
    _RELIGION_VALUES / _set_signo_zodiacal)."""

    @pytest.mark.asyncio
    async def test_valid_orientacion_sexual_is_kept(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(orientacion_sexual="bisexual"))
        )
        findings = await extract_demographics_with_ai([_post("soy bisexual")], username="x")
        assert findings.orientacion_sexual == "bisexual"

    @pytest.mark.asyncio
    async def test_invalid_orientacion_sexual_is_discarded(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(orientacion_sexual="queer"))
        )
        findings = await extract_demographics_with_ai([_post("texto cualquiera")], username="x")
        assert findings.orientacion_sexual is None

    @pytest.mark.asyncio
    async def test_valid_religion_is_kept(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(religion="budismo"))
        )
        findings = await extract_demographics_with_ai([_post("soy budista")], username="x")
        assert findings.religion == "budismo"

    @pytest.mark.asyncio
    async def test_invalid_religion_is_discarded(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(religion="pastafarismo"))
        )
        findings = await extract_demographics_with_ai([_post("texto cualquiera")], username="x")
        assert findings.religion is None

    @pytest.mark.asyncio
    async def test_signo_zodiacal_is_normalized_to_canonical_format(self, monkeypatch, respx_mock):
        """El modelo puede devolver el rango con capitalización o espacios
        distintos al ejemplo del prompt; debe normalizarse siempre al
        mismo formato canónico que usa la detección por regex/emoji."""
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(signo_zodiacal="Escorpio (23 oct - 21 nov)"))
        )
        findings = await extract_demographics_with_ai([_post("soy escorpio")], username="x")
        assert findings.signo_zodiacal == "escorpio (23 oct - 21 nov)"

    @pytest.mark.asyncio
    async def test_signo_zodiacal_with_only_the_sign_name_is_normalized(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(signo_zodiacal="Aries"))
        )
        findings = await extract_demographics_with_ai([_post("soy aries")], username="x")
        assert findings.signo_zodiacal == "aries (21 mar - 19 abr)"

    @pytest.mark.asyncio
    async def test_invalid_signo_zodiacal_is_discarded(self, monkeypatch, respx_mock):
        monkeypatch.setattr(settings, "mistral_api_key", "fake-key")
        respx_mock.post(MISTRAL_URL).mock(
            return_value=httpx.Response(200, json=_mock_content(signo_zodiacal="ofiuco"))
        )
        findings = await extract_demographics_with_ai([_post("texto cualquiera")], username="x")
        assert findings.signo_zodiacal is None
