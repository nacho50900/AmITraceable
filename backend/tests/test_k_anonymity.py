from app.nlp.demographic_extraction import DemographicFindings
from app.scoring.k_anonymity import (
    PopulationNarrowingStep,
    _risk_level,
    estimate_population_narrowing,
)


class TestRiskLevel:
    def test_bajo_at_and_above_100000(self):
        assert _risk_level(100_000) == "bajo"
        assert _risk_level(1_000_000) == "bajo"

    def test_medio_between_1000_and_100000(self):
        assert _risk_level(1_000) == "medio"
        assert _risk_level(99_999) == "medio"

    def test_alto_between_20_and_1000(self):
        assert _risk_level(20) == "alto"
        assert _risk_level(999) == "alto"

    def test_critico_below_20(self):
        assert _risk_level(19) == "critico"
        assert _risk_level(0) == "critico"


class TestEstimatePopulationNarrowing:
    def test_no_findings_returns_empty_list(self):
        findings = DemographicFindings()
        assert estimate_population_narrowing(findings) == []

    def test_sexo_produces_step_with_texto_source_by_default(self):
        findings = DemographicFindings(sexo="mujer", evidence={"sexo": ["https://x/1"]}, source={"sexo": "texto"})

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        step = steps[0]
        assert step.category == "sexo"
        assert step.attribute_label == "Sexo: mujer"
        assert step.remaining_population is not None
        assert step.remaining_population > 0
        assert step.risk_level == "bajo"
        assert step.source == "texto"
        assert step.evidence == ["https://x/1"]

    def test_edad_narrows_population_further_than_sexo_alone(self):
        only_sexo = DemographicFindings(sexo="mujer")
        with_edad = DemographicFindings(sexo="mujer", edad=24)

        pop_sexo_only = estimate_population_narrowing(only_sexo)[-1].remaining_population
        pop_with_edad = estimate_population_narrowing(with_edad)[-1].remaining_population

        assert pop_with_edad < pop_sexo_only

    def test_full_cascade_narrows_monotonically_reddit_style_example(self):
        """Reproduce el ejemplo de la conversación: mujer, 24 años, vive en
        León, estudia Medicina -> la población restante debe decrecer en
        cada paso de la cascada."""
        findings = DemographicFindings(
            sexo="mujer",
            edad=24,
            municipio="leon",
            estudios="medicina",
            evidence={
                "sexo": ["https://x/1"],
                "edad": ["https://x/1"],
                "municipio": ["https://x/1"],
                "estudios": ["https://x/1"],
            },
        )

        steps = estimate_population_narrowing(findings)

        assert [s.category for s in steps] == ["sexo", "edad", "ubicacion", "estudios"]
        populations = [s.remaining_population for s in steps]
        # Estrictamente decreciente en cada paso de la cascada
        assert all(populations[i] > populations[i + 1] for i in range(len(populations) - 1))
        assert steps[-1].risk_level in ("alto", "critico")

    def test_municipio_takes_priority_over_provincia_when_both_present(self):
        findings = DemographicFindings(municipio="leon", provincia="madrid")

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert "municipio" in steps[0].attribute_label.lower()
        assert "León" in steps[0].attribute_label or "Leon" in steps[0].attribute_label

    def test_provincia_used_when_no_municipio(self):
        findings = DemographicFindings(provincia="madrid")

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert "provincia" in steps[0].attribute_label.lower()

    def test_unknown_value_not_in_reference_tables_is_no_estimable(self):
        findings = DemographicFindings(estudios="una_carrera_que_no_existe")

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].remaining_population is None
        assert steps[0].risk_level == "no_estimable"
        assert steps[0].note is not None

    def test_unknown_province_is_no_estimable_not_a_crash(self):
        findings = DemographicFindings(provincia="provincia_inventada")

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].remaining_population is None
        assert steps[0].risk_level == "no_estimable"

    def test_universidad_and_empresa_are_always_no_estimable(self):
        findings = DemographicFindings(universidad="Salamanca", empresa="Acme")

        steps = estimate_population_narrowing(findings)

        categories = {s.category: s for s in steps}
        assert categories["universidad"].remaining_population is None
        assert categories["universidad"].risk_level == "no_estimable"
        assert categories["empresa"].remaining_population is None
        assert categories["empresa"].risk_level == "no_estimable"

    def test_source_imagen_propagates_to_location_step_with_extra_note(self):
        findings = DemographicFindings(
            provincia="madrid",
            evidence={"provincia": ["https://ig/1"]},
            source={"provincia": "imagen"},
        )

        steps = estimate_population_narrowing(findings)

        assert steps[0].source == "imagen"
        assert "imagen" in steps[0].note.lower()

    def test_source_defaults_to_texto_when_not_specified(self):
        findings = DemographicFindings(sexo="hombre")  # sin dict `source` relleno

        steps = estimate_population_narrowing(findings)

        assert steps[0].source == "texto"

    def test_ocupacion_step_present_and_estimable_for_known_value(self):
        findings = DemographicFindings(ocupacion="docente")

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "ocupacion"
        assert steps[0].remaining_population is not None

    def test_evidence_defaults_to_empty_list_when_missing(self):
        findings = DemographicFindings(sexo="hombre")  # sin entrada en evidence

        steps = estimate_population_narrowing(findings)

        assert steps[0].evidence == []

    def test_order_of_steps_follows_pipeline_order(self):
        findings = DemographicFindings(
            sexo="hombre", edad=30, provincia="madrid", estudios="derecho", ocupacion="abogado"
        )

        steps = estimate_population_narrowing(findings)

        assert [s.category for s in steps] == ["sexo", "edad", "ubicacion", "estudios", "ocupacion"]
