import pytest

from app.nlp.demographic_extraction import DemographicFindings
from app.data.ine_reference import MARITAL_STATUS_BY_SEX, MARITAL_STATUS_DISTRIBUTION, TOTAL_POPULATION_ES
from app.scoring.k_anonymity import (
    PopulationNarrowingStep,
    _risk_level,
    estimate_population_narrowing,
    final_remaining_population,
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
        # 1 - 0.508 (proporción INE de mujeres) = 49.2% de reducción
        # respecto al total de España (es el primer escalón de la cadena).
        assert step.reduction_percent == 49.2

    def test_reduction_percent_is_relative_to_previous_step_not_to_the_total(self):
        """El segundo escalón debe reducir respecto a lo que quedaba TRAS
        el primero, no respecto a la población total de España desde
        cero -- por eso no se puede derivar de `proportion` sin más."""
        findings = DemographicFindings(sexo="mujer", edad=24)

        steps = estimate_population_narrowing(findings)

        sexo_step = next(s for s in steps if s.category == "sexo")
        edad_step = next(s for s in steps if s.category == "edad")

        # El % de reducción de la edad, aplicado sobre lo que quedaba tras
        # el sexo, debe coincidir con la caída real observada entre ambos
        # escalones -- no con lo que reduciría la edad sobre el total.
        observed_drop = 1 - (edad_step.remaining_population / sexo_step.remaining_population)
        assert edad_step.reduction_percent == round(observed_drop * 100, 1)

    def test_reduction_percent_is_none_when_value_is_not_in_reference_tables(self):
        findings = DemographicFindings(estudios="una_carrera_que_no_existe")

        steps = estimate_population_narrowing(findings)

        assert steps[0].reduction_percent is None

    def test_reduction_percent_is_none_for_standalone_steps(self):
        findings = DemographicFindings(universidad="Salamanca", empresa="Acme")

        steps = estimate_population_narrowing(findings)

        for step in steps:
            assert step.reduction_percent is None

    def test_reduction_percent_present_for_location_step(self):
        findings = DemographicFindings(provincia="madrid")

        steps = estimate_population_narrowing(findings)

        assert steps[0].reduction_percent is not None
        assert 0 <= steps[0].reduction_percent <= 100

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

    def test_comunidad_autonoma_used_when_no_municipio_or_provincia(self):
        findings = DemographicFindings(
            comunidad_autonoma="canarias",
            evidence={"comunidad_autonoma": ["https://ig/1"]},
            source={"comunidad_autonoma": "imagen"},
        )

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        step = steps[0]
        assert step.category == "ubicacion"
        assert "comunidad autónoma" in step.attribute_label.lower()
        assert "canarias" in step.attribute_label.lower()
        assert step.remaining_population is not None
        assert step.remaining_population > 0
        assert step.source == "imagen"

    def test_provincia_takes_priority_over_comunidad_autonoma_when_both_present(self):
        findings = DemographicFindings(provincia="las palmas", comunidad_autonoma="canarias")

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert "provincia" in steps[0].attribute_label.lower()

    def test_unknown_comunidad_autonoma_is_no_estimable_not_a_crash(self):
        findings = DemographicFindings(comunidad_autonoma="comunidad_inventada")

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


class TestFinalRemainingPopulation:
    def test_none_when_no_chained_trait_detected(self):
        findings = DemographicFindings(universidad="UNED")  # solo un standalone, no encadenable
        steps = estimate_population_narrowing(findings)

        assert final_remaining_population(steps) is None

    def test_equals_last_chained_step_when_all_are_estimable(self):
        findings = DemographicFindings(sexo="hombre", edad=30, provincia="madrid")
        steps = estimate_population_narrowing(findings)

        location_step = next(s for s in steps if s.category == "ubicacion")
        assert final_remaining_population(steps) == location_step.remaining_population
        # Es MENOR que el de un solo rasgo aislado: la intersección de varios
        # rasgos siempre reduce (o deja igual) la población, nunca la aumenta.
        sexo_step = next(s for s in steps if s.category == "sexo")
        assert final_remaining_population(steps) <= sexo_step.remaining_population

    def test_no_estimable_intermediate_step_does_not_break_the_chain(self):
        """Si un rasgo intermedio (p.ej. ubicación) no es estimable, el
        siguiente rasgo encadenable debe seguir partiendo del acumulado
        anterior, no resetearse -- el resultado final debe ser el del
        último escalón realmente estimable (estudios), no None."""
        findings = DemographicFindings(
            sexo="hombre", provincia="provincia_no_existe_en_la_tabla", estudios="derecho"
        )
        steps = estimate_population_narrowing(findings)

        estudios_step = next(s for s in steps if s.category == "estudios")
        assert final_remaining_population(steps) == estudios_step.remaining_population
        assert final_remaining_population(steps) is not None

    def test_standalone_steps_are_ignored(self):
        findings = DemographicFindings(sexo="hombre", universidad="UNED", empresa="Acme")
        steps = estimate_population_narrowing(findings)

        sexo_step = next(s for s in steps if s.category == "sexo")
        assert final_remaining_population(steps) == sexo_step.remaining_population


class TestEstadoCivilStep:
    """El paso pedido explícitamente: la inferencia simbólica de IA sobre
    el estado civil (soltero/con_pareja/casado) debe aparecer en la tabla
    de estrechamiento y afectar al porcentaje, no quedarse solo en
    inferred_attributes."""

    def test_casado_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(
            estado_civil="casado", evidence={"estado_civil": ["bio"]}, source={"estado_civil": "ia_simbolica"}
        )

        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        step = steps[0]
        assert step.category == "estado_civil"
        assert step.attribute_label == "Casado/a"
        assert step.remaining_population is not None
        assert step.remaining_population > 0
        assert step.remaining_population < TOTAL_POPULATION_ES  # de verdad estrecha, no es un no-op
        assert step.source == "ia_simbolica"
        assert step.evidence == ["bio"]
        assert "simbólico" in step.note.lower()

    def test_con_pareja_and_soltero_and_viudo_also_produce_a_step(self):
        for value, expected_label in [
            ("con_pareja", "Tiene pareja (sin estar casado/a)"),
            ("soltero", "Soltero/a (sin pareja actualmente)"),
            ("viudo", "Viudo/a"),
        ]:
            steps = estimate_population_narrowing(DemographicFindings(estado_civil=value))
            assert len(steps) == 1
            assert steps[0].attribute_label == expected_label
            assert steps[0].remaining_population is not None

    def test_none_produces_no_step(self):
        """Si la IA no encontró ninguna señal en ningún sentido, no debe
        inventarse un paso -- a diferencia de sexo/edad, que siempre están
        presentes en un texto autodeclarado, aquí el valor por defecto (sin
        señal) es genuinamente 'no lo sé' ('desconocido'), no una de las
        tres categorías por defecto."""
        findings = DemographicFindings(estado_civil=None)

        steps = estimate_population_narrowing(findings)

        assert steps == []

    def test_the_three_categories_sum_to_the_total_population(self):
        """Las proporciones de MARITAL_STATUS_DISTRIBUTION deben sumar 1.0
        -- si no, alguna combinación de casado/con_pareja/soltero contaría
        a la misma persona dos veces, o a nadie."""
        total = sum(MARITAL_STATUS_DISTRIBUTION.values())
        assert total == pytest.approx(1.0)

    def test_participates_in_the_chain_after_other_traits(self):
        """Debe estrechar MÁS que sexo solo, no sustituirlo ni ser
        independiente de él -- es un paso más de la misma cadena."""
        only_sexo = DemographicFindings(sexo="mujer")
        with_relacion = DemographicFindings(sexo="mujer", estado_civil="casado")

        pop_sexo_only = estimate_population_narrowing(only_sexo)[-1].remaining_population
        pop_with_relacion = estimate_population_narrowing(with_relacion)[-1].remaining_population

        assert pop_with_relacion < pop_sexo_only

    def test_counts_towards_final_remaining_population(self):
        """Es una categoría ENCADENABLE (_CHAINED_CATEGORIES): debe poder
        ser, ella sola, el último escalón que determine
        final_remaining_population -- justo lo que pedía 'que afecte al
        porcentaje', no que se quede fuera del cálculo combinado."""
        findings = DemographicFindings(estado_civil="casado")
        steps = estimate_population_narrowing(findings)

        assert final_remaining_population(steps) == steps[0].remaining_population

    def test_reduction_percent_present(self):
        findings = DemographicFindings(estado_civil="casado")
        steps = estimate_population_narrowing(findings)

        assert steps[0].reduction_percent is not None
        assert 0 <= steps[0].reduction_percent <= 100

    def test_uses_exact_cross_tab_when_sexo_is_already_known(self):
        """Cuando también se conoce el sexo, debe usarse la proporción
        EXACTA de esa combinación (MARITAL_STATUS_BY_SEX), no la marginal
        (MARITAL_STATUS_DISTRIBUTION) -- son números distintos a propósito
        en la tabla, así que dan un remaining_population distinto."""
        only_estado_civil = estimate_population_narrowing(DemographicFindings(estado_civil="viudo"))
        with_sexo = estimate_population_narrowing(DemographicFindings(sexo="mujer", estado_civil="viudo"))

        estado_civil_step = next(s for s in with_sexo if s.category == "estado_civil")
        sexo_step = next(s for s in with_sexo if s.category == "sexo")

        # Población tras sexo+viudo (cruce exacto) = pop_mujeres * P(viudo|mujer)
        expected = round(sexo_step.remaining_population * MARITAL_STATUS_BY_SEX["mujer"]["viudo"])
        assert estado_civil_step.remaining_population == expected
        # Y debe ser DISTINTO de aplicar la marginal sobre pop_mujeres (lo que
        # se haría si no se usara la tabla cruzada), para confirmar que de
        # verdad se está usando MARITAL_STATUS_BY_SEX y no MARITAL_STATUS_DISTRIBUTION.
        marginal_equivalent = round(sexo_step.remaining_population * MARITAL_STATUS_DISTRIBUTION["viudo"])
        assert estado_civil_step.remaining_population != marginal_equivalent
        assert "EXACTO" in estado_civil_step.note

    def test_falls_back_to_marginal_distribution_when_sexo_unknown(self):
        findings = DemographicFindings(estado_civil="viudo")
        steps = estimate_population_narrowing(findings)

        expected = round(TOTAL_POPULATION_ES * MARITAL_STATUS_DISTRIBUTION["viudo"])
        assert steps[0].remaining_population == expected
        assert "EXACTO" not in steps[0].note

    def test_marital_status_by_sex_sums_to_one_per_sex(self):
        for sexo, distribution in MARITAL_STATUS_BY_SEX.items():
            assert sum(distribution.values()) == pytest.approx(1.0), sexo
