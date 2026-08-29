import pytest

from app.nlp.demographic_extraction import DemographicFindings
from app.data.ine_reference import (
    HOUSEHOLD_TYPE_DISTRIBUTION,
    LANGUAGE_BY_CCAA,
    MARITAL_STATUS_BY_SEX,
    MARITAL_STATUS_DISTRIBUTION,
    NATIONALITY_DISTRIBUTION,
    RELIGION_DISTRIBUTION,
    SEXUAL_ORIENTATION_DISTRIBUTION,
    SITUACION_LABORAL_DISTRIBUTION,
    SPORT_PRACTICE_BY_SEX,
    SPORT_PRACTICE_DISTRIBUTION,
    TOTAL_POPULATION_ES,
    ZODIAC_DISTRIBUTION,
)
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

    def test_edad_rango_step_uses_age_range_proportion_when_no_exact_age(self):
        findings = DemographicFindings(
            edad_rango_min=25,
            edad_rango_max=30,
            source={"edad_rango_min": "ia_estimada"},
            confidence={"edad_rango_min": 0.75},
        )

        steps = estimate_population_narrowing(findings)
        edad_step = next(s for s in steps if s.category == "edad")

        assert edad_step.value_raw == "25-30"
        assert edad_step.source == "ia_estimada"
        assert edad_step.confidence == 0.75
        assert edad_step.remaining_population is not None
        assert edad_step.note_code == "edad_estimada_por_tramo"

    def test_wider_edad_rango_narrows_population_less(self):
        """El punto central del rediseño: un rango más ancho debe narrowear
        MENOS (más población restante), nunca más -- ensanchar el rango
        nunca debe producir una falsa precisión."""
        narrow = DemographicFindings(edad_rango_min=25, edad_rango_max=27)
        wide = DemographicFindings(edad_rango_min=15, edad_rango_max=45)

        pop_narrow = estimate_population_narrowing(narrow)[-1].remaining_population
        pop_wide = estimate_population_narrowing(wide)[-1].remaining_population

        assert pop_wide > pop_narrow

    def test_edad_exacta_wins_over_edad_rango_when_both_present(self):
        """Nunca deberían convivir edad exacta y rango a la vez (ver
        ai_attribute_extraction.py), pero si por algún motivo llegaran
        ambas, la edad exacta debe ganar -- es la más precisa."""
        findings = DemographicFindings(
            edad=24,
            edad_rango_min=25,
            edad_rango_max=30,
            source={"edad": "texto", "edad_rango_min": "ia_estimada"},
            confidence={"edad_rango_min": 0.75},
        )

        steps = estimate_population_narrowing(findings)
        edad_step = next(s for s in steps if s.category == "edad")

        assert edad_step.value_raw == "24"
        assert edad_step.confidence is None
        assert edad_step.note_code == "edad_repartida_uniformemente"

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

    def test_con_pareja_and_soltero_and_viudo_and_divorciado_also_produce_a_step(self):
        for value, expected_label in [
            ("con_pareja", "Tiene pareja (sin estar casado/a)"),
            ("soltero", "Soltero/a (sin pareja actualmente)"),
            ("viudo", "Viudo/a"),
            ("divorciado", "Divorciado/a o separado/a"),
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


class TestNacionalidadStep:
    def test_espanola_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(nacionalidad="espanola", evidence={"nacionalidad": ["https://x/1"]})
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        step = steps[0]
        assert step.category == "nacionalidad"
        assert step.attribute_label == "Nacionalidad: española"
        assert step.remaining_population == round(TOTAL_POPULATION_ES * NATIONALITY_DISTRIBUTION["espanola"])
        assert step.evidence == ["https://x/1"]

    def test_extranjera_also_produces_a_step(self):
        findings = DemographicFindings(nacionalidad="extranjera")
        steps = estimate_population_narrowing(findings)

        assert steps[0].remaining_population == round(TOTAL_POPULATION_ES * NATIONALITY_DISTRIBUTION["extranjera"])

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(nacionalidad=None)) == []

    def test_counts_towards_final_remaining_population(self):
        findings = DemographicFindings(nacionalidad="espanola")
        steps = estimate_population_narrowing(findings)
        assert final_remaining_population(steps) == steps[0].remaining_population

    def test_distribution_sums_to_one(self):
        assert sum(NATIONALITY_DISTRIBUTION.values()) == pytest.approx(1.0)

    def test_source_defaults_to_texto(self):
        findings = DemographicFindings(nacionalidad="espanola")
        steps = estimate_population_narrowing(findings)
        assert steps[0].source == "texto"


class TestSituacionLaboralStep:
    def test_activo_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(situacion_laboral="activo")
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "situacion_laboral"
        assert steps[0].attribute_label == "Situación laboral: trabaja actualmente"
        assert steps[0].remaining_population == round(
            TOTAL_POPULATION_ES * SITUACION_LABORAL_DISTRIBUTION["activo"]
        )

    def test_all_five_categories_produce_a_step(self):
        for value in ("activo", "parado", "jubilado", "estudiante", "otro_inactivo"):
            steps = estimate_population_narrowing(DemographicFindings(situacion_laboral=value))
            assert len(steps) == 1
            assert steps[0].remaining_population is not None

    def test_is_distinct_from_ocupacion_category(self):
        """No debe confundirse con el sector profesional (`ocupacion`):
        ambos pueden coexistir como pasos separados de la cadena."""
        findings = DemographicFindings(situacion_laboral="activo", ocupacion="sanitario")
        steps = estimate_population_narrowing(findings)

        categories = {s.category for s in steps}
        assert {"situacion_laboral", "ocupacion"} <= categories

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(situacion_laboral=None)) == []

    def test_distribution_sums_to_one(self):
        assert sum(SITUACION_LABORAL_DISTRIBUTION.values()) == pytest.approx(1.0)


class TestTipoHogarStep:
    def test_unipersonal_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(tipo_hogar="unipersonal")
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "tipo_hogar"
        assert steps[0].attribute_label == "Tipo de hogar: vive solo/a"
        assert steps[0].remaining_population == round(
            TOTAL_POPULATION_ES * HOUSEHOLD_TYPE_DISTRIBUTION["unipersonal"]
        )

    def test_all_four_categories_produce_a_step(self):
        for value in ("unipersonal", "pareja_sin_hijos", "pareja_con_hijos", "monoparental"):
            steps = estimate_population_narrowing(DemographicFindings(tipo_hogar=value))
            assert len(steps) == 1
            assert steps[0].remaining_population is not None

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(tipo_hogar=None)) == []

    def test_distribution_sums_to_one(self):
        assert sum(HOUSEHOLD_TYPE_DISTRIBUTION.values()) == pytest.approx(1.0)


class TestLenguaMaternaStep:
    def test_no_estimable_when_ccaa_unknown(self):
        """Sin comunidad autónoma conocida, la lengua materna cooficial no
        se puede acotar (ver _resolve_ccaa_for_language) -- debe marcarse
        no_estimable en vez de aplicar una proporción nacional inventada."""
        findings = DemographicFindings(lengua_materna="catalan")
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "lengua_materna"
        assert steps[0].risk_level == "no_estimable"
        assert steps[0].remaining_population is None

    def test_estimable_when_comunidad_autonoma_matches(self):
        findings = DemographicFindings(lengua_materna="catalan", comunidad_autonoma="cataluna")
        steps = estimate_population_narrowing(findings)

        lengua_step = next(s for s in steps if s.category == "lengua_materna")
        assert lengua_step.risk_level != "no_estimable"
        assert lengua_step.remaining_population is not None

    def test_resolves_ccaa_from_provincia_when_comunidad_not_set_directly(self):
        """Si solo se conoce la provincia (no la comunidad autónoma
        directamente), debe resolverse vía PROVINCE_TO_CCAA -- ver
        _resolve_ccaa_for_language."""
        findings = DemographicFindings(lengua_materna="euskera", provincia="vizcaya")
        steps = estimate_population_narrowing(findings)

        lengua_step = next(s for s in steps if s.category == "lengua_materna")
        assert lengua_step.risk_level != "no_estimable"

    def test_uses_proportion_conditioned_on_ccaa_not_national(self):
        findings = DemographicFindings(lengua_materna="gallego", comunidad_autonoma="galicia")
        steps = estimate_population_narrowing(findings)

        location_step = next(s for s in steps if s.category == "ubicacion")
        lengua_step = next(s for s in steps if s.category == "lengua_materna")
        # Se aplica sobre la población YA acotada por el paso de ubicación
        # (comunidad_autonoma="galicia" activa también _step_location antes
        # en la cadena), no sobre el total nacional -- por eso la base es
        # location_step.remaining_population, no TOTAL_POPULATION_ES.
        expected = round(location_step.remaining_population * LANGUAGE_BY_CCAA["galicia"]["gallego"])
        assert lengua_step.remaining_population == expected

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(lengua_materna=None)) == []

    def test_counts_towards_final_remaining_population_when_estimable(self):
        findings = DemographicFindings(lengua_materna="catalan", comunidad_autonoma="cataluna")
        steps = estimate_population_narrowing(findings)
        assert final_remaining_population(steps) == steps[-1].remaining_population


class TestOrientacionSexualStep:
    def test_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(orientacion_sexual="bisexual", source={"orientacion_sexual": "texto"})
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "orientacion_sexual"
        assert steps[0].attribute_label == "Orientación sexual: Bisexual"
        assert steps[0].remaining_population == round(
            TOTAL_POPULATION_ES * SEXUAL_ORIENTATION_DISTRIBUTION["bisexual"]
        )
        assert steps[0].note_code == "orientacion_sexual_categoria_especial"

    def test_all_seven_categories_produce_a_step(self):
        for value in SEXUAL_ORIENTATION_DISTRIBUTION:
            steps = estimate_population_narrowing(DemographicFindings(orientacion_sexual=value))
            assert len(steps) == 1
            assert steps[0].remaining_population is not None

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(orientacion_sexual=None)) == []

    def test_counts_towards_final_remaining_population(self):
        findings = DemographicFindings(orientacion_sexual="gay")
        steps = estimate_population_narrowing(findings)
        assert final_remaining_population(steps) == steps[-1].remaining_population

    def test_distribution_sums_to_one(self):
        assert sum(SEXUAL_ORIENTATION_DISTRIBUTION.values()) == pytest.approx(1.0)


class TestReligionStep:
    def test_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(religion="ateismo", source={"religion": "texto"})
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "religion"
        assert steps[0].attribute_label == "Religión: Ateismo"
        assert steps[0].remaining_population == round(TOTAL_POPULATION_ES * RELIGION_DISTRIBUTION["ateismo"])
        assert steps[0].note_code == "religion_categoria_especial"

    def test_all_eight_categories_produce_a_step(self):
        for value in RELIGION_DISTRIBUTION:
            steps = estimate_population_narrowing(DemographicFindings(religion=value))
            assert len(steps) == 1
            assert steps[0].remaining_population is not None

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(religion=None)) == []

    def test_counts_towards_final_remaining_population(self):
        findings = DemographicFindings(religion="islam")
        steps = estimate_population_narrowing(findings)
        assert final_remaining_population(steps) == steps[-1].remaining_population


class TestSignoZodiacalStep:
    def test_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(
            signo_zodiacal="aries (21 mar - 19 abr)", source={"signo_zodiacal": "texto"}
        )
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "signo_zodiacal"
        assert steps[0].remaining_population == round(TOTAL_POPULATION_ES * ZODIAC_DISTRIBUTION["aries"])
        assert steps[0].note_code == "signo_zodiacal_note"

    def test_base_sign_is_extracted_from_the_stored_range_string(self):
        """`findings.signo_zodiacal` se guarda como 'signo (rango)' -- el
        paso debe usar solo el nombre del signo para buscar en
        ZODIAC_DISTRIBUTION, no la cadena completa con el rango."""
        findings = DemographicFindings(signo_zodiacal="escorpio (23 oct - 21 nov)")
        steps = estimate_population_narrowing(findings)
        assert steps[0].remaining_population == round(TOTAL_POPULATION_ES * ZODIAC_DISTRIBUTION["escorpio"])

    def test_all_twelve_signs_produce_a_step(self):
        for sign in ZODIAC_DISTRIBUTION:
            findings = DemographicFindings(signo_zodiacal=f"{sign} (rango cualquiera)")
            steps = estimate_population_narrowing(findings)
            assert len(steps) == 1
            assert steps[0].remaining_population is not None

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(signo_zodiacal=None)) == []

    def test_counts_towards_final_remaining_population(self):
        findings = DemographicFindings(signo_zodiacal="leo (23 jul - 22 ago)")
        steps = estimate_population_narrowing(findings)
        assert final_remaining_population(steps) == steps[-1].remaining_population

    def test_distribution_sums_to_one(self):
        assert sum(ZODIAC_DISTRIBUTION.values()) == pytest.approx(1.0)


class TestSpecialCategoryFieldsCombined:
    """Regresión: orientacion_sexual, religion y signo_zodiacal se
    detectaban en demographic_extraction.py/ai_attribute_extraction.py
    pero nunca llegaban a k_anonymity.py -- se perdían sin afectar al
    número final ni aparecer en el informe (Comandante, agosto 2026)."""

    def test_all_three_together_chain_correctly(self):
        findings = DemographicFindings(
            sexo="mujer",
            orientacion_sexual="bisexual",
            religion="ateismo",
            signo_zodiacal="aries (21 mar - 19 abr)",
        )
        steps = estimate_population_narrowing(findings)
        categories = {s.category for s in steps}
        assert {"sexo", "orientacion_sexual", "religion", "signo_zodiacal"} <= categories

        # El último escalón debe ser el más estrecho (cada paso multiplica
        # sobre el anterior).
        assert final_remaining_population(steps) == steps[-1].remaining_population
        assert steps[-1].remaining_population < steps[0].remaining_population


class TestPracticaDeportivaStep:
    def test_produces_a_step_that_narrows_population(self):
        findings = DemographicFindings(practica_deportiva="futbol", source={"practica_deportiva": "texto"})
        steps = estimate_population_narrowing(findings)

        assert len(steps) == 1
        assert steps[0].category == "practica_deportiva"
        assert steps[0].attribute_label == "Práctica deportiva: Fútbol"
        assert steps[0].remaining_population == round(
            TOTAL_POPULATION_ES * SPORT_PRACTICE_DISTRIBUTION["futbol"]
        )
        assert steps[0].note_code == "practica_deportiva_no_particion"

    def test_all_categories_produce_a_step(self):
        for value in SPORT_PRACTICE_DISTRIBUTION:
            steps = estimate_population_narrowing(DemographicFindings(practica_deportiva=value))
            assert len(steps) == 1
            assert steps[0].remaining_population is not None

    def test_ciclismo_padel_tenis_baloncesto_have_distinct_labels(self):
        """Las 4 modalidades añadidas en la primera ampliación de la tabla
        (fuente 2022, ver historial de ine_reference.py) también deben
        tener su propio label legible, no caer en el fallback genérico
        .title()."""
        expected = {
            "ciclismo": "Ciclismo",
            "padel": "Pádel",
            "tenis": "Tenis",
            "baloncesto": "Baloncesto",
        }
        for value, label in expected.items():
            steps = estimate_population_narrowing(DemographicFindings(practica_deportiva=value))
            assert steps[0].attribute_label == f"Práctica deportiva: {label}"

    def test_second_batch_of_modalities_have_distinct_labels(self):
        """Segunda ampliación de la tabla (2022): mismo criterio que el
        test anterior."""
        expected = {
            "futbol_sala": "Fútbol sala",
            "golf": "Golf",
            "yoga_pilates": "Yoga / pilates",
            "gimnasia_intensa": "Gimnasia intensa (aerobic/step/spinning)",
        }
        for value, label in expected.items():
            steps = estimate_population_narrowing(DemographicFindings(practica_deportiva=value))
            assert steps[0].attribute_label == f"Práctica deportiva: {label}"

    def test_full_2024_25_survey_expansion_all_have_distinct_labels(self):
        """Tercera ampliación: tabla 1.21 completa de la Encuesta de
        Hábitos Deportivos en España 2024/25 (41 modalidades en total,
        ver ine_reference.py) -- las 28 modalidades nuevas de esta
        ampliación también deben tener su propio label legible."""
        expected = {
            "baile_fitness": "Baile fitness (zumba)",
            "tenis_mesa": "Tenis de mesa",
            "atletismo": "Atletismo",
            "esqui": "Esquí / snowboard",
            "voleibol": "Voleibol",
            "boxeo": "Boxeo",
            "submarinismo": "Submarinismo / buceo",
            "pesca": "Pesca",
            "patinaje": "Patinaje",
            "petanca": "Petanca / bolos",
            "artes_marciales": "Artes marciales",
            "piraguismo_remo": "Piragüismo / remo",
            "badminton": "Bádminton",
            "pelota_vasca": "Pelota vasca (frontón)",
            "caza": "Caza",
            "motociclismo": "Motociclismo",
            "surf": "Surf",
            "automovilismo": "Automovilismo",
            "vela": "Vela",
            "hipica": "Hípica",
            "balonmano": "Balonmano",
            "triatlon": "Triatlón",
            "rugby": "Rugby",
            "lucha_defensa_personal": "Lucha / defensa personal",
            "esqui_nautico": "Esquí náutico",
            "squash": "Squash",
            "aeronautica": "Actividades aeronáuticas",
            "ajedrez": "Ajedrez",
        }
        assert len(expected) == 28
        for value, label in expected.items():
            steps = estimate_population_narrowing(DemographicFindings(practica_deportiva=value))
            assert steps[0].attribute_label == f"Práctica deportiva: {label}"

    def test_running_and_atletismo_are_distinct_categories(self):
        """Regresión: en un borrador anterior 'practico atletismo' caía en
        el grupo 'running' (ver historial de demographic_extraction.py).
        La encuesta 2024/25 (tabla 1.21) los trata como DOS filas
        separadas con población muy distinta -- deben seguir siendo
        categorías independientes, con proporciones distintas."""
        assert "atletismo" in SPORT_PRACTICE_DISTRIBUTION
        assert "running" in SPORT_PRACTICE_DISTRIBUTION
        assert SPORT_PRACTICE_DISTRIBUTION["atletismo"] != SPORT_PRACTICE_DISTRIBUTION["running"]

    def test_golf_uses_the_same_formula_as_every_other_modality(self):
        """Regresión conceptual: en un borrador anterior (basado en una
        cifra de prensa suelta sobre la edición 2022) golf era un caso
        especial que NO se multiplicaba por la tasa general de práctica
        deportiva, porque esa cifra de prensa venía ya expresada sobre
        población total. La tabla oficial 1.21 de la edición 2024/25
        desmiente eso: golf sale exactamente con la misma fórmula que
        cualquier otra modalidad (practicantes de golf ÷ total de
        practicantes de algún deporte × tasa general). Este test
        documenta que YA NO hay ningún caso especial en la tabla -- si
        alguien reintrodujera esa excepción por error, este test lo
        detectaría."""
        assert SPORT_PRACTICE_DISTRIBUTION["golf"] == 0.014

    def test_all_41_modalities_of_the_2024_25_survey_are_present(self):
        """Sanity check de cobertura completa: la tabla 1.21 de la
        encuesta 2024/25 tiene 41 modalidades detectables (se excluyen
        deliberadamente 'Total', que es la fila de cabecera, y 'Otro
        deporte', que no tiene una frase-ancla de práctica distinguible
        -- ver comentario en ine_reference.py)."""
        assert len(SPORT_PRACTICE_DISTRIBUTION) == 41

    def test_none_produces_no_step(self):
        assert estimate_population_narrowing(DemographicFindings(practica_deportiva=None)) == []

    def test_counts_towards_final_remaining_population(self):
        findings = DemographicFindings(sexo="hombre", practica_deportiva="musculacion")
        steps = estimate_population_narrowing(findings)
        assert final_remaining_population(steps) == steps[-1].remaining_population
        assert steps[-1].remaining_population < steps[0].remaining_population

    def test_distribution_is_deliberately_not_a_partition(self):
        """Regresión conceptual: a diferencia de ZODIAC_DISTRIBUTION o
        SEXUAL_ORIENTATION_DISTRIBUTION (que SÍ deben sumar 1, ver sus
        propios tests), SPORT_PRACTICE_DISTRIBUTION es de una encuesta de
        respuesta múltiple -- que la suma actual esté cerca de 1 (o lo
        supere, como ocurre ahora con 13 modalidades) es COINCIDENCIA/
        consecuencia esperada, no una señal de que en realidad sea una
        partición: cada persona puede sumar en varias modalidades a la
        vez, así que la suma puede superar 1 sin que eso sea un error.
        Este test documenta la intención (no debe forzarse a sumar
        exactamente 1 ni a quedarse por debajo), no impone un límite
        concreto de la suma."""
        total = sum(SPORT_PRACTICE_DISTRIBUTION.values())
        assert total > 0.5  # sanity check: no vacío ni con valores absurdamente bajos

    def test_uses_exact_sex_conditioned_proportion_when_sexo_is_already_known(self):
        """Mismo patrón que TestEstadoCivilStep.test_uses_exact_cross_tab_
        when_sexo_is_already_known (tabla 1.22 de la encuesta, en vez de
        la tabla 1.21 marginal): cuando también se conoce el sexo, debe
        usarse SPORT_PRACTICE_BY_SEX, no SPORT_PRACTICE_DISTRIBUTION --
        son números distintos a propósito, así que dan un
        remaining_population distinto."""
        with_sexo = estimate_population_narrowing(
            DemographicFindings(sexo="mujer", practica_deportiva="yoga_pilates")
        )
        sexo_step = next(s for s in with_sexo if s.category == "sexo")
        deporte_step = next(s for s in with_sexo if s.category == "practica_deportiva")

        expected = round(sexo_step.remaining_population * SPORT_PRACTICE_BY_SEX["yoga_pilates"]["mujer"])
        assert deporte_step.remaining_population == expected

        # Y debe ser DISTINTO de aplicar la marginal sobre pop_mujeres (lo
        # que se haría si no se usara la tabla condicionada), para
        # confirmar que de verdad se está usando SPORT_PRACTICE_BY_SEX y
        # no SPORT_PRACTICE_DISTRIBUTION. "yoga_pilates" es un buen caso
        # de prueba precisamente porque el sesgo por sexo es enorme (las
        # mujeres lo practican ~4 veces más).
        marginal_equivalent = round(sexo_step.remaining_population * SPORT_PRACTICE_DISTRIBUTION["yoga_pilates"])
        assert deporte_step.remaining_population != marginal_equivalent
        assert deporte_step.note_code == "practica_deportiva_ajustada_por_sexo"

    def test_falls_back_to_marginal_when_sexo_unknown(self):
        findings = DemographicFindings(practica_deportiva="yoga_pilates")
        steps = estimate_population_narrowing(findings)

        expected = round(TOTAL_POPULATION_ES * SPORT_PRACTICE_DISTRIBUTION["yoga_pilates"])
        assert steps[0].remaining_population == expected
        assert steps[0].note_code == "practica_deportiva_no_particion"

    def test_falls_back_to_marginal_when_modality_has_no_entry_for_that_sex(self):
        """'squash' no tiene clave 'mujer' en SPORT_PRACTICE_BY_SEX a
        propósito (la encuesta redondeó a 0,0% con esa muestra concreta,
        ver comentario en ine_reference.py) -- para una mujer que declara
        practicar squash, debe caer de vuelta a la marginal en vez de
        devolver una población de 0 (que sería una certeza que el dato
        real no respalda)."""
        assert "mujer" not in SPORT_PRACTICE_BY_SEX["squash"]

        steps = estimate_population_narrowing(DemographicFindings(sexo="mujer", practica_deportiva="squash"))
        deporte_step = next(s for s in steps if s.category == "practica_deportiva")

        assert deporte_step.remaining_population is not None
        assert deporte_step.remaining_population > 0
        assert deporte_step.note_code == "practica_deportiva_no_particion"

    def test_uses_exact_proportion_for_a_male_biased_sport_too(self):
        """Mismo test que el de yoga_pilates pero en la dirección
        contraria del sesgo (hombres, deporte muy masculinizado) -- para
        confirmar que el ajuste funciona en ambos sentidos, no solo
        cuando el sexo declarado coincide con el sexo mayoritario de un
        ejemplo concreto."""
        with_sexo = estimate_population_narrowing(
            DemographicFindings(sexo="hombre", practica_deportiva="caza")
        )
        sexo_step = next(s for s in with_sexo if s.category == "sexo")
        deporte_step = next(s for s in with_sexo if s.category == "practica_deportiva")

        expected = round(sexo_step.remaining_population * SPORT_PRACTICE_BY_SEX["caza"]["hombre"])
        assert deporte_step.remaining_population == expected
        assert deporte_step.note_code == "practica_deportiva_ajustada_por_sexo"

    def test_sport_practice_by_sex_does_not_need_to_sum_to_one(self):
        """A diferencia de MARITAL_STATUS_BY_SEX (partición, cada
        sub-diccionario SÍ debe sumar 1 -- ver ese test), esta tabla es de
        una encuesta de respuesta múltiple igual que SPORT_PRACTICE_DISTRIBUTION:
        no tiene sentido exigir que sume 1, y no debería forzarse."""
        for sexo, distribution in SPORT_PRACTICE_BY_SEX.items():
            assert sum(distribution.values()) < 1.0, sexo

