"""
Módulo 6 (nuevo): estima cuánta gente en España comparte cada combinación
de atributos autodeclarados detectados, en cascada, al estilo de una tabla
de k-anonimato.

Decisión de diseño importante: en vez de construir una base de datos
sintética con una fila por cada uno de los ~49M habitantes de España
(lo que rompería el diseño stateless/sin-BD del proyecto, ver docstring de
`app/models/schemas.py`), este módulo multiplica PROPORCIONES AGREGADAS del
INE en cadena, asumiendo independencia entre atributos salvo que exista una
tabla cruzada real (no es el caso aquí, ver limitación abajo).

Esto es matemáticamente equivalente a "cuántas filas quedarían" en una
tabla de microdatos si esas variables fueran independientes entre sí, y es
la aproximación estándar en estudios de riesgo de reidentificación cuando
no se dispone de microdatos individuales.

Limitación documentada (para la memoria): asumir independencia entre edad,
sexo, provincia, estudios y ocupación es una simplificación. En la
realidad hay correlación (p. ej. la distribución de edad varía algo entre
provincias, o ciertos estudios están más concentrados en unas edades que
otras). El resultado es una ESTIMACIÓN, no un conteo exacto; se marca así
explícitamente en el informe.

Umbrales de riesgo (inspirados en los estándares habituales de
k-anonimato / small-cell suppression, p. ej. HIPAA Safe Harbor usa k<11
como umbral de riesgo alto para variables demográficas):
- remaining >= 100_000  -> bajo
- remaining >= 1_000     -> medio
- remaining >= 20        -> alto
- remaining <  20        -> critico
"""
from dataclasses import dataclass

from app.data.ine_reference import (
    AGE_DISTRIBUTION_1Y,
    MUNICIPALITY_POPULATION,
    OCCUPATION_DISTRIBUTION,
    PROVINCE_POPULATION,
    SEX_DISTRIBUTION,
    STUDIES_DISTRIBUTION,
    TOTAL_POPULATION_ES,
)
from app.nlp.demographic_extraction import DemographicFindings


@dataclass
class PopulationNarrowingStep:
    attribute_label: str  # p.ej. "Sexo: hombre", "Vive en León"
    category: str  # sexo | edad | ubicacion | estudios | ocupacion | universidad | empresa
    remaining_population: int | None  # None si no estimable con las tablas actuales
    risk_level: str  # bajo | medio | alto | critico | no_estimable
    evidence: list[str]
    source: str = "texto"  # "texto" (autodeclaración) | "imagen" (geolocation.py)
    note: str | None = None


def _risk_level(remaining: float) -> str:
    if remaining >= 100_000:
        return "bajo"
    if remaining >= 1_000:
        return "medio"
    if remaining >= 20:
        return "alto"
    return "critico"


def _apply_proportion(
    remaining: float,
    proportion: float | None,
    label: str,
    category: str,
    evidence: list[str],
    source: str = "texto",
    note: str | None = None,
) -> tuple[float, PopulationNarrowingStep | None]:
    """Multiplica `remaining` por una proporción marginal del INE (asumiendo
    independencia respecto a los atributos ya aplicados) y construye el
    escalón del informe. Devuelve (nuevo_remaining, step_o_None)."""
    if proportion is None:
        return remaining, PopulationNarrowingStep(
            attribute_label=label,
            category=category,
            remaining_population=None,
            risk_level="no_estimable",
            evidence=evidence,
            source=source,
            note="No hay dato de referencia del INE para este valor concreto en la tabla actual.",
        )

    new_remaining = remaining * proportion
    return new_remaining, PopulationNarrowingStep(
        attribute_label=label,
        category=category,
        remaining_population=round(new_remaining),
        risk_level=_risk_level(new_remaining),
        evidence=evidence,
        source=source,
        note=note,
    )


def estimate_population_narrowing(findings: DemographicFindings) -> list[PopulationNarrowingStep]:
    steps: list[PopulationNarrowingStep] = []
    remaining = float(TOTAL_POPULATION_ES)

    if findings.sexo:
        remaining, step = _apply_proportion(
            remaining,
            SEX_DISTRIBUTION.get(findings.sexo),
            f"Sexo: {findings.sexo}",
            "sexo",
            findings.evidence.get("sexo", []),
            source=findings.source.get("sexo", "texto"),
        )
        if step:
            steps.append(step)

    if findings.edad is not None:
        remaining, step = _apply_proportion(
            remaining,
            AGE_DISTRIBUTION_1Y.get(findings.edad),
            f"Edad: {findings.edad} años",
            "edad",
            findings.evidence.get("edad", []),
            source=findings.source.get("edad", "texto"),
            note="Estimado repartiendo uniformemente la proporción de INE por tramos "
                 "quinquenales entre las edades de cada tramo (no hay tabla año a año "
                 "descargable directamente); ver ine_reference.py.",
        )
        if step:
            steps.append(step)

    # Ubicación: usa municipio si está disponible (más específico); si no,
    # provincia. Nunca los dos a la vez (el municipio ya está contenido en
    # la provincia; aplicar ambos contaría el filtro geográfico dos veces).
    #
    # La población de un municipio/provincia no es una "proporción sobre el
    # total nacional" en el mismo sentido que sexo/edad -- es un recuento
    # absoluto. Para combinarlo con los filtros ya aplicados (sexo, edad),
    # asumimos que ese municipio/provincia tiene una pirámide de edad/sexo
    # similar a la nacional (limitación documentada arriba) y reescalamos:
    # remaining_tras_geografia = poblacion_municipio * (remaining / TOTAL_ES)
    location = findings.municipio or findings.provincia
    if location:
        table = MUNICIPALITY_POPULATION if findings.municipio else PROVINCE_POPULATION
        pop = table.get(location)
        label = (
            f"Vive en municipio: {location.title()}"
            if findings.municipio
            else f"Vive en provincia: {location.title()}"
        )
        evidence_key = "municipio" if findings.municipio else "provincia"
        location_source = findings.source.get(evidence_key, "texto")

        if pop is None:
            steps.append(
                PopulationNarrowingStep(
                    attribute_label=label,
                    category="ubicacion",
                    remaining_population=None,
                    risk_level="no_estimable",
                    evidence=findings.evidence.get(evidence_key, []),
                    source=location_source,
                    note="No hay dato de población de referencia para este municipio/provincia en la tabla actual.",
                )
            )
        else:
            ratio_so_far = remaining / TOTAL_POPULATION_ES
            remaining = pop * ratio_so_far
            steps.append(
                PopulationNarrowingStep(
                    attribute_label=label,
                    category="ubicacion",
                    remaining_population=round(remaining),
                    risk_level=_risk_level(remaining),
                    evidence=findings.evidence.get(evidence_key, []),
                    source=location_source,
                    note="Asume distribución de edad/sexo similar a la media nacional (aproximación)."
                    + (" Ubicación estimada a partir de una imagen, no de texto autodeclarado: "
                       "menor fiabilidad que una autodeclaración explícita." if location_source == "imagen" else ""),
                )
            )

    if findings.estudios:
        remaining, step = _apply_proportion(
            remaining,
            STUDIES_DISTRIBUTION.get(findings.estudios),
            f"Estudió: {findings.estudios.title()}",
            "estudios",
            findings.evidence.get("estudios", []),
            source=findings.source.get("estudios", "texto"),
        )
        if step:
            steps.append(step)

    if findings.ocupacion:
        remaining, step = _apply_proportion(
            remaining,
            OCCUPATION_DISTRIBUTION.get(findings.ocupacion),
            f"Ocupación: {findings.ocupacion.title()}",
            "ocupacion",
            findings.evidence.get("ocupacion", []),
            source=findings.source.get("ocupacion", "texto"),
        )
        if step:
            steps.append(step)

    # Universidad y empresa concretas no son "proporciones nacionales": son
    # recuentos absolutos (nº de alumnos/empleados), y no tenemos esa tabla
    # en este MVP. Se listan como detectadas pero no estimables, en vez de
    # inventar un número, para mantener la honestidad del informe.
    if findings.universidad:
        steps.append(
            PopulationNarrowingStep(
                attribute_label=f"Universidad: {findings.universidad.title()}",
                category="universidad",
                remaining_population=None,
                risk_level="no_estimable",
                evidence=findings.evidence.get("universidad", []),
                note="Requeriría el nº de alumnos/egresados de esa universidad y titulación "
                     "concretos (trabajo futuro): dato no incluido en esta versión.",
            )
        )

    if findings.empresa:
        steps.append(
            PopulationNarrowingStep(
                attribute_label=f"Empresa: {findings.empresa}",
                category="empresa",
                remaining_population=None,
                risk_level="no_estimable",
                evidence=findings.evidence.get("empresa", []),
                note="Requeriría el nº de empleados de esa empresa concreta (trabajo futuro): "
                     "dato no incluido en esta versión.",
            )
        )

    return steps
