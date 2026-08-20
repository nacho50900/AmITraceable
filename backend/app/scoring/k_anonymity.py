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
    AGE_DISTRIBUTION_5Y,
    AUTONOMOUS_COMMUNITY_DISPLAY_NAMES,
    CCAA_POPULATION,
    HOUSEHOLD_TYPE_DISTRIBUTION,
    LANGUAGE_BY_CCAA,
    MARITAL_STATUS_BY_SEX,
    MARITAL_STATUS_DISTRIBUTION,
    MUNICIPALITY_POPULATION,
    NATIONALITY_DISTRIBUTION,
    OCCUPATION_DISTRIBUTION,
    PROVINCE_POPULATION,
    PROVINCE_TO_CCAA,
    SEX_DISTRIBUTION,
    SITUACION_LABORAL_DISTRIBUTION,
    STUDIES_DISTRIBUTION,
    TOTAL_POPULATION_ES,
)
from app.nlp.demographic_extraction import DemographicFindings
from app import note_codes


@dataclass
class PopulationNarrowingStep:
    attribute_label: str  # p.ej. "Sexo: hombre", "Vive en León"
    category: str  # sexo | edad | ubicacion | estudios | ocupacion | universidad | empresa
    remaining_population: int | None  # None si no estimable con las tablas actuales
    risk_level: str  # bajo | medio | alto | critico | no_estimable
    evidence: list[str]
    # "texto" (autodeclaración detectada por regex) | "imagen" (geolocation.py) |
    # "ia" (autodeclaración detectada por IA en texto/bio) | "ia_nombre" (estimación de
    # sexo por convención cultural del nombre público, no autodeclaración -- ver
    # app/nlp/ai_attribute_extraction.py)
    source: str = "texto"
    note: str | None = None
    # Código estable (ver app/note_codes.py) para que el frontend traduzca
    # `note` sin tener que parsear la frase en español -- `note` se
    # conserva tal cual para logs/descarga JSON. None si este paso no
    # lleva nota (la mayoría no la llevan).
    note_code: str | None = None
    # Valor "en crudo" de este atributo, sin la plantilla de `attribute_label`
    # ya montada (p.ej. "hombre", "24", "León") -- para que el frontend
    # pueda traducir la plantilla ("Sexo: ", "Vive en municipio: "...) e
    # interpolar el valor, en vez de traducir `attribute_label` ya unido
    # (que mezclaría plantilla y valor, y a veces el valor es un nombre
    # propio -- universidad, empresa, topónimo -- que no debe traducirse).
    value_raw: str | None = None
    # Solo relevante cuando category == "ubicacion": "municipio" | "provincia" |
    # "comunidad_autonoma" -- necesario porque los tres comparten `category`
    # pero cada uno usa una plantilla de traducción distinta en el frontend
    # ("Vive en municipio:" / "Vive en provincia:" / "Vive en comunidad
    # autónoma:"), y esa distinción no se puede recuperar de `value_raw` ni
    # de `source`.
    location_level: str | None = None
    # remaining_population / TOTAL_POPULATION_ES, ya calculado aquí para que el
    # frontend no tenga que conocer ni duplicar esa constante (usado para el
    # pictograma de población, ver PopulationPictogram.tsx). None si remaining_population
    # también lo es.
    proportion: float | None = None
    # % que ESTE rasgo concreto ha reducido la población respecto al
    # escalón ANTERIOR de la cadena (no respecto al total de España desde
    # cero) -- p.ej. si antes de este paso quedaban 20M de personas
    # compatibles y este rasgo las deja en 10M, reduction_percent = 50.0.
    # None en pasos "no estimables" (no hay proporción que aplicar) y en
    # los standalone (universidad/empresa), donde no hay una proporción
    # nacional de referencia con la que calcularlo.
    reduction_percent: float | None = None
    # Confianza (0-1) declarada por la IA para una estimación INDIRECTA,
    # cuando aplique -- de momento solo lo usa el tramo de edad estimado
    # (`edad_rango`, ver ai_attribute_extraction.py::_set_edad_rango). Se
    # expone como campo propio, no interpolado dentro de `note` (que se
    # mantiene siempre como texto fijo por categoría/fuente, ver
    # note_codes.py), para no romper el contrato de traducción por
    # note_code del frontend. None para el resto de pasos.
    confidence: float | None = None


# Categorías que participan en la cadena de estrechamiento (mismo criterio
# que _CHAINED_STEPS, ver más abajo -- se define aquí arriba porque
# `final_remaining_population` la necesita y así queda cerca de donde se usa).
_CHAINED_CATEGORIES = {
    "sexo", "edad", "ubicacion", "estudios", "ocupacion", "estado_civil",
    "nacionalidad", "situacion_laboral", "tipo_hogar", "lengua_materna",
}


def final_remaining_population(steps: list[PopulationNarrowingStep]) -> int | None:
    """Nº de personas en España que compartirían, EN CONJUNTO, todos los
    rasgos encadenables que se hayan podido estimar (sexo + edad + ubicación
    + estudios + ocupación) -- no la proporción de un rasgo aislado, sino la
    intersección de todos ellos. Para el informe: "en España hay X personas
    que comparten tus rasgos".

    Cada escalón de la cadena ya lleva en su propio `remaining_population`
    el acumulado incluyendo TODOS los anteriores (ver `estimate_population_narrowing`
    y `_apply_proportion`): un escalón "no estimable" no rompe la cadena, es
    un no-op (no se descuenta población por él, simplemente no se afina
    más). Por eso basta con quedarse con el ÚLTIMO escalón de categoría
    encadenable que SÍ tenga `remaining_population` -- es, matemáticamente,
    el resultado final de la cadena completa, sin tener que repetir aquí la
    lógica de encadenado.

    None si NINGÚN rasgo encadenable se pudo estimar (el informe no llegó a
    afinar nada más allá de la población total de España, así que no hay
    ningún número nuevo que mostrar aquí)."""
    candidates = [
        step.remaining_population
        for step in steps
        if step.category in _CHAINED_CATEGORIES and step.remaining_population is not None
    ]
    return candidates[-1] if candidates else None


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
    note_code: str | None = None,
    value_raw: str | None = None,
    confidence: float | None = None,
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
            note_code=note_codes.NO_INE_DATA_FOR_VALUE,
            value_raw=value_raw,
            confidence=confidence,
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
        note_code=note_code,
        value_raw=value_raw,
        proportion=new_remaining / TOTAL_POPULATION_ES,
        # `proportion` (el parámetro, no el campo del step) YA ES el factor
        # marginal de este rasgo por sí solo -- new_remaining/remaining --
        # así que la reducción respecto al escalón anterior es 1 menos ese
        # factor, sin falta de recalcular nada.
        reduction_percent=round((1 - proportion) * 100, 1),
        confidence=confidence,
    )


def _step_sexo(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if not findings.sexo:
        return remaining, None
    source = findings.source.get("sexo", "texto")
    note = None
    note_code = None
    if source == "ia_nombre":
        note = (
            "Estimado por convención cultural del nombre público de la cuenta, no por una "
            "autodeclaración explícita: fiabilidad menor (nombres unisex, apodos o "
            "transliteraciones pueden dar una estimación incorrecta)."
        )
        note_code = note_codes.SEXO_ESTIMADO_POR_NOMBRE
    return _apply_proportion(
        remaining,
        SEX_DISTRIBUTION.get(findings.sexo),
        f"Sexo: {findings.sexo}",
        "sexo",
        findings.evidence.get("sexo", []),
        source=source,
        note=note,
        note_code=note_code,
        # "hombre"/"mujer" -- conjunto cerrado, el frontend lo traduce con
        # su propia tabla (mismo patrón que risk_level/source).
        value_raw=findings.sexo,
    )


_ESTADO_CIVIL_LABELS = {
    "casado": "Casado/a",
    "con_pareja": "Tiene pareja (sin estar casado/a)",
    "divorciado": "Divorciado/a o separado/a",
    "soltero": "Soltero/a (sin pareja actualmente)",
    "viudo": "Viudo/a",
}


def _step_relacion(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if findings.estado_civil is None:
        return remaining, None

    # Si ya se conoce el sexo (se aplica antes en la cadena, ver
    # _CHAINED_STEPS), se usa la proporción EXACTA de esa combinación
    # concreta -- P(estado_civil | sexo), tabla real cruzada -- en vez de
    # aproximar multiplicando dos proporciones marginales asumiendo
    # independencia. Si no se conoce el sexo, se sigue usando la
    # distribución marginal (sin distinguir sexo) como aproximación.
    sex_distribution = MARITAL_STATUS_BY_SEX.get(findings.sexo)
    exact = sex_distribution is not None
    if exact:
        proportion = sex_distribution.get(findings.estado_civil)
    else:
        proportion = MARITAL_STATUS_DISTRIBUTION.get(findings.estado_civil)

    note = (
        "Inferido por IA a partir de contenido simbólico o indirecto (emojis, fechas, "
        "menciones recurrentes...), no de una autodeclaración explícita: fiabilidad menor "
        "que el resto de rasgos de esta lista."
    )
    note_code = note_codes.ESTADO_CIVIL_IA_SIMBOLICA
    if exact:
        note += (
            " Al conocerse también el sexo, se usa el porcentaje EXACTO de esa combinación "
            "concreta (estado civil condicionado a sexo, tabla real del INE), no una "
            "aproximación multiplicando proporciones independientes."
        )
        note_code = note_codes.ESTADO_CIVIL_IA_SIMBOLICA_EXACT_COMBO

    return _apply_proportion(
        remaining,
        proportion,
        _ESTADO_CIVIL_LABELS[findings.estado_civil],
        "estado_civil",
        findings.evidence.get("estado_civil", []),
        source=findings.source.get("estado_civil", "ia_simbolica"),
        note=note,
        note_code=note_code,
        value_raw=findings.estado_civil,
    )


def _step_edad(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if findings.edad is not None:
        return _apply_proportion(
            remaining,
            AGE_DISTRIBUTION_1Y.get(findings.edad),
            f"Edad: {findings.edad} años",
            "edad",
            findings.evidence.get("edad", []),
            source=findings.source.get("edad", "texto"),
            note="Estimado repartiendo uniformemente la proporción de INE por tramos "
                 "quinquenales entre las edades de cada tramo (no hay tabla año a año "
                 "descargable directamente); ver ine_reference.py.",
            note_code=note_codes.EDAD_REPARTIDA_UNIFORMEMENTE,
            # Un número no necesita traducción; se manda como string para que
            # el frontend lo interpole igual que el resto de `value_raw`.
            value_raw=str(findings.edad),
        )

    if findings.edad_rango is not None:
        # No hay edad EXACTA autodeclarada, pero sí una estimación
        # INDIRECTA por tramo quinquenal (ver
        # ai_attribute_extraction.py::_set_edad_rango) que ya superó su
        # propio umbral de confianza mínima antes de llegar aquí -- se usa
        # la proporción de INE del tramo completo (AGE_DISTRIBUTION_5Y),
        # no la derivada año a año, porque no se conoce la edad exacta
        # dentro del tramo.
        return _apply_proportion(
            remaining,
            AGE_DISTRIBUTION_5Y.get(findings.edad_rango),
            f"Edad aproximada: {findings.edad_rango} años",
            "edad",
            findings.evidence.get("edad_rango", []),
            source=findings.source.get("edad_rango", "ia_estimada"),
            note="Tramo de edad estimado por IA a partir de pistas indirectas del texto "
                 "(años de graduación, curso que menciona estar haciendo, referencias "
                 "generacionales...), no de una autodeclaración explícita: fiabilidad menor. "
                 "Se usa el tramo quinquenal de población del INE en vez de un año exacto, "
                 "por ser una estimación menos precisa que una edad autodeclarada.",
            note_code=note_codes.EDAD_ESTIMADA_POR_TRAMO,
            value_raw=findings.edad_rango,
            confidence=findings.confidence.get("edad_rango"),
        )

    return remaining, None


def _location_no_estimable_step(
    label: str, evidence: list[str], source: str, value_raw: str, location_level: str
) -> PopulationNarrowingStep:
    return PopulationNarrowingStep(
        attribute_label=label,
        category="ubicacion",
        remaining_population=None,
        risk_level="no_estimable",
        evidence=evidence,
        source=source,
        note="No hay dato de población de referencia para este municipio/provincia en la tabla actual.",
        note_code=note_codes.LOCATION_NO_POPULATION_DATA,
        value_raw=value_raw,
        location_level=location_level,
    )


def _location_note(source: str) -> tuple[str, str]:
    """Devuelve (note, note_code). `imagen` es la única condición que altera
    el mensaje -- mismo patrón que el resto de notas de este módulo."""
    base = "Asume distribución de edad/sexo similar a la media nacional (aproximación)."
    if source == "imagen":
        base += (
            " Ubicación estimada a partir de una imagen, no de texto autodeclarado: "
            "menor fiabilidad que una autodeclaración explícita."
        )
        return base, note_codes.LOCATION_NOTE_IMAGEN
    return base, note_codes.LOCATION_NOTE_BASE


def _step_location(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    """Usa municipio si está disponible (más específico); si no, provincia;
    si no, comunidad autónoma (menos específico, solo lo rellena
    geolocation.py -- ver `_assign_geolocated_region` en report/generator.py
    -- cuando la imagen solo permite identificar una comunidad con varias
    provincias, p.ej. Canarias). Nunca más de uno a la vez: cada nivel ya
    contiene al siguiente, y aplicar varios contaría el filtro geográfico
    más de una vez.

    La población de un municipio/provincia/comunidad no es una "proporción
    sobre el total nacional" en el mismo sentido que sexo/edad -- es un
    recuento absoluto. Para combinarlo con los filtros ya aplicados (sexo,
    edad), asumimos que esa zona tiene una pirámide de edad/sexo similar a
    la nacional (limitación documentada en el docstring del módulo) y
    reescalamos: remaining_tras_geografia = poblacion_zona * (remaining / TOTAL_ES)
    """
    if findings.municipio:
        location, table = findings.municipio, MUNICIPALITY_POPULATION
        evidence_key = location_level = "municipio"
        display_value = location.title()
        label = f"Vive en municipio: {display_value}"
    elif findings.provincia:
        location, table = findings.provincia, PROVINCE_POPULATION
        evidence_key = location_level = "provincia"
        display_value = location.title()
        label = f"Vive en provincia: {display_value}"
    elif findings.comunidad_autonoma:
        location, table = findings.comunidad_autonoma, CCAA_POPULATION
        evidence_key = location_level = "comunidad_autonoma"
        display_value = AUTONOMOUS_COMMUNITY_DISPLAY_NAMES.get(location, location.title())
        label = f"Vive en comunidad autónoma: {display_value}"
    else:
        return remaining, None

    source = findings.source.get(evidence_key, "texto")
    evidence = findings.evidence.get(evidence_key, [])

    population = table.get(location)
    if population is None:
        return remaining, _location_no_estimable_step(label, evidence, source, display_value, location_level)

    new_remaining = population * (remaining / TOTAL_POPULATION_ES)
    note, note_code = _location_note(source)
    return new_remaining, PopulationNarrowingStep(
        attribute_label=label,
        category="ubicacion",
        remaining_population=round(new_remaining),
        risk_level=_risk_level(new_remaining),
        evidence=evidence,
        source=source,
        note=note,
        note_code=note_code,
        # Topónimo -- nombre propio, el frontend lo interpola tal cual, sin
        # traducirlo (ver docstring de `value_raw` en PopulationNarrowingStep).
        value_raw=display_value,
        location_level=location_level,
        # new_remaining / remaining = population / TOTAL_POPULATION_ES (se
        # simplifica el factor remaining/TOTAL_POPULATION_ES de ambos
        # lados) -- es el factor marginal de esta ubicación por sí sola.
        reduction_percent=round((1 - population / TOTAL_POPULATION_ES) * 100, 1),
    )


def _step_estudios(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if not findings.estudios:
        return remaining, None
    return _apply_proportion(
        remaining,
        STUDIES_DISTRIBUTION.get(findings.estudios),
        f"Estudió: {findings.estudios.title()}",
        "estudios",
        findings.evidence.get("estudios", []),
        source=findings.source.get("estudios", "texto"),
        value_raw=findings.estudios,
    )


def _step_ocupacion(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if not findings.ocupacion:
        return remaining, None
    return _apply_proportion(
        remaining,
        OCCUPATION_DISTRIBUTION.get(findings.ocupacion),
        f"Ocupación: {findings.ocupacion.title()}",
        "ocupacion",
        findings.evidence.get("ocupacion", []),
        source=findings.source.get("ocupacion", "texto"),
        value_raw=findings.ocupacion,
    )


_NATIONALITY_LABELS = {
    "espanola": "Nacionalidad: española",
    "extranjera": "Nacionalidad: extranjera",
}


def _step_nacionalidad(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if not findings.nacionalidad:
        return remaining, None
    return _apply_proportion(
        remaining,
        NATIONALITY_DISTRIBUTION.get(findings.nacionalidad),
        _NATIONALITY_LABELS[findings.nacionalidad],
        "nacionalidad",
        findings.evidence.get("nacionalidad", []),
        source=findings.source.get("nacionalidad", "texto"),
        value_raw=findings.nacionalidad,
    )


_SITUACION_LABORAL_LABELS = {
    "activo": "Situación laboral: trabaja actualmente",
    "parado": "Situación laboral: en desempleo",
    "jubilado": "Situación laboral: jubilado/a o pensionista",
    "estudiante": "Situación laboral: estudiante",
    "otro_inactivo": "Situación laboral: inactivo/a (labores del hogar u otra situación)",
}


def _step_situacion_laboral(
    findings: DemographicFindings, remaining: float
) -> tuple[float, PopulationNarrowingStep | None]:
    if not findings.situacion_laboral:
        return remaining, None
    return _apply_proportion(
        remaining,
        SITUACION_LABORAL_DISTRIBUTION.get(findings.situacion_laboral),
        _SITUACION_LABORAL_LABELS[findings.situacion_laboral],
        "situacion_laboral",
        findings.evidence.get("situacion_laboral", []),
        source=findings.source.get("situacion_laboral", "texto"),
        note="Distinto del sector profesional (ocupación): aquí es si la persona trabaja, "
             "busca trabajo, está jubilada o estudia.",
        note_code=note_codes.SITUACION_LABORAL_NOTE,
        value_raw=findings.situacion_laboral,
    )


_HOUSEHOLD_LABELS = {
    "unipersonal": "Tipo de hogar: vive solo/a",
    "pareja_sin_hijos": "Tipo de hogar: vive en pareja, sin hijos en el hogar",
    "pareja_con_hijos": "Tipo de hogar: vive en pareja, con hijos en el hogar",
    "monoparental": "Tipo de hogar: familia monoparental (un solo progenitor con hijos)",
}


def _step_tipo_hogar(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if not findings.tipo_hogar:
        return remaining, None
    return _apply_proportion(
        remaining,
        HOUSEHOLD_TYPE_DISTRIBUTION.get(findings.tipo_hogar),
        _HOUSEHOLD_LABELS[findings.tipo_hogar],
        "tipo_hogar",
        findings.evidence.get("tipo_hogar", []),
        source=findings.source.get("tipo_hogar", "texto"),
        note="Proporción de HOGARES del INE usada como aproximación de la proporción de "
             "personas (ver ine_reference.py) -- misma limitación ya asumida para ubicación.",
        note_code=note_codes.TIPO_HOGAR_NOTE,
        value_raw=findings.tipo_hogar,
    )


_LANGUAGE_LABELS = {
    "catalan": "Lengua materna/habitual: catalán",
    "euskera": "Lengua materna/habitual: euskera",
    "gallego": "Lengua materna/habitual: gallego",
    "valenciano": "Lengua materna/habitual: valenciano",
}


def _resolve_ccaa_for_language(findings: DemographicFindings) -> str | None:
    """La tabla LANGUAGE_BY_CCAA está condicionada a la comunidad autónoma
    (ver ine_reference.py) -- hace falta resolverla aunque el paso de
    ubicación haya detectado provincia/municipio en vez de la comunidad
    directamente. No hay mapeo municipio->provincia en este MVP (ver
    ine_reference.py), así que solo se resuelve desde comunidad_autonoma o
    provincia; si solo se conoce el municipio, esta señal no se aplica."""
    if findings.comunidad_autonoma:
        return findings.comunidad_autonoma
    if findings.provincia:
        return PROVINCE_TO_CCAA.get(findings.provincia)
    return None


def _step_lengua(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    if not findings.lengua_materna:
        return remaining, None

    label = _LANGUAGE_LABELS[findings.lengua_materna]
    evidence = findings.evidence.get("lengua_materna", [])
    source = findings.source.get("lengua_materna", "texto")

    ccaa = _resolve_ccaa_for_language(findings)
    ccaa_distribution = LANGUAGE_BY_CCAA.get(ccaa) if ccaa else None
    proportion = ccaa_distribution.get(findings.lengua_materna) if ccaa_distribution else None

    if proportion is None:
        return remaining, PopulationNarrowingStep(
            attribute_label=label,
            category="lengua_materna",
            remaining_population=None,
            risk_level="no_estimable",
            evidence=evidence,
            source=source,
            note="Solo se puede acotar si también se conoce la comunidad autónoma de "
                 "residencia (esta lengua es casi residual fuera de su territorio "
                 "cooficial, así que una proporción nacional no aportaría nada útil).",
            note_code=note_codes.LENGUA_NO_ESTIMABLE,
            value_raw=findings.lengua_materna,
        )

    return _apply_proportion(
        remaining,
        proportion,
        label,
        "lengua_materna",
        evidence,
        source=source,
        note="Proporción calculada SOLO dentro de la comunidad autónoma ya conocida "
             "(P(lengua | comunidad autónoma), no a nivel nacional).",
        note_code=note_codes.LENGUA_WITHIN_CCAA,
        value_raw=findings.lengua_materna,
    )


def _step_universidad(findings: DemographicFindings) -> PopulationNarrowingStep | None:
    """Universidad y empresa concretas no son "proporciones nacionales": son
    recuentos absolutos (nº de alumnos/empleados), y no tenemos esa tabla en
    este MVP. Se listan como detectadas pero no estimables, en vez de
    inventar un número, para mantener la honestidad del informe."""
    if not findings.universidad:
        return None
    return PopulationNarrowingStep(
        attribute_label=f"Universidad: {findings.universidad.title()}",
        category="universidad",
        remaining_population=None,
        risk_level="no_estimable",
        evidence=findings.evidence.get("universidad", []),
        note="Requeriría el nº de alumnos/egresados de esa universidad y titulación "
             "concretos (trabajo futuro): dato no incluido en esta versión.",
        note_code=note_codes.UNIVERSIDAD_NO_ESTIMABLE,
        # Nombre propio -- no se traduce, el frontend lo interpola tal cual.
        value_raw=findings.universidad.title(),
    )


def _step_empresa(findings: DemographicFindings) -> PopulationNarrowingStep | None:
    if not findings.empresa:
        return None
    return PopulationNarrowingStep(
        attribute_label=f"Empresa: {findings.empresa}",
        category="empresa",
        remaining_population=None,
        risk_level="no_estimable",
        evidence=findings.evidence.get("empresa", []),
        note="Requeriría el nº de empleados de esa empresa concreta (trabajo futuro): "
             "dato no incluido en esta versión.",
        note_code=note_codes.EMPRESA_NO_ESTIMABLE,
        value_raw=findings.empresa,
    )


# Pasos que estrechan `remaining` en cadena, en este orden concreto (el
# orden importa: cada paso condiciona al siguiente, ver docstring del
# módulo sobre la asunción de independencia).
_CHAINED_STEPS = (
    _step_sexo, _step_edad, _step_location, _step_estudios, _step_ocupacion,
    _step_nacionalidad, _step_situacion_laboral, _step_tipo_hogar,
    # Depende de que _step_location ya haya podido resolver comunidad
    # autónoma o provincia (ver _resolve_ccaa_for_language) -- por eso va
    # después, aunque no dependa del PROPIO remaining de location, solo de
    # los campos ya presentes en `findings`.
    _step_lengua,
    # Al final: es la señal menos fiable de la cadena (inferencia simbólica
    # por IA, no autodeclaración -- ver docstring de
    # DemographicFindings.estado_civil), así que refina lo que ya se haya
    # estrechado con rasgos más sólidos, en vez de condicionarlos.
    _step_relacion,
)

# Pasos independientes de `remaining`, que no participan en la cadena de
# estrechamiento (no hay tabla de proporción nacional para ellos).
_STANDALONE_STEPS = (_step_universidad, _step_empresa)


def estimate_population_narrowing(findings: DemographicFindings) -> list[PopulationNarrowingStep]:
    steps: list[PopulationNarrowingStep] = []
    remaining = float(TOTAL_POPULATION_ES)

    for step_fn in _CHAINED_STEPS:
        remaining, step = step_fn(findings, remaining)
        if step:
            steps.append(step)

    for standalone_fn in _STANDALONE_STEPS:
        step = standalone_fn(findings)
        if step:
            steps.append(step)

    return steps
