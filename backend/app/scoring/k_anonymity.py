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
    RELIGION_DISTRIBUTION,
    SEX_DISTRIBUTION,
    SEXUAL_ORIENTATION_DISTRIBUTION,
    SITUACION_LABORAL_DISTRIBUTION,
    SPORT_PRACTICE_DISTRIBUTION,
    STUDIES_DISTRIBUTION,
    TOTAL_POPULATION_ES,
    ZODIAC_DISTRIBUTION,
    EYE_COLOR_DISTRIBUTION,
    HAIR_COLOR_DISTRIBUTION,
    SKIN_TONE_DISTRIBUTION,
    age_range_proportion,
)
from app.nlp.demographic_extraction import DemographicFindings
from app.models.schemas import ManualAttribute
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
    # cuando aplique -- de momento solo lo usa el rango de edad estimado
    # (`edad_rango_min`/`edad_rango_max`, ver
    # ai_attribute_extraction.py::_set_edad_rango). Se
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
    # Categorías especiales del art. 9 RGPD (orientacion_sexual, religion)
    # y signo_zodiacal -- SÍ participan en la cadena que afina el número
    # final, igual que el resto: si la persona lo autodeclaró en público,
    # es información real que reduce el conjunto de posibles individuos.
    # Ver _step_orientacion_sexual/_step_religion/_step_signo_zodiacal
    # para el razonamiento completo de por qué se procesan pese a ser
    # datos sensibles (spoiler: la herramienta existe para mostrarle a la
    # persona su propia exposición, no para perfilar a terceros).
    "orientacion_sexual", "religion", "signo_zodiacal",
    # NO es partición (ver docstring de _step_practica_deportiva y el
    # comentario de SPORT_PRACTICE_DISTRIBUTION en ine_reference.py), pero
    # eso no impide que narrowee -- una proporción marginal sigue siendo
    # una proporción de población válida para multiplicar en la cadena.
    "practica_deportiva",
    # Rasgos físicos manuales
    "color_ojos", "color_pelo", "color_piel",
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

    if findings.edad_rango_min is not None and findings.edad_rango_max is not None:
        # No hay edad EXACTA autodeclarada, pero sí un RANGO estimado
        # INDIRECTAMENTE (ver ai_attribute_extraction.py::_set_edad_rango)
        # que ya superó su propio umbral de confianza mínima antes de
        # llegar aquí. A diferencia de la primera versión (que encajaba la
        # estimación en un tramo quinquenal FIJO de AGE_DISTRIBUTION_5Y),
        # el rango aquí es de ancho LIBRE -- lo eligió el propio modelo
        # según cuánta certeza tenía -- así que se usa
        # `age_range_proportion` (suma AGE_DISTRIBUTION_1Y año a año sobre
        # el rango exacto dado), no un tramo prefijado.
        label = f"Edad aproximada: {findings.edad_rango_min}-{findings.edad_rango_max} años"
        return _apply_proportion(
            remaining,
            age_range_proportion(findings.edad_rango_min, findings.edad_rango_max),
            label,
            "edad",
            findings.evidence.get("edad_rango_min", []),
            source=findings.source.get("edad_rango_min", "ia_estimada"),
            note="Rango de edad estimado por IA a partir de pistas indirectas del texto "
                 "(años de graduación, curso que menciona estar haciendo, referencias "
                 "generacionales...), no de una autodeclaración explícita: fiabilidad menor. "
                 "El ancho del rango lo decide la propia IA según cuánta certeza tenía -- un "
                 "rango más amplio reduce menos la población, pero es más honesto que fingir "
                 "precisión sobre una pista débil.",
            note_code=note_codes.EDAD_ESTIMADA_POR_TRAMO,
            value_raw=f"{findings.edad_rango_min}-{findings.edad_rango_max}",
            confidence=findings.confidence.get("edad_rango_min"),
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


_SPORT_LABELS = {
    "musculacion": "Musculación / gimnasio",
    "senderismo": "Senderismo / montañismo",
    "running": "Running / atletismo",
    "natacion": "Natación",
    "futbol": "Fútbol",
    "ciclismo": "Ciclismo",
    "padel": "Pádel",
    "tenis": "Tenis",
    "baloncesto": "Baloncesto",
}


def _step_practica_deportiva(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    """A diferencia del resto de pasos, SPORT_PRACTICE_DISTRIBUTION (ver
    ine_reference.py) NO es una partición -- son proporciones MARGINALES
    de una encuesta de respuesta múltiple, no probabilidades mutuamente
    excluyentes (una persona puede practicar varios deportes a la vez).
    `_apply_proportion` no exige que la tabla sume 1 (nunca lo ha exigido,
    ver OCCUPATION_DISTRIBUTION, que tampoco suma 1 -- ahí es porque solo
    cubre un subconjunto de ocupaciones, aquí es porque el propio dato de
    origen no es una partición), así que no hace falta ningún ajuste
    especial aquí: el cálculo es idéntico al resto de pasos."""
    if not findings.practica_deportiva:
        return remaining, None
    label = _SPORT_LABELS.get(findings.practica_deportiva, findings.practica_deportiva.title())
    return _apply_proportion(
        remaining,
        SPORT_PRACTICE_DISTRIBUTION.get(findings.practica_deportiva),
        f"Práctica deportiva: {label}",
        "practica_deportiva",
        findings.evidence.get("practica_deportiva", []),
        source=findings.source.get("practica_deportiva", "texto"),
        note="Proporción marginal de la Encuesta de Hábitos Deportivos en España (no es "
             "una partición: la encuesta es de respuesta múltiple, una persona puede "
             "practicar varios deportes a la vez, así que este dato por sí solo no implica "
             "que sea el ÚNICO deporte que practica).",
        note_code=note_codes.PRACTICA_DEPORTIVA_NO_PARTICION,
        value_raw=findings.practica_deportiva,
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


_ORIENTACION_SEXUAL_LABELS = {
    "heterosexual": "Heterosexual",
    "gay": "Gay",
    "lesbiana": "Lesbiana",
    "bisexual": "Bisexual",
    "pansexual": "Pansexual",
    "asexual": "Asexual",
    "homosexual": "Homosexual",
}


def _step_orientacion_sexual(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    """Orientación sexual autodeclarada de forma EXPLÍCITA (nunca inferida
    de forma indirecta -- ver el requisito de autodeclaración literal en
    `demographic_extraction.py::_try_detect_orientacion_sexual` y en el
    campo homónimo del prompt de IA en `ai_attribute_extraction.py`).

    Es una categoría especial del art. 9 RGPD. Se procesa de todos modos
    porque el propósito de esta herramienta es precisamente mostrarle a
    la persona qué tan expuesto queda un dato que ELLA MISMA publicó en
    abierto -- no perfilar a terceros ni inferir nada que no haya dicho
    explícitamente. Se marca en la nota para que quede claro en el
    informe que se trata de una categoría especial, no un dato más.

    `SEXUAL_ORIENTATION_DISTRIBUTION` (ver ine_reference.py) es una
    estimación contextual para España basada en CIS/observatorios, NO un
    censo oficial del INE -- España no pregunta esto en el censo."""
    if not findings.orientacion_sexual:
        return remaining, None
    label = _ORIENTACION_SEXUAL_LABELS.get(
        findings.orientacion_sexual, findings.orientacion_sexual.title()
    )
    return _apply_proportion(
        remaining,
        SEXUAL_ORIENTATION_DISTRIBUTION.get(findings.orientacion_sexual),
        f"Orientación sexual: {label}",
        "orientacion_sexual",
        findings.evidence.get("orientacion_sexual", []),
        source=findings.source.get("orientacion_sexual", "texto"),
        note="Categoría especial de datos (art. 9 RGPD): se calcula únicamente "
             "porque la propia persona lo autodeclaró de forma explícita en "
             "público, nunca por inferencia indirecta. La proporción usada es "
             "una estimación contextual para España (CIS/observatorios), no un "
             "censo oficial del INE.",
        note_code=note_codes.ORIENTACION_SEXUAL_CATEGORIA_ESPECIAL,
        value_raw=findings.orientacion_sexual,
    )


def _step_religion(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    """Religión autodeclarada explícitamente en texto o mediante un
    símbolo/emoji inequívoco -- ver `_try_detect_religion` y el campo
    homónimo del prompt de IA. Misma categoría especial del art. 9 RGPD
    que `orientacion_sexual`, mismo razonamiento para procesarla (ver
    docstring de `_step_orientacion_sexual`)."""
    if not findings.religion:
        return remaining, None
    return _apply_proportion(
        remaining,
        RELIGION_DISTRIBUTION.get(findings.religion),
        f"Religión: {findings.religion.title()}",
        "religion",
        findings.evidence.get("religion", []),
        source=findings.source.get("religion", "texto"),
        note="Categoría especial de datos (art. 9 RGPD): se calcula únicamente "
             "porque la propia persona lo autodeclaró o lo indicó con un "
             "símbolo/emoji explícito en público, nunca por inferencia "
             "indirecta. La proporción usada es una estimación contextual para "
             "España (CIS y fuentes comunitarias), no un censo oficial del "
             "INE -- España no recoge la afiliación religiosa en el censo.",
        note_code=note_codes.RELIGION_CATEGORIA_ESPECIAL,
        value_raw=findings.religion,
    )


def _step_signo_zodiacal(findings: DemographicFindings, remaining: float) -> tuple[float, PopulationNarrowingStep | None]:
    """El signo zodiacal, más allá de la creencia astrológica en sí (sin
    base científica), revela de facto un rango real de ~30 días de fecha
    de nacimiento -- por eso SÍ afina la población igual que cualquier
    otro rasgo, no es un dato "de broma". No es categoría especial RGPD
    (a diferencia de orientación sexual/religión): la fecha de nacimiento
    aproximada no está en el art. 9.

    `findings.signo_zodiacal` se guarda como 'signo (rango de fechas)'
    (ver `_try_detect_signo_zodiacal`); aquí solo hace falta el nombre
    del signo (antes del primer paréntesis) para buscarlo en
    `ZODIAC_DISTRIBUTION`, que usa claves sin el rango."""
    if not findings.signo_zodiacal:
        return remaining, None
    signo = findings.signo_zodiacal.split(" (")[0]
    return _apply_proportion(
        remaining,
        ZODIAC_DISTRIBUTION.get(signo),
        f"Signo zodiacal: {findings.signo_zodiacal.title()}",
        "signo_zodiacal",
        findings.evidence.get("signo_zodiacal", []),
        source=findings.source.get("signo_zodiacal", "texto"),
        note="La distribución es uniforme (~8,33% cada signo, un doceavo del "
             "año) porque las fechas de nacimiento se reparten de forma "
             "prácticamente uniforme a lo largo del año -- no es un dato "
             "inventado: revela directamente un rango real de ~30 días de "
             "nacimiento, aunque la creencia astrológica en sí no tenga base "
             "científica.",
        note_code=note_codes.SIGNO_ZODIACAL_NOTE,
        value_raw=findings.signo_zodiacal,
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
    # Autodeclaraciones explícitas (o símbolo/emoji inequívoco), igual de
    # sólidas que sexo/nacionalidad/etc en cuanto a fiabilidad de la
    # señal -- van aquí, antes de _step_relacion (la señal MENOS fiable
    # de la cadena), aunque el orden no cambie el resultado final (la
    # multiplicación de proporciones es conmutativa, ver docstring del
    # módulo): es solo para que el orden de aparición en el informe siga
    # yendo de más a menos fiable.
    _step_orientacion_sexual, _step_religion, _step_signo_zodiacal,
    # Autodeclaración explícita igual de sólida que las anteriores (misma
    # exigencia de verbo de PRÁCTICA, no simple mención -- ver
    # demographic_extraction.py::_SPORT_PRACTICE_RE); va aquí por el mismo
    # motivo que el bloque de arriba, no porque dependa de nada previo.
    _step_practica_deportiva,
    # Al final: es la señal menos fiable de la cadena (inferencia simbólica
    # por IA, no autodeclaración -- ver docstring de
    # DemographicFindings.estado_civil), así que refina lo que ya se haya
    # estrechado con rasgos más sólidos, en vez de condicionarlos.
    _step_relacion,
)

# Pasos independientes de `remaining`, que no participan en la cadena de
# estrechamiento (no hay tabla de proporción nacional para ellos).
_STANDALONE_STEPS = (_step_universidad, _step_empresa)


def estimate_population_narrowing(findings: DemographicFindings, manual_attributes: list[ManualAttribute] | None = None) -> list[PopulationNarrowingStep]:
    steps: list[PopulationNarrowingStep] = []
    remaining = float(TOTAL_POPULATION_ES)

    for step_fn in _CHAINED_STEPS:
        remaining, step = step_fn(findings, remaining)
        if step:
            steps.append(step)
            
    if manual_attributes:
        for attr in manual_attributes:
            if attr.category == "color_ojos":
                remaining, step = _apply_proportion(
                    remaining,
                    EYE_COLOR_DISTRIBUTION.get(attr.value),
                    f"Color de ojos: {attr.value.title()}",
                    attr.category,
                    [],
                    source="manual",
                    note="Rasgo físico añadido manualmente. Proporción estimada contextualmente.",
                    note_code=None,
                    value_raw=attr.value,
                )
                if step: steps.append(step)
            elif attr.category == "color_pelo":
                remaining, step = _apply_proportion(
                    remaining,
                    HAIR_COLOR_DISTRIBUTION.get(attr.value),
                    f"Color de pelo: {attr.value.title()}",
                    attr.category,
                    [],
                    source="manual",
                    note="Rasgo físico añadido manualmente. Proporción estimada contextualmente.",
                    note_code=None,
                    value_raw=attr.value,
                )
                if step: steps.append(step)
            elif attr.category == "color_piel":
                remaining, step = _apply_proportion(
                    remaining,
                    SKIN_TONE_DISTRIBUTION.get(attr.value),
                    f"Color de piel: {attr.value.title()}",
                    attr.category,
                    [],
                    source="manual",
                    note="Rasgo físico añadido manualmente. Proporción estimada contextualmente.",
                    note_code=None,
                    value_raw=attr.value,
                )
                if step: steps.append(step)

    for standalone_fn in _STANDALONE_STEPS:
        step = standalone_fn(findings)
        if step:
            steps.append(step)

    return steps
