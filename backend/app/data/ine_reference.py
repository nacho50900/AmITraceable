"""
Tablas de distribución poblacional agregadas, usadas por
`scoring/k_anonymity.py` para estimar cuánta gente en España comparte un
conjunto de características (edad, sexo, provincia, estudios...).

IMPORTANTE - alcance de estos datos:
Los valores de este fichero son APROXIMADOS (órdenes de magnitud correctos,
tomados de cifras públicas del INE de 2024-2025), pensados para que la
arquitectura funcione end-to-end en el MVP del TFG. Para la versión final
de la memoria, se recomienda sustituir estas constantes por una carga desde
CSV descargados directamente de:

- Población por sexo/edad/provincia (Padrón continuo / Estadística Continua
  de Población): https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177095
- Población por municipios: https://www.ine.es/jaxiT3/Tabla.htm?t=2917
- Nivel de estudios por edad/sexo (Censo 2021 / EPA):
  https://www.ine.es/dyngs/INEbase/es/categoria.htm?c=Estadistica_P&cid=1254734710990
- Ocupación (CNO-11) por sexo: Encuesta de Población Activa (EPA)

No se ha construido una base de datos sintética persona-a-persona (ver nota
de diseño en `scoring/k_anonymity.py`): estas son distribuciones agregadas,
no microdatos individuales.

SOBRE LA VIGENCIA DE ESTOS DATOS (`_LAST_VERIFIED` más abajo):
cada tabla lleva asociada la fecha del dato del INE en el que se basó (no
la fecha en que se escribió este fichero). `stale_tables()` compara esa
fecha contra hoy y avisa si ha pasado demasiado tiempo -- ver esa función
para el criterio de umbral por tipo de fuente (anual vs. plurianual). Es
un AVISO, no una descarga automática: cuando salte, hay que revisar a mano
si el INE ha publicado cifras más recientes y actualizar la tabla y su
fecha en `_LAST_VERIFIED` a la vez. Para las tablas con una fuente clara y
estructurada en la API del INE, `backend/scripts/update_ine_reference.py`
automatiza la parte de DESCARGAR el dato nuevo (no de aplicarlo -- eso
sigue siendo una decisión humana, ver docstring de ese script).
"""

from datetime import date, timedelta

# Fecha del dato de origen de cada tabla (NO la fecha en que se escribió
# este fichero) -- usada por `stale_tables()`. `None` significa "sin fuente
# con fecha de publicación periódica conocida", ver esa misma tabla más
# abajo para el porqué en cada caso -- esas no se comprueban.
_LAST_VERIFIED: dict[str, date | None] = {
    "TOTAL_POPULATION_ES": date(2025, 1, 1),
    "SEX_DISTRIBUTION": date(2025, 1, 1),
    "MARITAL_STATUS_DISTRIBUTION": date(2026, 8, 11),
    "MARITAL_STATUS_BY_SEX": date(2026, 8, 11),
    # Tabla 01003 del INE (Padrón continuo, población año a año), año
    # 2022 -- ver comentario junto a AGE_DISTRIBUTION_1Y para el porqué
    # del desfase con TOTAL_POPULATION_ES (2025). AGE_DISTRIBUTION_5Y se
    # deriva de esta misma tabla (ver _build_age_distribution_5y_from_1y),
    # así que comparten fecha de verificación.
    "AGE_DISTRIBUTION_1Y": date(2022, 1, 1),
    "AGE_DISTRIBUTION_5Y": date(2022, 1, 1),
    "PROVINCE_POPULATION": date(2026, 8, 11),
    # Sin tabla INE concreta citada al construir esta -- es un orden de
    # magnitud estimado a mano, no una cifra derivada de una fuente con
    # fecha de publicación. No tiene sentido marcarla "caducada" frente a
    # una fuente que nunca existió.
    "OCCUPATION_DISTRIBUTION": None,
    # STUDIES_DISTRIBUTION tiene fuente real (histórico de egresados por
    # rama desde 1985 + detalle reciente por titulación, Ministerio de
    # Ciencia/Universidades) -- no es del INE ni tiene API en vivo, así
    # que `update_ine_reference.py` no la recalcula sola; se refresca con
    # `scripts/update_studies_distribution.py` (intenta descargar solo,
    # cae a instrucciones manuales si falla) cuando el Ministerio
    # publique una edición más reciente de cualquiera de los ficheros.
    "STUDIES_DISTRIBUTION": date(2026, 8, 11),  # fecha de esta ejecución -- el histórico llega hasta el curso 2023-2024, el detalle por titulación hasta 2023 (egresados) / 2024 (matriculados)
    "NATIONALITY_DISTRIBUTION": date(2026, 8, 11),
    "SITUACION_LABORAL_DISTRIBUTION": date(2025, 10, 1),  # EPA T4 2025
    "HOUSEHOLD_TYPE_DISTRIBUTION": date(2024, 1, 1),
    # ECEPOV es una encuesta puntual del INE, sin periodicidad anual fija
    # (la última es de 2021, la anterior de 2018) -- un umbral de "1 año"
    # saltaría permanentemente sin que haya nada nuevo que revisar. Se
    # comprueba con un umbral mucho más largo (ver STALE_THRESHOLDS).
    "LANGUAGE_BY_CCAA": date(2021, 1, 1),
    # Encuesta de Hábitos Deportivos en España (Ministerio de Educación,
    # FP y Deportes + CSD + INE), tabla 1.21. Edición usada: 2024/25
    # (fichero oficial descargado del portal de estadísticas del
    # Ministerio en esta sesión, con el desglose completo de las 41
    # modalidades que trae esa tabla -- reemplaza un primer borrador
    # basado en cifras sueltas de prensa sobre la edición 2022, que solo
    # cubría 9 modalidades). Periodicidad quinquenal: no se espera una
    # edición más reciente hasta dentro de varios años.
    "SPORT_PRACTICE_DISTRIBUTION": date(2025, 1, 1),
    # Tabla 1.22 de la misma encuesta y edición (2024/25) -- mismo
    # fichero, mismo día de descarga.
    "SPORT_PRACTICE_BY_SEX": date(2025, 1, 1),
    "SPORT_PRACTICE_BY_AGE_BAND": date(2025, 1, 1),
    # INE/EPA, indicador "Nivel de formación de la población adulta", año 2024.
    "EDUCATION_LEVEL_DISTRIBUTION": date(2024, 1, 1),
    # Ver comentario junto a RAMA_ESTUDIOS_DISTRIBUTION: mezcla 2015/16
    # (público) + 2019/20 (privado) -- se usa la fecha más antigua de las
    # dos como fecha de verificación, no la más reciente, para no
    # aparentar más actualidad de la que realmente tiene el dato.
    "RAMA_ESTUDIOS_DISTRIBUTION": date(2015, 9, 1),
    "SPORT_PRACTICE_BY_EDUCATION_LEVEL": date(2025, 1, 1),
}

# Umbral de antigüedad (días) a partir del cual `stale_tables()` avisa,
# distinto según la periodicidad real de la fuente -- un umbral único de
# "1 año" para todo generaría avisos constantes en tablas que el propio
# INE no actualiza cada año (Censo, ECEPOV/ECH), y dejaría pasar demasiado
# tiempo en las que sí se actualizan anualmente (Padrón, EPA).
_STALE_THRESHOLD_ANNUAL = timedelta(days=400)  # Padrón, EPA: margen de ~1 mes sobre 1 año
_STALE_THRESHOLD_MULTIYEAR = timedelta(days=6 * 365)  # Censo (~cada 10 años, pero se usan notas de prensa intermedias), ECEPOV/ECH (sin periodicidad fija)

_STALE_THRESHOLDS: dict[str, timedelta] = {
    "TOTAL_POPULATION_ES": _STALE_THRESHOLD_ANNUAL,
    "SEX_DISTRIBUTION": _STALE_THRESHOLD_ANNUAL,
    "AGE_DISTRIBUTION_1Y": _STALE_THRESHOLD_ANNUAL,
    "AGE_DISTRIBUTION_5Y": _STALE_THRESHOLD_ANNUAL,
    "PROVINCE_POPULATION": _STALE_THRESHOLD_ANNUAL,
    "NATIONALITY_DISTRIBUTION": _STALE_THRESHOLD_ANNUAL,
    "SITUACION_LABORAL_DISTRIBUTION": _STALE_THRESHOLD_ANNUAL,
    # STUDIES_DISTRIBUTION: el Ministerio publica una edición nueva del
    # Excel de matriculados por titulación cada curso -- mismo umbral
    # anual que el resto de fuentes con periodicidad de curso/año.
    "STUDIES_DISTRIBUTION": _STALE_THRESHOLD_ANNUAL,
    "MARITAL_STATUS_DISTRIBUTION": _STALE_THRESHOLD_MULTIYEAR,
    "MARITAL_STATUS_BY_SEX": _STALE_THRESHOLD_MULTIYEAR,
    "HOUSEHOLD_TYPE_DISTRIBUTION": _STALE_THRESHOLD_MULTIYEAR,
    "LANGUAGE_BY_CCAA": _STALE_THRESHOLD_MULTIYEAR,
    # Quinquenal por diseño (ver comentario en _LAST_VERIFIED) -- un
    # umbral anual saltaría constantemente sin que haya nada nuevo que
    # revisar, igual que ECEPOV/Censo.
    "SPORT_PRACTICE_DISTRIBUTION": _STALE_THRESHOLD_MULTIYEAR,
    "SPORT_PRACTICE_BY_SEX": _STALE_THRESHOLD_MULTIYEAR,
    "SPORT_PRACTICE_BY_AGE_BAND": _STALE_THRESHOLD_MULTIYEAR,
    "EDUCATION_LEVEL_DISTRIBUTION": _STALE_THRESHOLD_ANNUAL,
    "RAMA_ESTUDIOS_DISTRIBUTION": _STALE_THRESHOLD_MULTIYEAR,
    "SPORT_PRACTICE_BY_EDUCATION_LEVEL": _STALE_THRESHOLD_MULTIYEAR,
}


def stale_tables(as_of: date | None = None) -> list[tuple[str, date, int]]:
    """Tablas de este fichero cuyo dato de origen (`_LAST_VERIFIED`) ha
    superado su umbral de antigüedad esperado (`_STALE_THRESHOLDS`).

    Devuelve una lista de (nombre_tabla, fecha_verificacion, dias_de_antiguedad),
    ordenada de más a menos antigua. Lista vacía si todo está dentro de
    plazo (o si `as_of` es anterior a todas las fechas registradas).

    NO descarga nada ni compara contra el INE en vivo -- es una
    comprobación puramente local de "¿cuánto tiempo ha pasado desde la
    fecha que documentamos como origen de este dato?". Pensada para
    llamarse una vez al arrancar el backend (ver `_lifespan` en
    `main.py`) y dejar un aviso en el log, igual que ya se hace con la
    disponibilidad de los modelos de visión."""
    as_of = as_of or date.today()
    stale = []
    for name, last_verified in _LAST_VERIFIED.items():
        if last_verified is None:
            continue
        age = as_of - last_verified
        threshold = _STALE_THRESHOLDS.get(name, _STALE_THRESHOLD_ANNUAL)
        if age > threshold:
            stale.append((name, last_verified, age.days))
    return sorted(stale, key=lambda row: row[2], reverse=True)


# Población residente en España a 1 de enero de 2025 (INE, Estadística
# Continua de Población / Censo Anual de Población).
TOTAL_POPULATION_ES = 49_128_297

# REFERENCIAS CONTEXTUALES DE TERCEROS (NO DEL INE, NO USADAS PARA
# k-anonimato): estas cifras tienen un uso estrictamente contextual en la
# sección de "información inferida" del informe, para ayudar al usuario a
# situar qué tan raro/normal es un atributo, no para calcular riesgo de
# reidentificación estadístico. Las tablas oficiales del INE siguen siendo
# la fuente prioritaria para age/sex/province/studies/occupation.
#
# FUENTES DOCUMENTADAS Y ENFOCADO A ESPAÑA (NO a Europa general):
# 1) Religiones / afiliación religiosa en España:
#    - CIS (Centro de Investigaciones Sociológicas): barómetros de opinión y
#      religión en España, que reportan la composición religiosa del país.
#    - FCJE (Federación de Comunidades Judías de España): comunidad judía en
#      España, estimada en decenas de miles (no es censo del INE, pero sí es la
#      referencia española más directa para la comunidad judía).
#    - Para islam y catolicismo, la referencia primaria es la encuesta del CIS
#      y datos de observatorios religiosos españoles, no un censo del INE por
#      confesión.
# 2) Signo zodiacal:
#    - no existe un censo o encuesta oficial del INE por signo zodiacal. Por
#      eso se usa una contextualización matemática: con una natalidad distribuida
#      sin sesgo fuerte por mes, cada signo es aprox. 1/12 del total.
# 3) Orientación sexual:
#    - no hay variable oficial del INE sobre orientación sexual. La referencia
#      fiable en España es la evidencia sociológica/encuestas LGBT+ españolas y
#      europeas, que situan la población LGBTIQ+ en torno a ~4-8%, y la
#      mayoría de la población en la categoría heterosexual.
#
# IMPORTANTÍSIMO: estas cantidades son CONTEXTUALES, NO oficiales del INE, y
# NO se usan en `k_anonymity.py` ni en `remaining_population` para riesgo de
# reidentificación. Se usan solo para mostrar en el informe: "si la población
# de España es N, este atributo afecta a X personas aproximadas".
#
# Fórmula de conteo contextual:
#   personas_aproximadas = TOTAL_POPULATION_ES * porcentaje
# Ejemplo: 49_128_297 * 0.001 = ~49_128 personas (judíos en España, orden de
# magnitud aproximado según fuentes comunitarias y barómetros).
RELIGION_DISTRIBUTION = {
    "judaismo": 0.001,      # ~0,1% = ~49k personas en España (fuente comunitaria: FCJE / contexto religioso español)
    "islam": 0.022,         # ~2,2% = ~1,1M personas en España (fuente: CIS/observatorios religiosos)
    "catolicismo": 0.63,    # ~63% ≈ 31M personas (contexto religioso español, CIS)
    "cristianismo": 0.67,   # ~67% (contexto general de identidad cristiana, no censo oficial)
    "budismo": 0.005,       # ~0,5% ≈ 245k personas
    "hinduismo": 0.003,     # ~0,3% ≈ 147k personas
    "ateismo": 0.18,        # ~18% ≈ 8,8M personas
    "agnosticismo": 0.08,   # ~8% ≈ 3,9M personas
}

# Signo zodiacal: distribución contextual uniforme en España por fecha de
# nacimiento, no oficial. 1/12 exacto (no 0,0833 redondeado -- 12 × 0,0833
# = 0,9996, no 1,0: detectado por un test de regresión que comprueba que
# la distribución suma 1, ver test_k_anonymity.py::TestSignoZodiacalStep).
ZODIAC_DISTRIBUTION = {
    "aries": 1 / 12,
    "tauro": 1 / 12,
    "geminis": 1 / 12,
    "cancer": 1 / 12,
    "leo": 1 / 12,
    "virgo": 1 / 12,
    "libra": 1 / 12,
    "escorpio": 1 / 12,
    "sagitario": 1 / 12,
    "capricornio": 1 / 12,
    "acuario": 1 / 12,
    "piscis": 1 / 12,
}

# Orientación sexual: referencia contextual para España, no oficial. La
# mayoría es heterosexual y una minoría LGBTIQ+ se mueve alrededor del 4-8%.
# "heterosexual" es la categoría residual: se ajustó de 0,94 a 0,914 para
# que el conjunto sume exactamente 1 (con 0,94 sumaba 1,026 -- "homosexual"
# se añadió como término de autodeclaración distinto de "gay"/"lesbiana"
# sin volver a cuadrar el resto de la tabla; detectado por un test de
# regresión, ver test_k_anonymity.py::TestOrientacionSexualStep). Se ajusta
# la categoría mayoritaria/residual en vez de las minoritarias porque estas
# últimas son las que llevan más trabajo de estimación detrás.
SEXUAL_ORIENTATION_DISTRIBUTION = {
    "heterosexual": 0.914,
    "gay": 0.022,
    "lesbiana": 0.015,
    "bisexual": 0.017,
    "pansexual": 0.002,
    "asexual": 0.002,
    "homosexual": 0.028,
}

# Reparto por sexo. Aproximado (España tiene ligera mayoría femenina por
# esperanza de vida más alta en edades avanzadas).
SEX_DISTRIBUTION = {
    "hombre": 0.492,
    "mujer": 0.508,
}

# Estado civil (simplificado a 5 categorías, ver DemographicFindings.estado_civil).
# No es una tabla del INE tal cual -- se DERIVA combinando dos fuentes
# reales del INE que miden cosas distintas:
#   1. Censo Anual de Población 2024 (INE, nota de prensa de dic. 2025):
#      estado civil LEGAL a 1 de enero de 2024 -- soltero 34,9%, casado
#      45,8%, divorciado/separado 7,8%, viudo 7,0%.
#   2. ECEPOV 2021 (INE): ~70% de la población de 16+ años "tiene pareja"
#      en sentido amplio (no solo matrimonio).
# "casado", "viudo" y "divorciado" se toman directamente de (1) -- son
# categorías legales inequívocas, no hace falta derivarlas. "con_pareja"
# (pareja sin estar casado/a: parejas de hecho, relaciones sin convivencia
# legal...) se deriva como la diferencia entre "tiene pareja en sentido
# amplio" (2) y "casado" (1): 0.70 - 0.458 = 0.242, y "soltero" como el
# complementario de todo lo anterior: 1 - 0.458 - 0.242 - 0.078 - 0.070 =
# 0.152.
#
# NOTA sobre "divorciado" (0.078): el Censo mide "divorciado/separado" como
# UNA sola categoría legal, sin distinguir si esa persona tiene pareja
# actual o no -- una parte de ese 7,8% podría en realidad tener ya una
# nueva pareja informal y estar contada también dentro del "con_pareja" de
# (2). Se trata aquí como categoría propia y excluyente de las demás
# (igual que el resto de esta tabla) porque es lo que el usuario
# autodeclara ("estoy divorciado/a"), no porque el Censo garantice que sea
# mutuamente excluyente con "con_pareja" -- limitación heredada de que no
# existe una única encuesta pública que cruce las 5 categorías a la vez
# (misma limitación ya documentada arriba para "con_pareja"/"soltero").
# BUG ENCONTRADO Y CORREGIDO en esta sesión (detectado por
# test_the_three_categories_sum_to_the_total_population /
# test_marital_status_by_sex_sums_to_one_per_sex en CI): al aplicar esta
# tabla, `_normalize_marital_status` en update_ine_reference.py
# redondeaba las 5 categorías de forma INDEPENDIENTE en vez de calcular
# "soltero" como el complementario de las otras cuatro YA REDONDEADAS --
# que es justo lo que dice el comentario de más abajo que debía pasar.
# El resultado sumaba 1.001 en vez de 1.0. Se ha ajustado "soltero"
# (0.153 -> 0.152) para que absorba ese residuo, y se ha corregido
# `_normalize_marital_status` en el script para que la próxima vez que
# se aplique esta tabla no vuelva a pasar. Ver también
# MARITAL_STATUS_BY_SEX["hombre"] más abajo, mismo bug.
MARITAL_STATUS_DISTRIBUTION = {
    "casado": 0.458,
    "con_pareja": 0.242,
    "divorciado": 0.078,
    "soltero": 0.152,
    "viudo": 0.070,
}

# Estado civil CONDICIONADO por sexo (aprox. 2024, INE -- Población
# residente por sexo y estado civil), es decir P(estado_civil | sexo). A
# diferencia de MARITAL_STATUS_DISTRIBUTION (marginal, ignora el sexo), esta
# tabla da la proporción REAL de cada combinación sexo+estado_civil, no una
# aproximación asumiendo independencia -- que no lo son: las mujeres son
# viudas en una proporción bastante mayor que los hombres (esperanza de vida
# más alta, y suelen ser algo más jóvenes que su pareja). Se usa en
# k_anonymity.py::_step_relacion SOLO cuando también se conoce el sexo de la
# persona (aplicado antes en la cadena): en ese caso da el porcentaje EXACTO
# de esa combinación concreta, en vez de multiplicar dos proporciones
# marginales por separado. Cada sub-diccionario suma 1.0 (son P(x|sexo), no
# P(x) sin más).
#
# "divorciado" por sexo NO viene de una tabla INE cruzada real (no se
# encontró una desagregada por sexo al construir esta tabla) -- se aproxima
# repartiendo el 7,8% nacional (ver MARITAL_STATUS_DISTRIBUTION) dentro de
# cada sexo con el mismo criterio que "con_pareja"/"soltero": se resta
# proporcionalmente de esos dos, manteniendo intactos "casado" y "viudo"
# (esos sí son cifras legales directas). Doble aproximación (nacional +
# reparto), documentada así para que quede claro que es menos fiable que el
# resto de esta tabla.
MARITAL_STATUS_BY_SEX = {
    "hombre": {
        "casado": 0.464,
        "con_pareja": 0.270,
        "divorciado": 0.072,
        "soltero": 0.169,
        "viudo": 0.025,
    },
    "mujer": {
        "casado": 0.452,
        "con_pareja": 0.216,
        "divorciado": 0.083,
        "soltero": 0.136,
        "viudo": 0.113,
    },
}

# Proporción de población por EDAD EXACTA (0-100), año a año.
#
# Fuente: INE, "Cifras de Población" / Padrón continuo, tabla 01003
# ("Población por edad -año a año-, Españoles/Extranjeros, Sexo y Año",
# https://www.ine.es/jaxi/Tabla.htm?path=/t20/e245/p08/l0/&file=01003.px),
# columna "Ambos sexos", año 2022 (última con datos año-a-año descargados
# en esta sesión -- el selector de esa tabla no es un CSV descargable
# directamente por URL, hay que exportarlo a mano desde el navegador).
# Reemplaza un primer borrador que DERIVABA esta distribución repartiendo
# uniformemente AGE_DISTRIBUTION_5Y dentro de cada tramo quinquenal (ver
# `_build_age_distribution_1y_approx`, más abajo, que se conserva solo
# como referencia histórica de esa aproximación) -- esa aproximación
# llegó a producir tasas de práctica deportiva por tramo de edad
# superiores al 100% al cruzarla con la Encuesta de Hábitos Deportivos
# 2024/25 (el tramo 15-24, el más pequeño y sensible a errores, es donde
# más se notaba), señal de que el reparto uniforme dentro de cada tramo
# de 5 años no se ajustaba lo bastante bien a la pirámide real en esa
# franja concreta.
#
# Desfase de año: esta tabla es de 2022, TOTAL_POPULATION_ES es de 2025
# (Censo a 1 de enero de 2025, ~1,65M más de población total, sobre todo
# por migración neta). Se usa esta distribución como PROPORCIÓN (no como
# cifra absoluta) aplicada sobre TOTAL_POPULATION_ES, asumiendo que la
# FORMA de la pirámide de edad no cambia mucho en 2-3 años aunque el
# total crezca -- la misma clase de aproximación que ya hacía la versión
# anterior (uniforme dentro de tramo), pero based on datos reales de
# población en vez de un reparto artificial, lo cual evita el problema
# de tasas por encima del 100%.
AGE_DISTRIBUTION_1Y = {
    0: 0.006754, 1: 0.007176, 2: 0.007585, 3: 0.008025, 4: 0.00858,
    5: 0.009041, 6: 0.009293, 7: 0.009466, 8: 0.00938, 9: 0.009937,
    10: 0.010194, 11: 0.010428, 12: 0.01061, 13: 0.011093, 14: 0.010699,
    15: 0.010668, 16: 0.010434, 17: 0.010417, 18: 0.010385, 19: 0.010158,
    20: 0.010192, 21: 0.010332, 22: 0.010174, 23: 0.009977, 24: 0.010215,
    25: 0.010194, 26: 0.010329, 27: 0.01053, 28: 0.010908, 29: 0.011237,
    30: 0.011217, 31: 0.011394, 32: 0.011588, 33: 0.011808, 34: 0.011959,
    35: 0.012257, 36: 0.012659, 37: 0.013144, 38: 0.013579, 39: 0.014339,
    40: 0.01494, 41: 0.015655, 42: 0.016074, 43: 0.016757, 44: 0.01696,
    45: 0.017238, 46: 0.01709, 47: 0.016952, 48: 0.016504, 49: 0.016411,
    50: 0.016122, 51: 0.015884, 52: 0.015673, 53: 0.015516, 54: 0.015575,
    55: 0.015163, 56: 0.015011, 57: 0.015167, 58: 0.014381, 59: 0.013854,
    60: 0.01349, 61: 0.013479, 62: 0.013056, 63: 0.012697, 64: 0.012334,
    65: 0.011445, 66: 0.011052, 67: 0.010477, 68: 0.010417, 69: 0.01028,
    70: 0.009564, 71: 0.00923, 72: 0.009482, 73: 0.009783, 74: 0.00882,
    75: 0.008304, 76: 0.008505, 77: 0.007927, 78: 0.007635, 79: 0.006405,
    80: 0.005736, 81: 0.006785, 82: 0.004307, 83: 0.004672, 84: 0.004973,
    85: 0.005099, 86: 0.004606, 87: 0.004193, 88: 0.00385, 89: 0.003411,
    90: 0.002812, 91: 0.002406, 92: 0.001897, 93: 0.001531, 94: 0.001134,
    95: 0.000877, 96: 0.000633, 97: 0.00046, 98: 0.000319, 99: 0.000221,
    100: 0.000414,
}


def _build_age_distribution_5y_from_1y() -> dict[str, float]:
    """Deriva AGE_DISTRIBUTION_5Y sumando AGE_DISTRIBUTION_1Y dentro de
    cada tramo quinquenal -- garantiza que ambas tablas sean consistentes
    entre sí por construcción (antes era al revés: 1Y se derivaba de 5Y
    repartiendo uniformemente, ver comentario en AGE_DISTRIBUTION_1Y de
    arriba sobre por qué se invirtió)."""
    bands: dict[str, float] = {}
    for age, proportion in AGE_DISTRIBUTION_1Y.items():
        band = age_bin(age)
        bands[band] = bands.get(band, 0.0) + proportion
    return bands


def age_bin(age: int) -> str:
    """Convierte una edad concreta en su tramo quinquenal de AGE_DISTRIBUTION_5Y."""
    if age >= 85:
        return "85+"
    lower = (age // 5) * 5
    return f"{lower}-{lower + 4}"


# Distribución de edad en tramos de 5 años, proporción sobre el total --
# derivada de AGE_DISTRIBUTION_1Y (ver función de arriba), no al revés.
# Se mantiene por si se necesita el agregado por tramo en algún otro
# sitio; k_anonymity.py usa AGE_DISTRIBUTION_1Y directamente.
AGE_DISTRIBUTION_5Y = _build_age_distribution_5y_from_1y()


def age_range_proportion(min_age: int, max_age: int) -> float:
    """Suma la proporción de AGE_DISTRIBUTION_1Y entre min_age y max_age
    (ambos incluidos). A diferencia de `age_bin` (que encaja una edad en
    uno de los tramos FIJOS de AGE_DISTRIBUTION_5Y), esto admite un rango
    de ANCHO ARBITRARIO -- pensado para estimaciones de edad por IA donde
    el propio rango se ensancha o estrecha según la confianza real de la
    pista (ver `ai_attribute_extraction.py::_set_edad_rango`), en vez de
    forzarlo a encajar en un tramo quinquenal prefijado. Si `min_age`
    o `max_age` caen fuera de 0-100 (rango de AGE_DISTRIBUTION_1Y), esas
    edades fuera de rango simplemente no suman nada (`.get(edad, 0.0)`),
    no lanzan error -- un rango que se salga un poco por los extremos
    (p. ej. 95-110) sigue siendo válido, solo pierde algo de precisión en
    la cola que no existe en la tabla."""
    return sum(AGE_DISTRIBUTION_1Y.get(age, 0.0) for age in range(min_age, max_age + 1))


# Claves canónicas de las comunidades autónomas cuyo nombre se repite
# muchas veces a lo largo de este fichero (como clave de provincia -- cuando
# coincide -- de comunidad, de nombre legible y de múltiples alias). Se
# definen como constantes en vez de repetir el literal (SonarCloud:
# "Define a constant instead of duplicating this literal").
_CCAA_LA_RIOJA = "la rioja"
_CCAA_CASTILLA_Y_LEON = "castilla y leon"
_CCAA_CASTILLA_LA_MANCHA = "castilla la mancha"
_CCAA_COMUNIDAD_VALENCIANA = "comunidad valenciana"
_CCAA_PAIS_VASCO = "pais vasco"


# Población por provincia (aprox. 2024, INE - Estadística Continua de
# Población). Cubre una selección representativa; añade más si tu análisis
# lo necesita. Claves en minúsculas, sin tildes para facilitar el matching.
PROVINCE_POPULATION = {
    "madrid": 7_113_886,
    "barcelona": 5_959_941,
    "valencia": 2_763_996,
    "sevilla": 1_977_664,
    "alicante": 2_033_566,
    "malaga": 1_791_183,
    "murcia": 1_586_989,
    "cadiz": 1_261_420,
    "vizcaya": 1_167_233,
    "a coruna": 1_135_623,
    "baleares": 1_249_844,
    "las palmas": 1_171_547,
    "santa cruz de tenerife": 1_087_319,
    "zaragoza": 998_443,
    "asturias": 1_015_128,
    "pontevedra": 947_818,
    "granada": 945_797,
    "tarragona": 875_530,
    "gerona": 830_429,
    "castellon": 627_620,
    "toledo": 755_081,
    "badajoz": 665_155,
    "cordoba": 773_163,
    "jaen": 618_143,
    "navarra": 683_854,
    "almeria": 770_554,
    "guipuzcoa": 733_149,
    "valladolid": 528_644,
    "cantabria": 593_623,
    "leon": 448_030,
    "lerida": 458_226,
    "huelva": 538_789,
    "burgos": 362_663,
    "caceres": 388_190,
    "salamanca": 328_446,
    _CCAA_LA_RIOJA: 326_803,
    "lugo": 326_022,
    "orense": 305_278,
    "albacete": 390_751,
    "guadalajara": 285_839,
    "ciudad real": 494_848,
    "alava": 341_961,
    "huesca": 230_087,
    "zamora": 165_564,
    "avila": 160_738,
    "palencia": 158_702,
    "segovia": 158_251,
    "teruel": 136_091,
    "cuenca": 199_859,
    "soria": 90_183,
    "ceuta": 83_567,
    "melilla": 87_067,
}

# Población por municipio (aprox.). Solo capitales/ciudades grandes de
# ejemplo; amplía según necesites. Cuando se detecta un municipio, se usa
# ESTA tabla en lugar de la de provincia (más específica), no ambas a la vez.
MUNICIPALITY_POPULATION = {
    "madrid": 3_330_000,
    "barcelona": 1_660_000,
    "valencia": 800_000,
    "sevilla": 690_000,
    "zaragoza": 680_000,
    "malaga": 590_000,
    "murcia": 460_000,
    "bilbao": 345_000,
    "leon": 122_000,
    "salamanca": 143_000,
    "avila": 57_000,
    "valladolid": 296_000,
    "burgos": 174_000,
    "santander": 172_000,
    "vitoria": 253_000,
    "gijon": 267_000,
    "oviedo": 220_000,
    "pamplona": 205_000,
    "santiago de compostela": 98_000,
    "logrono": 152_000,
    "caceres": 96_000,
    "segovia": 51_000,
    "soria": 39_000,
    "teruel": 35_000,
}

# Mapeo comunidad autónoma -> provincias del INE que la componen (claves de
# PROVINCE_POPULATION). Necesario porque `app/vision/geolocation.py` estima
# ubicación a nivel de COMUNIDAD AUTÓNOMA (es la granularidad de la columna
# "region" del dataset OSV-5M/Nominatim), mientras que PROVINCE_POPULATION
# está a nivel de PROVINCIA -- sin este mapeo, ninguna estimación por imagen
# podía casar nunca con la tabla de provincias y todo salía "no estimable"
# (ver k_anonymity.py: se añade un nivel "comunidad_autonoma" intermedio,
# menos específico que provincia pero más que nada, para las comunidades con
# varias provincias donde no se puede elegir una sola sin más información).
AUTONOMOUS_COMMUNITY_PROVINCES: dict[str, list[str]] = {
    "andalucia": ["sevilla", "cadiz", "malaga", "granada", "cordoba", "jaen", "almeria", "huelva"],
    "aragon": ["zaragoza", "huesca", "teruel"],
    "asturias": ["asturias"],
    "baleares": ["baleares"],
    "canarias": ["las palmas", "santa cruz de tenerife"],
    "cantabria": ["cantabria"],
    _CCAA_CASTILLA_Y_LEON: ["valladolid", "leon", "burgos", "salamanca", "zamora", "avila", "palencia", "segovia", "soria"],
    _CCAA_CASTILLA_LA_MANCHA: ["toledo", "ciudad real", "albacete", "guadalajara", "cuenca"],
    "cataluna": ["barcelona", "tarragona", "gerona", "lerida"],
    _CCAA_COMUNIDAD_VALENCIANA: ["valencia", "alicante", "castellon"],
    "extremadura": ["badajoz", "caceres"],
    "galicia": ["a coruna", "pontevedra", "lugo", "orense"],
    "madrid": ["madrid"],
    "murcia": ["murcia"],
    "navarra": ["navarra"],
    _CCAA_PAIS_VASCO: ["vizcaya", "guipuzcoa", "alava"],
    _CCAA_LA_RIOJA: [_CCAA_LA_RIOJA],
    "ceuta": ["ceuta"],
    "melilla": ["melilla"],
}

# Nombre legible de cada comunidad autónoma (para las etiquetas del
# informe), a partir de su clave interna en AUTONOMOUS_COMMUNITY_PROVINCES.
AUTONOMOUS_COMMUNITY_DISPLAY_NAMES: dict[str, str] = {
    "andalucia": "Andalucía",
    "aragon": "Aragón",
    "asturias": "Asturias",
    "baleares": "Islas Baleares",
    "canarias": "Canarias",
    "cantabria": "Cantabria",
    _CCAA_CASTILLA_Y_LEON: "Castilla y León",
    _CCAA_CASTILLA_LA_MANCHA: "Castilla-La Mancha",
    "cataluna": "Cataluña",
    _CCAA_COMUNIDAD_VALENCIANA: "Comunidad Valenciana",
    "extremadura": "Extremadura",
    "galicia": "Galicia",
    "madrid": "Comunidad de Madrid",
    "murcia": "Región de Murcia",
    "navarra": "Comunidad Foral de Navarra",
    _CCAA_PAIS_VASCO: "País Vasco",
    _CCAA_LA_RIOJA: "La Rioja",
    "ceuta": "Ceuta",
    "melilla": "Melilla",
}

# Alias reconocidos -> clave canónica de AUTONOMOUS_COMMUNITY_PROVINCES. Ya
# normalizados (sin tildes, minúsculas) porque `resolve_autonomous_community`
# normaliza el valor de entrada con el mismo criterio antes de buscar aquí.
# Incluye variantes en español (con/sin "comunidad de"/"principado de"...) Y
# en inglés, porque el "region" de OSV-5M/Nominatim puede venir en cualquiera
# de los dos idiomas según cómo se generaron los metadatos del dataset.
_AUTONOMOUS_COMMUNITY_ALIASES: dict[str, str] = {
    "andalucia": "andalucia",
    "andalusia": "andalucia",
    "aragon": "aragon",
    "asturias": "asturias",
    "principado de asturias": "asturias",
    "baleares": "baleares",
    "islas baleares": "baleares",
    "illes balears": "baleares",
    "balearic islands": "baleares",
    "canarias": "canarias",
    "islas canarias": "canarias",
    "canary islands": "canarias",
    "cantabria": "cantabria",
    _CCAA_CASTILLA_Y_LEON: _CCAA_CASTILLA_Y_LEON,
    "castilla-y-leon": _CCAA_CASTILLA_Y_LEON,
    "castile and leon": _CCAA_CASTILLA_Y_LEON,
    "castille and leon": _CCAA_CASTILLA_Y_LEON,
    _CCAA_CASTILLA_LA_MANCHA: _CCAA_CASTILLA_LA_MANCHA,
    "castilla-la mancha": _CCAA_CASTILLA_LA_MANCHA,
    "castile-la mancha": _CCAA_CASTILLA_LA_MANCHA,
    "cataluna": "cataluna",
    "catalunya": "cataluna",
    "catalonia": "cataluna",
    _CCAA_COMUNIDAD_VALENCIANA: _CCAA_COMUNIDAD_VALENCIANA,
    "comunitat valenciana": _CCAA_COMUNIDAD_VALENCIANA,
    "valencian community": _CCAA_COMUNIDAD_VALENCIANA,
    "region of valencia": _CCAA_COMUNIDAD_VALENCIANA,
    "extremadura": "extremadura",
    "galicia": "galicia",
    "madrid": "madrid",
    "comunidad de madrid": "madrid",
    "community of madrid": "madrid",
    "murcia": "murcia",
    "region de murcia": "murcia",
    "region of murcia": "murcia",
    "murcia region": "murcia",
    "navarra": "navarra",
    "comunidad foral de navarra": "navarra",
    "navarre": "navarra",
    _CCAA_PAIS_VASCO: _CCAA_PAIS_VASCO,
    "euskadi": _CCAA_PAIS_VASCO,
    "basque country": _CCAA_PAIS_VASCO,
    _CCAA_LA_RIOJA: _CCAA_LA_RIOJA,
    "rioja": _CCAA_LA_RIOJA,
    "ceuta": "ceuta",
    "melilla": "melilla",
}


def resolve_autonomous_community(raw_region: str) -> str | None:
    """Normaliza `raw_region` (tal como viene de geolocation.py, en español
    o inglés, con o sin tildes/mayúsculas) y devuelve la clave canónica de
    AUTONOMOUS_COMMUNITY_PROVINCES, o None si no se reconoce (p.ej. otro
    país, o "desconocido" cuando el índice FAISS no tenía dato de región).
    Requiere COINCIDENCIA EXACTA tras normalizar -- para eso sirve (un
    campo estructurado que ya trae solo el nombre de la región). Para
    texto libre en una frase, usar `resolve_autonomous_community_in_text`."""
    normalized = _strip_accents(raw_region).strip().lower()
    return _AUTONOMOUS_COMMUNITY_ALIASES.get(normalized)


def resolve_autonomous_community_in_text(text: str) -> str | None:
    """Como `resolve_autonomous_community`, pero para una frase libre (p.ej.
    "canarias, aunque he vivido fuera"): busca si algún alias conocido
    aparece como SUBCADENA del texto ya normalizado -- mismo criterio de
    coincidencia por subcadena que ya usan las tablas de provincia/
    municipio en demographic_extraction.py y ai_attribute_extraction.py.
    Usado por la detección por regex de autodeclaraciones tipo "vivo en
    Canarias" (comunidad autónoma completa, sin provincia concreta)."""
    normalized = _strip_accents(text).strip().lower()
    matched_alias = next((alias for alias in _AUTONOMOUS_COMMUNITY_ALIASES if alias in normalized), None)
    return _AUTONOMOUS_COMMUNITY_ALIASES.get(matched_alias) if matched_alias else None


def _strip_accents(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _build_ccaa_population() -> dict[str, int]:
    """Suma la población de las provincias de cada comunidad autónoma a
    partir de PROVINCE_POPULATION (que cubre las 50 provincias + Ceuta y
    Melilla), en vez de mantener una segunda cifra hardcodeada que podría
    quedar desincronizada si se actualiza PROVINCE_POPULATION."""
    return {
        ccaa: sum(PROVINCE_POPULATION[province] for province in provinces)
        for ccaa, provinces in AUTONOMOUS_COMMUNITY_PROVINCES.items()
    }


# Población por comunidad autónoma, derivada de PROVINCE_POPULATION (ver
# _build_ccaa_population). Usada por k_anonymity.py cuando la geolocalización
# por imagen solo puede identificar la comunidad autónoma (varias provincias
# posibles) y no una provincia concreta.
CCAA_POPULATION: dict[str, int] = _build_ccaa_population()


# Reverso de AUTONOMOUS_COMMUNITY_PROVINCES: provincia -> comunidad autónoma
# a la que pertenece. Usado por report/generator.py para agrupar varias
# fotos por comunidad autónoma aunque cada una se haya resuelto a un nivel
# distinto (p.ej. una foto dio directamente "Sevilla" y otra dio
# "Andalusia": ambas deben sumar como señal de la misma comunidad).
PROVINCE_TO_CCAA: dict[str, str] = {
    province: ccaa for ccaa, provinces in AUTONOMOUS_COMMUNITY_PROVINCES.items() for province in provinces
}


# Proporción de la población adulta (25-64) con una titulación/ámbito de
# estudio concreto.
#
# RESUELTO con el MÉTODO DE DOS PASOS (versión final, no el de respaldo
# de una sesión anterior): el INE no publica esta granularidad de
# titulación concreta (ver HISTORIAL DE INVESTIGACIÓN más abajo), pero sí
# es automatizable combinando dos fuentes reales del Ministerio de
# Ciencia/Universidades (ninguna con API en vivo -- se descargan a mano o
# vía el intento de descarga automática de
# `scripts/update_studies_distribution.py`, que cae a instrucciones
# manuales si falla):
#
#  1. TOTAL POR RAMA, ROBUSTO: suma de EGRESADOS (graduados, no
#     matriculados -- cada persona cuenta una sola vez, el año que se
#     titula) desde el curso 1985-1986 hasta 2023-2024, fichero "Series
#     históricas de estudiantes universitarios... Total SUE... Egresados
#     por nivel de estudio, sexo y rama de enseñanza" -- 8.622.396
#     egresados acumulados en total, repartidos en las 5 ramas amplias
#     (Ciencias de la Salud, Ciencias Sociales y Jurídicas, Ingeniería y
#     Arquitectura, Ciencias, Artes y Humanidades).
#  2. REPARTO DENTRO DE CADA RAMA: usando el detalle reciente por
#     titulación concreta (Egresados 2015-2023, con Matriculados
#     2015-2024 como respaldo si una titulación no tiene egresados en ese
#     rango), se calculó qué proporción de cada rama corresponde a cada
#     una de las 14 categorías de aquí abajo (p. ej. dentro de "Ciencias
#     de la Salud": medicina/enfermeria/farmacia/veterinaria), asumiendo
#     que ese reparto reciente aproxima el reparto histórico de la rama.
#
# total_estimado[carrera] = total_historico_egresados[rama] x reparto_reciente[carrera dentro de la rama]
# proporción_final[carrera] = total_estimado[carrera] / población_25_64
#
# población_25_64 (27.659.231) se calculó de verdad, no se asumió un
# "40%": TOTAL_POPULATION_ES x suma de AGE_DISTRIBUTION_5Y para los
# tramos 25-29 a 60-64, ambos ya en este mismo fichero.
#
# COMPROBACIÓN DE CONSISTENCIA hecha antes de aplicar esto: la suma de
# los totales históricos de las 5 ramas (8.622.396) cuadra EXACTAMENTE
# con el bloque "Total" del propio fichero histórico (misma cifra) --
# confirma que el parseo de columnas/filas del Excel (una tabla ancha con
# 6 bloques de 39 columnas -uno por año- y una jerarquía anidada de filas
# por Universidad > Nivel > Sexo) está cogiendo la fila y los rangos de
# columna correctos.
#
# LIMITACIONES CONOCIDAS: clasificación del reparto por PALABRAS CLAVE,
# no código oficial (a diferencia de OCCUPATION_DISTRIBUTION con la
# CNO-11 -- ver `_CNO11_SUBGRUPO_TO_APP_CATEGORY` en
# update_ine_reference.py para contraste; algún caso límite conocido,
# p. ej. "Bioinformática" cae también en "ingenieria informatica" por
# compartir subcadena); dobles grados cuentan enteros en las dos
# categorías que combinan, no repartidos; el reparto DENTRO de cada rama
# usa datos recientes (2015-2023) aplicados al total histórico completo
# (desde 1985), asumiendo que la popularidad relativa de cada carrera
# dentro de su rama no ha cambiado radicalmente en 40 años -- una
# aproximación, no una medición directa. Ver
# `scripts/update_studies_distribution.py` para el cálculo completo,
# reproducible con `--matriculados --egresados --egresados-historico-rama`.
#
# HISTORIAL DE INVESTIGACIÓN (para contexto, ya no aplica): el INE no
# tiene tabla anual con esta granularidad (solo la EILU, puntual); el
# SIIU del Ministerio de Ciencia/Universidades sí publica anualmente por
# "campo de estudio", pero sin API confirmada -- lo que sí resultó
# accesible y suficiente fue este otro fichero de series históricas por
# titulación, encontrado y descargado por Nacho. Ver el docstring de
# `scripts/update_ine_reference.py` (secciones "INVESTIGACIÓN de
# STUDIES_DISTRIBUTION..." y "BÚSQUEDA FUERA DEL INE...") para el
# historial completo de intentos previos, incluido el método de respaldo
# (reparto reciente x 40% asumido) usado antes de conseguir el fichero
# histórico por rama.
# Nivel de formación MÁXIMO alcanzado por la población, en los 3 tramos
# de la clasificación CNED-2014/ISCED que usa el INE/EPA: "superior"
# (universidad + FP grado superior + doctorado, niveles 5-8),
# "secundaria_superior" (bachillerato + FP grado medio + postsecundaria no
# superior, niveles 3-4) y "secundaria_o_inferior" (ESO, primaria, sin
# estudios, niveles 0-2). DISTINTO de STUDIES_DISTRIBUTION (más abajo),
# que es la CARRERA UNIVERSITARIA concreta, no el nivel alcanzado.
#
# Fuente: INE/EPA (Encuesta de Población Activa), indicador "Nivel de
# formación de la población adulta", año 2024, por sexo:
#   hombres: secundaria_o_inferior 38,3% / secundaria_superior 23,0% / superior 38,7%
#   mujeres: secundaria_o_inferior 31,9% / secundaria_superior 22,8% / superior 45,3%
# Combinado ponderando por SEX_DISTRIBUTION (misma técnica que
# MARITAL_STATUS_BY_SEX -- ver su comentario -- para obtener un marginal
# único a partir de cifras oficiales por sexo).
#
# LIMITACIÓN IMPORTANTE (documentada a propósito, no oculta): el
# indicador EPA cubre población de 25 a 64 años, NO toda la población de
# 15+ que es la referencia habitual del resto de este fichero. Se aplica
# aquí igualmente como aproximación de la población general porque:
# (a) es la única cifra oficial de nivel educativo con desglose por sexo
# fácilmente verificable, y (b) es el rango de edad estándar que usan
# INE/Eurostat precisamente porque medir "nivel COMPLETADO" en gente más
# joven (todavía cursando estudios) o considerar por igual a gente mucho
# mayor (con un contexto de acceso a la educación muy distinto en su
# época) sesga la comparación en ambos sentidos. El resultado es una
# aproximación razonable para adultos en edad típica de trabajar, pero
# probablemente sobre-estima "superior" en el extremo joven (15-24, donde
# mucha gente aún no ha terminado) y lo infra-estima en el extremo mayor
# (65+, generación con mucho menor acceso histórico a estudios
# superiores) -- mismo tipo de aproximación que ya se acepta en otras
# tablas de este fichero (p. ej. golf/2022, ver historial de
# SPORT_PRACTICE_DISTRIBUTION), documentada en vez de escondida.
EDUCATION_LEVEL_DISTRIBUTION = {
    "secundaria_o_inferior": 0.3505,
    "secundaria_superior": 0.229,
    "superior": 0.4205,
}


STUDIES_DISTRIBUTION = {
    "medicina": 0.0128,
    "enfermeria": 0.0231,
    "derecho": 0.0308,
    "ingenieria informatica": 0.0300,
    "ingenieria industrial": 0.0114,
    "administracion de empresas": 0.0328,
    "psicologia": 0.0331,
    "magisterio": 0.0524,
    "arquitectura": 0.0127,
    "farmacia": 0.0048,
    "biologia": 0.0201,
    "periodismo": 0.0064,
    "economia": 0.0101,
    "veterinaria": 0.0027,
}

# Rama de conocimiento oficial de cada una de las 14 carreras de
# STUDIES_DISTRIBUTION, según la clasificación de 5 ramas del Real
# Decreto 1393/2007 (modificado por RD 43/2015), art. único: Artes y
# Humanidades / Ciencias / Ciencias de la Salud / Ciencias Sociales y
# Jurídicas / Ingeniería y Arquitectura -- la adscripción de cada título
# concreto se registra en el RUCT (Registro de Universidades, Centros y
# Títulos) del Ministerio.
#
# SOLO INFORMATIVO -- NUNCA genera un paso de estrechamiento de
# población propio (ver _step_rama_estudios en k_anonymity.py): quien
# estudia "derecho" ya está, con probabilidad 1, dentro de "Ciencias
# Sociales y Jurídicas" -- aplicar la proporción de la rama ENCIMA de la
# proporción de la carrera concreta (STUDIES_DISTRIBUTION) contaría el
# mismo hecho dos veces y estrecharía la población sin ninguna
# justificación estadística (el suceso "estudia derecho" no es
# independiente del suceso "está en la rama CS y J", es un subconjunto
# exacto). Por eso esta tabla es solo un mapeo de exhibición: se usa
# para RELLENAR `rama_estudios` cuando `estudios` ya se conoce (mismo
# patrón que `nivel_estudios` se infiere de `estudios`), pero el paso de
# k-anonimato de `rama_estudios` se salta por completo en ese caso.
#
# Dos casos verificados explícitamente por no ser obvios (búsqueda
# dedicada, no asumidos): "psicologia" está adscrita oficialmente a
# Ciencias de la Salud como rama PRINCIPAL (BOE-A-2022-12576, código
# RUCT 2502443), aunque muchas facultades también imparten créditos
# básicos de Ciencias Sociales y Jurídicas como rama secundaria -- se
# usa aquí solo la principal. "veterinaria" está adscrita a Ciencias de
# la Salud (confirmado en documentación oficial de admisión de la
# Universidad de Zaragoza y en el propio catálogo de asignaturas de la
# USC), NO a "Ciencias" como podría parecer a primera vista.
STUDIES_TO_RAMA = {
    "medicina": "ciencias_salud",
    "enfermeria": "ciencias_salud",
    "farmacia": "ciencias_salud",
    "psicologia": "ciencias_salud",
    "veterinaria": "ciencias_salud",
    "derecho": "ciencias_sociales_juridicas",
    "administracion de empresas": "ciencias_sociales_juridicas",
    "magisterio": "ciencias_sociales_juridicas",
    "periodismo": "ciencias_sociales_juridicas",
    "economia": "ciencias_sociales_juridicas",
    "ingenieria informatica": "ingenieria_arquitectura",
    "ingenieria industrial": "ingenieria_arquitectura",
    "arquitectura": "ingenieria_arquitectura",
    "biologia": "ciencias",
}

# Proporción de la población que ha estudiado en cada rama de
# conocimiento (marginal, no depende de qué carrera concreta). Fuente:
# Ministerio de Ciencia/Universidades, Estadística de Estudiantes
# Universitarios (EEU) -- matriculados de Grado por rama de enseñanza,
# universidades PÚBLICAS presenciales curso 2015/16 (última serie
# encontrada con desglose completo de las 5 ramas en una sola fuente)
# más universidades PRIVADAS curso 2019/20 (idem, fuente distinta).
#
# LIMITACIÓN DOCUMENTADA: mezcla dos cursos académicos distintos
# (2015/16 público + 2019/20 privado) porque no se encontró una única
# fuente con el desglose completo de las 5 ramas para ambos tipos de
# universidad en el mismo curso. Se valida cruzando con una cifra
# independiente de otro informe (EEU, nuevo ingreso curso 2021/22:
# Ciencias Sociales y Jurídicas 46,4% de las nuevas matrículas, Ciencias
# la menor con 6,3%) -- las proporciones calculadas aquí (46,66% y
# 6,02% respectivamente) coinciden lo bastante bien como para dar
# confianza en el orden de magnitud, a pesar del desfase de cursos.
# Como con otras tablas de este fichero basadas en matriculación/
# egresados en vez of población total, es una aproximación de "quién ha
# pasado por la universidad recientemente", no un censo de toda la
# población adulta que alguna vez estudió cada rama.
RAMA_ESTUDIOS_DISTRIBUTION = {
    "ciencias_sociales_juridicas": 0.4666,
    "ingenieria_arquitectura": 0.1903,
    "ciencias_salud": 0.1806,
    "artes_humanidades": 0.1023,
    "ciencias": 0.0602,
}

# Proporción de la población ocupada por gran categoría profesional
# (aprox., basado en agregados de la EPA/CNO-11).
#
# CANDIDATO DE AUTOMATIZACIÓN LOCALIZADO (sin aplicar todavía): tabla INE
# 65134, "Ocupados por sexo y ocupación" (EPA, subgrupos CNO-11), con un
# mapeo subgrupo->categoría diseñado a mano en
# `scripts/update_ine_reference.py` (`_CNO11_SUBGRUPO_TO_APP_CATEGORY`).
# Sin verificar todavía contra la API real -- ver el docstring de ese
# script (cabecera, sección "INVESTIGACIÓN de STUDIES_DISTRIBUTION y
# OCCUPATION_DISTRIBUTION") para el detalle completo y las limitaciones
# conocidas. Corre `python scripts/update_ine_reference.py` para comparar
# estos valores contra el INE.
OCCUPATION_DISTRIBUTION = {
    "docente": 0.049,
    "sanitario": 0.055,
    "desarrollador de software": 0.018,
    "ingeniero": 0.020,
    "abogado": 0.008,
    "comercial": 0.049,
    "hosteleria": 0.044,
    "administracion publica": 0.065,
    "construccion": 0.060,
    "transporte": 0.030,
}

# Práctica deportiva declarada, por modalidad. Fuente: Encuesta de Hábitos
# Deportivos en España 2022 (Ministerio de Cultura y Deporte / Consejo
# Superior de Deportes, con el INE colaborando en el diseño muestral) --
# estadística oficial de periodicidad quinquenal. Resultados detallados:
# https://www.csd.gob.es/sites/default/files/media/files/2022-12/Encuesta%20de%20H%C3%A1bitos%20Deportivos%20en%20Espa%C3%B1a%202022%20Resultados%20detallados.pdf
# -- cifras por modalidad verificadas cruzando dos fuentes secundarias
# independientes que citan directamente esa publicación oficial
# (esciclismo.com/actualidad/-/73789.html y valgo.es), ambas coincidiendo
# en los mismos porcentajes.
#
# IMPORTANTE -- esta tabla NO es una partición (a diferencia de
# OCCUPATION_DISTRIBUTION o STUDIES_DISTRIBUTION, donde cada persona
# tiene un único valor): la encuesta es de respuesta múltiple, una misma
# persona puede practicar varias modalidades a la vez (la propia
# publicación documenta correlaciones entre modalidades, p. ej. quien
# hace ciclismo también hace senderismo con más frecuencia que la
# población general), así que los porcentajes de abajo son proporciones
# MARGINALES ("de toda la población, qué fracción practica ESTA
# modalidad en concreto"), no probabilidades mutuamente excluyentes -- no
# tiene sentido comprobar que sumen 1, y no deberían forzarse a sumarlo.
#
# Fuente: Encuesta de Hábitos Deportivos en España 2024/25 (Ministerio de
# Educación, FP y Deportes + CSD + INE), tabla 1.21 "Personas que
# practicaron deporte en el último año por modalidad deportiva y
# frecuencia" -- a diferencia de un primer borrador de esta tabla (basado
# en cifras de prensa sobre la edición 2022, que solo cubrían 9
# modalidades porque la prensa nunca cita la tabla completa), esta
# versión sale directamente del fichero oficial descargado del portal de
# estadísticas del Ministerio, con el dato exacto de practicantes en
# miles de TODAS las modalidades que desglosa la encuesta.
#
# Cálculo: (practicantes de esta modalidad en miles ÷ 26.605, total de
# practicantes de ALGÚN deporte en miles) × 0.627 (% de la población de
# 15+ años que practicó algún deporte en el último año, cifra 2024/25 --
# confirmada por prensa oficial del CSD, no viene en este fichero
# concreto). El primer factor es la cuota de esta modalidad DENTRO de
# quienes hacen deporte; el segundo la convierte a proporción sobre la
# POBLACIÓN TOTAL, que es lo que necesita _apply_proportion.
#
# Se excluyen deliberadamente dos filas de la tabla original:
#   - "Total" -- es la fila de cabecera (26.605 miles = 100% de
#     practicantes), no una modalidad.
#   - "Otro deporte" -- es un cajón de sastre sin identidad propia; no
#     existe ninguna frase-ancla de práctica que distinga a alguien
#     declarando "otro deporte" de cualquier otra frase genérica del
#     texto, así que no se puede detectar con el mismo criterio que el
#     resto (frase-ancla específica de la modalidad).
#
# "yoga_pilates" y "baile_fitness" son aproximaciones -- ver comentario
# en cada clave: la encuesta agrupa actividades bajo un paraguas más
# amplio que la frase-ancla concreta que la gente usa para declararlas en
# primera persona (p. ej. "gimnasia suave" también incluye "gimnasia de
# mantenimiento" sin más detalle, que no tiene una frase-ancla propia
# distinguible de una mención genérica). El resto de claves SÍ tiene
# correspondencia 1:1 exacta con una fila de la encuesta.
SPORT_PRACTICE_DISTRIBUTION = {
    "yoga_pilates": 0.183,           # "Gimnasia suave" -- yoga, pilates, tai-chi, gimnasia de mantenimiento (aproximación, ver nota arriba)
    "gimnasia_intensa": 0.171,       # "Gimnasia intensa" -- aerobic, step, spinning (distinto de baile_fitness, ver esa clave)
    "senderismo": 0.156,             # "Senderismo, montañismo"
    "musculacion": 0.147,            # "Musculación, halterofilia"
    "natacion": 0.140,               # "Natación"
    "ciclismo": 0.134,               # "Ciclismo"
    "running": 0.111,                # "Carrera a pie, running, marcha" -- DISTINTO de "atletismo" (ver esa clave), son dos filas separadas en la encuesta
    "padel": 0.096,                  # "Pádel"
    "futbol": 0.068,                 # "Fútbol 11 y 7"
    "baloncesto": 0.041,             # "Baloncesto"
    "baile_fitness": 0.039,          # "Otra act. fís. con música" -- zumba, baile fitness, aerobic con coreografía (aproximación, ver nota arriba)
    "futbol_sala": 0.039,            # "Fútbol sala, fútbol playa" -- DISTINTO de "futbol" (11 y 7) de arriba
    "ajedrez": 0.038,                # "Ajedrez" -- sedentario, pero la propia encuesta oficial lo cuenta como modalidad deportiva (federado en España)
    "tenis": 0.033,                  # "Tenis"
    "tenis_mesa": 0.033,             # "Tenis de mesa" -- ping-pong
    "atletismo": 0.029,              # "Atletismo" -- DISTINTO de "running" de arriba
    "esqui": 0.028,                  # "Deportes de invierno" -- esquí, snowboard
    "voleibol": 0.026,               # "Voleibol"
    "boxeo": 0.021,                  # "Boxeo"
    "submarinismo": 0.020,           # "Actividades subacuáticas" -- buceo, submarinismo
    "pesca": 0.019,                  # "Pesca"
    "patinaje": 0.019,               # "Patinaje, monopatín"
    "petanca": 0.014,                # "Petanca o bolos"
    "golf": 0.014,                   # "Golf, pitch and putt, minigolf" -- ya no es un caso especial (ver ADR): esta tabla oficial lo trata igual que el resto, no como % directo sobre población total
    "artes_marciales": 0.012,        # "Artes marciales" -- DISTINTO de "lucha_defensa_personal" (ver esa clave)
    "piraguismo_remo": 0.012,        # "Piragüismo, remo, descensos"
    "badminton": 0.011,              # "Bádminton"
    "pelota_vasca": 0.011,           # "Frontón, frontenis, trinquete"
    "caza": 0.010,                   # "Caza"
    "motociclismo": 0.009,           # "Motociclismo"
    "surf": 0.008,                   # "Surf"
    "automovilismo": 0.006,          # "Automovilismo"
    "vela": 0.005,                   # "Vela"
    "hipica": 0.005,                 # "Hípica"
    "balonmano": 0.005,              # "Balonmano"
    "triatlon": 0.004,               # "Triatlón"
    "rugby": 0.004,                  # "Rugby, rugby 7"
    "lucha_defensa_personal": 0.003, # "Lucha o defensa personal" -- DISTINTO de "artes_marciales" de arriba
    "esqui_nautico": 0.003,          # "Esquí náutico, motonáutica"
    "squash": 0.002,                 # "Squash"
    "aeronautica": 0.002,            # "Actividades aeronáuticas" -- parapente, ala delta, paracaidismo
}

# Práctica deportiva CONDICIONADA por sexo (tabla 1.22 de la misma
# encuesta y edición, "Personas que practicaron deporte en el último año
# por modalidad deportiva, sexo, edad y nivel de estudios"), es decir
# P(practica X | sexo). Mismo patrón que MARITAL_STATUS_BY_SEX (ver
# arriba) -- se usa en k_anonymity.py::_step_practica_deportiva SOLO
# cuando también se conoce el sexo de la persona (aplicado antes en la
# cadena): da la proporción REAL de esa combinación concreta en vez de
# aplicar la marginal de SPORT_PRACTICE_DISTRIBUTION sin distinguir sexo.
# El efecto es grande: p. ej. "caza" lo practican los hombres ~32 veces
# más que las mujeres; "yoga_pilates" lo practican las mujeres ~4 veces
# más que los hombres. Aplicar la marginal a alguien que ya declaró su
# sexo desperdicia esa señal.
#
# A DIFERENCIA de MARITAL_STATUS_BY_SEX (donde cada sub-diccionario suma
# 1.0 porque es una partición sobre categorías excluyentes), aquí cada
# sub-diccionario NO suma 1.0 -- mismo motivo que SPORT_PRACTICE_DISTRIBUTION
# (encuesta de respuesta múltiple, ver esa tabla arriba): son proporciones
# marginales dentro de cada sexo, no una partición.
#
# Cálculo: la tabla 1.22 da, para cada modalidad, el % de practicantes DE
# ESE SEXO (no de la población total) que hacen esa modalidad -- p. ej.
# "19,0% de los hombres que practican algún deporte juegan al fútbol".
# Para convertirlo a P(fútbol | hombre) sobre la POBLACIÓN masculina total
# (no solo los practicantes), se multiplica por la tasa de práctica
# deportiva DENTRO de cada sexo (practicantes de ese sexo ÷ población de
# ese sexo). Esa tasa no viene en esta tabla -- se ha derivado aquí mismo
# a partir de datos YA presentes en este fichero (TOTAL_POPULATION_ES,
# SEX_DISTRIBUTION y age_range_proportion(0, 14) para estimar población
# 15+), asumiendo que el reparto hombre/mujer es igual dentro de la
# población 15+ que en la población total -- aproximación razonable (el
# desequilibrio de sexos por esperanza de vida se concentra sobre todo en
# edades muy avanzadas, no en el corte "menor/mayor de 15 años"), no una
# cifra oficial de "población española de 15+ por sexo":
#   población_15+ = TOTAL_POPULATION_ES × (1 − age_range_proportion(0, 14))
#   población_hombres_15+ = población_15+ × SEX_DISTRIBUTION["hombre"]
#   tasa_hombres = 13.661.000 (practicantes hombres, tabla 1.21) ÷ población_hombres_15+ = 0,6467
#   población_mujeres_15+ = población_15+ × SEX_DISTRIBUTION["mujer"]
#   tasa_mujeres = 12.945.000 (practicantes mujeres, tabla 1.21) ÷ población_mujeres_15+ = 0,5935
# Y luego, por modalidad: P(X | sexo) = (%_de_practicantes_de_ese_sexo_que_hacen_X ÷ 100) × tasa_ese_sexo
#
# CASOS OMITIDOS A PROPÓSITO: cuando la tabla 1.22 redondea el % de un
# sexo a 0,0 (solo pasa con "squash" y mujeres -- muestra demasiado
# pequeña para el diseño muestral de la encuesta en esa combinación
# concreta), NO se incluye esa clave de sexo en el sub-diccionario. Un
# 0,0% redondeado no significa "cero mujeres practican squash", significa
# "por debajo del umbral de detección de esta encuesta" -- forzar un 0.0
# literal aquí haría que el escalón de estrechamiento mostrara "0
# personas comparten tus rasgos", una certeza que el dato real no
# respalda. Al faltar la clave, _step_practica_deportiva cae de vuelta a
# la marginal de SPORT_PRACTICE_DISTRIBUTION para ese caso -- ver esa
# función.
SPORT_PRACTICE_BY_SEX = {
    "yoga_pilates": {"hombre": 0.0731, "mujer": 0.2849},
    "gimnasia_intensa": {"hombre": 0.1474, "mujer": 0.1899},
    "senderismo": {"hombre": 0.1714, "mujer": 0.1377},
    "musculacion": {"hombre": 0.1908, "mujer": 0.1015},
    "natacion": {"hombre": 0.1365, "mujer": 0.1407},
    "ciclismo": {"hombre": 0.1966, "mujer": 0.0700},
    "running": {"hombre": 0.1306, "mujer": 0.0902},
    "padel": {"hombre": 0.1345, "mujer": 0.0576},
    "futbol": {"hombre": 0.1229, "mujer": 0.0137},
    "baloncesto": {"hombre": 0.0595, "mujer": 0.0220},
    "baile_fitness": {"hombre": 0.0103, "mujer": 0.0665},
    "futbol_sala": {"hombre": 0.0731, "mujer": 0.0059},
    "ajedrez": {"hombre": 0.0576, "mujer": 0.0172},
    "tenis": {"hombre": 0.0479, "mujer": 0.0178},
    "tenis_mesa": {"hombre": 0.0485, "mujer": 0.0166},
    "atletismo": {"hombre": 0.0407, "mujer": 0.0166},
    "esqui": {"hombre": 0.0349, "mujer": 0.0202},
    "voleibol": {"hombre": 0.0265, "mujer": 0.0255},
    "boxeo": {"hombre": 0.0285, "mujer": 0.0125},
    "submarinismo": {"hombre": 0.0252, "mujer": 0.0148},
    "patinaje": {"hombre": 0.0175, "mujer": 0.0208},
    "pesca": {"hombre": 0.0323, "mujer": 0.0059},
    "petanca": {"hombre": 0.0194, "mujer": 0.0089},
    "golf": {"hombre": 0.0188, "mujer": 0.0083},
    "badminton": {"hombre": 0.0116, "mujer": 0.0113},
    "artes_marciales": {"hombre": 0.0168, "mujer": 0.0059},
    "piraguismo_remo": {"hombre": 0.0162, "mujer": 0.0065},
    "pelota_vasca": {"hombre": 0.0168, "mujer": 0.0059},
    "caza": {"hombre": 0.0194, "mujer": 0.0006},
    "motociclismo": {"hombre": 0.0142, "mujer": 0.0024},
    "surf": {"hombre": 0.0097, "mujer": 0.0059},
    "automovilismo": {"hombre": 0.0110, "mujer": 0.0012},
    "vela": {"hombre": 0.0071, "mujer": 0.0030},
    "balonmano": {"hombre": 0.0071, "mujer": 0.0030},
    "hipica": {"hombre": 0.0052, "mujer": 0.0047},
    "triatlon": {"hombre": 0.0065, "mujer": 0.0018},
    "rugby": {"hombre": 0.0058, "mujer": 0.0018},
    "lucha_defensa_personal": {"hombre": 0.0045, "mujer": 0.0024},
    "esqui_nautico": {"hombre": 0.0039, "mujer": 0.0012},
    "squash": {"hombre": 0.0045},  # sin "mujer": redondeaba a 0,0 en la encuesta -- ver nota arriba
    "aeronautica": {"hombre": 0.0013, "mujer": 0.0030},
}


# Práctica deportiva CONDICIONADA por tramo de edad (misma tabla 1.22 que
# SPORT_PRACTICE_BY_SEX, columnas de edad en vez de sexo), es decir
# P(practica X | tramo de edad). Mismo patrón, mismo motivo -- ver el
# comentario largo de SPORT_PRACTICE_BY_SEX arriba para el porqué general
# (no reinventarlo aquí).
#
# Los tramos de la encuesta (15-24, 25-54, 55+) coinciden EXACTAMENTE con
# fronteras de quinquenios estándar del INE, así que la conversión a
# población total no necesita aproximar nada de reparto dentro de tramo:
#   tasa_15_24 = 4.459.000 (practicantes, tabla 1.21) ÷ (TOTAL_POPULATION_ES × age_range_proportion(15, 24)) = 0,8816
#   tasa_25_54 = 14.813.000 ÷ (TOTAL_POPULATION_ES × age_range_proportion(25, 54)) = 0,7171
#   tasa_55_mas = 7.333.000 ÷ (TOTAL_POPULATION_ES × age_range_proportion(55, 100)) = 0,4412
# Y por modalidad: P(X | tramo) = (%_de_practicantes_de_ese_tramo_que_hacen_X ÷ 100) × tasa_ese_tramo
#
# NOTA HISTÓRICA IMPORTANTE: un primer intento de esto usaba
# age_range_proportion() derivado de la APROXIMACIÓN uniforme dentro de
# tramo que tenía AGE_DISTRIBUTION_1Y en su momento -- daba una tasa
# imposible del 100,85% para el tramo 15-24 (más practicantes que
# población). Fue precisamente ESTE cálculo el que forzó a sustituir
# AGE_DISTRIBUTION_1Y por datos reales año-a-año del INE (tabla 01003,
# ver el comentario junto a esa tabla) en vez de seguir aproximando.
#
# CASOS OMITIDOS A PROPÓSITO (mismo criterio que SPORT_PRACTICE_BY_SEX):
# "automovilismo" y "triatlon" no tienen clave "55_mas" -- la encuesta
# redondeó su % en ese tramo a 0,0 (muestra insuficiente en esa
# combinación concreta), no "cero personas de 55+ los practican". Al
# faltar la clave, _step_practica_deportiva cae de vuelta a la marginal
# de SPORT_PRACTICE_DISTRIBUTION para ese caso.

SPORT_PRACTICE_BY_AGE_BAND = {
    "aeronautica": {"15_24": 0.0018, "25_54": 0.0029, "55_mas": 0.0009},
    "ajedrez": {"15_24": 0.0934, "25_54": 0.0423, "55_mas": 0.015},
    "artes_marciales": {"15_24": 0.0353, "25_54": 0.0136, "55_mas": 0.0022},
    "atletismo": {"15_24": 0.0732, "25_54": 0.0359, "55_mas": 0.0066},
    "automovilismo": {"15_24": 0.0212, "25_54": 0.0079},
    "badminton": {"15_24": 0.0432, "25_54": 0.0108, "55_mas": 0.0026},
    "baile_fitness": {"15_24": 0.067, "25_54": 0.043, "55_mas": 0.0269},
    "baloncesto": {"15_24": 0.1516, "25_54": 0.0423, "55_mas": 0.0053},
    "balonmano": {"15_24": 0.0264, "25_54": 0.0029, "55_mas": 0.0013},
    "boxeo": {"15_24": 0.0723, "25_54": 0.0222, "55_mas": 0.0031},
    "caza": {"15_24": 0.0132, "25_54": 0.0086, "55_mas": 0.0106},
    "ciclismo": {"15_24": 0.1569, "25_54": 0.1721, "55_mas": 0.0803},
    "esqui": {"15_24": 0.0503, "25_54": 0.0373, "55_mas": 0.0088},
    "esqui_nautico": {"15_24": 0.0026, "25_54": 0.0036, "55_mas": 0.0018},
    "futbol": {"15_24": 0.2336, "25_54": 0.076, "55_mas": 0.0079},
    "futbol_sala": {"15_24": 0.149, "25_54": 0.0416, "55_mas": 0.0035},
    "gimnasia_intensa": {"15_24": 0.2671, "25_54": 0.223, "55_mas": 0.0781},
    "golf": {"15_24": 0.0282, "25_54": 0.0122, "55_mas": 0.0115},
    "hipica": {"15_24": 0.0123, "25_54": 0.0057, "55_mas": 0.0026},
    "lucha_defensa_personal": {"15_24": 0.0071, "25_54": 0.0036, "55_mas": 0.0018},
    "motociclismo": {"15_24": 0.0141, "25_54": 0.0115, "55_mas": 0.0031},
    "musculacion": {"15_24": 0.3015, "25_54": 0.1929, "55_mas": 0.0446},
    "natacion": {"15_24": 0.1913, "25_54": 0.1513, "55_mas": 0.1121},
    "padel": {"15_24": 0.2248, "25_54": 0.1212, "55_mas": 0.0269},
    "patinaje": {"15_24": 0.0573, "25_54": 0.0244, "55_mas": 0.0013},
    "pelota_vasca": {"15_24": 0.0317, "25_54": 0.0115, "55_mas": 0.0053},
    "pesca": {"15_24": 0.0326, "25_54": 0.0208, "55_mas": 0.0141},
    "petanca": {"15_24": 0.0441, "25_54": 0.0129, "55_mas": 0.0066},
    "piraguismo_remo": {"15_24": 0.0247, "25_54": 0.0158, "55_mas": 0.0026},
    "rugby": {"15_24": 0.0176, "25_54": 0.0029, "55_mas": 0.0004},
    "running": {"15_24": 0.1499, "25_54": 0.1506, "55_mas": 0.0521},
    "senderismo": {"15_24": 0.1402, "25_54": 0.1915, "55_mas": 0.1182},
    "squash": {"15_24": 0.0009, "25_54": 0.0043, "55_mas": 0.0004},
    "submarinismo": {"15_24": 0.0317, "25_54": 0.0301, "55_mas": 0.0044},
    "surf": {"15_24": 0.0282, "25_54": 0.0093, "55_mas": 0.0004},
    "tenis": {"15_24": 0.0899, "25_54": 0.0366, "55_mas": 0.0119},
    "tenis_mesa": {"15_24": 0.0917, "25_54": 0.0351, "55_mas": 0.0119},
    "triatlon": {"15_24": 0.0071, "25_54": 0.0065},
    "vela": {"15_24": 0.0088, "25_54": 0.005, "55_mas": 0.004},
    "voleibol": {"15_24": 0.1296, "25_54": 0.0215, "55_mas": 0.0013},
    "yoga_pilates": {"15_24": 0.1534, "25_54": 0.1915, "55_mas": 0.1831},
}


# Práctica deportiva CONDICIONADA por nivel de estudios (misma tabla 1.22
# que SPORT_PRACTICE_BY_SEX/BY_AGE_BAND, columnas de nivel de estudios en
# vez de sexo/edad), es decir P(practica X | nivel_estudios). Mismo
# patrón, mismo motivo -- ver el comentario largo de SPORT_PRACTICE_BY_SEX
# para el porqué general.
#
# A diferencia de sexo (binario limpio) y edad (tramos que coinciden con
# quinquenios INE exactos), aquí la conversión a población total exige
# apoyarse en EDUCATION_LEVEL_DISTRIBUTION (ver esa tabla arriba, y su
# comentario sobre la limitación de basarse en población 25-64, no 15+):
#   población_15_más = TOTAL_POPULATION_ES × (1 − age_range_proportion(0, 14))
#   población_tramo = población_15_más × EDUCATION_LEVEL_DISTRIBUTION[tramo]
#   tasa_secundaria_o_inferior = 6.289.000 (practicantes, tabla 1.21) ÷ población_tramo = 0,4238
#   tasa_secundaria_superior = 7.067.000 ÷ población_tramo = 0,7289
#   tasa_superior = 13.250.000 ÷ población_tramo = 0,7443
# Y por modalidad: P(X | tramo) = (%_de_practicantes_de_ese_tramo_que_hacen_X ÷ 100) × tasa_ese_tramo
#
# A diferencia de SPORT_PRACTICE_BY_SEX y SPORT_PRACTICE_BY_AGE_BAND, esta
# tabla no tuvo ningún caso de "0,0% redondeado" que omitir -- las 41
# modalidades tienen dato en los 3 tramos.

SPORT_PRACTICE_BY_EDUCATION_LEVEL = {
    "aeronautica": {"secundaria_o_inferior": 0.0008, "secundaria_superior": 0.0029, "superior": 0.003},
    "ajedrez": {"secundaria_o_inferior": 0.0195, "secundaria_superior": 0.0445, "superior": 0.0491},
    "artes_marciales": {"secundaria_o_inferior": 0.0093, "secundaria_superior": 0.016, "superior": 0.0112},
    "atletismo": {"secundaria_o_inferior": 0.0148, "secundaria_superior": 0.0372, "superior": 0.0357},
    "automovilismo": {"secundaria_o_inferior": 0.0047, "secundaria_superior": 0.0051, "superior": 0.0082},
    "badminton": {"secundaria_o_inferior": 0.0085, "secundaria_superior": 0.0175, "superior": 0.0112},
    "baile_fitness": {"secundaria_o_inferior": 0.0174, "secundaria_superior": 0.0496, "superior": 0.0528},
    "baloncesto": {"secundaria_o_inferior": 0.0335, "secundaria_superior": 0.0576, "superior": 0.038},
    "balonmano": {"secundaria_o_inferior": 0.003, "secundaria_superior": 0.0117, "superior": 0.0037},
    "boxeo": {"secundaria_o_inferior": 0.0165, "secundaria_superior": 0.0292, "superior": 0.0194},
    "caza": {"secundaria_o_inferior": 0.0089, "secundaria_superior": 0.0146, "superior": 0.0082},
    "ciclismo": {"secundaria_o_inferior": 0.0737, "secundaria_superior": 0.1662, "superior": 0.1675},
    "esqui": {"secundaria_o_inferior": 0.0089, "secundaria_superior": 0.0313, "superior": 0.0417},
    "esqui_nautico": {"secundaria_o_inferior": 0.0013, "secundaria_superior": 0.0029, "superior": 0.0037},
    "futbol": {"secundaria_o_inferior": 0.0636, "secundaria_superior": 0.094, "superior": 0.0581},
    "futbol_sala": {"secundaria_o_inferior": 0.0335, "secundaria_superior": 0.0561, "superior": 0.0357},
    "gimnasia_intensa": {"secundaria_o_inferior": 0.0763, "secundaria_superior": 0.2034, "superior": 0.233},
    "golf": {"secundaria_o_inferior": 0.0064, "secundaria_superior": 0.0175, "superior": 0.0179},
    "hipica": {"secundaria_o_inferior": 0.0051, "secundaria_superior": 0.0058, "superior": 0.0045},
    "lucha_defensa_personal": {"secundaria_o_inferior": 0.0034, "secundaria_superior": 0.0044, "superior": 0.003},
    "motociclismo": {"secundaria_o_inferior": 0.0068, "secundaria_superior": 0.008, "superior": 0.0104},
    "musculacion": {"secundaria_o_inferior": 0.0631, "secundaria_superior": 0.1779, "superior": 0.2017},
    "natacion": {"secundaria_o_inferior": 0.0771, "secundaria_superior": 0.1618, "superior": 0.1816},
    "padel": {"secundaria_o_inferior": 0.042, "secundaria_superior": 0.1057, "superior": 0.1377},
    "patinaje": {"secundaria_o_inferior": 0.0119, "secundaria_superior": 0.0226, "superior": 0.0238},
    "pelota_vasca": {"secundaria_o_inferior": 0.0047, "secundaria_superior": 0.0211, "superior": 0.0119},
    "pesca": {"secundaria_o_inferior": 0.0229, "secundaria_superior": 0.0277, "superior": 0.0119},
    "petanca": {"secundaria_o_inferior": 0.0114, "secundaria_superior": 0.0182, "superior": 0.0141},
    "piraguismo_remo": {"secundaria_o_inferior": 0.003, "secundaria_superior": 0.0138, "superior": 0.0171},
    "rugby": {"secundaria_o_inferior": 0.0038, "secundaria_superior": 0.0036, "superior": 0.0037},
    "running": {"secundaria_o_inferior": 0.05, "secundaria_superior": 0.121, "superior": 0.1585},
    "senderismo": {"secundaria_o_inferior": 0.0716, "secundaria_superior": 0.1669, "superior": 0.2218},
    "squash": {"secundaria_o_inferior": 0.0004, "secundaria_superior": 0.0015, "superior": 0.0045},
    "submarinismo": {"secundaria_o_inferior": 0.0072, "secundaria_superior": 0.0262, "superior": 0.0283},
    "surf": {"secundaria_o_inferior": 0.0017, "secundaria_superior": 0.0095, "superior": 0.0127},
    "tenis": {"secundaria_o_inferior": 0.0161, "secundaria_superior": 0.0386, "superior": 0.0447},
    "tenis_mesa": {"secundaria_o_inferior": 0.0127, "secundaria_superior": 0.0445, "superior": 0.0432},
    "triatlon": {"secundaria_o_inferior": 0.0008, "secundaria_superior": 0.0073, "superior": 0.0045},
    "vela": {"secundaria_o_inferior": 0.0034, "secundaria_superior": 0.0036, "superior": 0.0074},
    "voleibol": {"secundaria_o_inferior": 0.0259, "secundaria_superior": 0.0379, "superior": 0.0208},
    "yoga_pilates": {"secundaria_o_inferior": 0.1051, "secundaria_superior": 0.1786, "superior": 0.2516},
}


# Reparto por nacionalidad (española vs. extranjera), INE -- Censo Anual de
# Población a 1 de enero de 2025, mismo corte temporal que
# TOTAL_POPULATION_ES (42.216.326 nacionalidad española + 6.911.971
# nacionalidad extranjera, de 49.128.297 totales). A diferencia de
# color de piel/origen étnico (ver ADR-17 en
# docs/src/09_architecture_decisions.adoc), la nacionalidad LEGAL no es
# dato de categoría especial del art. 9 RGPD -- es un dato administrativo
# habitual (el que aparece en cualquier check-in de hotel o contrato), así
# que sí se incluye aquí. No se desagrega por país concreto (solo
# español/extranjero): una tabla por nacionalidad específica reduciría
# demasiado la población de referencia con muestras pequeñas y esta
# versión no tiene los recuentos por país necesarios para hacerlo con
# fiabilidad.
NATIONALITY_DISTRIBUTION = {
    "espanola": 0.851,
    "extranjera": 0.149,
}

# Situación laboral -- INE, Encuesta de Población Activa (EPA), 4º
# trimestre de 2025 (tasa de actividad 58,94%, tasa de paro 9,93% sobre la
# población activa), reexpresada como proporción de la población de 16 o
# más años (no del total de España: la EPA no cubre a los menores de 16,
# y son justo el universo de referencia de lo que puede autodeclararse en
# redes sociales). Distinta de OCCUPATION_DISTRIBUTION (que es SECTOR
# profesional, no situación): aquí lo que importa es si la persona
# trabaja, busca trabajo, está jubilada o estudia, y son categorías
# excluyentes entre sí en el momento de la encuesta.
#   - activo (ocupado): 58,94% * (1 - 9,93%) = 53,08%, redondeado a 0.531
#   - parado: 58,94% * 9,93% = 5,85%, redondeado a 0.059
#   - inactivos (41,06% restante), repartidos con el orden de magnitud
#     habitual de sus subcategorías en la EPA (jubilados/pensionistas
#     ~57% de los inactivos, estudiantes ~20%, resto -- labores del hogar,
#     incapacidad permanente, otras situaciones -- ~23%): jubilado 0.234,
#     estudiante 0.082, otro_inactivo 0.094.
SITUACION_LABORAL_DISTRIBUTION = {
    "activo": 0.531,
    "parado": 0.059,
    "jubilado": 0.234,
    "estudiante": 0.082,
    "otro_inactivo": 0.094,
}

# Tipo de hogar en el que reside la persona -- INE, Encuesta Continua de
# Hogares (ECH), aprox. 2019-2024 (proporción de HOGARES, no de personas;
# se usa igualmente como proxy de la proporción de PERSONAS que viven en
# cada tipo de hogar, aproximación ya asumida en el resto de este módulo
# para ubicación -- ver docstring de scoring/k_anonymity.py sobre asumir
# distribución similar a la media nacional). unipersonal 25,7% y
# pareja_con_hijos 33,4% son cifras directas de la ECH; pareja_sin_hijos se
# deriva de "10,3M hogares de pareja (con o sin hijos) sobre ~18,2M
# hogares totales" menos pareja_con_hijos; monoparental de "~1,9M hogares
# monoparentales sobre ~19,4M hogares totales" (2024); "otro" es el resto
# (hogares complejos, varios núcleos familiares, corresidentes sin
# vínculo familiar).
#
# CANDIDATO DE AUTOMATIZACIÓN LOCALIZADO (sin aplicar todavía): tabla
# PC-Axis del INE "t20/p274/serie/prov/p01/l0/01013.px" (ECH, tipo de
# hogar x tipo de edificio, nacional), con mapeo exhaustivo a estas 5
# categorías en `scripts/update_ine_reference.py`
# (`_TIPO_HOGAR_TO_APP_CATEGORY`). Sin verificar todavía contra la API
# real -- ver el docstring de ese script (cabecera, sección
# "INVESTIGACIÓN de HOUSEHOLD_TYPE_DISTRIBUTION") para el detalle
# completo. Corre `python scripts/update_ine_reference.py` para comparar
# estos valores contra el INE.
HOUSEHOLD_TYPE_DISTRIBUTION = {
    "unipersonal": 0.257,
    "pareja_sin_hijos": 0.232,
    "pareja_con_hijos": 0.334,
    "monoparental": 0.100,
    "otro": 0.077,
}

# Lengua materna/habitual cooficial, CONDICIONADA a la comunidad autónoma
# de residencia -- P(lengua | CCAA), no P(lengua) a nivel nacional (el
# catalán fuera de Cataluña/Baleares/C. Valenciana es residual, así que una
# proporción nacional sería casi inútil como filtro). Fuente: Encuesta de
# Características Esenciales de la Población y Viviendas (INE, 2021),
# "lengua materna" por CCAA. Solo se aplica en k_anonymity.py cuando la
# lengua autodeclarada coincide con la cooficial de la CCAA ya conocida por
# otro medio (ubicación) -- mismo patrón que MARITAL_STATUS_BY_SEX
# (requiere conocer primero otro atributo). Cada sub-diccionario NO suma
# 1.0 a propósito: son solo la cooficial vs. "todo lo demás" (castellano u
# otras), no un desglose completo de todas las lenguas de esa CCAA.
#
# INVESTIGADO fuera del INE (sin automatizar): los institutos autonómicos
# con lengua cooficial hacen sus propias encuestas sociolingüísticas (p.
# ej. IDESCAT en Cataluña, quinquenal) con mejor periodicidad que la
# ECEPOV del INE, pero automatizarlo de verdad exigiría repetir esta
# investigación por separado para cada una de las 6 CCAA con lengua
# cooficial (cada una con su propio instituto/encuesta/periodicidad) --
# no se ha considerado que compense en esta pasada. Ver el docstring de
# `scripts/update_ine_reference.py` (sección "BÚSQUEDA FUERA DEL INE...")
# para el detalle.
LANGUAGE_BY_CCAA = {
    "cataluna": {"catalan": 0.555, "castellano_u_otra": 0.445},
    "islas baleares": {"catalan": 0.429, "castellano_u_otra": 0.571},
    "comunidad valenciana": {"valenciano": 0.352, "castellano_u_otra": 0.648},
    "pais vasco": {"euskera": 0.337, "castellano_u_otra": 0.663},
    "galicia": {"gallego": 0.828, "castellano_u_otra": 0.172},
    "navarra": {"euskera": 0.146, "castellano_u_otra": 0.854},
}


# Referencias contextuales para rasgos físicos (añadidos manualmente).
EYE_COLOR_DISTRIBUTION = {
    'marron': 0.55,
    'verde': 0.15,
    'azul': 0.15,
    'miel': 0.10,
    'negro': 0.05,
}

HAIR_COLOR_DISTRIBUTION = {
    'moreno': 0.45,
    'castano': 0.40,
    'rubio': 0.10,
    'pelirrojo': 0.02,
    'canoso': 0.03,
}

SKIN_TONE_DISTRIBUTION = {
    'claro': 0.60,
    'medio': 0.35,
    'oscuro': 0.05,
}
