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
"""

# Población residente en España a 1 de enero de 2025 (INE, Estadística
# Continua de Población / Censo Anual de Población).
TOTAL_POPULATION_ES = 49_128_297

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
        "casado": 0.470,
        "con_pareja": 0.211,
        "divorciado": 0.078,
        "soltero": 0.206,
        "viudo": 0.035,
    },
    "mujer": {
        "casado": 0.447,
        "con_pareja": 0.194,
        "divorciado": 0.078,
        "soltero": 0.179,
        "viudo": 0.102,
    },
}

# Distribución de edad en tramos de 5 años, proporción sobre el total.
# Pirámide poblacional aproximada de España (envejecida, con menos peso en
# tramos jóvenes). Suma ~1.0.
AGE_DISTRIBUTION_5Y = {
    "0-4": 0.038, "5-9": 0.042, "10-14": 0.046, "15-19": 0.045,
    "20-24": 0.045, "25-29": 0.052, "30-34": 0.062, "35-39": 0.070,
    "40-44": 0.081, "45-49": 0.084, "50-54": 0.078, "55-59": 0.072,
    "60-64": 0.064, "65-69": 0.056, "70-74": 0.052, "75-79": 0.042,
    "80-84": 0.030, "85+": 0.041,
}


def age_bin(age: int) -> str:
    """Convierte una edad concreta en su tramo quinquenal de AGE_DISTRIBUTION_5Y.
    Se mantiene por si se necesita el agregado por tramo en algún otro sitio,
    pero `AGE_DISTRIBUTION_1Y` (más abajo) es lo que usa k_anonymity.py."""
    if age >= 85:
        return "85+"
    lower = (age // 5) * 5
    return f"{lower}-{lower + 4}"


def _build_age_distribution_1y() -> dict[int, float]:
    """Deriva una proporción por EDAD EXACTA (año a año) a partir de
    AGE_DISTRIBUTION_5Y, repartiendo uniformemente la proporción de cada
    tramo quinquenal entre las edades que lo componen.

    Nota de precisión: el INE sí publica población año a año (tabla
    "Población por edad (año a año), Españoles/Extranjeros, Sexo y Año",
    https://www.ine.es/jaxi/Tabla.htm?path=%2Ft20%2Fe245%2Fp08%2Fl0%2F&file=01003.px),
    pero es un selector interactivo, no un CSV descargable directamente por
    URL, así que aquí se DERIVA a partir de los tramos de 5 años en vez de
    usar el dato exacto. El reparto uniforme dentro de cada tramo es una
    aproximación razonable (la pirámide de población no varía mucho entre
    edades consecutivas), pero si se quiere máxima precisión, sustituye
    esta función por una carga directa de esa tabla del INE exportada a CSV.
    """
    distribution: dict[int, float] = {}
    for band, proportion in AGE_DISTRIBUTION_5Y.items():
        ages = range(85, 101) if band == "85+" else range(int(band.split("-")[0]), int(band.split("-")[0]) + 5)
        ages = list(ages)
        per_age = proportion / len(ages)
        for age in ages:
            distribution[age] = per_age
    return distribution


# Proporción de población por EDAD EXACTA (0-100), derivada de
# AGE_DISTRIBUTION_5Y (ver docstring de _build_age_distribution_1y). Es lo
# que usa scoring/k_anonymity.py para no agrupar edades en tramos de 5 años.
AGE_DISTRIBUTION_1Y = _build_age_distribution_1y()


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
    "madrid": 7_100_000,
    "barcelona": 5_800_000,
    "valencia": 2_650_000,
    "sevilla": 1_950_000,
    "alicante": 1_950_000,
    "malaga": 1_750_000,
    "murcia": 1_570_000,
    "cadiz": 1_240_000,
    "vizcaya": 1_170_000,
    "a coruna": 1_120_000,
    "baleares": 1_260_000,
    "las palmas": 1_130_000,
    "santa cruz de tenerife": 1_060_000,
    "zaragoza": 980_000,
    "asturias": 1_000_000,
    "pontevedra": 940_000,
    "granada": 920_000,
    "tarragona": 830_000,
    "gerona": 770_000,
    "castellon": 590_000,
    "toledo": 730_000,
    "badajoz": 660_000,
    "cordoba": 780_000,
    "jaen": 610_000,
    "navarra": 670_000,
    "almeria": 730_000,
    "guipuzcoa": 720_000,
    "valladolid": 519_000,
    "cantabria": 585_000,
    "leon": 438_000,
    "lerida": 430_000,
    "huelva": 520_000,
    "burgos": 355_000,
    "caceres": 385_000,
    "salamanca": 336_000,
    _CCAA_LA_RIOJA: 320_000,
    "lugo": 327_000,
    "orense": 305_000,
    "albacete": 385_000,
    "guadalajara": 265_000,
    "ciudad real": 495_000,
    "alava": 335_000,
    "huesca": 225_000,
    "zamora": 165_000,
    "avila": 158_000,
    "palencia": 155_000,
    "segovia": 154_000,
    "teruel": 134_000,
    "cuenca": 195_000,
    "soria": 88_000,
    "ceuta": 83_000,
    "melilla": 87_000,
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
# estudio concreto. MUY aproximado (basado en órdenes de magnitud de
# graduados universitarios en España por rama, ~40% de esa franja tiene
# estudios superiores). Ajusta con datos reales del Censo/EPA si tu TFG
# necesita precisión aquí.
STUDIES_DISTRIBUTION = {
    "medicina": 0.006,
    "enfermeria": 0.010,
    "derecho": 0.014,
    "ingenieria informatica": 0.012,
    "ingenieria industrial": 0.008,
    "administracion de empresas": 0.018,
    "psicologia": 0.010,
    "magisterio": 0.016,
    "arquitectura": 0.003,
    "farmacia": 0.004,
    "biologia": 0.005,
    "periodismo": 0.004,
    "economia": 0.010,
    "veterinaria": 0.002,
}

# Proporción de la población ocupada por gran categoría profesional
# (aprox., basado en agregados de la EPA/CNO-11).
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
    "espanola": 0.859,
    "extranjera": 0.141,
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
LANGUAGE_BY_CCAA = {
    "cataluna": {"catalan": 0.555, "castellano_u_otra": 0.445},
    "islas baleares": {"catalan": 0.429, "castellano_u_otra": 0.571},
    "comunidad valenciana": {"valenciano": 0.352, "castellano_u_otra": 0.648},
    "pais vasco": {"euskera": 0.337, "castellano_u_otra": 0.663},
    "galicia": {"gallego": 0.828, "castellano_u_otra": 0.172},
    "navarra": {"euskera": 0.146, "castellano_u_otra": 0.854},
}

