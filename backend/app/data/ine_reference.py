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
    "la rioja": 320_000,
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
    "castilla y leon": ["valladolid", "leon", "burgos", "salamanca", "zamora", "avila", "palencia", "segovia", "soria"],
    "castilla la mancha": ["toledo", "ciudad real", "albacete", "guadalajara", "cuenca"],
    "cataluna": ["barcelona", "tarragona", "gerona", "lerida"],
    "comunidad valenciana": ["valencia", "alicante", "castellon"],
    "extremadura": ["badajoz", "caceres"],
    "galicia": ["a coruna", "pontevedra", "lugo", "orense"],
    "madrid": ["madrid"],
    "murcia": ["murcia"],
    "navarra": ["navarra"],
    "pais vasco": ["vizcaya", "guipuzcoa", "alava"],
    "la rioja": ["la rioja"],
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
    "castilla y leon": "Castilla y León",
    "castilla la mancha": "Castilla-La Mancha",
    "cataluna": "Cataluña",
    "comunidad valenciana": "Comunidad Valenciana",
    "extremadura": "Extremadura",
    "galicia": "Galicia",
    "madrid": "Comunidad de Madrid",
    "murcia": "Región de Murcia",
    "navarra": "Comunidad Foral de Navarra",
    "pais vasco": "País Vasco",
    "la rioja": "La Rioja",
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
    "castilla y leon": "castilla y leon",
    "castilla-y-leon": "castilla y leon",
    "castile and leon": "castilla y leon",
    "castille and leon": "castilla y leon",
    "castilla la mancha": "castilla la mancha",
    "castilla-la mancha": "castilla la mancha",
    "castile-la mancha": "castilla la mancha",
    "cataluna": "cataluna",
    "catalunya": "cataluna",
    "catalonia": "cataluna",
    "comunidad valenciana": "comunidad valenciana",
    "comunitat valenciana": "comunidad valenciana",
    "valencian community": "comunidad valenciana",
    "region of valencia": "comunidad valenciana",
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
    "pais vasco": "pais vasco",
    "euskadi": "pais vasco",
    "basque country": "pais vasco",
    "la rioja": "la rioja",
    "rioja": "la rioja",
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
