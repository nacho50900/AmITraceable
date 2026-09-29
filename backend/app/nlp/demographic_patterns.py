"""
Bancos temáticos de expresiones regulares para la detección de
autodeclaraciones demográficas (ver `demographic_extraction.py`).

Por qué existe este módulo: `merge_findings` (ai_attribute_extraction.py)
da PRIORIDAD a la regex -- la IA solo rellena lo que la regex NO encontró,
nunca lo corrige. Un falso positivo de regex llega, por tanto, tal cual
al estimador de k-anonimato. De ahí que los bancos sean deliberadamente
ESTRICTOS (frases-ancla en primera persona, listas cerradas, exclusión de
frases sobre terceros) en vez de permisivos.

Convenciones:
- Los patrones se escriben sobre texto NORMALIZADO (minúsculas y sin
  tildes ni eñes, ver `normalizar`): así 'terminé'/'termine' o
  'máster'/'master' se comportan igual, que era el fallo más frecuente
  con el texto real de los usuarios.
- Un banco es una tupla de `(valor_canonico, patron_compilado)`; gana la
  primera entrada que encaja (el orden importa igual que en una cadena
  if/elif: lo específico va antes que lo genérico).
- Las frases sobre terceros ("mi madre es jubilada") se descartan con
  `es_propio`: una coincidencia no cuenta si un tercero ("mi madre", "mi
  jefe"...) aparece ANTES en la misma cláusula.
"""
import re
import unicodedata

Banco = tuple[tuple[str, "re.Pattern[str]"], ...]


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes/eñes/diéresis (mismo criterio que
    `_strip_accents` de demographic_extraction.py, más `lower()`)."""
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c)).lower()


def _banco(*entradas: tuple[str, str]) -> Banco:
    return tuple((valor, re.compile(patron)) for valor, patron in entradas)


def primer_valor(banco: Banco, clausula: str) -> str | None:
    """Valor canónico de la primera entrada del banco que encaja en la
    cláusula (y no habla de un tercero), o None."""
    for valor, patron in banco:
        match = patron.search(clausula)
        if match and es_propio(clausula, match.start()):
            return valor
    return None


def buscar_propios(patron: "re.Pattern[str]", clausula: str):
    """Coincidencias del patrón en la cláusula que NO hablan de un tercero."""
    return [m for m in patron.finditer(clausula) if es_propio(clausula, m.start())]


# ---------------------------------------------------------------------------
# Frases sobre terceros
# ---------------------------------------------------------------------------

_TERCERA_PERSONA_RE = re.compile(
    r"\b(?:mi|mis|tu|tus|su|sus|nuestr[oa]s?)\s+(?:(?:querid|viej|joven|ex|difunt|futur)[oa]s?\s+)?"
    r"(?:padre|madre|padres|abuel[oa]s?|suegr[oa]s?|tio|tia|tios|tias|vecin[oa]s?|amig[oa]s?|hermano|hermana|"
    r"hermanos|marido|mujer|esposo|esposa|novio|novia|pareja|jefe|jefa|companer[oa]s?|primo|prima|cunad[oa]|"
    r"yerno|nuera|hij[oa]s?|nieto|nieta|sobrin[oa]s?|ex)\b"
)
_NUMERO_COMPUESTO_RE = re.compile(r"\b(treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa) y (\w+)\b")
_SEPARADOR_CLAUSULAS_RE = re.compile(r"[.;!?\n]+|,| y | pero | aunque | mientras ")
_SEPARADOR_SIN_Y_RE = re.compile(r"[.;!?\n]+|,| pero | aunque | mientras ")


def clausulas(texto_normalizado: str, separar_por_y: bool = True) -> list[str]:
    """Divide el texto en cláusulas. 'treinta y cuatro' se protege como
    'treinta_y_cuatro' para no partirlo por la 'y'. Con `separar_por_y=False`
    no se parte por 'y' (topónimos como 'castilla y leon')."""
    protegido = _NUMERO_COMPUESTO_RE.sub(r"\1_y_\2", texto_normalizado)
    separador = _SEPARADOR_CLAUSULAS_RE if separar_por_y else _SEPARADOR_SIN_Y_RE
    return [c.strip() for c in separador.split(protegido) if c.strip()]


def es_propio(clausula: str, inicio_coincidencia: int) -> bool:
    """False si un tercero ('mi madre', 'mi jefe'...) aparece ANTES de la
    coincidencia en su cláusula: 'mi madre es jubilada' -> jubilada no es del
    autor, pero 'soy mujer casada con mi marido' -> 'soy mujer' sí."""
    return _TERCERA_PERSONA_RE.search(clausula[:inicio_coincidencia]) is None


# ---------------------------------------------------------------------------
# Edad
# ---------------------------------------------------------------------------

_UNIDADES = {
    "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}
_NUMEROS_ESPECIALES = {
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
}
_UNIDADES_RE = "|".join(_UNIDADES)
_VEINTI_RE = re.compile(rf"veinti({_UNIDADES_RE})")
_DECENA_Y_UNIDAD_RE = re.compile(
    rf"(treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa) y ({_UNIDADES_RE})"
)


def numero_en_palabras(texto: str) -> int | None:
    """'veinticuatro' -> 24, 'treinta y cuatro' -> 34; None si no es un número."""
    texto = texto.strip().replace("_y_", " y ")
    if texto in _NUMEROS_ESPECIALES:
        return _NUMEROS_ESPECIALES[texto]
    veinti = _VEINTI_RE.fullmatch(texto)
    if veinti:
        return 20 + _UNIDADES[veinti.group(1)]
    compuesto = _DECENA_Y_UNIDAD_RE.fullmatch(texto)
    if compuesto:
        return _NUMEROS_ESPECIALES[compuesto.group(1)] + _UNIDADES[compuesto.group(2)]
    return None


_EDAD_ANCLADA = tuple(
    re.compile(p)
    for p in (
        r"\b(?:tengo|cumplo|cumpli|cumplimos)\s+(\d{1,2})\s+anos\b",
        r"\b(?:cumplo|cumpli)\s+(\d{2})\b(?=\s*(?:anos|ya\b|hoy|ayer|manana|esta semana|[.,!?]|$))",
        r"\b(?:acabo de cumplir|recien cumplidos?|estrene los)\s+(?:los\s+)?(\d{2})\b",
        r"\bmi edad (?:es|son|actual es)\s+(?:de\s+)?(\d{1,2})\b",
        r"\bedad\s*[:=]\s*(\d{1,2})\b",
        r"\bi(?:'m| am|m)\s+(?:a\s+)?(\d{1,2})[- ]?(?:years?[- ]?old|yo|y/o)\b",
    )
)
_EDAD_EN_PALABRAS_RE = re.compile(r"\b(?:tengo|cumplo|cumpli|con)\s+([a-z]+(?:_y_[a-z]+)?)\s+anos\b")
# "N años" suelto: lo mantiene la versión original (test 'con mis 30 años'),
# pero se descartan las duraciones ('hace 15 años', 'dentro de 20 años').
_EDAD_GENERICA_RE = re.compile(r"\b(\d{1,2})\s+anos\b")
_NO_ES_EDAD_RE = re.compile(
    r"\b(?:hace|hacia|durante|dentro de|llevo|llevamos|hara|tras|despues de|mas de|menos de|casi|cada|"
    r"ultimos|proximos|otros|unos|desde|hasta|de aqui a|a los|en|por)\s*$"
)
# Convención de Reddit: "(24M)", "[29F]", "yo 29F". Se comprueba sobre el
# texto ORIGINAL porque la mayúscula distingue "24M" de "24 m" (metros).
_EDAD_SEXO_REDDIT_RE = re.compile(r"[(\[]\s*(\d{2})\s*([MF])\s*[)\]]|\b(?:[Yy]o|[Ss]oy|I am|I'm|Im)\s+(\d{2})\s?([MF])\b")

EDAD_MIN, EDAD_MAX = 12, 100


def _edad_valida(valor: int | None) -> int | None:
    return valor if valor is not None and EDAD_MIN <= valor <= EDAD_MAX else None


def edad_y_sexo_reddit(texto: str) -> tuple[int | None, str | None]:
    match = _EDAD_SEXO_REDDIT_RE.search(texto)
    if not match:
        return None, None
    edad = int(match.group(1) or match.group(3))
    letra = match.group(2) or match.group(4)
    return _edad_valida(edad), ("hombre" if letra == "M" else "mujer")


def edad_autodeclarada(texto: str) -> int | None:
    """Edad literal escrita por el propio usuario, o None. Orden: primero
    las frases-ancla (más fiables), después el 'N años' suelto."""
    edad, _ = edad_y_sexo_reddit(texto)
    if edad is not None:
        return edad

    for clausula in clausulas(normalizar(texto)):
        for patron in _EDAD_ANCLADA:
            for match in buscar_propios(patron, clausula):
                if _edad_valida(int(match.group(1))) is not None:
                    return int(match.group(1))

        for match in buscar_propios(_EDAD_EN_PALABRAS_RE, clausula):
            valor = _edad_valida(numero_en_palabras(match.group(1)))
            if valor is not None:
                return valor

        for match in buscar_propios(_EDAD_GENERICA_RE, clausula):
            prefijo = clausula[max(0, match.start() - 18) : match.start()]
            if _NO_ES_EDAD_RE.search(prefijo):
                continue
            valor = _edad_valida(int(match.group(1)))
            if valor is not None:
                return valor
    return None


# ---------------------------------------------------------------------------
# Sexo
# ---------------------------------------------------------------------------

_SOY = r"\bsoy\s+(?:un\s+|una\s+|el\s+|la\s+|otr[oa]\s+)?"
_SEXO_HOMBRE_RE = re.compile(
    _SOY + r"(?:chico|chaval|hombre|tio|senor|varon|caballero|padre|papa|abuelo|hijo|hermano|marido|esposo)\b"
    r"|\b(?:como hombre que soy|siendo (?:un )?hombre|al ser (?:un )?hombre)\b"
    r"|\bi(?:'m| am|m)\s+(?:a\s+|an\s+)?(?:\d{1,2}[- ]?(?:years?[- ]?old|yo|y/o)\s+)?"
    r"(?:guy|man|male|dude|dad|father|husband|boy)\b"
)
_SEXO_MUJER_RE = re.compile(
    _SOY + r"(?:chica|mujer|tia|senora|madre|mama|abuela|chavala|hija|hermana|esposa)\b"
    r"|\b(?:como mujer que soy|siendo (?:una )?mujer|al ser (?:una )?mujer)\b"
    r"|\bi(?:'m| am|m)\s+(?:a\s+|an\s+)?(?:\d{1,2}[- ]?(?:years?[- ]?old|yo|y/o)\s+)?"
    r"(?:girl|woman|female|lady|mom|mother|wife)\b"
)


def sexo_autodeclarado(texto: str) -> str | None:
    """'hombre' | 'mujer' | None. Si el texto contiene autodeclaraciones de
    AMBOS sexos (cita, ironía, varias personas) no se adivina: None."""
    _, sexo_reddit = edad_y_sexo_reddit(texto)
    if sexo_reddit is not None:
        return sexo_reddit

    partes = clausulas(normalizar(texto))
    es_hombre = any(buscar_propios(_SEXO_HOMBRE_RE, c) for c in partes)
    es_mujer = any(buscar_propios(_SEXO_MUJER_RE, c) for c in partes)
    if es_hombre == es_mujer:
        return None
    return "hombre" if es_hombre else "mujer"


# ---------------------------------------------------------------------------
# Ubicación
# ---------------------------------------------------------------------------

# Anclas de RESIDENCIA en primera persona (con el candidato a topónimo en el
# grupo 1). Deliberadamente NO incluye 'soy de X' (origen, no residencia) ni
# 'vive en X' (tercera persona) ni 'vivo cerca de X' (impreciso).
_RESIDENCIA_ANCLAS: tuple["re.Pattern[str]", ...] = tuple(
    re.compile(p)
    for p in (
        r"\bvivo(?: actualmente| ahora| ahora mismo| ya| aqui)? en ([a-z ]+)",
        r"\bvivimos(?: actualmente| ahora| ya| aqui)? en ([a-z ]+)",
        r"\b(?:resido|residimos)(?: actualmente| ahora)? en ([a-z ]+)",
        r"\b(?:estoy|estamos) viviendo en ([a-z ]+)",
        r"\b(?:me|nos) (?:he |hemos )?(?:mude|mudado|mudamos) a ([a-z ]+)",
        r"\bsoy vecin[oa] de ([a-z ]+)",
        r"\bafincad[oa] en ([a-z ]+)",
        r"\bllevo \w+ anos viviendo en ([a-z ]+)",
    )
)


def candidatos_residencia(texto: str) -> list[str]:
    """Trozos de texto (normalizados) que siguen a una frase-ancla de
    residencia, en orden de aparición de las anclas."""
    candidatos: list[str] = []
    for clausula in clausulas(normalizar(texto), separar_por_y=False):
        for patron in _RESIDENCIA_ANCLAS:
            for match in buscar_propios(patron, clausula):
                candidatos.append(match.group(1).strip())
    return candidatos


def contiene_toponimo(candidato: str, toponimo: str) -> bool:
    """Coincidencia por PALABRA COMPLETA ('leon' no encaja en 'leones')."""
    return re.search(rf"\b{re.escape(toponimo)}\b", candidato) is not None


# ---------------------------------------------------------------------------
# Estudios
# ---------------------------------------------------------------------------

_ESTUDIOS_ANCLAS: tuple["re.Pattern[str]", ...] = tuple(
    re.compile(p)
    for p in (
        r"\b(?:estudio|estudiando|estudie|estudiamos|estudiante(?: de| en)?|voy a estudiar)\b\s*"
        r"(?:(?:la |el |una |un |los |las )?(?:carrera|grado|licenciatura|master)?\s*(?:de |en |del )?)?([a-z ]{3,80})",
        r"\b(?:graduad[oa]|licenciad[oa]|diplomad[oa]|titulad[oa]|egresad[oa]|doctorad[oa])\s+"
        r"(?:universitari[oa]\s+)?(?:en|de|por)\s+([a-z ]{3,80})",
        r"\bme (?:he )?(?:licencie|gradue|diplome|titule)\s+(?:en|de)\s+([a-z ]{3,80})",
        r"\b(?:cursando|cursar|curso|hago|estoy haciendo|hice|estoy en)\s+(?:el |la |un |una )?"
        r"((?:grado|carrera|licenciatura|ingenieria|master|doctorado)[a-z ]{0,80})",
        r"\bmi carrera (?:es|fue)?\s*(?:de |en )?([a-z ]{3,80})",
    )
)

# canónico (claves de STUDIES_DISTRIBUTION) -> alias. Lo que no está aquí y
# tampoco es una clave literal cae en el respaldo de demographic_extraction.
ESTUDIOS_SINONIMOS: Banco = _banco(
    ("medicina", r"\bmedicina\b"),
    ("enfermeria", r"\benfermeria\b"),
    ("derecho", r"\bderecho\b|\bleyes\b"),
    (
        "ingenieria informatica",
        r"ingenieria informatica|\binformatica\b|ingenieria del software|ingenieria de (?:los )?computadores"
        r"|ciencias? de la computacion|\bciberseguridad\b|\bsoftware\b|sistemas de informacion",
    ),
    ("ingenieria industrial", r"ingenieria industrial|organizacion industrial"),
    (
        "administracion de empresas",
        r"administracion (?:y direccion )?de empresas|\bade\b|\bdade\b|ciencias empresariales|\bempresariales\b",
    ),
    ("psicologia", r"\bpsicologia\b"),
    ("magisterio", r"\bmagisterio\b|educacion (?:primaria|infantil)|profesorado de (?:primaria|infantil)"),
    (
        "arquitectura",
        r"\barquitectura\b(?!\s+(?:de|del)\s+(?:software|computador\w*|ordenador\w*|redes|sistemas|datos|la informacion))",
    ),
    ("farmacia", r"\bfarmacia\b"),
    ("biologia", r"\bbiologia\b|ciencias biologicas"),
    ("periodismo", r"\bperiodismo\b"),
    ("economia", r"\beconomia\b|ciencias economicas"),
    ("veterinaria", r"\bveterinaria\b"),
)


def candidatos_estudios(texto: str) -> list[str]:
    """Trozos de texto (normalizados) que siguen a una frase-ancla de
    estudios ('estudio...', 'graduado en...', 'me licencié en...',
    'cursando el grado de...')."""
    candidatos: list[str] = []
    for clausula in clausulas(normalizar(texto), separar_por_y=False):
        for patron in _ESTUDIOS_ANCLAS:
            candidatos.extend(m.group(1).strip() for m in buscar_propios(patron, clausula))
    return candidatos


def estudios_autodeclarados(texto: str) -> str | None:
    for candidato in candidatos_estudios(texto):
        valor = primer_valor(ESTUDIOS_SINONIMOS, candidato)
        if valor is not None:
            return valor
    return None


# ---------------------------------------------------------------------------
# Nivel de estudios (sobre texto normalizado: 'terminé' == 'termine')
# ---------------------------------------------------------------------------

_TERMINE = r"(?:termine|acabe|complete|finalice|he terminado|he acabado)"
_NIVEL_SUPERIOR_NEGADO_RE = re.compile(
    r"\bno (?:he )?(?:termin\w+|acab\w+|complet\w+|finaliz\w+)\s+(?:la |el |mi |mis )?"
    r"(?:carrera|universidad|grado|master|doctorado|licenciatura|ingenieria)\b"
)
_NIVEL_SUPERIOR_RE = re.compile(
    r"\b(?:soy universitari[oa]|tengo una carrera(?: universitaria)?|tengo un grado(?: universitario)?\b(?!\s+(?:medio|superior))|"
    r"soy graduad[oa] (?:en|universitari[oa])|soy licenciad[oa]|tengo una licenciatura|"
    rf"{_TERMINE} (?:la |el |mi )?(?:carrera|universidad|grado(?!\s+(?:medio|superior))|licenciatura|ingenieria|master|doctorado)|"
    r"me (?:he )?(?:licencie|gradue|titule|doctore) en|"
    r"tengo (?:un |el )?(?:master|doctorado|posgrado|titulo universitario)|"
    r"(?:hice|estudie|curse) (?:un |el |la )?(?:master|doctorado|carrera|grado(?!\s+(?:medio|superior)))|"
    r"soy doctor(?:a)? en|soy doctorand[oa]|soy tecnic[oa] superior|"
    r"(?:tengo|termine|acabe|hice|estudie|curse) (?:un |el |la )?(?:ciclo formativo de grado superior|"
    r"grado superior(?: de fp)?|cfgs|fp (?:de )?(?:grado )?superior))\b"
)
_NIVEL_SECUNDARIA_SUPERIOR_RE = re.compile(
    rf"\b(?:tengo (?:el |un |mi )?bachillerato|{_TERMINE} (?:el |la )?(?:bachillerato|fp)|"
    r"hice (?:el )?bachillerato|soy bachiller|"
    r"(?:tengo|termine|acabe|hice|estudie|curse) (?:un |el |la )?(?:ciclo formativo de grado medio|grado medio(?: de fp)?|"
    r"cfgm|fp (?:de )?(?:grado )?medi[oa])|soy tecnic[oa] (?:de grado medio|medio))\b"
)
_NIVEL_SECUNDARIA_O_INFERIOR_RE = re.compile(
    r"\b(?:(?:solo )?tengo (?:solo )?(?:la )?(?:eso|secundaria|primaria|graduado escolar)|"
    r"no (?:he )?(?:termin\w+|acab\w+|complet\w+) (?:la eso|el instituto|secundaria|bachillerato)|"
    r"(?:solo )?(?:tengo )?estudios primarios|no tengo estudios|sin estudios|"
    r"(?:abandone|deje) (?:los estudios|el instituto|la eso|el colegio|la escuela))\b"
)


def nivel_estudios_autodeclarado(texto: str) -> str | None:
    """Nivel COMPLETADO (no en curso). 'No terminé la carrera' NO cuenta
    como superior (la versión anterior lo marcaba como tal)."""
    for clausula in clausulas(normalizar(texto)):
        if buscar_propios(_NIVEL_SECUNDARIA_O_INFERIOR_RE, clausula):
            return "secundaria_o_inferior"
        if _NIVEL_SUPERIOR_NEGADO_RE.search(clausula):
            continue
        if buscar_propios(_NIVEL_SUPERIOR_RE, clausula):
            return "superior"
        if buscar_propios(_NIVEL_SECUNDARIA_SUPERIOR_RE, clausula):
            return "secundaria_superior"
    return None


# ---------------------------------------------------------------------------
# Ocupación (sector profesional)
# ---------------------------------------------------------------------------

_ANCLA_OFICIO = (
    r"\b(?:soy|somos|trabajo|trabajamos|trabajando|ejerzo|me dedico a ser|mi profesion es|mi trabajo es|curro|currando)"
    r"\s+(?:(?:de|como|un|una|el|la|mi|buen[oa]?|muy)\s+)*"
)


def _oficio(*roles: str) -> str:
    return _ANCLA_OFICIO + r"(?:" + "|".join(roles) + r")\b"


def _sector(*sectores: str) -> str:
    return (
        r"\b(?:trabajo|trabajamos|curro|me dedico)\s+(?:en|a|de)\s+(?:el |la |al )?(?:sector )?(?:del? )?"
        r"(?:" + "|".join(sectores) + r")\b"
    )


# Las claves son las de OCCUPATION_DISTRIBUTION. El orden importa: 'desarrollador
# de software' va antes que 'ingeniero' ('ingeniero de software' es desarrollo).
OCUPACION_ANCLADA: Banco = _banco(
    ("desarrollador de software", _oficio(
        r"programador(?:a)?", r"desarrollador(?:a)?", r"developer", r"dev", r"informatic[oa]",
        r"ingenier[oa] (?:de|del) software", r"software engineer", r"(?:backend|frontend|full ?stack) (?:developer|engineer)",
    )),
    ("docente", _oficio(r"docente", r"profe", r"profesor(?:a)?", r"maestr[oa]", r"catedratic[oa]", r"educador(?:a)?")),
    ("docente", r"\bdoy clases\b"),
    ("sanitario", _oficio(
        r"medic[oa]", r"enfermer[oa]", r"doctor(?:a)?", r"fisioterapeuta", r"farmaceutic[oa]", r"matron(?:a)?",
        r"celador(?:a)?", r"sanitari[oa]", r"odontolog[oa]", r"dentista", r"tcae", r"auxiliar de enfermeria",
    )),
    ("ingeniero", _oficio(r"ingenier[oa]")),
    ("abogado", _oficio(r"abogad[oa]", r"letrad[oa]", r"procurador(?:a)?")),
    ("comercial", _oficio(r"comercial", r"vendedor(?:a)?", r"dependient[oa]", r"agente comercial", r"representante de ventas")),
    ("comercial", _sector(r"ventas", r"comercio")),
    ("hosteleria", _oficio(r"camarer[oa]", r"cociner[oa]", r"barista", r"chef", r"bartender", r"jefe de sala")),
    ("hosteleria", _sector(r"hosteleria")),
    ("administracion publica", _oficio(
        r"funcionari[oa]", r"policia(?: local| nacional)?", r"guardia civil", r"bombero(?:a)?", r"militar",
    )),
    ("administracion publica", _sector(r"administracion publica")),
    ("administracion publica", r"\btrabajo para (?:el )?(?:ayuntamiento|gobierno|junta|diputacion|ministerio)\b"),
    ("construccion", _oficio(
        r"albanil", r"electricista", r"fontaner[oa]", r"encofrador(?:a)?", r"peon de obra", r"jefe de obra", r"maestro de obra",
    )),
    ("construccion", _sector(r"construccion")),
    ("transporte", _oficio(
        r"camioner[oa]", r"taxista", r"repartidor(?:a)?", r"conductor(?:a)?(?: de \w+)?", r"piloto", r"azafat[oa]",
        r"maquinista", r"mensajer[oa]",
    )),
    ("transporte", _sector(r"transporte", r"logistica")),
)
def ocupacion_autodeclarada(texto: str) -> str | None:
    """Sector profesional autodeclarado con frase-ancla en primera persona
    ('soy X', 'trabajo de/como X', 'curro de X'). La palabra suelta NO basta:
    'anuncio comercial' o 'transporte público' no son ocupaciones."""
    for clausula in clausulas(normalizar(texto)):
        valor = primer_valor(OCUPACION_ANCLADA, clausula)
        if valor is not None:
            return valor
    return None


# ---------------------------------------------------------------------------
# Universidad
# ---------------------------------------------------------------------------

_ANCLA_UNIVERSIDAD = (
    r"(?:estudio|estudiando|estudie|estudiante|graduad[oa]|licenciad[oa]|egresad[oa]|alumn[oa]|matriculad[oa]|"
    r"cursando|mi uni(?:versidad)?(?: es)?|voy a la|soy de la|estoy en la)(?:[a-z0-9 ]{0,40}?)\b(?:la |el )?"
)


def _uni(*alias: str) -> str:
    return _ANCLA_UNIVERSIDAD + r"(?:" + "|".join(alias) + r")\b"


# Nombre corto coherente con el que ya guarda 'universidad de X' (solo 'X').
UNIVERSIDADES: Banco = _banco(
    ("UNED", _uni(r"uned")),
    ("Oviedo", _uni(r"uniovi")),
    ("Politécnica de Madrid", _uni(r"upm")),
    ("Complutense de Madrid", _uni(r"ucm")),
    ("Autónoma de Madrid", _uni(r"uam")),
    ("Carlos III de Madrid", _uni(r"uc3m")),
    ("Rey Juan Carlos", _uni(r"urjc")),
    ("Alcalá", _uni(r"uah")),
    ("Politécnica de Valencia", _uni(r"upv")),
    ("Granada", _uni(r"ugr")),
    ("Salamanca", _uni(r"usal")),
    ("Autónoma de Barcelona", _uni(r"uab")),
    ("Politécnica de Cataluña", _uni(r"upc")),
    ("Pompeu Fabra", _uni(r"upf")),
    ("País Vasco", _uni(r"ehu")),
    ("A Coruña", _uni(r"udc")),
    ("Vigo", _uni(r"uvigo")),
    ("Santiago de Compostela", _uni(r"usc")),
    ("La Laguna", _uni(r"ull")),
    ("Las Palmas de Gran Canaria", _uni(r"ulpgc")),
    ("Zaragoza", _uni(r"unizar")),
    ("Cantabria", _uni(r"unican")),
    ("Extremadura", _uni(r"unex")),
    ("Castilla-La Mancha", _uni(r"uclm")),
    ("Pablo de Olavide", _uni(r"upo")),
    ("Burgos", _uni(r"ubu")),
    ("León", _uni(r"unileon")),
    ("La Rioja", _uni(r"unirioja")),
    ("Miguel Hernández", _uni(r"umh")),
    ("Jaume I", _uni(r"uji")),
    ("Islas Baleares", _uni(r"uib")),
    ("Lleida", _uni(r"udl")),
    ("Girona", _uni(r"udg")),
)
# "Universidad Complutense de Madrid", "Universidad Politécnica de Valencia":
# el nombre propio tras 'universidad' cuando NO es 'universidad de X' (ese
# caso ya lo cubre la regex original). Se aplica sobre el texto ORIGINAL
# porque exige mayúsculas en el nombre propio.
UNIVERSIDAD_NOMBRE_PROPIO_RE = re.compile(
    r"\b[Uu]niversidad\s+((?!de\b|del\b)[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+"
    r"(?:\s+(?:de|del|la|y)\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+|\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+){0,3})"
)


def universidad_autodeclarada(texto: str) -> str | None:
    for clausula in clausulas(normalizar(texto)):
        valor = primer_valor(UNIVERSIDADES, clausula)
        if valor is not None:
            return valor
    match = UNIVERSIDAD_NOMBRE_PROPIO_RE.search(texto)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Empresa (sobre texto original: exige mayúscula inicial en el nombre)
# ---------------------------------------------------------------------------

_NOMBRE_EMPRESA = (
    r"[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ&\-]*"
    r"(?:\s+(?:(?:de|del|la|las|los|el|y|&)\s+)?[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ&\-]*){0,3}"
)
EMPRESA_RE = re.compile(
    r"\b(?i:trabajo|trabajamos|trabajando|curro|currando|laboro)(?i:\s+(?:actualmente|ahora|desde\s+\w+))?\s+"
    r"(?i:en|para)\s+(" + _NOMBRE_EMPRESA + r")"
    r"|\b(?i:soy\s+emplead[oa]\s+de|mi\s+empresa\s+es|hago\s+pr[áa]cticas\s+en|soy\s+becari[oa]\s+en)\s+("
    + _NOMBRE_EMPRESA + r")"
    r"|\b(?i:i\s+work\s+(?:at|for))\s+(" + _NOMBRE_EMPRESA + r")"
)


def empresa_autodeclarada(texto: str, toponimos_conocidos: frozenset[str]) -> str | None:
    """Empresa donde dice trabajar. Se descartan los topónimos de las
    tablas del INE ('trabajo en Madrid' es un lugar, no una empresa)."""
    for match in EMPRESA_RE.finditer(texto):
        nombre = next(g for g in match.groups() if g).strip(" .-")
        if normalizar(nombre) in toponimos_conocidos:
            continue
        return nombre
    return None


# ---------------------------------------------------------------------------
# Nacionalidad
# ---------------------------------------------------------------------------

_GENTILICIOS = (
    r"marroqui(?:es)?", r"rumano|rumana", r"colombian[oa]", r"venezolan[oa]", r"ecuatorian[oa]", r"peruan[oa]",
    r"argentin[oa]", r"bolivian[oa]", r"chin[oa]", r"ucranian[oa]", r"italian[oa]", r"aleman(?:a)?",
    r"britanic[oa]", r"ingles(?:a)?", r"frances(?:a)?", r"portugues(?:a)?", r"brasilen[oa]", r"paraguay[oa]",
    r"uruguay[oa]", r"chilen[oa]", r"dominican[oa]", r"cuban[oa]", r"mexican[oa]", r"hondurenio|hondurena",
    r"salvadoren[oa]", r"guatemalteco|guatemalteca", r"nicaraguense", r"panamen[oa]", r"costarricense",
    r"haitian[oa]", r"senegales(?:a)?", r"argelin[oa]", r"nigerian[oa]", r"ghanes(?:a)?", r"pakistani",
    r"indi[oa]", r"bangladesi", r"filipin[oa]", r"rus[oa]", r"bulgar[oa]", r"polac[oa]", r"lituan[oa]",
    r"georgian[oa]", r"holandes(?:a)?", r"belga", r"suiz[oa]", r"suec[oa]", r"noruego|noruega", r"irlandes(?:a)?",
    r"estadounidense", r"canadiense", r"japones(?:a)?", r"corean[oa]", r"turc[oa]", r"siri[oa]", r"libanes(?:a)?",
    r"egipci[oa]", r"camerunes(?:a)?", r"guinean[oa]",
)
_PAISES = (
    "marruecos", "rumania", "colombia", "venezuela", "ecuador", "peru", "argentina", "bolivia", "china", "ucrania",
    "italia", "alemania", "reino unido", "inglaterra", "francia", "portugal", "brasil", "paraguay", "uruguay",
    "chile", "republica dominicana", "cuba", "mexico", "honduras", "el salvador", "guatemala", "nicaragua",
    "panama", "costa rica", "haiti", "senegal", "argelia", "nigeria", "ghana", "pakistan", "india", "bangladesh",
    "filipinas", "rusia", "bulgaria", "polonia", "lituania", "holanda", "paises bajos", "belgica", "suiza",
    "suecia", "noruega", "irlanda", "estados unidos", "eeuu", "canada", "japon", "corea", "turquia", "siria",
    "libano", "egipto", "camerun", "guinea",
)
NACIONALIDAD: Banco = _banco(
    (
        "espanola",
        r"\b(?:soy|somos)\s+(?:100 ?% )?espanol(?:a|es|as)?\b|\bnacionalidad espanola\b|\bsoy de espana\b"
        r"|\borgullos[oa] de ser espanol(?:a)?\b",
    ),
    (
        "extranjera",
        r"\b(?:soy|somos)\s+(?:un |una )?(?:100 ?% )?(?:" + "|".join(_GENTILICIOS) + r")\b"
        r"|\b(?:soy|somos) extranjer[oa]s?\b|\bnacionalidad extranjera\b"
        r"|\bnacionalidad (?:" + "|".join(_GENTILICIOS) + r")\b"
        r"|\bsoy de (?:" + "|".join(_PAISES) + r")\b",
    ),
)


def nacionalidad_autodeclarada(texto: str) -> str | None:
    for clausula in clausulas(normalizar(texto)):
        valor = primer_valor(NACIONALIDAD, clausula)
        if valor is not None:
            return valor
    return None


# ---------------------------------------------------------------------------
# Situación laboral
# ---------------------------------------------------------------------------

_PAREJA_O_HOGAR = r"(?:pareja|novi[oa]|marido|mujer|espos[oa])"
# Orden deliberado (igual que antes): lo específico antes que 'activo'.
SITUACION_LABORAL: Banco = _banco(
    (
        "parado",
        r"\b(?:estoy (?:en (?:el )?paro|desemplead[oa]|sin (?:trabajo|empleo|curro))|"
        r"sin (?:trabajo|empleo) desde|(?:busco|buscando|estoy buscando|en busca de) (?:trabajo|empleo|curro)|"
        r"desemplead[oa]|cobrando (?:el )?paro|apuntad[oa] al paro)\b",
    ),
    ("jubilado", r"\b(?:jubilad[oa]|prejubilad[oa]|pensionista|me jubile|mi jubilacion)\b"),
    (
        "estudiante",
        r"\b(?:soy estudiante|estudiante a tiempo completo|estudiante (?:de|en|universitari[oa])|soy universitari[oa]|"
        r"estudio en la (?:uni|universidad|facultad|escuela|instituto)|"
        r"estoy estudiando (?:en|la carrera|un grado|el grado|un master|bachillerato|la eso|fp)|"
        r"estoy en (?:\d[ºo]?|primero|segundo|tercero|cuarto|quinto|ultimo) (?:de|curso)|"
        r"estoy cursando (?:el )?(?:bachillerato|la eso|un ciclo|fp|el grado|la carrera))\b",
    ),
    (
        "otro_inactivo",
        r"\b(?:ama de casa|amo de casa|labores del hogar|mis labores|me dedico (?:a la casa|al hogar)|"
        r"incapacidad permanente|incapacitad[oa] permanente|invalidez permanente|pension (?:por incapacidad|de invalidez))\b",
    ),
    (
        "activo",
        r"\b(?:trabajo (?:en|de|para|como)|soy autonom[oa]|tengo (?:un )?(?:trabajo|empleo|curro)|"
        r"soy (?:funcionari[oa]|emplead[oa]|asalariad[oa]|empresari[oa]|freelance)|mi (?:empleo|empresa)|"
        r"salgo del trabajo|voy al trabajo)\b",
    ),
)


def situacion_laboral_autodeclarada(texto: str) -> str | None:
    for clausula in clausulas(normalizar(texto)):
        valor = primer_valor(SITUACION_LABORAL, clausula)
        if valor is not None:
            return valor
    return None


# ---------------------------------------------------------------------------
# Lengua materna
# ---------------------------------------------------------------------------

LENGUA_MATERNA: Banco = _banco(
    (
        "catalan",
        r"\b(?:mi lengua materna es(?: el)? catalan|hablo(?: el)? catalan|catalanoparlante|catalanohablante|"
        r"en casa (?:hablamos|hablo|parlamos) catalan|mi idioma (?:materno|habitual) es(?: el)? catalan)\b",
    ),
    (
        "euskera",
        r"\b(?:mi lengua materna es(?: el)? (?:euskera|euskara|vasco)|hablo(?: el)? (?:euskera|euskara|vasco)|"
        r"euskaldun|euskaldunak?|en casa (?:hablamos|hablo) (?:euskera|euskara|vasco)|euskaraz)\b",
    ),
    (
        "gallego",
        r"\b(?:mi lengua materna es(?: el)? (?:gallego|galego)|hablo(?: el)? (?:gallego|galego)|galegofalante|"
        r"galegoparlante|en casa (?:hablamos|hablo) (?:gallego|galego))\b",
    ),
    (
        "valenciano",
        r"\b(?:mi lengua materna es(?: el)? valenciano|hablo(?: el)? valenciano|en casa (?:hablamos|hablo) valenciano)\b",
    ),
)


def lengua_materna_autodeclarada(texto: str) -> str | None:
    for clausula in clausulas(normalizar(texto)):
        valor = primer_valor(LENGUA_MATERNA, clausula)
        if valor is not None:
            return valor
    return None


# ---------------------------------------------------------------------------
# Tipo de hogar (señales que se combinan sobre todos los posts)
# ---------------------------------------------------------------------------

HOGAR_SOLO_RE = re.compile(r"\b(?:vivo sol[oa]|vivo yo sol[oa]|vivo por mi cuenta|vivo en solitario)\b")
HOGAR_CON_PAREJA_RE = re.compile(
    rf"\b(?:vivo con mi {_PAREJA_O_HOGAR}|convivo con mi {_PAREJA_O_HOGAR}|vivimos (?:mi|con mi) {_PAREJA_O_HOGAR}(?: y yo)?|"
    rf"mi {_PAREJA_O_HOGAR} y yo (?:vivimos|compartimos)|vivimos juntos|vivimos juntas)\b"
)
HOGAR_MONOPARENTAL_RE = re.compile(
    r"\b(?:madre soltera|padre soltero|familia monoparental|familia monomarental|madre sola con|padre solo con|"
    r"criando sola a|criando solo a)\b"
)
HOGAR_HIJOS_RE = re.compile(
    r"\b(?:mis? hij[oa]s?|mis (?:ninos|ninas|peques|pequenos|gemelos|mellizos)|mi (?:bebe|nino|nina)|"
    r"soy (?:padre|madre|papa|mama) de)\b"
)


# ---------------------------------------------------------------------------
# Religión y signo zodiacal (texto)
# ---------------------------------------------------------------------------

_ANCLA_CREENCIA = (
    r"\b(?:soy|somos|me considero|me declaro|profeso|practico|mi religion es|de religion|soy practicante)\s+"
    r"(?:(?:muy|bastante|profundamente|100 ?%|un|una|el|la)\s+)*"
)
# Todo exige frase-ancla en primera persona: 'colegio católico' o 'Cristiano
# Ronaldo' no declaran la religión del autor.
_CREENCIAS = (
    ("judaismo", r"judi[oa]|judaismo|de religion judia"),
    ("islam", r"musulman(?:a)?|islam|del islam"),
    ("catolicismo", r"catolic[oa]"),
    (
        "cristianismo",
        r"cristian[oa]|evangelic[oa]|protestante|ortodox[oa]|testigos? de jehova|mormon(?:a)?|adventista|pentecostal",
    ),
    ("budismo", r"budista|budismo"),
    ("hinduismo", r"hinduista|hinduismo"),
    ("ateismo", r"ate[oa]|ateismo"),
    ("agnosticismo", r"agnostic[oa]"),
)
RELIGION_ANCLADA: Banco = _banco(
    *((valor, _ANCLA_CREENCIA + r"(?:" + alias + r")\b") for valor, alias in _CREENCIAS)
)


def religion_autodeclarada(texto: str) -> str | None:
    for clausula in clausulas(normalizar(texto)):
        valor = primer_valor(RELIGION_ANCLADA, clausula)
        if valor is not None:
            return valor
    return None


# 'leo' (verbo), 'libra' (verbo/unidad), 'cancer' (enfermedad) y 'acuario'
# (el del zoo) solo cuentan como signo si se declaran como tal.
SIGNOS_AMBIGUOS = frozenset({"leo", "libra", "cancer", "acuario"})
SIGNO_ANCLADO_RE = re.compile(
    r"\b(?:soy|mi signo(?: del zodiaco)?(?: es)?|signo(?: de)?|ascendente|del signo|nacid[oa] bajo)\s+(?:de\s+)?"
    r"(?:un |una )?(leo|libra|cancer|acuario)\b"
)
