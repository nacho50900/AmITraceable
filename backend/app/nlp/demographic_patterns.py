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

Banco = tuple[tuple[str, re.Pattern[str]], ...]

# Valores canónicos que se repiten en varios bancos.
_HOMBRE = "hombre"
_MUJER = "mujer"
_DOCENTE = "docente"
_COMERCIAL = "comercial"
_HOSTELERIA = "hosteleria"
_ADMIN_PUBLICA = "administracion publica"
_CONSTRUCCION = "construccion"
_TRANSPORTE = "transporte"
_MOTIVO_NACIMIENTO = "año de nacimiento"
_MOTIVO_GRADUACION = "graduación"


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


def buscar_propios(patron: re.Pattern[str], clausula: str):
    """Coincidencias del patrón en la cláusula que NO hablan de un tercero."""
    return [m for m in patron.finditer(clausula) if es_propio(clausula, m.start())]


# ---------------------------------------------------------------------------
# Frases sobre terceros
# ---------------------------------------------------------------------------

_TERCERA_PERSONA_RE = re.compile(
    r"\b(?:mi|mis|tu|tus|su|sus|nuestr[oa]s?)\s+(?:(?:querid|viej|joven|ex|difunt|futur)[oa]s?\s+)?"
    r"(?:padre|madre|mama|papa|padres|abuel[oa]s?|suegr[oa]s?|tio|tia|tios|tias|vecin[oa]s?|amig[oa]s?|hermano|hermana|"
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


def clausulas_con_original(texto: str, separar_por_y: bool = True) -> list[tuple[str, str]]:
    """Pares `(cláusula original, cláusula normalizada)`. La normalización
    (minúsculas, sin tildes) conserva la longitud del texto salvo en casos
    raros (ligaduras Unicode), así que los mismos índices sirven para
    recuperar del original el nombre propio, con mayúscula y tilde, que la
    regex encontró sobre el normalizado. Si la longitud cambiara, se devuelve
    el normalizado en ambos lados (nunca se desalinean índices)."""
    normalizado = normalizar(texto)
    original = texto if len(normalizado) == len(texto) else normalizado
    separador = _SEPARADOR_CLAUSULAS_RE if separar_por_y else _SEPARADOR_SIN_Y_RE
    pares: list[tuple[str, str]] = []
    inicio = 0
    for corte in [*separador.finditer(normalizado), None]:
        fin = corte.start() if corte else len(normalizado)
        if normalizado[inicio:fin].strip():
            pares.append((original[inicio:fin].strip(), normalizado[inicio:fin].strip()))
        if corte:
            inicio = corte.end()
    return pares


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
    return _edad_valida(edad), (_HOMBRE if letra == "M" else _MUJER)


def _edad_anclada(clausula: str) -> int | None:
    for patron in _EDAD_ANCLADA:
        for match in buscar_propios(patron, clausula):
            if _edad_valida(int(match.group(1))) is not None:
                return int(match.group(1))
    return None


def _edad_en_palabras(clausula: str) -> int | None:
    for match in buscar_propios(_EDAD_EN_PALABRAS_RE, clausula):
        valor = _edad_valida(numero_en_palabras(match.group(1)))
        if valor is not None:
            return valor
    return None


def _edad_generica(clausula: str) -> int | None:
    """'N años' suelto, descartando duraciones ('hace 15 años')."""
    for match in buscar_propios(_EDAD_GENERICA_RE, clausula):
        prefijo = clausula[max(0, match.start() - 18) : match.start()]
        if _NO_ES_EDAD_RE.search(prefijo):
            continue
        valor = _edad_valida(int(match.group(1)))
        if valor is not None:
            return valor
    return None


def edad_autodeclarada(texto: str) -> int | None:
    """Edad literal escrita por el propio usuario, o None. Orden: primero
    las frases-ancla (más fiables), después la edad en palabras y por último
    el 'N años' suelto."""
    edad, _ = edad_y_sexo_reddit(texto)
    if edad is not None:
        return edad

    for clausula in clausulas(normalizar(texto)):
        edad = _edad_anclada(clausula) or _edad_en_palabras(clausula) or _edad_generica(clausula)
        if edad is not None:
            return edad
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
    return _HOMBRE if es_hombre else _MUJER


# ---------------------------------------------------------------------------
# Ubicación
# ---------------------------------------------------------------------------

# Anclas de RESIDENCIA en primera persona (con el candidato a topónimo en el
# grupo 1). Deliberadamente NO incluye 'soy de X' (origen, no residencia) ni
# 'vive en X' (tercera persona) ni 'vivo cerca de X' (impreciso).
_RESIDENCIA_ANCLAS: tuple[re.Pattern[str], ...] = tuple(
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

_ESTUDIOS_ANCLAS: tuple[re.Pattern[str], ...] = tuple(
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
    (_DOCENTE, _oficio(_DOCENTE, r"profe", r"profesor(?:a)?", r"maestr[oa]", r"catedratic[oa]", r"educador(?:a)?")),
    (_DOCENTE, r"\bdoy clases\b"),
    ("sanitario", _oficio(
        r"medic[oa]", r"enfermer[oa]", r"doctor(?:a)?", r"fisioterapeuta", r"farmaceutic[oa]", r"matron(?:a)?",
        r"celador(?:a)?", r"sanitari[oa]", r"odontolog[oa]", r"dentista", r"tcae", r"auxiliar de enfermeria",
    )),
    ("ingeniero", _oficio(r"ingenier[oa]")),
    ("abogado", _oficio(r"abogad[oa]", r"letrad[oa]", r"procurador(?:a)?")),
    (_COMERCIAL, _oficio(_COMERCIAL, r"vendedor(?:a)?", r"dependient[oa]", r"agente comercial", r"representante de ventas")),
    (_COMERCIAL, _sector(r"ventas", r"comercio")),
    (_HOSTELERIA, _oficio(r"camarer[oa]", r"cociner[oa]", r"barista", r"chef", r"bartender", r"jefe de sala")),
    (_HOSTELERIA, _sector(r"hosteleria")),
    (_ADMIN_PUBLICA, _oficio(
        r"funcionari[oa]", r"policia(?: local| nacional)?", r"guardia civil", r"bombero(?:a)?", r"militar",
    )),
    (_ADMIN_PUBLICA, _sector(r"administracion publica")),
    (_ADMIN_PUBLICA, r"\btrabajo para (?:el )?(?:ayuntamiento|gobierno|junta|diputacion|ministerio)\b"),
    (_CONSTRUCCION, _oficio(
        r"albanil", r"electricista", r"fontaner[oa]", r"encofrador(?:a)?", r"peon de obra", r"jefe de obra", r"maestro de obra",
    )),
    (_CONSTRUCCION, _sector(r"construccion")),
    (_TRANSPORTE, _oficio(
        r"camioner[oa]", r"taxista", r"repartidor(?:a)?", r"conductor(?:a)?(?: de \w+)?", r"piloto", r"azafat[oa]",
        r"maquinista", r"mensajer[oa]",
    )),
    (_TRANSPORTE, _sector(r"transporte", r"logistica")),
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
_ALIAS_UNIVERSIDADES: tuple[tuple[str, str], ...] = (
    ("UNED", "uned"),
    ("Oviedo", "uniovi"),
    ("Politécnica de Madrid", "upm"),
    ("Complutense de Madrid", "ucm"),
    ("Autónoma de Madrid", "uam"),
    ("Carlos III de Madrid", "uc3m"),
    ("Rey Juan Carlos", "urjc"),
    ("Alcalá", "uah"),
    ("Politécnica de Valencia", "upv"),
    ("Granada", "ugr"),
    ("Salamanca", "usal"),
    ("Autónoma de Barcelona", "uab"),
    ("Politécnica de Cataluña", "upc"),
    ("Pompeu Fabra", "upf"),
    ("País Vasco", "ehu"),
    ("A Coruña", "udc"),
    ("Vigo", "uvigo"),
    ("Santiago de Compostela", "usc"),
    ("La Laguna", "ull"),
    ("Las Palmas de Gran Canaria", "ulpgc"),
    ("Zaragoza", "unizar"),
    ("Cantabria", "unican"),
    ("Extremadura", "unex"),
    ("Castilla-La Mancha", "uclm"),
    ("Pablo de Olavide", "upo"),
    ("Burgos", "ubu"),
    ("León", "unileon"),
    ("La Rioja", "unirioja"),
    ("Miguel Hernández", "umh"),
    ("Jaume I", "uji"),
    ("Islas Baleares", "uib"),
    ("Lleida", "udl"),
    ("Girona", "udg"),
)
UNIVERSIDADES: Banco = _banco(*((n, _uni(a)) for n, a in _ALIAS_UNIVERSIDADES))
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


# ---------------------------------------------------------------------------
# Edad indirecta: fecha/año de nacimiento y años de graduación
# ---------------------------------------------------------------------------
# A diferencia de `edad_autodeclarada` ("tengo 24 años"), aquí la edad se
# DEDUCE de un año: "nací en 1999" (2 edades posibles según el cumpleaños),
# "me gradué en 2019" (edad típica al graduarse + años transcurridos). Se
# devuelve una edad EXACTA solo con fecha de nacimiento completa; en el
# resto, un RANGO. Los rangos de graduación son deliberadamente anchos: la
# incertidumbre la absorbe el ancho, no una confianza inventada (mismo
# principio que `ai_attribute_extraction._set_edad_rango`, motivado por un
# caso real donde una estimación estrecha y equivocada se coló).

from dataclasses import dataclass
from datetime import date

_MES_A_NUMERO = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_MESES_RE = "|".join(_MES_A_NUMERO)
_ANIO = r"(19\d{2}|20[0-2]\d)"

_FECHA_NACIMIENTO_RE = re.compile(
    r"\b(?:naci|nacid[oa]|fecha de nacimiento|cumpleanos|mi cumple)\b[^.\n]{0,30}?"
    r"(?:(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})|(\d{1,2}) de (" + _MESES_RE + r") de (\d{4}))"
)
_ANIO_NACIMIENTO_RE = tuple(
    re.compile(p)
    for p in (
        r"\b(?:naci|soy nacid[oa]|yo naci)\s+(?:en|el)\s+(?:el\s+)?(?:ano\s+)?(19\d{2}|20[01]\d)\b",
        r"\bsoy del\s+(?:ano\s+)?(\d{2}|19\d{2}|20[01]\d)\b(?!\s*(?:%|euros|km|kg))",
        r"\bquinta del\s+(\d{2}|19\d{2}|20[01]\d)\b",
        r"\bi was born (?:in|on)\b[^.\n]{0,15}?\b(19\d{2}|20[01]\d)\b",
    )
)

# (patrón, edad mínima y máxima al ocurrir el hito). Grupo 1 = año del hito.
_HITOS_ACADEMICOS: tuple[tuple[re.Pattern[str], int, int], ...] = tuple(
    (re.compile(p), lo, hi)
    for p, lo, hi in (
        (
            r"\b(?:me gradue|me licencie|me diplome|me titule|termine|acabe|finalice)\s+(?:de |en |con |la |el |mi )*"
            r"(?:carrera|grado|universidad|licenciatura|ingenieria|diplomatura|uni)\b[^.\n]{0,25}?\b(?:en|el|del)?\s*" + _ANIO,
            21, 27,
        ),
        (r"\b(?:me gradue|termine|acabe|finalice|hice|cursee?)\s+(?:el |mi |un )+master\b[^.\n]{0,25}?\b(?:en|el|del)?\s*" + _ANIO, 22, 32),
        (r"\b(?:defendi|termine|acabe|finalice|hice)\s+(?:la |mi |el )*(?:tesis|doctorado)\b[^.\n]{0,25}?\b(?:en|el|del)?\s*" + _ANIO, 25, 40),
        (r"\b(?:de|en) la promocion\s+(?:de |del )?" + _ANIO, 17, 27),
        (r"\bpromocion\s+(?:de |del )?" + _ANIO, 17, 27),
        (r"\b(?:hice|aprobe|pase|me presente a)\s+(?:la\s+)?(?:selectividad|ebau|pau|evau)\b[^.\n]{0,20}?\b(?:en|el|del)?\s*" + _ANIO, 17, 21),
        (r"\b(?:termine|acabe|finalice)\s+(?:el\s+)?bachillerato\b[^.\n]{0,20}?\b(?:en|el|del)?\s*" + _ANIO, 17, 19),
        (
            r"\b(?:empece|comence|entre en|inicie)\s+(?:la\s+|el\s+|en la\s+)?(?:carrera|uni|universidad|grado)\b[^.\n]{0,20}?\b(?:en|el|del)?\s*" + _ANIO,
            17, 25,
        ),
        (r"\bi graduated(?: from [a-z ]{2,40})? in\s+" + _ANIO, 21, 27),
    )
)

_ANCHO_MAXIMO_RANGO = 20


@dataclass(frozen=True)
class EdadIndirecta:
    exacta: int | None
    minima: int | None
    maxima: int | None
    motivo: str  # p. ej. "fecha de nacimiento", _MOTIVO_NACIMIENTO, _MOTIVO_GRADUACION


def _anio_completo(dos_cifras_o_cuatro: str, hoy: date) -> int | None:
    """'99' -> 1999, '05' -> 2005, '2001' -> 2001; None si la edad resultante
    cae fuera de 12-100 (para 'soy del 20' o cosas que no son un año)."""
    valor = int(dos_cifras_o_cuatro)
    candidatos = [valor] if valor >= 1000 else [1900 + valor, 2000 + valor]
    for anio in candidatos:
        if EDAD_MIN <= hoy.year - anio <= EDAD_MAX:
            return anio
    return None


def _edad_exacta(nacimiento: date, hoy: date) -> int:
    return hoy.year - nacimiento.year - ((hoy.month, hoy.day) < (nacimiento.month, nacimiento.day))


def _rango_por_anio(anio: int, edad_al_hito: tuple[int, int], hoy: date) -> tuple[int, int] | None:
    minima = max(hoy.year - anio + edad_al_hito[0] - 1, EDAD_MIN)
    maxima = min(hoy.year - anio + edad_al_hito[1], EDAD_MAX)
    return (minima, maxima) if minima <= maxima else None


def _fecha_nacimiento(clausula: str, hoy: date) -> date | None:
    """Fecha de nacimiento completa declarada en la cláusula, o None."""
    m = _FECHA_NACIMIENTO_RE.search(clausula)
    if not m or not es_propio(clausula, m.start()):
        return None
    numerica = m.group(1) is not None
    anio = _anio_completo(m.group(3) if numerica else m.group(6), hoy)
    if anio is None:
        return None
    mes = int(m.group(2)) if numerica else _MES_A_NUMERO[m.group(5)]
    dia = int(m.group(1)) if numerica else int(m.group(4))
    try:
        fecha = date(anio, mes, dia)
    except ValueError:
        return None
    return fecha if fecha <= hoy else None


def _edad_por_fecha(clausula: str, hoy: date) -> int | None:
    fecha = _fecha_nacimiento(clausula, hoy)
    if fecha is None:
        return None
    edad = _edad_exacta(fecha, hoy)
    return edad if EDAD_MIN <= edad <= EDAD_MAX else None


def _rangos_de_clausula(clausula: str, hoy: date) -> list[tuple[int, int, str]]:
    """Rangos `(mínima, máxima, motivo)` deducidos de años de nacimiento y
    de hitos académicos de la cláusula."""
    rangos: list[tuple[int, int, str]] = []
    for patron in _ANIO_NACIMIENTO_RE:
        for m in buscar_propios(patron, clausula):
            anio = _anio_completo(m.group(1), hoy)
            rango = _rango_por_anio(anio, (0, 0), hoy) if anio else None
            if rango:
                rangos.append((*rango, _MOTIVO_NACIMIENTO))
    for patron, lo, hi in _HITOS_ACADEMICOS:
        for m in buscar_propios(patron, clausula):
            rango = _rango_por_anio(int(m.group(1)), (lo, hi), hoy)
            if rango:
                rangos.append((*rango, _MOTIVO_GRADUACION))
    return rangos


def _intersectar(rangos: list[tuple[int, int, str]]) -> EdadIndirecta | None:
    if not rangos:
        return None
    minima = max(r[0] for r in rangos)
    maxima = min(r[1] for r in rangos)
    if minima > maxima or maxima - minima > _ANCHO_MAXIMO_RANGO:
        return None
    motivo = _MOTIVO_NACIMIENTO if any(r[2] == _MOTIVO_NACIMIENTO for r in rangos) else _MOTIVO_GRADUACION
    return EdadIndirecta(None, minima, maxima, motivo)


def edad_indirecta(textos: list[str], hoy: date) -> EdadIndirecta | None:
    """Edad deducida de fechas/años que el autor da sobre sí mismo. Si hay
    fecha de nacimiento completa, edad exacta; si no, la intersección de
    todos los rangos encontrados. Contradicciones (intersección vacía) o
    rangos de más de `_ANCHO_MAXIMO_RANGO` años se descartan: no aportan."""
    rangos: list[tuple[int, int, str]] = []
    for texto in textos:
        for clausula in clausulas(normalizar(texto), separar_por_y=False):
            edad = _edad_por_fecha(clausula, hoy)
            if edad is not None:
                return EdadIndirecta(edad, None, None, "fecha de nacimiento")
            rangos += _rangos_de_clausula(clausula, hoy)
    return _intersectar(rangos)


# ---------------------------------------------------------------------------
# Biografía (fragmentos sin frase-ancla)
# ---------------------------------------------------------------------------
# Una bio habla del autor POR DEFINICIÓN, así que aquí se relajan las
# frase-ancla que sí exigen las publicaciones: "Católica | Madrid",
# "Ingeniera @ Indra", "23 años", "Madre de 2" son declaraciones aunque no
# lleven "soy". Solo se aplica al pseudo-post de tipo "bio" (ver
# `report/generator.py::_posts_with_bio_pseudo_post`) y solo rellena campos
# que las reglas normales no hayan fijado ya (ver `bio_atributos`).

_SEPARADOR_BIO_RE = re.compile(r"[|·•\n]+|\s[-–—/]\s|\s{3,}")

_BIO_RELIGION: Banco = _banco(
    *((valor, r"\b(?:" + alias + r")\b") for valor, alias in _CREENCIAS)
)

_BIO_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("desarrollador de software", (
        r"programador(?:a)?", r"desarrollador(?:a)?", r"developer", r"dev", r"software engineer",
        r"ingenier[oa] (?:de|del) software", r"(?:backend|frontend|full ?stack)",
    )),
    (_DOCENTE, (_DOCENTE, r"profe", r"profesor(?:a)?", r"maestr[oa]", r"catedratic[oa]", r"educador(?:a)?")),
    ("sanitario", (
        r"medic[oa]", r"enfermer[oa]", r"fisioterapeuta", r"farmaceutic[oa]", r"matron(?:a)?", r"sanitari[oa]",
        r"odontolog[oa]", r"dentista", r"tcae",
    )),
    ("ingeniero", (r"ingenier[oa]", r"ing\.")),
    ("abogado", (r"abogad[oa]", r"letrad[oa]", r"procurador(?:a)?")),
    (_COMERCIAL, (_COMERCIAL, r"vendedor(?:a)?", r"dependient[oa]")),
    (_HOSTELERIA, (r"camarer[oa]", r"cociner[oa]", r"barista", r"chef", r"bartender")),
    (_ADMIN_PUBLICA, (r"funcionari[oa]", r"policia(?: local| nacional)?", r"guardia civil", r"bombero(?:a)?", r"militar")),
    (_CONSTRUCCION, (r"albanil", r"electricista", r"fontaner[oa]", r"encofrador(?:a)?", r"jefe de obra")),
    (_TRANSPORTE, (r"camioner[oa]", r"taxista", r"repartidor(?:a)?", r"piloto", r"azafat[oa]", r"maquinista")),
)
_BIO_OCUPACION: Banco = _banco(
    *((valor, r"(?<![@\w])(?:" + "|".join(roles) + r")(?!\w)") for valor, roles in _BIO_ROLES)
)
# Títulos que implican formación superior en España (habilitan `nivel_estudios`).
_BIO_TITULO_SUPERIOR_RE = re.compile(
    r"(?<![@\w])(?:phd|msc|mba|doctorand[oa]|doctor(?:a)? en|graduad[oa]|licenciad[oa]|ingenier[oa]|arquitect[oa]|"
    r"abogad[oa]|medic[oa]|farmaceutic[oa]|dentista|ing\.|lic\.)(?!\w)"
)

_BIO_EDAD_RE = re.compile(r"(?<![\w+/])(\d{2})\s*(?:anos|yo|y/o)\b(?!\s+(?:de|en|con)\s+\w)")
_BIO_UBICACION_RE = re.compile(r"📍\s*([^\n|·•,]{2,40})")
_BIO_EMPRESA_RE = re.compile(r"(?:@\s+|\bat\s+|\bAt\s+)((?:[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ&\-]*)(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ&\-]*){0,2})")
_BIO_ESTUDIOS_RE = re.compile(r"(?<!\w)(?:grado|carrera|licenciatura|master|doctorado|degree)\s+(?:en|de|in)\s+([a-z ]{3,60})")
_BIO_SITUACION: Banco = _banco(
    ("estudiante", r"(?<!\w)(?:estudiante|universitari[oa])(?!\w)"),
    ("jubilado", r"(?<!\w)(?:jubilad[oa]|prejubilad[oa])(?!\w)"),
    ("parado", r"(?<!\w)(?:en paro|desemplead[oa]|buscando trabajo|buscando empleo)(?!\w)"),
    ("activo", r"(?<!\w)(?:autonom[oa]|emprendedor(?:a)?|freelance|empresari[oa]|funcionari[oa])(?!\w)"),
)
_BIO_SEXO_MUJER_RE = re.compile(r"(?<!\w)(?:madre|mama|mami|esposa|abuela|chica|chavala)(?!\w)")
_BIO_SEXO_HOMBRE_RE = re.compile(r"(?<!\w)(?:padre|papa|papi|esposo|marido|abuelo|chico|chaval)(?!\w)")


def _alias_universidad_bio(nombre: str, alias: str) -> tuple[str, str]:
    """Alias largos valen sueltos ('Uniovi'); los cortos ('uam', 'ugr') solo
    con '@' delante, para no confundirlos con otras palabras."""
    return (nombre, rf"(?<![\w])@?{alias}(?!\w)" if len(alias) >= 5 else rf"(?<![\w])@{alias}(?!\w)")


_POSESIVO_AL_FINAL_RE = re.compile(r"\b(?:mi|mis|su|sus|tu|tus|nuestr[oa]s?)\s*$")


def _sexo_propio_en_bio(patron: re.Pattern[str], fragmento: str) -> bool:
    """'Madre de 2' declara el sexo del autor; 'mi madre es mi heroína' no.
    `es_propio` mira lo que va ANTES de la coincidencia, pero aquí la propia
    coincidencia ('madre') es el tercero, precedido de un posesivo."""
    m = patron.search(fragmento)
    if not m:
        return False
    return es_propio(fragmento, m.start()) and not _POSESIVO_AL_FINAL_RE.search(fragmento[: m.start()])


def _fragmentos_bio(texto: str) -> list[tuple[str, str]]:
    """Fragmentos de la bio (partida por `|`, `·`, `•`, saltos de línea o
    ' - '), cada uno como `(original, normalizado)`."""
    fragmentos: list[tuple[str, str]] = []
    for trozo in _SEPARADOR_BIO_RE.split(texto):
        if trozo.strip():
            fragmentos.append((trozo.strip(), normalizar(trozo).strip()))
    return fragmentos


def _bio_campos_simples(norm: str, universidades: Banco, encontrados: dict[str, object]) -> None:
    bancos = (
        ("religion", _BIO_RELIGION), ("ocupacion", _BIO_OCUPACION),
        ("situacion_laboral", _BIO_SITUACION), ("universidad", universidades),
    )
    for clave, banco in bancos:
        valor = primer_valor(banco, norm)
        if valor is not None:
            encontrados.setdefault(clave, valor)
    if _BIO_TITULO_SUPERIOR_RE.search(norm):
        encontrados.setdefault("nivel_estudios", "superior")


def _bio_edad(norm: str) -> int | None:
    m = _BIO_EDAD_RE.search(norm)
    if not m or not es_propio(norm, m.start()) or not EDAD_MIN <= int(m.group(1)) <= EDAD_MAX:
        return None
    prefijo = norm[max(0, m.start() - 18) : m.start()]
    return None if _NO_ES_EDAD_RE.search(prefijo) else int(m.group(1))


def _bio_sexo(mujer: bool, hombre: bool) -> str | None:
    if mujer == hombre:
        return None
    return _MUJER if mujer else _HOMBRE


def bio_atributos(texto: str) -> dict[str, object]:
    """Atributos de una biografía sin exigir frase-ancla. Claves posibles:
    'religion', 'ocupacion', 'nivel_estudios', 'situacion_laboral', 'sexo',
    'universidad', 'edad' (str -> valor canónico o int), 'empresa',
    'ubicacion' y 'estudios' (listas de candidatos normalizados, para
    resolverlos con las mismas tablas INE que las publicaciones)."""
    encontrados: dict[str, object] = {}
    universidades = _banco(*(_alias_universidad_bio(n, a) for n, a in _ALIAS_UNIVERSIDADES))
    ubicaciones: list[str] = []
    estudios: list[str] = []
    mujer = hombre = False

    for original, norm in _fragmentos_bio(texto):
        _bio_campos_simples(norm, universidades, encontrados)
        edad = _bio_edad(norm)
        if edad is not None:
            encontrados.setdefault("edad", edad)
        mujer = mujer or _sexo_propio_en_bio(_BIO_SEXO_MUJER_RE, norm)
        hombre = hombre or _sexo_propio_en_bio(_BIO_SEXO_HOMBRE_RE, norm)
        empresa = _BIO_EMPRESA_RE.search(original)
        if empresa:
            encontrados.setdefault("empresa", empresa.group(1).strip(" .-"))
        ubicaciones += [normalizar(m.group(1)).strip() for m in _BIO_UBICACION_RE.finditer(original)]
        estudios += [m.group(1).strip() for m in _BIO_ESTUDIOS_RE.finditer(norm)]

    sexo = _bio_sexo(mujer, hombre)
    if sexo:
        encontrados["sexo"] = sexo
    if ubicaciones:
        encontrados["ubicacion"] = ubicaciones
    if estudios:
        encontrados["estudios"] = estudios
    return encontrados
