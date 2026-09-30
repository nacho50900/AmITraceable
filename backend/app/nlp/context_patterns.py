"""
Bancos de regex para el CONTEXTO que rodea a la persona: dónde vive con más
detalle que provincia/municipio, cuándo no va a estar en casa, a quién más
expone (terceros y menores) y qué relación de pareja declara.

Mismas convenciones que `lifestyle_patterns.py` (texto normalizado, por
cláusula, descarte de terceros con `dp.es_propio`), con una diferencia:
los nombres propios ("mi novia Laura") solo se distinguen por la mayúscula,
así que esos patrones corren sobre la cláusula ORIGINAL (ver
`dp.clausulas_con_original`), y el resto sobre la normalizada.

Sobre terceros y menores: el informe habla del autor, pero un post que
nombra a su hijo o cuenta que su madre está enferma expone también a otra
persona que no ha dado su consentimiento. Por eso el valor del informe solo
lleva la INICIAL del nombre ("Nombra a su pareja (L.)"), nunca el nombre.

La relación de pareja ("relacion") es una inferencia blanda aparte: NO
rellena `DemographicFindings.estado_civil`, que por diseño solo lo fija la IA
simbólica (`source["estado_civil"] = "ia_simbolica"`, ver
demographic_extraction.py) para no presentar como dato duro algo que suele
inferirse de pistas indirectas.
"""
import re

from app.nlp import demographic_patterns as dp
from app.nlp.lifestyle_patterns import CONDICIONES, Hallazgo

# Literales que se repiten en varios bancos.
_UBICACION_DETALLADA = "ubicacion_detallada"
_VIAJE_FUTURO = "viaje_futuro"
_MENOR = "menor"
_TERCERO = "tercero"
_PAREJA = "su pareja"
_HIJO = "su hijo/a"

CONF_DIRECTA = 0.7
CONF_INDICIO = 0.55

_MESES = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
_PALABRAS = r"[a-z]+(?: [a-z]+){0,2}"


def _titulo(valor: str) -> str:
    return valor.strip().title()


def _inicial(nombre: str) -> str:
    return nombre.strip()[:1].upper() + "."


# ---------------------------------------------------------------------------
# Ubicación detallada
# ---------------------------------------------------------------------------

_BARRIO_RE = re.compile(
    r"\bvivo en (?:el |la |los |las )?(?:barrio|zona|urbanizacion|distrito|pueblo) (?:de |del |de la |de los )?(" + _PALABRAS + r")"
)
_CERCA_RE = re.compile(r"\bvivo (?:muy |bastante )?cerca (?:de la |del |de los |de las |de )(" + _PALABRAS + r")")
_TRABAJA_ZONA_RE = re.compile(
    r"\b(?:trabajo|curro|tengo la oficina) (?:en|por) (?:la |el )?(?:zona|calle|avenida|plaza|barrio|poligono) (?:de |del )?(" + _PALABRAS + r")"
)
_FRECUENTA_RE = re.compile(
    r"\b(?:voy (?:siempre|todos los dias|cada dia|a diario|cada semana) (?:a la |al |a )|suelo ir (?:a la |al |a )|mi sitio (?:de siempre|favorito) es (?:la |el )?)(" + _PALABRAS + r")"
)
_UBICACION_TIEMPO_REAL_RE = re.compile(r"\b(?:ubicacion en tiempo real|comparto mi ubicacion|mi ubicacion actual|live location)\b")
_PIN_RE = re.compile(r"📍\s*([^\n,|]{2,40})")


def _ubicacion(original: str, norm: str) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []
    etiquetas = (
        (_BARRIO_RE, "Menciona el barrio o la zona donde vive: {}"),
        (_CERCA_RE, "Dice que vive cerca de: {}"),
        (_TRABAJA_ZONA_RE, "Indica la zona donde trabaja: {}"),
        (_FRECUENTA_RE, "Menciona un lugar que frecuenta: {}"),
    )
    for patron, plantilla in etiquetas:
        m = patron.search(norm)
        if m and dp.es_propio(norm, m.start()):
            hallazgos.append(Hallazgo(_UBICACION_DETALLADA, plantilla.format(_titulo(m.group(1))), CONF_DIRECTA))
    if _UBICACION_TIEMPO_REAL_RE.search(norm):
        hallazgos.append(Hallazgo(_UBICACION_DETALLADA, "Comparte su ubicación en tiempo real", CONF_DIRECTA))
    pin = _PIN_RE.search(original)
    if pin:
        hallazgos.append(
            Hallazgo(_UBICACION_DETALLADA, f"Etiqueta un lugar concreto con 📍: {pin.group(1).strip()}", CONF_INDICIO)
        )
    return hallazgos


# ---------------------------------------------------------------------------
# Viajes futuros y ausencias de casa
# ---------------------------------------------------------------------------

_VERBO_VIAJE_RE = re.compile(
    r"\b(?:me voy|nos vamos|me marcho|nos marchamos|salgo|salimos|viajo|viajamos|volamos|vuelo|me escapo|"
    r"nos escapamos|cojo el avion|cogemos el avion|de viaje|de vacaciones|de escapada|de puente|"
    r"(?:tengo|tenemos) (?:el )?(?:vuelo|billete|billetes|reserva|hotel)|reserve|reservamos|"
    r"estare(?:mos)? (?:de viaje|de vacaciones))\b"
)
_FECHA_FUTURA_RE = re.compile(
    r"\b(?:del \d{1,2} al \d{1,2}(?: de " + _MESES + r")?|(?:el )?\d{1,2} de " + _MESES + r"|"
    r"(?:en|a finales de|a principios de|a mediados de|el proximo|este) " + _MESES + r"|"
    r"(?<!la )(?<!esta )(?<!cada )(?<!hasta )manana|pasado manana|esta noche|"
    r"este (?:finde|fin de semana|puente|sabado|domingo|viernes)|"
    r"el (?:proximo )?(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)|la semana que viene|"
    r"el mes que viene|el proximo (?:mes|fin de semana|puente)|"
    r"en (?:\d+|un par de|unos) (?:dias|semanas|meses)|dentro de \d+ (?:dias|semanas|meses))\b"
)
_PASADO_RE = re.compile(
    r"\b(?:volvi|volvimos|estuve|estuvimos|fui|fuimos|ayer|hace \d+|el ano pasado|el mes pasado|la semana pasada|"
    r"acabo de|acabamos de|de vuelta|ya estoy|ya estamos|recuerdo)\b"
)
_VIVIENDA_VACIA_RE = re.compile(
    r"\b(?:casa vacia|casa sola|piso vacio|sin nadie en casa|nadie en casa|dejo la casa sola|dejamos la casa sola)\b"
)
_FUERA_RE = re.compile(
    r"\b(?:estare fuera|estaremos fuera|voy a estar fuera|vamos a estar fuera|no estare en casa|no estaremos en casa)\b"
)


def _viajes(norm: str) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []
    verbo = _VERBO_VIAJE_RE.search(norm)
    fecha = _FECHA_FUTURA_RE.search(norm)
    if verbo and fecha and not _PASADO_RE.search(norm) and dp.es_propio(norm, verbo.start()):
        hallazgos.append(
            Hallazgo(_VIAJE_FUTURO, f"Anuncia un viaje o ausencia próxima (fecha: {fecha.group(0)})", CONF_DIRECTA)
        )
    if _VIVIENDA_VACIA_RE.search(norm):
        hallazgos.append(Hallazgo(_VIAJE_FUTURO, "Indica que su vivienda quedará sola o vacía", CONF_DIRECTA))
    if _FUERA_RE.search(norm):
        hallazgos.append(Hallazgo(_VIAJE_FUTURO, "Indica que estará fuera de casa unos días", CONF_INDICIO))
    return hallazgos


# ---------------------------------------------------------------------------
# Terceros y menores
# ---------------------------------------------------------------------------

_RELACIONES = {
    "hijo": _HIJO, "hija": _HIJO, "marido": _PAREJA, "mujer": _PAREJA, "esposa": _PAREJA,
    "esposo": _PAREJA, "novio": _PAREJA, "novia": _PAREJA, "pareja": _PAREJA,
    "madre": "su madre", "padre": "su padre", "hermano": "un hermano/a", "hermana": "un hermano/a",
    "abuelo": "un abuelo/a", "abuela": "un abuelo/a", "suegro": "un suegro/a", "suegra": "un suegro/a",
    "primo": "un primo/a", "prima": "un primo/a", "tio": "un tío/a", "tia": "un tío/a",
    "sobrino": "un sobrino/a", "sobrina": "un sobrino/a", "amigo": "un amigo/a", "amiga": "un amigo/a",
}
_MENORES = frozenset({"bebe", "nino", "nina", "peque"})
_REL_ORIGINAL = "|".join(
    ["hij[oa]", "marido", "mujer", "esposa", "esposo", "novi[oa]", "pareja", "madre", "padre", "herman[oa]",
     "abuel[oa]", "suegr[oa]", "prim[oa]", "t[ií][oa]", "sobrin[oa]", "amig[oa]", "beb[eé]", "ni[ñn][oa]", "peque"]
)
_NOMBRE_PROPIO_RE = re.compile(
    r"(?i:\bmi\s+(" + _REL_ORIGINAL + r")(?:\s+se llama)?)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñü]{2,})"
)
_EDAD_MENOR_RE = re.compile(
    r"\bmi\s+(?:hij[oa]|bebe|nin[oa]|peque|nieto|nieta|sobrin[oa])s?\s+(?:de |tiene |acaba de cumplir |cumple )?(\d{1,2})\s+(anos|meses)\b"
)
_COLEGIO_RE = re.compile(
    r"\b(?:mi|mis|nuestr[oa]s?)\s+(?:hij[oa]s?|nin[oa]s?|peques?)(?: [a-z]{3,12})?\s+(?:va|van|estudia|estudian|esta|estan)\s+"
    r"(?:al|en el|en la|a la|en)\s+(?:colegio|cole|guarderia|escuela infantil|instituto|ceip|ies)\b(?:\s+(" + _PALABRAS + r"))?"
)
_RECOGIDA_RE = re.compile(
    r"\b(?:recojo|llevo|dejo)\s+(?:a\s+)?(?:mi\s+|mis\s+)?(?:hij[oa]s?|nin[oa]s?|peques?)\s+(?:del|al|en el)\s+(?:cole|colegio|guarderia|escuela)\b"
)
_SALUD_TERCERO_RE = re.compile(
    r"\bmi\s+(madre|padre|hermano|hermana|abuelo|abuela|pareja|novio|novia|marido|mujer|hijo|hija|amigo|amiga)\s+"
    r"(?:(?:tiene|padece|sufre(?: de)?)\s+(?:un |una )?(?:" + "|".join(CONDICIONES) + r")|"
    r"esta (?:ingresad[oa]|enferm[oa]|en el hospital)|murio|ha muerto|ha fallecido|fallecio|se opera|le van a operar)\b"
)


def _terceros_y_menores(original: str, norm: str) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []

    for m in _NOMBRE_PROPIO_RE.finditer(original):
        if not dp.es_propio(norm, m.start()):
            continue
        relacion = dp.normalizar(m.group(1))
        inicial = _inicial(m.group(2))
        if relacion in _MENORES:
            hallazgos.append(Hallazgo(_MENOR, f"Nombra a un menor de su entorno ({inicial})", CONF_DIRECTA))
        else:
            hallazgos.append(Hallazgo(_TERCERO, f"Nombra a {_RELACIONES[relacion]} ({inicial})", CONF_DIRECTA))

    edad = _EDAD_MENOR_RE.search(norm)
    if edad and (edad.group(2) == "meses" or int(edad.group(1)) < 18):
        hallazgos.append(Hallazgo(_MENOR, "Indica la edad de un menor de su entorno", CONF_DIRECTA))

    colegio = _COLEGIO_RE.search(norm)
    if colegio:
        detalle = f": {_titulo(colegio.group(1))}" if colegio.group(1) else ""
        hallazgos.append(Hallazgo(_MENOR, f"Menciona el colegio o la guardería de un menor{detalle}", CONF_DIRECTA))
    if _RECOGIDA_RE.search(norm):
        hallazgos.append(Hallazgo(_MENOR, "Menciona la rutina de llevar o recoger a un menor del colegio", CONF_DIRECTA))

    salud = _SALUD_TERCERO_RE.search(norm)
    if salud:
        hallazgos.append(
            Hallazgo(_TERCERO, "Menciona la salud o el fallecimiento de un familiar o allegado (dato de terceros)", CONF_INDICIO)
        )
    return hallazgos


# ---------------------------------------------------------------------------
# Relación de pareja (inferencia blanda, no toca `estado_civil`)
# ---------------------------------------------------------------------------

_RELACION: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (valor, re.compile(patron))
    for valor, patron in (
        ("Menciona a su pareja actual", r"\b(?:tengo (?:un |una )?(?:novi[oa]|pareja)|mi (?:novi[oa]|pareja)\b|estoy (?:saliendo|en pareja|con mi pareja)|salgo con\b|mi chic[oa]\b)"),
        ("Menciona a su cónyuge o su boda", r"\b(?:estoy casad[oa]|me case|nos casamos|mi boda|mi luna de miel|mi (?:marido|esposa|esposo)\b)"),
        ("Se declara soltero/a", r"\b(?:estoy solter[oa]|soy solter[oa]|sin pareja|soltería)\b"),
        ("Menciona una separación o divorcio", r"\b(?:me (?:he )?divorci(?:e|ado|ada)|estoy divorciad[oa]|soy divorciad[oa]|mi ex\b|me separe|estoy separad[oa]|mi divorcio|mi separacion)"),
        ("Se declara viudo/a", r"\b(?:soy viud[oa]|enviude)\b"),
        ("Menciona su aniversario de pareja", r"\b(?:nuestro (?:primer )?aniversario|\d+ anos (?:juntos|con mi (?:pareja|novi[oa]|marido|mujer)))\b"),
    )
)


def _relacion(norm: str) -> list[Hallazgo]:
    return [Hallazgo("relacion", valor, CONF_DIRECTA - 0.1) for valor, patron in _RELACION if patron.search(norm)]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def detectar_contexto(texto: str) -> list[Hallazgo]:
    """Ubicación detallada, viajes/ausencias, terceros, menores y relación
    de pareja de un texto. Sin duplicados."""
    hallazgos: list[Hallazgo] = []
    for original, norm in dp.clausulas_con_original(texto, separar_por_y=False):
        hallazgos += _ubicacion(original, norm)
        hallazgos += _viajes(norm)
        hallazgos += _terceros_y_menores(original, norm)
        hallazgos += _relacion(norm)
    return list(dict.fromkeys(hallazgos))
