"""
Extracción de datos demográficos AUTODECLARADOS por el propio usuario en su
texto (p. ej. "tengo 24 años", "vivo en León", "estudio Medicina").

Esto es distinto de `attribute_inference.py`, que infiere atributos de
forma indirecta a partir de en qué comunidades/hashtags participa el
usuario. Aquí se buscan menciones EXPLÍCITAS en primera persona, que son
las que alimentan el estimador de k-anonimato (`scoring/k_anonymity.py`):
si no sabemos la edad o provincia exactas que el usuario ha escrito sobre
sí mismo, no tiene sentido intentar "adivinarlas" para ese cálculo — el
objetivo de este módulo es solo capturar lo que el usuario YA ha revelado
literalmente sobre sí mismo en su propio texto público.

Como el resto del proyecto: heurísticas simples y explicables (regex) en
vez de NER/modelos más agresivos, para mantener el resultado auditable y
el alcance defensivo.
"""
import re
import unicodedata
from dataclasses import dataclass, field

from app.data.ine_reference import (
    AUTONOMOUS_COMMUNITY_PROVINCES,
    MUNICIPALITY_POPULATION,
    OCCUPATION_DISTRIBUTION,
    PROVINCE_POPULATION,
    STUDIES_DISTRIBUTION,
    resolve_autonomous_community_in_text,
)
from app.models.schemas import InferredAttribute, SocialPost


def _strip_accents(text: str) -> str:
    """Quita tildes/diéresis para poder comparar contra las claves de las
    tablas de `ine_reference.py`, que están sin acentuar (p. ej. 'leon',
    'avila'). Necesario porque el texto real de los usuarios sí lleva
    tildes ('León', 'Ávila')."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


@dataclass
class DemographicFindings:
    sexo: str | None = None
    edad: int | None = None
    provincia: str | None = None
    municipio: str | None = None
    # Se rellena en dos casos: (1) autodeclaración explícita de una
    # comunidad autónoma COMPLETA en el propio texto (p.ej. "vivo en
    # Canarias", sin decir la isla/provincia) -- lo detecta este mismo
    # módulo, ver `_match_location`; o (2) geolocation.py (vía
    # report/generator.py) cuando la estimación por imagen solo identifica
    # una comunidad con VARIAS provincias posibles y no hay autodeclaración
    # de texto que la sustituya. En ambos casos es porque no se puede
    # reducir a una sola provincia sin inventar información -- ver
    # AUTONOMOUS_COMMUNITY_PROVINCES en ine_reference.py.
    comunidad_autonoma: str | None = None
    # Permalinks de fotos cuyo pie de foto indica que la persona está DE
    # VIAJE/VACACIONES en ese sitio (no en su lugar de residencia habitual)
    # -- ver app/nlp/travel_detection.py (regex) y ai_attribute_extraction.py
    # (campo "fotos_de_viaje" del prompt). NUNCA lo rellena este módulo (solo
    # detecta autodeclaraciones de sexo/edad/ubicación/etc., no intención de
    # viaje); se deja aquí para que report/generator.py tenga un único sitio
    # donde consultar "qué fotos NO debo usar para inferir dónde vive esta
    # persona". No participa en merge_findings (no es un campo INE, es una
    # exclusión que se une con travel_detection.py, no se sobrescribe).
    travel_permalinks: set[str] = field(default_factory=set)
    estudios: str | None = None
    ocupacion: str | None = None
    universidad: str | None = None
    empresa: str | None = None
    # Nacionalidad autodeclarada: "espanola" | "extranjera" | None. A
    # diferencia de origen étnico/racial (ver ADR-17), la nacionalidad
    # LEGAL no es dato de categoría especial del art. 9 RGPD -- ver
    # NATIONALITY_DISTRIBUTION en ine_reference.py.
    nacionalidad: str | None = None
    # Situación laboral autodeclarada: "activo" (trabaja) | "parado" |
    # "jubilado" | "estudiante" | "otro_inactivo" | None. Distinto de
    # `ocupacion` (que es SECTOR profesional, p.ej. "sanitario") -- aquí
    # es si la persona trabaja, busca trabajo, está jubilada o estudia.
    situacion_laboral: str | None = None
    # Tipo de hogar en el que vive: "unipersonal" | "pareja_sin_hijos" |
    # "pareja_con_hijos" | "monoparental" | None. A diferencia del resto de
    # campos de esta clase, NO se detecta con una única regex por post: se
    # necesita combinar señales de varios posts (menciones a pareja/hijos
    # pueden aparecer en publicaciones distintas) -- ver
    # `_detect_household_type`, llamada una vez al final sobre TODOS los
    # posts, no dentro del bucle principal como el resto.
    tipo_hogar: str | None = None
    # Lengua materna/habitual cooficial autodeclarada: "catalan" |
    # "euskera" | "gallego" | "valenciano" | None. Solo tiene sentido como
    # filtro de población SI también se conoce la comunidad autónoma (ver
    # LANGUAGE_BY_CCAA en ine_reference.py y k_anonymity.py::_step_lengua) --
    # el catalán fuera de Cataluña/Baleares es una señal casi inútil.
    lengua_materna: str | None = None
    # Estado civil / relación de pareja. A diferencia de sexo/edad/ubicación
    # (autodeclaraciones EXPLÍCITAS), este campo SOLO lo rellena la IA
    # razonando sobre contenido simbólico/indirecto (emojis, fechas, estilo
    # de escritura -- p. ej. una bio "18/05/20🧡👸✨" sugiere un aniversario
    # de pareja; "mi marido y yo" sugiere matrimonio; "mi difunto marido"
    # sugiere viudedad). Valores: "soltero" | "con_pareja" (pareja sin
    # estar casado/a) | "casado" | "viudo" | None (sin señal en ningún
    # sentido -- "desconocido", no "soltero" por defecto). Se
    # eleva a campo propio (en vez de vivir solo en `soft_inferences`)
    # porque SÍ debe participar en el estimador de k-anonimato
    # (scoring/k_anonymity.py -> MARITAL_STATUS_DISTRIBUTION), a petición
    # explícita: que aparezca en la tabla de "qué se puede inferir sobre
    # ti" y afecte al porcentaje de población restante, no solo en la
    # lista genérica de atributos inferidos. Se marca con
    # `source["estado_civil"] = "ia_simbolica"` (nunca "texto" ni "ia" a
    # secas) para que el informe deje claro que es una inferencia
    # simbólica de fiabilidad menor, no una autodeclaración.
    estado_civil: str | None = None
    # Orientación sexual autodeclarada explícitamente (p. ej. "soy
    # heterosexual", "soy gay"). Detectada por regex si aparece literal;
    # también puede ser rellenada por la IA si viene de inferencia directa
    # del texto. Valores libres en minúscula: "heterosexual", "gay",
    # "lesbiana", "bisexual", "pansexual", "asexual", "homosexual".
    orientacion_sexual: str | None = None
    # Signo zodiacal autodeclarado o indicado mediante emoji zodiacal
    # (♈ Aries, ♉ Tauro, …). Incluye el margen de fechas de nacimiento
    # que implica, p. ej. "aries (21 mar - 19 abr)". Solo lo rellena la IA
    # (el emoji por sí solo no es suficiente texto para regex).
    signo_zodiacal: str | None = None
    # Creencia religiosa autodeclarada o indicada mediante emoji/símbolo
    # (✡️ judaísmo, ☪️ islam, ✝️/🕊️ cristianismo, etc.). Solo lo rellena
    # la IA.
    religion: str | None = None
    # permalinks de los posts que dispararon cada detección, para trazabilidad
    evidence: dict[str, list[str]] = field(default_factory=dict)
    # procedencia de cada dato detectado: "texto" (autodeclaración escrita,
    # por defecto) o "imagen" (estimada vía app/vision/geolocation.py). Solo
    # se rellena explícitamente cuando algo viene de imagen; lo que viene de
    # este módulo es siempre "texto".
    source: dict[str, str] = field(default_factory=dict)
    # Inferencias BLANDAS detectadas por IA a partir de contenido simbólico
    # o indirecto (emojis, fechas, estilo de escritura...) -- p. ej.
    # "18/05/20🧡👸✨" en la biografía sugiriendo el aniversario de una
    # relación de pareja. A diferencia de TODO lo demás en esta clase, esto
    # NO es una autodeclaración explícita ni un dato del INE con el que
    # estrechar población: son señales probabilísticas de baja certeza, con
    # su propia confianza (0-1), pensadas para la lista general de
    # "atributos inferidos" del informe (`InferredAttribute`,
    # `report.inferred_attributes`), no para el estimador de k-anonimato.
    # Vive aquí (y no en un objeto de retorno aparte) por la misma razón
    # pragmática que `travel_permalinks`: sale de la MISMA llamada a
    # Mistral que rellena el resto de este dataclass (ver
    # app/nlp/ai_attribute_extraction.py), y así se evita una segunda
    # llamada a la IA solo para esto. No participa en `merge_findings` (no
    # es un campo INE): report/generator.py lo lee directamente.
    soft_inferences: list[InferredAttribute] = field(default_factory=list)


_AGE_RE = re.compile(r"\b(?:tengo|con)\s+(\d{1,2})\s+años\b|\b(\d{1,2})\s+años\b", re.I)
_SEX_MALE_RE = re.compile(r"\b(soy un chico|soy un chaval|soy hombre)\b", re.I)
_SEX_FEMALE_RE = re.compile(r"\b(soy una chica|soy mujer)\b", re.I)
_UNIVERSITY_RE = re.compile(r"\buniversidad de (\w+)", re.I)
# Nota: se usa [Tt]rabajo (clase de caracteres en la primera letra) en vez
# de un grupo con flag "(?i:trabajo)", porque ese grupo no contiene ninguna
# alternancia y Sonar lo marca como "unnecessarily grouped subpattern"
# (python:regex). Cubre el caso real que nos importa (mayúscula al empezar
# frase: "Trabajo en Indra..."), aunque ya no cubre variantes en mayúsculas
# intermedias tipo "TRABAJO" -- caso que no aparece en el uso real de bios
# de redes sociales y no está cubierto por los tests existentes.
_COMPANY_RE = re.compile(r"\b[Tt]rabajo (?:en|para)\s+([A-Z][\wÁÉÍÓÚáéíóú]+)")
_STUDY_VERB_RE = re.compile(r"\b(?:estudio|estudiante de|graduad[oa] en)\s+([a-záéíóúñ ]+)", re.I)

# Nacionalidad. No se intenta cubrir todas las nacionalidades del mundo
# (inabarcable con regex y fuera del alcance de este MVP): se cubren la
# autodeclaración explícita ("nacionalidad española/extranjera") y los
# gentilicios de las nacionalidades más numerosas en España a fecha de
# escribir esto (ver NATIONALITY_DISTRIBUTION en ine_reference.py) -- si
# no aparece ninguno de estos patrones, se queda en None (no se asume
# "española por defecto": eso sería inventar un dato que el usuario no ha
# declarado).
_NATIONALITY_ESPANOLA_RE = re.compile(
    r"\b(soy español|soy española|nacionalidad española|de nacionalidad española)\b", re.I
)
_NATIONALITY_EXTRANJERA_RE = re.compile(
    r"\b(soy extranjero|soy extranjera|nacionalidad extranjera|de nacionalidad extranjera|"
    r"soy marroqu[ií]|soy colombian[oa]|soy rumano|soy rumana|soy venezolan[oa]|"
    r"soy peruan[oa]|soy ecuatorian[oa]|soy argentin[oa]|soy bolivian[oa]|soy chin[oa])\b",
    re.I,
)

# Situación laboral (distinto de `ocupacion`, que es SECTOR profesional).
_EMPLOYMENT_PARADO_RE = re.compile(r"\b(estoy en paro|desemplead[oa]|buscando empleo|busco trabajo)\b", re.I)
_EMPLOYMENT_JUBILADO_RE = re.compile(r"\b(jubilad[oa]|pensionista)\b", re.I)
_EMPLOYMENT_ESTUDIANTE_RE = re.compile(r"\b(soy estudiante|estudiante a tiempo completo)\b", re.I)
_EMPLOYMENT_ACTIVO_RE = re.compile(
    r"\b(trabajo (?:en|de|para|como)|soy autónomo|soy autónoma|tengo trabajo)\b", re.I
)

# Tipo de hogar: señales que se combinan sobre TODOS los posts (no una
# regex "ganadora" por post, ver `_detect_household_type`).
_HOUSEHOLD_ALONE_RE = re.compile(r"\bvivo sol[oa]\b", re.I)
_HOUSEHOLD_WITH_PARTNER_RE = re.compile(
    r"\bvivo con mi (pareja|novio|novia|marido|mujer|esposo|esposa)\b", re.I
)
_HOUSEHOLD_MONOPARENTAL_RE = re.compile(r"\b(madre soltera|padre soltero|familia monoparental)\b", re.I)
_HOUSEHOLD_CHILDREN_MENTION_RE = re.compile(r"\bmis? hij[oa]s?\b", re.I)

# Lengua materna/habitual cooficial. Igual que con nacionalidad, no se
# intenta cubrir todo el espectro dialectal (asturiano, aragonés, aranés...
# -- ver ADR correspondiente si se amplía en el futuro): solo las 4 lenguas
# cooficiales con tabla de referencia en LANGUAGE_BY_CCAA.
_LANGUAGE_CATALAN_RE = re.compile(r"\b(mi lengua materna es el catalán|hablo catalán|catalanoparlante)\b", re.I)
_LANGUAGE_EUSKERA_RE = re.compile(r"\b(mi lengua materna es el euskera|hablo euskera|euskaldun)\b", re.I)
_LANGUAGE_GALLEGO_RE = re.compile(r"\b(mi lengua materna es el gallego|hablo gallego|galegofalante)\b", re.I)
_LANGUAGE_VALENCIANO_RE = re.compile(r"\b(mi lengua materna es el valenciano|hablo valenciano)\b", re.I)


# Orientación sexual: se cubren las autodeclaraciones explícitas más
# comunes en bios de redes sociales.
_SEXUALITY_RE = re.compile(
    r"\b(soy heterosexual|soy hetero|soy gay|soy lesbiana|soy bisexual|"
    r"soy pansexual|soy asexual|soy homosexual|"
    r"heterosexual|homosexual|bisexual|pansexual|asexual)\b",
    re.I,
)
# Mapa de emojis zodiacales a (nombre_signo, rango_fechas). Se mantiene como
# dict (no regex) porque los emojis son caracteres Unicode puntuales.
_ZODIAC_EMOJI_MAP: dict[str, tuple[str, str]] = {
    "\u2648": ("aries",       "21 mar - 19 abr"),  # ♈
    "\u2649": ("tauro",       "20 abr - 20 may"),  # ♉
    "\u264a": ("geminis",     "21 may - 20 jun"),  # ♊
    "\u264b": ("cancer",      "21 jun - 22 jul"),  # ♋
    "\u264c": ("leo",         "23 jul - 22 ago"),  # ♌
    "\u264d": ("virgo",       "23 ago - 22 sep"),  # ♍
    "\u264e": ("libra",       "23 sep - 22 oct"),  # ♎
    "\u264f": ("escorpio",    "23 oct - 21 nov"),  # ♏
    "\u2650": ("sagitario",   "22 nov - 21 dic"),  # ♐
    "\u2651": ("capricornio", "22 dic - 19 ene"),  # ♑
    "\u2652": ("acuario",     "20 ene - 18 feb"),  # ♒
    "\u2653": ("piscis",      "19 feb - 20 mar"),  # ♓
}


def extract_demographics(posts: list[SocialPost]) -> DemographicFindings:
    findings = DemographicFindings()

    for post in posts:
        text = post.text or ""
        if not text:
            continue

        _try_detect_edad(text, post.permalink, findings)
        _try_detect_sexo(text, post.permalink, findings)
        _try_detect_location(text, post.permalink, findings)
        _try_detect_estudios(text, post.permalink, findings)
        _try_detect_ocupacion(text, post.permalink, findings)
        _try_detect_universidad(text, post.permalink, findings)
        _try_detect_empresa(text, post.permalink, findings)
        _try_detect_nacionalidad(text, post.permalink, findings)
        _try_detect_situacion_laboral(text, post.permalink, findings)
        _try_detect_lengua_materna(text, post.permalink, findings)
        _try_detect_orientacion_sexual(text, post.permalink, findings)
        _try_detect_signo_zodiacal(text, post.permalink, findings)

    _detect_household_type(posts, findings)
    _mark_all_detected_as_texto(findings)
    return findings


def _mark_all_detected_as_texto(findings: DemographicFindings) -> None:
    """Todo lo detectado por este módulo viene de texto autodeclarado (por
    definición: es lo único que procesa). Se marca explícitamente para que
    el frontend pueda distinguirlo de lo que venga de geolocation.py."""
    for attr_name in (
        "sexo", "edad", "provincia", "municipio", "comunidad_autonoma",
        "estudios", "ocupacion", "universidad", "empresa",
        "nacionalidad", "situacion_laboral", "tipo_hogar", "lengua_materna",
        "orientacion_sexual", "signo_zodiacal",
    ):
        if getattr(findings, attr_name) is not None:
            findings.source[attr_name] = "texto"


def _try_detect_edad(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.edad is not None:
        return

    match = _AGE_RE.search(text)
    if not match:
        return

    age = int(match.group(1) or match.group(2))
    if 12 <= age <= 100:  # descarta falsos positivos ("100 años de historia")
        findings.edad = age
        findings.evidence.setdefault("edad", []).append(permalink)


def _try_detect_sexo(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.sexo is not None:
        return

    if _SEX_MALE_RE.search(text):
        findings.sexo = "hombre"
    elif _SEX_FEMALE_RE.search(text):
        findings.sexo = "mujer"
    else:
        return

    findings.evidence.setdefault("sexo", []).append(permalink)


def _try_detect_estudios(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.estudios is not None:
        return

    match = _STUDY_VERB_RE.search(text)
    if not match:
        return

    candidate = _strip_accents(match.group(1).strip().lower())
    matched = next((k for k in STUDIES_DISTRIBUTION if k in candidate), None)
    if matched:
        findings.estudios = matched
        findings.evidence.setdefault("estudios", []).append(permalink)


def _try_detect_ocupacion(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.ocupacion is not None:
        return

    lowered = _strip_accents(text.lower())
    matched = next((k for k in OCCUPATION_DISTRIBUTION if k in lowered), None)
    if matched:
        findings.ocupacion = matched
        findings.evidence.setdefault("ocupacion", []).append(permalink)


def _try_detect_universidad(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.universidad is not None:
        return

    match = _UNIVERSITY_RE.search(text)
    if match:
        findings.universidad = match.group(1)
        findings.evidence.setdefault("universidad", []).append(permalink)


def _try_detect_empresa(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.empresa is not None:
        return

    match = _COMPANY_RE.search(text)
    if match:
        findings.empresa = match.group(1)
        findings.evidence.setdefault("empresa", []).append(permalink)


def _try_detect_nacionalidad(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.nacionalidad is not None:
        return

    if _NATIONALITY_ESPANOLA_RE.search(text):
        findings.nacionalidad = "espanola"
    elif _NATIONALITY_EXTRANJERA_RE.search(text):
        findings.nacionalidad = "extranjera"
    else:
        return

    findings.evidence.setdefault("nacionalidad", []).append(permalink)


def _try_detect_situacion_laboral(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.situacion_laboral is not None:
        return

    # Orden deliberado: "parado"/"jubilado"/"estudiante" son más
    # específicos y menos ambiguos que "activo" (que podría dar un falso
    # positivo con menciones genéricas al trabajo que no describen la
    # situación laboral actual de la persona, p.ej. citando el trabajo de
    # otra persona) -- se comprueban primero.
    if _EMPLOYMENT_PARADO_RE.search(text):
        findings.situacion_laboral = "parado"
    elif _EMPLOYMENT_JUBILADO_RE.search(text):
        findings.situacion_laboral = "jubilado"
    elif _EMPLOYMENT_ESTUDIANTE_RE.search(text):
        findings.situacion_laboral = "estudiante"
    elif _EMPLOYMENT_ACTIVO_RE.search(text):
        findings.situacion_laboral = "activo"
    else:
        return

    findings.evidence.setdefault("situacion_laboral", []).append(permalink)


def _try_detect_lengua_materna(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.lengua_materna is not None:
        return

    if _LANGUAGE_CATALAN_RE.search(text):
        findings.lengua_materna = "catalan"
    elif _LANGUAGE_EUSKERA_RE.search(text):
        findings.lengua_materna = "euskera"
    elif _LANGUAGE_GALLEGO_RE.search(text):
        findings.lengua_materna = "gallego"
    elif _LANGUAGE_VALENCIANO_RE.search(text):
        findings.lengua_materna = "valenciano"
    else:
        return

    findings.evidence.setdefault("lengua_materna", []).append(permalink)


def _collect_household_signals(posts: list[SocialPost]) -> tuple[bool, bool, bool, bool, list[str]]:
    """Recorre TODOS los posts (no uno solo -- ver docstring de
    `_detect_household_type`) acumulando qué señales de tipo de hogar
    aparecen en cualquiera de ellos, y en qué publicaciones concretas."""
    lives_alone = False
    lives_with_partner = False
    mentions_children = False
    monoparental_explicit = False
    evidence: list[str] = []

    for post in posts:
        text = post.text or ""
        if not text:
            continue
        if _HOUSEHOLD_MONOPARENTAL_RE.search(text):
            monoparental_explicit = True
            evidence.append(post.permalink)
        if _HOUSEHOLD_ALONE_RE.search(text):
            lives_alone = True
            evidence.append(post.permalink)
        if _HOUSEHOLD_WITH_PARTNER_RE.search(text):
            lives_with_partner = True
            evidence.append(post.permalink)
        if _HOUSEHOLD_CHILDREN_MENTION_RE.search(text):
            mentions_children = True
            evidence.append(post.permalink)

    return lives_alone, lives_with_partner, mentions_children, monoparental_explicit, evidence


def _resolve_household_type(
    lives_alone: bool, lives_with_partner: bool, mentions_children: bool, monoparental_explicit: bool
) -> str | None:
    """Decide la categoría de tipo de hogar a partir de las señales ya
    recogidas (ver `_collect_household_signals`). Orden de prioridad: una
    autodeclaración EXPLÍCITA de "familia monoparental" pesa más que
    combinar señales indirectas, y "vivo solo/a" es inequívoco por sí solo
    (no necesita combinarse con nada). Devuelve None si ninguna
    combinación es reconocible -- nunca se adivina."""
    if monoparental_explicit or (mentions_children and lives_alone):
        return "monoparental"
    if lives_alone:
        return "unipersonal"
    if lives_with_partner and mentions_children:
        return "pareja_con_hijos"
    if lives_with_partner:
        return "pareja_sin_hijos"
    return None


def _detect_household_type(posts: list[SocialPost], findings: DemographicFindings) -> None:
    """A diferencia del resto de detectores de este módulo, el tipo de
    hogar necesita combinar señales que pueden aparecer en publicaciones
    DISTINTAS (p.ej. "vivo con mi pareja" en una bio y "mis hijos" en un
    post posterior) -- por eso se hace en una pasada propia sobre todos los
    posts en vez de una regex "ganadora" por post como el resto de
    `_try_detect_*`."""
    if findings.tipo_hogar is not None:
        return

    lives_alone, lives_with_partner, mentions_children, monoparental_explicit, evidence = (
        _collect_household_signals(posts)
    )
    tipo_hogar = _resolve_household_type(lives_alone, lives_with_partner, mentions_children, monoparental_explicit)
    if tipo_hogar is None:
        return  # ninguna combinación reconocible -- se queda en None, no se adivina

    findings.tipo_hogar = tipo_hogar
    findings.evidence.setdefault("tipo_hogar", []).extend(dict.fromkeys(evidence))  # sin duplicados, conserva orden


def _try_detect_location(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.provincia is not None or findings.municipio is not None or findings.comunidad_autonoma is not None:
        return
    _match_location(text, permalink, findings)


def _match_location(text: str, permalink: str, findings: DemographicFindings) -> None:
    lowered = _strip_accents(text.lower())

    # Municipio primero (más específico); si hay match, no hace falta
    # comprobar provincia por separado para ese mismo texto.
    m = re.search(r"\bvivo en ([a-z ]+)", lowered)
    candidate = m.group(1).strip() if m else None

    if not candidate:
        return

    muni_match = next((k for k in MUNICIPALITY_POPULATION if k in candidate), None)
    if muni_match:
        findings.municipio = muni_match
        findings.evidence.setdefault("municipio", []).append(permalink)
        return

    prov_match = next((k for k in PROVINCE_POPULATION if k in candidate), None)
    if prov_match:
        findings.provincia = prov_match
        findings.evidence.setdefault("provincia", []).append(permalink)
        return

    # Ni municipio ni provincia concreta: puede que haya nombrado una
    # comunidad autónoma COMPLETA (p.ej. "vivo en Canarias", "vivo en
    # Andalucía"). Si esa comunidad tiene una sola provincia, no hay
    # ambigüedad y se usa directamente esa provincia (más específico); si
    # tiene varias, se guarda al nivel de comunidad autónoma.
    ccaa = resolve_autonomous_community_in_text(candidate)
    if ccaa is None:
        return

    provinces = AUTONOMOUS_COMMUNITY_PROVINCES[ccaa]
    if len(provinces) == 1:
        findings.provincia = provinces[0]
        findings.evidence.setdefault("provincia", []).append(permalink)
    else:
        findings.comunidad_autonoma = ccaa
        findings.evidence.setdefault("comunidad_autonoma", []).append(permalink)


def _try_detect_orientacion_sexual(text: str, permalink: str, findings: DemographicFindings) -> None:
    """Detecta autodeclaraciones explícitas de orientación sexual.
    El match más específico ('soy heterosexual') tiene preferencia
    sobre el token suelto ('heterosexual'), ya que el token suelto podría
    aparecer al hablar de otra persona; se toma el primer grupo capturado."""
    if findings.orientacion_sexual is not None:
        return

    match = _SEXUALITY_RE.search(text)
    if not match:
        return

    raw = match.group(0).lower()
    # Normalizar a valor canónico
    if "hetero" in raw:
        value = "heterosexual"
    elif "gay" in raw:
        value = "gay"
    elif "lesbiana" in raw:
        value = "lesbiana"
    elif "bisexual" in raw:
        value = "bisexual"
    elif "pansexual" in raw:
        value = "pansexual"
    elif "asexual" in raw:
        value = "asexual"
    elif "homosexual" in raw:
        value = "homosexual"
    else:
        return

    findings.orientacion_sexual = value
    findings.evidence.setdefault("orientacion_sexual", []).append(permalink)


def _try_detect_signo_zodiacal(text: str, permalink: str, findings: DemographicFindings) -> None:
    """Detecta emojis de signos zodiacales en el texto y almacena el
    signo junto con su rango de fechas de nacimiento implícito."""
    if findings.signo_zodiacal is not None:
        return

    for emoji_char, (signo, rango) in _ZODIAC_EMOJI_MAP.items():
        if emoji_char in text:
            findings.signo_zodiacal = f"{signo} ({rango})"
            findings.evidence.setdefault("signo_zodiacal", []).append(permalink)
            return
