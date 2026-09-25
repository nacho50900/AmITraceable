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
    SPORT_PRACTICE_DISTRIBUTION,
    STUDIES_DISTRIBUTION,
    STUDIES_TO_RAMA,
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
    # Rango de edad estimado (p. ej. min=20, max=35). A diferencia de
    # `edad` (autodeclaración EXPLÍCITA, por regex o IA leyendo una frase
    # literal tipo "tengo 24 años"), estos dos campos los rellena
    # ÚNICAMENTE ai_attribute_extraction.py mediante razonamiento
    # INDIRECTO/simbólico sobre pistas sueltas (año de graduación, curso
    # que menciona estar haciendo, jerga generacional...) -- mismo
    # principio que `estado_civil`.
    #
    # A diferencia de la primera versión de este mecanismo (que encajaba
    # la estimación en uno de los tramos FIJOS de AGE_DISTRIBUTION_5Y),
    # el ANCHO del rango es libre y lo decide el propio modelo: si la
    # pista es débil, debe ENSANCHAR el rango hasta tener una confianza
    # alta genuina de que la edad real cae dentro -- un rango de 20 años
    # con confianza alta es preferible a uno estrecho con confianza baja
    # (ver el prompt exacto en `ai_attribute_extraction.py::_SYSTEM_PROMPT`,
    # sección 'edad_estimada', y `_set_edad_rango`). Cambio motivado por un
    # caso real en producción (Comandante, agosto 2026): con un tramo
    # quinquenal fijo y un umbral de confianza demasiado permisivo, se
    # coló una estimación de 30 años para una persona de 21 -- dejar que
    # el propio rango absorba la incertidumbre, en vez de forzar un umbral
    # arbitrario sobre un valor puntual, es más honesto y más difícil de
    # acertar mal: `AGE_DISTRIBUTION_1Y`/`age_range_proportion` en
    # ine_reference.py calculan la proporción de población de CUALQUIER
    # rango, no solo de los tramos quinquenales predefinidos.
    #
    # Solo se calculan si `edad` sigue siendo None (una edad exacta
    # autodeclarada es siempre más precisa y la sustituye por completo,
    # nunca conviven `edad` y `edad_rango_min`/`edad_rango_max`). NUNCA
    # los rellena este módulo (regex): aquí solo se detectan
    # autodeclaraciones literales.
    edad_rango_min: int | None = None
    edad_rango_max: int | None = None
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
    # Nivel de formación alcanzado (tramos INE/EPA -- CNED-2014): "superior"
    # (universidad, grado, licenciatura, master, doctorado, FP de grado
    # superior) | "secundaria_superior" (bachillerato, FP de grado medio) |
    # "secundaria_o_inferior" (ESO, primaria, sin estudios) | None. DISTINTO
    # de `estudios` (que es la CARRERA concreta -- "medicina", "derecho" --
    # solo para quien ya cursa/cursó estudios superiores). Si `estudios` ya
    # se detectó, `nivel_estudios` se infiere automáticamente como
    # "superior" sin necesidad de una frase-ancla propia (ver
    # _try_detect_nivel_estudios) -- nombrar una carrera concreta ya es
    # evidencia suficiente de haber cursado educación superior.
    nivel_estudios: str | None = None
    # Rama de conocimiento oficial (RD 1393/2007): "ciencias_sociales_juridicas"
    # | "ingenieria_arquitectura" | "ciencias_salud" | "artes_humanidades" |
    # "ciencias" | None. DISTINTO de `estudios` (la carrera concreta) y de
    # `nivel_estudios` (el nivel alcanzado) -- esto es la categoría AMPLIA a
    # la que pertenece la carrera, con vocabulario más amplio que las 14
    # carreras de STUDIES_DISTRIBUTION (decenas de carreras adicionales que
    # no tienen proporción propia pero sí rama reconocible). Si `estudios`
    # ya se detectó, se infiere automáticamente vía STUDIES_TO_RAMA -- pero
    # SOLO como dato informativo, nunca genera su propio paso de
    # estrechamiento en ese caso (ver _step_rama_estudios en
    # k_anonymity.py): la proporción de la rama ya está contenida en la de
    # la carrera concreta, aplicar ambas contaría el mismo hecho dos veces.
    rama_estudios: str | None = None
    ocupacion: str | None = None
    universidad: str | None = None
    empresa: str | None = None
    # Práctica deportiva autodeclarada (p. ej. "musculacion", "running",
    # ver claves de SPORT_PRACTICE_DISTRIBUTION en ine_reference.py). A
    # diferencia de la `aficion` que ya existe (per-foto, texto libre, la
    # detecta Moondream2 en imágenes, nunca narrowea población -- ver
    # `VisualDescriptionCodes`), este es un campo NUEVO, a nivel de cuenta,
    # detectado en TEXTO (regex aquí + IA en ai_attribute_extraction.py),
    # deliberadamente independiente y sin tocar el mecanismo de `aficion`
    # existente: son cosas distintas (una foto puede sugerir una afición
    # puntual sin que la cuenta entera declare practicar deporte de forma
    # regular, y viceversa). Requiere una autodeclaración de PRÁCTICA
    # (verbos como "juego al...", "hago...", "voy a...") -- una simple
    # mención del deporte (p. ej. "vi el partido de fútbol") NO cuenta,
    # para evitar falsos positivos con comentarios de espectador, mucho
    # más frecuentes que autodeclaraciones reales de práctica.
    practica_deportiva: str | None = None
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
    # que implica, p. ej. "aries (21 mar - 19 abr)". También puede
    # detectarse con una mención textual explícita del signo, no solo con el
    # emoji.
    signo_zodiacal: str | None = None
    # Creencia religiosa autodeclarada o indicada mediante emoji/símbolo
    # (✡️ judaísmo, ☪️ islam, ✝️/🕊️ cristianismo, etc.).
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
    # Confianza (0-1) de estimaciones que no son autodeclaraciones exactas
    # y necesitan comunicar su propia fiabilidad al informe -- de momento
    # solo la usan `edad_rango_min`/`edad_rango_max` (mismo valor de
    # confianza duplicado bajo ambas claves, ver `merge_findings` en
    # ai_attribute_extraction.py, para que el mecanismo genérico de
    # traspaso por nombre de campo funcione sin caso especial). Dict
    # genérico por campo (no un float suelto) para poder ampliarse a
    # otros campos probabilísticos en el futuro sin cambiar la forma de
    # la clase otra vez.
    confidence: dict[str, float] = field(default_factory=dict)


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

# Nivel de estudios (tramos INE/EPA, ver _try_detect_nivel_estudios para
# el porqué del orden y de exigir "superior"/"medio" explícito en la FP).
_NIVEL_ESTUDIOS_SUPERIOR_RE = re.compile(
    r"\b(soy universitari[oa]|tengo una carrera universitaria|"
    r"tengo un grado universitario|soy graduad[oa] en|"
    r"termine la carrera|acabe la carrera|termine la universidad|"
    r"tengo una licenciatura|soy licenciad[oa]|"
    r"tengo un master|hice un master|termine un master|"
    r"tengo un doctorado|soy doctorand[oa]|"
    r"tengo un ciclo formativo de grado superior|"
    r"tengo un grado superior de fp|soy tecnico superior|soy tecnica superior)\b",
    re.I,
)
_NIVEL_ESTUDIOS_SECUNDARIA_SUPERIOR_RE = re.compile(
    r"\b(tengo el bachillerato|termine bachillerato|termine el bachillerato|"
    r"tengo un ciclo formativo de grado medio|tengo un grado medio de fp|"
    r"soy tecnico de grado medio|soy tecnica de grado medio|termine la fp)\b",
    re.I,
)
_NIVEL_ESTUDIOS_SECUNDARIA_O_INFERIOR_RE = re.compile(
    r"\b(solo tengo la eso|tengo la eso|no termine el instituto|"
    r"solo estudios primarios|no tengo estudios|abandone los estudios|"
    r"no termine la eso|no termine secundaria)\b",
    re.I,
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
# comunes en bios de redes sociales, incluyendo variantes ortográficas y
# expresiones frecuentes en textos cortos.
_SEXUALITY_RE = re.compile(
    r"\b(?:soy\s+)?(?:"
    r"hetero(?:\s*[- ]?\s*sex(?:ual|uial|uel))?|"
    r"heterosexual|heterosexuial|heterosexuel|"
    r"gay|lesbiana|bisexual|pansexual|asexual|homosexual|"
    r"bi(?:\s*[- ]?\s*sexual|sexu(?:al|uial|uel))|"
    r"queer"
    r")\b",
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
_ZODIAC_TEXT_MAP: dict[str, str] = {name: f"{name} ({rango})" for name, rango in _ZODIAC_EMOJI_MAP.values()}
_ZODIAC_TEXT_RE = re.compile(
    r"\b(?:soy\s+)?(?:aries|tauro|geminis|cancer|leo|virgo|libra|escorpio|"
    r"sagitario|capricornio|acuario|piscis)\b",
    re.I,
)

_RELIGION_EMOJI_MAP: dict[str, str] = {
    "✡": "judaismo",
    "✡️": "judaismo",
    "🔯": "judaismo",
    "☪": "islam",
    "☪️": "islam",
    "✝": "cristianismo",
    "✝️": "cristianismo",
    "☦": "cristianismo",
    "☸": "budismo",
    "☸️": "budismo",
    "🕉": "hinduismo",
    "🕉️": "hinduismo",
    "ॐ": "hinduismo",
    "📿": "catolicismo",
    "🕊": "cristianismo",
    "🕊️": "cristianismo",
}
_RELIGION_TEXT_RE = re.compile(
    r"\b(?:soy\s+)?(?:jud[ií]o|jud[ií]a|judio|judia|musulm[aá]n|musulmana|"
    r"cat[oó]lico|cat[oó]lica|catolico|catolica|cristiano|cristiana|budista|"
    r"hinduista|ateo|atea|agn[oó]stico|agn[oó]stica|islam|juda[ií]smo)\b",
    re.I,
)
_RELIGION_TEXT_MAP = {
    "judio": "judaismo",
    "judia": "judaismo",
    "judío": "judaismo",
    "judía": "judaismo",
    "judaismo": "judaismo",
    "musulman": "islam",
    "musulmana": "islam",
    "islam": "islam",
    "catolico": "catolicismo",
    "catolica": "catolicismo",
    "cristiano": "cristianismo",
    "cristiana": "cristianismo",
    "budista": "budismo",
    "hinduista": "hinduismo",
    "ateo": "ateismo",
    "atea": "ateismo",
    "agnostico": "agnosticismo",
    "agnostica": "agnosticismo",
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
        _try_detect_nivel_estudios(text, post.permalink, findings)
        _try_detect_rama_estudios(text, post.permalink, findings)
        _try_detect_ocupacion(text, post.permalink, findings)
        _try_detect_practica_deportiva(text, post.permalink, findings)
        _try_detect_universidad(text, post.permalink, findings)
        _try_detect_empresa(text, post.permalink, findings)
        _try_detect_nacionalidad(text, post.permalink, findings)
        _try_detect_situacion_laboral(text, post.permalink, findings)
        _try_detect_lengua_materna(text, post.permalink, findings)
        _try_detect_orientacion_sexual(text, post.permalink, findings)
        _try_detect_signo_zodiacal(text, post.permalink, findings)
        _try_detect_religion(text, post.permalink, findings)

    _detect_household_type(posts, findings)
    _mark_all_detected_as_texto(findings)
    return findings


def _mark_all_detected_as_texto(findings: DemographicFindings) -> None:
    """Todo lo detectado por este módulo viene de texto autodeclarado (por
    definición: es lo único que procesa). Se marca explícitamente para que
    el frontend pueda distinguirlo de lo que venga de geolocation.py."""
    for attr_name in (
        "sexo", "edad", "provincia", "municipio", "comunidad_autonoma",
        "estudios", "nivel_estudios", "rama_estudios", "ocupacion", "universidad", "empresa",
        "nacionalidad", "situacion_laboral", "tipo_hogar", "lengua_materna",
        "orientacion_sexual", "signo_zodiacal", "religion",
        "practica_deportiva",
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


def _try_detect_nivel_estudios(text: str, permalink: str, findings: DemographicFindings) -> None:
    """Requiere una autodeclaración de haber CURSADO/COMPLETADO ese nivel
    (verbos como "tengo...", "termine...", "soy licenciado/a en...") --
    "estoy estudiando en la universidad" (en curso, no completado) NO
    cuenta, mismo criterio que practica_deportiva con espectador vs.
    práctica real: contar aspiraciones o estudios en curso como si ya
    estuvieran completados sobre-estimaría el nivel real de la persona.

    Orden de comprobación: de mayor a menor nivel. Los ciclos de
    Formación Profesional son el caso más delicado -- un Ciclo Formativo
    de Grado Superior (CFGS) cuenta como "superior" en la clasificación
    CNED-2014/ISCED que usa el INE (nivel 5, no 3-4), mientras que un
    Ciclo Formativo de Grado Medio (CFGM) y el Bachillerato caen en
    "secundaria_superior" -- de ahí que las frases-ancla exijan
    "superior"/"medio" explícito en vez de un "tengo un ciclo formativo"
    genérico sin cualificar, que sería ambiguo entre los dos niveles."""
    if findings.nivel_estudios is not None:
        return

    # Nombrar una carrera universitaria concreta (ver _try_detect_estudios
    # / STUDIES_DISTRIBUTION) ya implica nivel "superior" sin necesidad de
    # una frase-ancla propia de nivel_estudios.
    if findings.estudios is not None:
        findings.nivel_estudios = "superior"
        findings.evidence.setdefault("nivel_estudios", []).append(permalink)
        return

    if _NIVEL_ESTUDIOS_SUPERIOR_RE.search(text):
        findings.nivel_estudios = "superior"
    elif _NIVEL_ESTUDIOS_SECUNDARIA_SUPERIOR_RE.search(text):
        findings.nivel_estudios = "secundaria_superior"
    elif _NIVEL_ESTUDIOS_SECUNDARIA_O_INFERIOR_RE.search(text):
        findings.nivel_estudios = "secundaria_o_inferior"
    else:
        return

    findings.evidence.setdefault("nivel_estudios", []).append(permalink)


# Vocabulario AMPLIADO de rama de conocimiento (RD 1393/2007) -- carreras
# adicionales que no tienen proporción propia en STUDIES_DISTRIBUTION (14
# carreras concretas) pero sí una rama reconocible. Reutiliza el mismo
# `_STUDY_VERB_RE` que _try_detect_estudios (misma frase-ancla de
# práctica: "estudio...", "estudiante de...", "graduado en...") contra un
# candidato de texto más amplio, en vez de un regex nuevo.
#
# ORDEN: cuando un término es sustring de otro más largo ("quimica"
# dentro de "ingenieria quimica", "historia" dentro de "historia del
# arte"), el más específico va SIEMPRE primero -- mismo motivo que el
# orden de alternancia en _SPORT_PRACTICE_RE (ver ese comentario). Un
# diccionario normal de Python conserva el orden de inserción (3.7+), así
# que ese orden es el que determina qué entrada gana en `next()`.
_RAMA_ESTUDIOS_VOCABULARY: dict[str, str] = {
    # Ingeniería y Arquitectura -- variantes de "ingenieria X" antes de
    # cualquier término corto que pudiera ser sustring de una de ellas.
    "ingenieria de telecomunicaciones": "ingenieria_arquitectura",
    "ingenieria aeroespacial": "ingenieria_arquitectura",
    "ingenieria electronica": "ingenieria_arquitectura",
    "ingenieria mecanica": "ingenieria_arquitectura",
    "ingenieria quimica": "ingenieria_arquitectura",
    "ingenieria agronoma": "ingenieria_arquitectura",
    "ingenieria civil": "ingenieria_arquitectura",
    # Ciencias Sociales y Jurídicas.
    "comunicacion audiovisual": "ciencias_sociales_juridicas",
    "relaciones laborales": "ciencias_sociales_juridicas",
    "ciencias politicas": "ciencias_sociales_juridicas",
    "trabajo social": "ciencias_sociales_juridicas",
    "criminologia": "ciencias_sociales_juridicas",
    "sociologia": "ciencias_sociales_juridicas",
    "publicidad": "ciencias_sociales_juridicas",
    "turismo": "ciencias_sociales_juridicas",
    # Ciencias de la Salud.
    "terapia ocupacional": "ciencias_salud",
    "optica y optometria": "ciencias_salud",
    "odontologia": "ciencias_salud",
    "fisioterapia": "ciencias_salud",
    "logopedia": "ciencias_salud",
    "podologia": "ciencias_salud",
    "nutricion": "ciencias_salud",
    # Artes y Humanidades -- "historia del arte" antes que "historia" a
    # secas (sustring).
    "traduccion e interpretacion": "artes_humanidades",
    "historia del arte": "artes_humanidades",
    "bellas artes": "artes_humanidades",
    "humanidades": "artes_humanidades",
    "filologia": "artes_humanidades",
    "filosofia": "artes_humanidades",
    "historia": "artes_humanidades",
    # Ciencias -- "ingenieria quimica" ya se comprobó arriba, así que
    # "quimica" a secas aquí solo dispara si NO era esa combinación.
    "ciencias ambientales": "ciencias",
    "matematicas": "ciencias",
    "bioquimica": "ciencias",
    "geologia": "ciencias",
    "quimica": "ciencias",
    "fisica": "ciencias",
}


def _try_detect_rama_estudios(text: str, permalink: str, findings: DemographicFindings) -> None:
    """SOLO rellena el campo -- NUNCA aplica un paso de estrechamiento
    propio cuando se infiere desde `estudios` (ver comentario en
    STUDIES_TO_RAMA y en _step_rama_estudios): la proporción de la rama
    ya está contenida en la de la carrera concreta."""
    if findings.rama_estudios is not None:
        return

    if findings.estudios is not None:
        findings.rama_estudios = STUDIES_TO_RAMA[findings.estudios]
        findings.evidence.setdefault("rama_estudios", []).extend(findings.evidence.get("estudios", []))
        return

    match = _STUDY_VERB_RE.search(text)
    if not match:
        return

    candidate = _strip_accents(match.group(1).strip().lower())
    matched_rama = next((rama for keyword, rama in _RAMA_ESTUDIOS_VOCABULARY.items() if keyword in candidate), None)
    if matched_rama:
        findings.rama_estudios = matched_rama
        findings.evidence.setdefault("rama_estudios", []).append(permalink)


def _try_detect_ocupacion(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.ocupacion is not None:
        return

    lowered = _strip_accents(text.lower())
    matched = next((k for k in OCCUPATION_DISTRIBUTION if k in lowered), None)
    if matched:
        findings.ocupacion = matched
        findings.evidence.setdefault("ocupacion", []).append(permalink)


# Práctica deportiva: a diferencia de OCCUPATION_DISTRIBUTION (donde basta
# con que la palabra aparezca en el texto, ver _try_detect_ocupacion de
# arriba), aquí NO basta con que se mencione el deporte -- "vi el partido
# de futbol" es una mención de espectador, mucho más frecuente en redes
# sociales que una autodeclaración real de práctica, y un simple
# substring match la marcaría como falso positivo. Por eso cada modalidad
# tiene sus propias frases-ancla de PRÁCTICA (verbos como "juego al...",
# "hago...", "salgo a...", "voy a...") en vez de comparar contra las
# claves de SPORT_PRACTICE_DISTRIBUTION por subcadena. Un grupo con
# nombre por modalidad (en vez de una tabla de mapeo aparte) para que
# `match.lastgroup` sea directamente la clave de la modalidad detectada.
#
# ORDEN DE ALTERNANCIA: cuando la frase-ancla de una modalidad es
# literalmente un PREFIJO de la de otra ("tenis" dentro de "tenis de
# mesa", "futbol" dentro de "futbol sala", "esqui" dentro de "esqui
# nautico"), la más específica va SIEMPRE primero -- si no, "juego al
# tenis de mesa" haría match como "tenis" (el \b encaja justo antes del
# espacio, antes de llegar a probar "tenis_mesa"), sin llegar nunca a la
# alternativa correcta. Ver test_futbol_sala_is_not_confused_with_futbol
# para el caso que motivó esta regla.
_SPORT_PRACTICE_RE = re.compile(
    r"\b(?:"
    r"(?P<futbol_sala>juego (?:al |a )?futbol sala\b|juego (?:al |a )?futbito\b|"
    r"practico futbol sala\b|practico futbito\b|entreno (?:al |a )?futbol sala\b)|"
    r"(?P<futbol>juego (?:al |a )?futbol\b|soy futbolista\b|practico futbol\b|entreno (?:al |a )?futbol\b)|"
    r"(?P<running>hago running\b|soy runner\b|salgo a correr\b|"
    r"corro (?:todas las semanas|cada semana|a diario|con regularidad)\b|"
    r"practico running\b)|"
    r"(?P<atletismo>practico atletismo\b|hago atletismo\b|soy atleta\b|entreno atletismo\b)|"
    r"(?P<natacion>hago natacion\b|voy a nadar\b|nado en la piscina\b|"
    r"practico natacion\b|soy nadador\b|soy nadadora\b)|"
    r"(?P<senderismo>hago senderismo\b|voy de senderismo\b|practico montanismo\b|"
    r"practico senderismo\b|salgo a hacer rutas\b|hago rutas de montana\b)|"
    r"(?P<musculacion>voy al gimnasio\b|hago musculacion\b|levanto pesas\b|"
    r"hago pesas\b|entreno en el gimnasio\b|hago crossfit\b)|"
    r"(?P<ciclismo>hago ciclismo\b|salgo en bici\b|salgo con la bici\b|"
    r"monto en bici (?:todas las semanas|cada semana|con regularidad)\b|"
    r"practico ciclismo\b|soy ciclista\b)|"
    r"(?P<padel>juego (?:al |a )?padel\b|practico padel\b)|"
    r"(?P<tenis_mesa>juego (?:al |a )?tenis de mesa\b|practico tenis de mesa\b|"
    r"juego (?:al |a )?ping pong\b|practico ping pong\b)|"
    r"(?P<tenis>juego (?:al |a )?tenis\b|practico tenis\b)|"
    r"(?P<baloncesto>juego (?:al |a )?baloncesto\b|practico baloncesto\b|"
    r"soy jugador de baloncesto\b|soy jugadora de baloncesto\b)|"
    r"(?P<balonmano>juego (?:al |a )?balonmano\b|practico balonmano\b|entreno balonmano\b)|"
    r"(?P<voleibol>juego (?:al |a )?voleibol\b|practico voleibol\b|"
    r"juego (?:al |a )?voley\b|practico voley\b)|"
    r"(?P<rugby>juego (?:al |a )?rugby\b|practico rugby\b)|"
    r"(?P<pelota_vasca>juego (?:al |a )?fronton\b|practico fronton\b|"
    r"juego (?:al |a )?frontenis\b|practico frontenis\b|"
    r"juego (?:a la |a )?pelota vasca\b|practico pelota vasca\b)|"
    r"(?P<petanca>juego a la petanca\b|juego a petanca\b|practico petanca\b)|"
    r"(?P<patinaje>hago patinaje\b|practico patinaje\b|salgo a patinar\b|voy a patinar\b|"
    r"hago monopatin\b|practico monopatin\b)|"
    r"(?P<motociclismo>practico motociclismo\b|hago motocross\b|compito en motocross\b|"
    r"soy piloto de motociclismo\b)|"
    r"(?P<automovilismo>practico automovilismo\b|hago rallies\b|compito en rallies\b|"
    r"soy piloto de carreras\b)|"
    r"(?P<aeronautica>hago parapente\b|practico parapente\b|hago ala delta\b|"
    r"practico ala delta\b|hago paracaidismo\b|practico paracaidismo\b)|"
    r"(?P<squash>juego (?:al |a )?squash\b|practico squash\b)|"
    r"(?P<badminton>juego (?:al |a )?badminton\b|practico badminton\b)|"
    r"(?P<golf>juego (?:al |a )?golf\b|practico golf\b|soy golfista\b)|"
    r"(?P<surf>hago surf\b|practico surf\b|salgo a hacer surf\b|soy surfista\b)|"
    r"(?P<vela>practico vela\b|hago vela\b|navego en velero\b)|"
    r"(?P<esqui_nautico>hago esqui nautico\b|practico esqui nautico\b|"
    r"hago motonautica\b|practico motonautica\b)|"
    r"(?P<piraguismo_remo>hago piraguismo\b|practico piraguismo\b|hago remo\b|"
    r"practico remo\b|hago kayak\b|practico kayak\b)|"
    r"(?P<submarinismo>hago submarinismo\b|practico submarinismo\b|hago buceo\b|"
    r"practico buceo\b|voy a bucear\b|soy buceador\b|soy buceadora\b)|"
    r"(?P<esqui>hago esqui\b|practico esqui\b|voy a esquiar\b|"
    r"hago snowboard\b|practico snowboard\b)|"
    r"(?P<triatlon>hago triatlon\b|practico triatlon\b|compito en triatlon\b|soy triatleta\b)|"
    r"(?P<boxeo>hago boxeo\b|practico boxeo\b|entreno boxeo\b|"
    r"soy boxeador\b|soy boxeadora\b)|"
    r"(?P<artes_marciales>practico artes marciales\b|hago artes marciales\b|"
    r"hago karate\b|practico karate\b|hago judo\b|practico judo\b|"
    r"hago taekwondo\b|practico taekwondo\b|hago kung fu\b|practico kung fu\b)|"
    r"(?P<lucha_defensa_personal>practico defensa personal\b|hago defensa personal\b|"
    r"practico lucha libre\b|hago lucha libre\b|practico jiu jitsu\b|hago jiu jitsu\b|"
    r"practico bjj\b|hago bjj\b)|"
    r"(?P<caza>voy de caza\b|salgo de caza\b|practico caza\b|"
    r"soy cazador\b|soy cazadora\b)|"
    r"(?P<pesca>voy de pesca\b|salgo a pescar\b|practico pesca\b|"
    r"soy pescador\b|soy pescadora\b)|"
    r"(?P<hipica>practico hipica\b|hago hipica\b|monto a caballo\b|"
    r"practico equitacion\b|hago equitacion\b)|"
    r"(?P<ajedrez>juego (?:al |a )?ajedrez\b|practico ajedrez\b|compito en ajedrez\b)|"
    # yoga_pilates: aproxima la categoría "gimnasia suave" de la encuesta
    # (ver nota en ine_reference.py) -- yoga, pilates y tai-chi son las
    # formas más habituales en que la gente lo declara en primera
    # persona; "gimnasia de mantenimiento" sin más detalle no tiene una
    # frase-ancla propia porque es demasiado genérica para distinguirla
    # de una mención de espectador o de otra actividad.
    r"(?P<yoga_pilates>hago yoga\b|practico yoga\b|voy a clases de yoga\b|"
    r"hago pilates\b|practico pilates\b|voy a clases de pilates\b|"
    r"hago tai chi\b|practico tai chi\b)|"
    # baile_fitness: aproxima "Otra actividad física con música" de la
    # encuesta -- zumba es, con diferencia, la forma más habitual de
    # declarar esta categoría en primera persona.
    r"(?P<baile_fitness>hago zumba\b|voy a clases de zumba\b|"
    r"hago baile fitness\b|hago bailoterapia\b)|"
    # gimnasia_intensa: aproxima la categoría homónima de la encuesta
    # (aerobic/step/spinning) -- DISTINTA de "baile_fitness" de arriba
    # (que es la propia encuesta la que las separa en dos filas). "hago
    # crossfit" queda deliberadamente en el grupo de musculacion de
    # arriba, no aquí, para no detectar dos veces la misma frase.
    r"(?P<gimnasia_intensa>hago aerobic\b|hago step\b|"
    r"hago spinning\b|voy a spinning\b|"
    r"voy a clases dirigidas de gimnasia\b)"
    r")",
    re.I,
)


def _try_detect_practica_deportiva(text: str, permalink: str, findings: DemographicFindings) -> None:
    if findings.practica_deportiva is not None:
        return

    lowered = _strip_accents(text.lower())
    match = _SPORT_PRACTICE_RE.search(lowered)
    if not match:
        return

    findings.practica_deportiva = match.lastgroup
    findings.evidence.setdefault("practica_deportiva", []).append(permalink)



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
    """Detecta emojis o nombres de signos zodiacales en el texto y almacena
    el signo junto con su rango de fechas de nacimiento implícito."""
    if findings.signo_zodiacal is not None:
        return

    for emoji_char, (signo, rango) in _ZODIAC_EMOJI_MAP.items():
        if emoji_char in text:
            findings.signo_zodiacal = f"{signo} ({rango})"
            findings.evidence.setdefault("signo_zodiacal", []).append(permalink)
            return

    normalized_text = _strip_accents(text.lower())
    match = _ZODIAC_TEXT_RE.search(normalized_text)
    if not match:
        return

    raw_name = match.group(0).strip()
    sign = raw_name.split()[-1]
    value = _ZODIAC_TEXT_MAP.get(sign)
    if value is None:
        return
    findings.signo_zodiacal = value
    findings.evidence.setdefault("signo_zodiacal", []).append(permalink)


def _try_detect_religion(text: str, permalink: str, findings: DemographicFindings) -> None:
    """Detecta símbolos o menciones explícitas de religión en texto."""
    if findings.religion is not None:
        return

    for emoji_char, religion in _RELIGION_EMOJI_MAP.items():
        if emoji_char in text:
            findings.religion = religion
            findings.evidence.setdefault("religion", []).append(permalink)
            return

    normalized_text = _strip_accents(text.lower())
    match = _RELIGION_TEXT_RE.search(normalized_text)
    if not match:
        return

    raw = match.group(0).strip()
    religion = _RELIGION_TEXT_MAP.get(raw.split()[-1])
    if religion is None:
        return

    findings.religion = religion
    findings.evidence.setdefault("religion", []).append(permalink)
