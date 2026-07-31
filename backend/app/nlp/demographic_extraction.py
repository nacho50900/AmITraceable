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
from app.models.schemas import SocialPost


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
    # permalinks de los posts que dispararon cada detección, para trazabilidad
    evidence: dict[str, list[str]] = field(default_factory=dict)
    # procedencia de cada dato detectado: "texto" (autodeclaración escrita,
    # por defecto) o "imagen" (estimada vía app/vision/geolocation.py). Solo
    # se rellena explícitamente cuando algo viene de imagen; lo que viene de
    # este módulo es siempre "texto".
    source: dict[str, str] = field(default_factory=dict)


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

    _mark_all_detected_as_texto(findings)
    return findings


def _mark_all_detected_as_texto(findings: DemographicFindings) -> None:
    """Todo lo detectado por este módulo viene de texto autodeclarado (por
    definición: es lo único que procesa). Se marca explícitamente para que
    el frontend pueda distinguirlo de lo que venga de geolocation.py."""
    for attr_name in (
        "sexo", "edad", "provincia", "municipio", "comunidad_autonoma",
        "estudios", "ocupacion", "universidad", "empresa",
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
