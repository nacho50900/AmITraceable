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
    MUNICIPALITY_POPULATION,
    OCCUPATION_DISTRIBUTION,
    PROVINCE_POPULATION,
    STUDIES_DISTRIBUTION,
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
    estudios: str | None = None
    ocupacion: str | None = None
    universidad: str | None = None
    empresa: str | None = None
    # permalinks de los posts que dispararon cada detección, para trazabilidad
    evidence: dict[str, list[str]] = field(default_factory=dict)


_AGE_RE = re.compile(r"\b(?:tengo|con)\s+(\d{1,2})\s+años\b|\b(\d{1,2})\s+años\b", re.I)
_SEX_MALE_RE = re.compile(r"\b(soy un chico|soy un chaval|soy hombre)\b", re.I)
_SEX_FEMALE_RE = re.compile(r"\b(soy una chica|soy mujer)\b", re.I)
_UNIVERSITY_RE = re.compile(r"\buniversidad de (\w+)", re.I)
_COMPANY_RE = re.compile(r"\btrabajo (?:en|para)\s+([A-Z][\wÁÉÍÓÚáéíóú]+)")
_STUDY_VERB_RE = re.compile(r"\b(?:estudio|estudiante de|graduad[oa] en)\s+([a-záéíóúñ ]+)", re.I)


def extract_demographics(posts: list[SocialPost]) -> DemographicFindings:
    findings = DemographicFindings()

    for post in posts:
        text = post.text or ""
        if not text:
            continue

        if findings.edad is None:
            m = _AGE_RE.search(text)
            if m:
                age_str = m.group(1) or m.group(2)
                age = int(age_str)
                if 12 <= age <= 100:  # descarta falsos positivos ("100 años de historia")
                    findings.edad = age
                    findings.evidence.setdefault("edad", []).append(post.permalink)

        if findings.sexo is None:
            if _SEX_MALE_RE.search(text):
                findings.sexo = "hombre"
                findings.evidence.setdefault("sexo", []).append(post.permalink)
            elif _SEX_FEMALE_RE.search(text):
                findings.sexo = "mujer"
                findings.evidence.setdefault("sexo", []).append(post.permalink)

        if findings.provincia is None or findings.municipio is None:
            _match_location(text, post.permalink, findings)

        if findings.estudios is None:
            m = _STUDY_VERB_RE.search(text)
            if m:
                candidate = _strip_accents(m.group(1).strip().lower())
                matched = next((k for k in STUDIES_DISTRIBUTION if k in candidate), None)
                if matched:
                    findings.estudios = matched
                    findings.evidence.setdefault("estudios", []).append(post.permalink)

        if findings.ocupacion is None:
            lowered = _strip_accents(text.lower())
            matched = next((k for k in OCCUPATION_DISTRIBUTION if k in lowered), None)
            if matched:
                findings.ocupacion = matched
                findings.evidence.setdefault("ocupacion", []).append(post.permalink)

        if findings.universidad is None:
            m = _UNIVERSITY_RE.search(text)
            if m:
                findings.universidad = m.group(1)
                findings.evidence.setdefault("universidad", []).append(post.permalink)

        if findings.empresa is None:
            m = _COMPANY_RE.search(text)
            if m:
                findings.empresa = m.group(1)
                findings.evidence.setdefault("empresa", []).append(post.permalink)

    return findings


def _match_location(text: str, permalink: str, findings: DemographicFindings) -> None:
    lowered = _strip_accents(text.lower())

    # Municipio primero (más específico); si hay match, no hace falta
    # comprobar provincia por separado para ese mismo texto.
    m = re.search(r"\bvivo en ([a-z ]+)", lowered)
    candidate = m.group(1).strip() if m else None

    if candidate:
        muni_match = next((k for k in MUNICIPALITY_POPULATION if k in candidate), None)
        if muni_match and findings.municipio is None:
            findings.municipio = muni_match
            findings.evidence.setdefault("municipio", []).append(permalink)
            return

        prov_match = next((k for k in PROVINCE_POPULATION if k in candidate), None)
        if prov_match and findings.provincia is None:
            findings.provincia = prov_match
            findings.evidence.setdefault("provincia", []).append(permalink)
