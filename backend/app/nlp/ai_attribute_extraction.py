"""
Extracción de datos demográficos autodeclarados usando un LLM (Mistral AI),
como complemento de las regex de `demographic_extraction.py`.

Motivación: las regex cubren un vocabulario fijo ("estudio X", "estudiante
de X", "vivo en X"...). Cualquier redacción real que no encaje en esas
plantillas (p. ej. "voy a 2º de Enfermería", "curso el grado en
Enfermería", "trabajo de comercial en una aseguradora") se pierde en
silencio. Este módulo pide a un LLM que lea el texto igual que lo haría una
persona y detecte el mismo tipo de autodeclaraciones explícitas -- no
inferencias ni suposiciones -- devolviendo un JSON estructurado.

Decisión de diseño importante: el LLM NO decide a qué categoría del INE
pertenece un valor. Solo propone el texto literal más cercano (p. ej.
"enfermeria"); la normalización final contra las tablas de
`ine_reference.py` (mismo criterio de coincidencia por subcadena que ya
usan las regex, ver `_set_normalized`) la sigue haciendo este módulo, no el
LLM. Así el cálculo de k-anonimato sigue anclado a categorías auditables
del INE: si el LLM alucinase una categoría inventada, simplemente no
coincide con ninguna clave conocida y no se estima nada (igual que ya pasa
hoy con las regex cuando no hay coincidencia), en vez de colar un número de
población falso.

Cuándo se ejecuta: automáticamente dentro del pipeline principal
(`report/generator.py`), en cada análisis -- a diferencia de
`ai_analysis.py` (conclusiones priorizadas en lenguaje natural), que sigue
siendo un botón aparte que el usuario pulsa bajo demanda.

Nota RGPD (para la memoria): esto envía el TEXTO CRUDO de las
publicaciones a un proveedor externo (Mistral AI, UE) en cada análisis, no
solo agregados como hace `ai_analysis.py`. Es una excepción consciente al
principio de minimización que se sigue en el resto del proyecto,
justificada porque es indispensable para la propia función (detectar
autodeclaraciones en lenguaje natural libre) y porque el usuario ya ha
dado consentimiento OAuth explícito sobre su propio contenido.

También se envían la biografía y el nombre público de la cuenta (si la
plataforma los expone), por el mismo motivo de minimización justificada.
El nombre público NO es una autodeclaración -- es una convención cultural
del nombre (p. ej. "Ana" sugiere sexo femenino en español), así que es una
señal mucho más débil y con más falsos positivos (nombres unisex,
transliteraciones, apodos) que una frase explícita como "soy mujer". Por
eso se pide al modelo en un campo JSON APARTE ("sexo_por_nombre") y se
marca con una procedencia distinta ("ia_nombre") en vez de mezclarla sin
más con `sexo` -- ver `_to_findings` y `k_anonymity.py`, donde esa
procedencia añade una nota de menor fiabilidad al informe.

Degradación: si no hay `MISTRAL_API_KEY` configurada, o la llamada falla
por cualquier motivo (cuota agotada, timeout, respuesta con forma
inesperada...), se devuelven unos `DemographicFindings` vacíos y el
pipeline sigue con normalidad solo con lo que hayan encontrado las regex --
mismo principio de "módulo opcional que nunca rompe el resto" que
`vision/geolocation.py` y `ai_analysis.py`.

También se pide un campo aparte "fotos_de_viaje": publicaciones cuyo texto
indica que la persona está de viaje/vacaciones/de paso por un sitio. No es
una autodeclaración de atributo (no rellena provincia/municipio), sino una
lista de EXCLUSIÓN que usa report/generator.py para no confundir "dónde fue
tomada esta foto" con "dónde vive esta persona" al combinar geolocalización
de imagen con inferencia de residencia -- ver
app/nlp/travel_detection.py (su equivalente por regex, que actúa como red
de seguridad cuando no hay IA disponible) y `DemographicFindings.travel_permalinks`.
"""
import json
import logging

import httpx

from app.config import settings
from app.data.ine_reference import (
    AUTONOMOUS_COMMUNITY_PROVINCES,
    MUNICIPALITY_POPULATION,
    OCCUPATION_DISTRIBUTION,
    PROVINCE_POPULATION,
    STUDIES_DISTRIBUTION,
    resolve_autonomous_community,
)
from app.models.schemas import SocialPost
from app.nlp.demographic_extraction import DemographicFindings, _strip_accents

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Campos "simples" (valor libre, sin normalizar contra una tabla del INE).
_FREE_TEXT_FIELDS = ("universidad", "empresa")
# Todos los campos que puede rellenar este módulo, en el mismo orden que
# `DemographicFindings`, usado por `merge_findings`.
_ALL_FIELDS = (
    "sexo", "edad", "provincia", "municipio", "comunidad_autonoma",
    "estudios", "ocupacion", "universidad", "empresa",
)

_SYSTEM_PROMPT = (
    "Eres un extractor de datos. Se te da: (1) el nombre público y la biografía de una "
    "cuenta, y (2) una lista de sus publicaciones, cada una precedida por su "
    "identificador entre corchetes, p. ej. [abc123] texto de la publicación. En la "
    "biografía y las publicaciones, busca ÚNICAMENTE autodeclaraciones EXPLÍCITAS en "
    "primera persona sobre la propia persona -- nunca sobre otras personas mencionadas, y "
    "nunca inferencias o suposiciones tuyas. Responde EXCLUSIVAMENTE con un JSON con esta "
    "forma exacta, sin texto adicional ni backticks:\n"
    '{"sexo": "hombre"|"mujer"|null, "edad": <entero>|null, "provincia": <string>|null, '
    '"municipio": <string>|null, "comunidad_autonoma": <string>|null, "estudios": <string>|null, '
    '"ocupacion": <string>|null, "universidad": <string>|null, "empresa": <string>|null, '
    '"sexo_por_nombre": "hombre"|"mujer"|null, '
    '"fotos_de_viaje": [<identificador_de_publicacion>, ...], '
    '"evidence": {"<nombre_de_campo>": "<identificador_de_publicacion_o_bio>"}}\n'
    "Usa null si no hay una declaración explícita y clara para ese campo. No inventes "
    "datos que no estén literalmente en el texto. El campo 'evidence' debe indicar, para "
    "cada campo que no sea null, el identificador exacto (el que va entre corchetes, o la "
    "palabra 'bio' si viene de la biografía) que lo prueba. 'comunidad_autonoma' es DISTINTO "
    "de 'provincia': úsalo SOLO cuando la persona diga que vive en una comunidad autónoma "
    "española COMPLETA sin especificar la provincia concreta (p. ej. 'vivo en Canarias', 'soy "
    "de Andalucía', 'vivo en el País Vasco'); si además especifica la provincia o ciudad "
    "(p. ej. 'vivo en Las Palmas', 'vivo en Sevilla'), usa 'provincia' o 'municipio' en su "
    "lugar y deja 'comunidad_autonoma' en null. 'sexo_por_nombre' es distinto "
    "de 'sexo': aquí NO busques una autodeclaración, sino tu mejor estimación de qué sexo "
    "sugiere culturalmente el NOMBRE PÚBLICO de la cuenta en español (p. ej. 'Ana' -> "
    "'mujer'); usa null si el nombre es ambiguo, es un alias/apodo sin relación con un "
    "nombre real, o no se te ha proporcionado nombre. 'fotos_de_viaje' es una lista aparte, "
    "sin relación con las autodeclaraciones anteriores: incluye ahí el identificador de "
    "CUALQUIER publicación cuyo texto indique que la persona está de viaje, de vacaciones, "
    "de paso, o visitando temporalmente un sitio que NO es necesariamente donde vive (p. ej. "
    "'De viaje en Roma', 'Unos días en la playa', 'Visitando a mi prima en Londres'). El "
    "objetivo es señalar publicaciones que NO deben usarse para deducir dónde vive la "
    "persona habitualmente, aunque la foto en sí esté geolocalizada con confianza. Si un "
    "texto no da ninguna pista de viaje/vacaciones, NO lo incluyas en esa lista."
)


class AiExtractionUnavailable(Exception):
    """Excepción interna: se captura siempre dentro de este módulo, nunca se
    propaga al llamador -- el pipeline principal no debe romperse porque la
    IA falle o no esté configurada."""


def _posts_prompt(posts: list[SocialPost]) -> str:
    lines = []
    for post in posts:
        text = (post.text or "").strip()
        if not text:
            continue
        # Recorte defensivo por publicación: acota el tamaño del prompt
        # (y por tanto el coste/cuota) sin depender de que el texto de
        # origen ya venga acotado.
        snippet = text[:600]
        lines.append(f"[{post.permalink}] {snippet}")
    return "\n".join(lines)


def _profile_prompt(username: str, full_name: str | None, bio: str | None) -> str:
    lines = [f"Nombre de usuario/handle: {username}"]
    if full_name:
        lines.append(f"Nombre público mostrado por la cuenta: {full_name}")
    if bio:
        lines.append(f"Biografía: {bio.strip()[:600]}")
    return "\n".join(lines)


async def extract_demographics_with_ai(
    posts: list[SocialPost],
    username: str,
    full_name: str | None = None,
    bio: str | None = None,
) -> DemographicFindings:
    """Punto de entrada del módulo. Nunca lanza excepciones: cualquier fallo
    (sin API key, red, cuota, forma de respuesta inesperada) se traduce en
    unos `DemographicFindings` vacíos para que el pipeline siga solo con lo
    que hayan encontrado las regex.

    `bio` se pasa aquí también por separado (además de ya poder venir
    inyectada como pseudo-post desde report/generator.py para que las
    regex la analicen igual que un post) porque aquí sirve además de
    contexto para que el modelo entienda mejor el resto del perfil, no solo
    como una fuente más de autodeclaraciones."""
    if not settings.mistral_api_key:
        return DemographicFindings()

    posts_text = _posts_prompt(posts)
    profile_text = _profile_prompt(username, full_name, bio)
    if not posts_text and not profile_text:
        return DemographicFindings()

    prompt = f"{profile_text}\n\nPublicaciones:\n{posts_text}" if posts_text else profile_text

    try:
        parsed = await _call_mistral(prompt)
    except AiExtractionUnavailable as exc:
        logger.warning("Extracción de atributos con IA no disponible: %s", exc)
        return DemographicFindings()

    return _to_findings(parsed)


async def _call_mistral(prompt_text: str) -> dict:
    payload = {
        "model": settings.mistral_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(MISTRAL_API_URL, json=payload, headers=headers)
    except httpx.RequestError as exc:
        raise AiExtractionUnavailable(f"error de red: {exc}") from exc

    if response.status_code != 200:
        raise AiExtractionUnavailable(f"HTTP {response.status_code}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AiExtractionUnavailable(f"respuesta con forma inesperada: {exc}") from exc


def _set_evidence(findings: DemographicFindings, field: str, evidence_map: dict) -> None:
    permalink = evidence_map.get(field) if isinstance(evidence_map, dict) else None
    findings.evidence.setdefault(field, [])
    if isinstance(permalink, str) and permalink:
        findings.evidence[field].append(permalink)
    findings.source[field] = "ia"


def _set_normalized(
    findings: DemographicFindings, parsed: dict, field: str, distribution: dict, evidence_map: dict
) -> None:
    raw = parsed.get(field)
    if not isinstance(raw, str) or not raw.strip():
        return
    candidate = _strip_accents(raw.strip().lower())
    matched = next((k for k in distribution if k in candidate), None)
    if matched:
        setattr(findings, field, matched)
        _set_evidence(findings, field, evidence_map)


def _match_place(raw: object, distribution: dict) -> str | None:
    """Busca `raw` (texto libre, p.ej. "Vivo en Las Palmas") como subcadena
    de alguna clave de `distribution` (municipio o provincia). None si
    `raw` no es un string usable o no hay coincidencia."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = _strip_accents(raw.strip().lower())
    return next((k for k in distribution if k in candidate), None)


def _set_comunidad_autonoma(findings: DemographicFindings, ccaa_raw: object, evidence_map: dict) -> None:
    """Procesa el campo 'comunidad_autonoma' del modelo (ver _SYSTEM_PROMPT):
    solo se usa cuando el texto no dio ya municipio/provincia concreta. Si
    la comunidad tiene una sola provincia, no hay ambigüedad y se guarda
    como provincia (más específico); si tiene varias, se guarda a nivel de
    comunidad autónoma."""
    if not isinstance(ccaa_raw, str) or not ccaa_raw.strip():
        return

    ccaa = resolve_autonomous_community(ccaa_raw)
    if ccaa is None:
        return

    provinces = AUTONOMOUS_COMMUNITY_PROVINCES[ccaa]
    if len(provinces) == 1:
        findings.provincia = provinces[0]
        _set_evidence(findings, "provincia", evidence_map)
    else:
        findings.comunidad_autonoma = ccaa
        _set_evidence(findings, "comunidad_autonoma", evidence_map)


def _set_location(findings: DemographicFindings, parsed: dict, evidence_map: dict) -> None:
    # Municipio primero (más específico), igual que en demographic_extraction.py.
    municipio = _match_place(parsed.get("municipio"), MUNICIPALITY_POPULATION)
    if municipio:
        findings.municipio = municipio
        _set_evidence(findings, "municipio", evidence_map)
        return

    provincia = _match_place(parsed.get("provincia"), PROVINCE_POPULATION)
    if provincia:
        findings.provincia = provincia
        _set_evidence(findings, "provincia", evidence_map)
        return

    # Ninguna coincidencia de municipio/provincia: puede que el modelo haya
    # devuelto una comunidad autónoma COMPLETA en su propio campo.
    _set_comunidad_autonoma(findings, parsed.get("comunidad_autonoma"), evidence_map)


def _to_findings(parsed: dict) -> DemographicFindings:
    if not isinstance(parsed, dict):
        return DemographicFindings()

    findings = DemographicFindings()
    evidence_map = parsed.get("evidence") or {}

    sexo = parsed.get("sexo")
    if sexo in ("hombre", "mujer"):
        findings.sexo = sexo
        _set_evidence(findings, "sexo", evidence_map)
    else:
        # Solo se usa la estimación por nombre si no hay autodeclaración
        # explícita -- es una señal más débil (ver docstring del módulo) y
        # nunca debe pisar una frase literal tipo "soy mujer".
        sexo_por_nombre = parsed.get("sexo_por_nombre")
        if sexo_por_nombre in ("hombre", "mujer"):
            findings.sexo = sexo_por_nombre
            findings.evidence.setdefault("sexo", []).append("nombre público de la cuenta")
            findings.source["sexo"] = "ia_nombre"

    edad = parsed.get("edad")
    if isinstance(edad, int) and not isinstance(edad, bool) and 12 <= edad <= 100:
        findings.edad = edad
        _set_evidence(findings, "edad", evidence_map)

    _set_normalized(findings, parsed, "estudios", STUDIES_DISTRIBUTION, evidence_map)
    _set_normalized(findings, parsed, "ocupacion", OCCUPATION_DISTRIBUTION, evidence_map)
    _set_location(findings, parsed, evidence_map)

    for field in _FREE_TEXT_FIELDS:
        value = parsed.get(field)
        if isinstance(value, str) and value.strip():
            setattr(findings, field, value.strip())
            _set_evidence(findings, field, evidence_map)

    fotos_de_viaje = parsed.get("fotos_de_viaje")
    if isinstance(fotos_de_viaje, list):
        findings.travel_permalinks = {p for p in fotos_de_viaje if isinstance(p, str) and p.strip()}

    return findings


def merge_findings(regex_findings: DemographicFindings, ai_findings: DemographicFindings) -> DemographicFindings:
    """Combina lo detectado por regex (determinista y gratuito) con lo
    detectado por IA. Las regex tienen prioridad: la IA solo rellena los
    campos que las regex NO encontraron -- nunca sobrescribe una
    autodeclaración ya confirmada por coincidencia exacta de patrón."""
    for field in _ALL_FIELDS:
        if getattr(regex_findings, field) is None and getattr(ai_findings, field) is not None:
            setattr(regex_findings, field, getattr(ai_findings, field))
            if field in ai_findings.evidence:
                regex_findings.evidence[field] = ai_findings.evidence[field]
            # Se preserva la procedencia tal cual la puso _to_findings
            # ("ia" para autodeclaraciones en texto, "ia_nombre" para la
            # estimación por nombre público) -- nunca se aplana a "ia" a
            # secas, o se perdería la distinción de fiabilidad.
            regex_findings.source[field] = ai_findings.source.get(field, "ia")
    return regex_findings
