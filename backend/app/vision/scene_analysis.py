"""
Análisis del CONTENIDO de cada foto (qué se ve: objetos, actividades,
aficiones, señales de relación de pareja) vía un modelo de
visión-lenguaje LOCAL (Moondream2, cargado con `transformers` igual que
DINOv2 en geolocation.py) -- complementario y arquitectónicamente distinto
de ese módulo, que solo compara SIMILITUD VISUAL contra un índice para
estimar dónde se tomó la foto, sin "entender" qué hay en ella.

Por qué local y no una API externa (p. ej. Mistral Pixtral, usado en una
versión anterior de este módulo): el tier gratuito de la API de Mistral
limita a 2 peticiones/minuto -- con una foto por publicación (y varias por
carrusel, ver InstagramClient), un perfil normal agota ese límite en
segundos, y la inmensa mayoría de fotos se quedarían sin analizar
(silenciosamente, por diseño best-effort). Un modelo local no tiene ese
límite -- solo el de tu propia CPU/GPU -- y además las fotos nunca salen
del servidor, mejor alineado con el diseño RGPD del resto del proyecto
(procesamiento en memoria, sin persistencia, sin terceros).

Modelo elegido: Moondream2 (~1.8B parámetros, `vikhyatk/moondream2` en
HuggingFace), diseñado específicamente para responder preguntas sobre
imágenes (VQA) de forma eficiente incluso en CPU, sin necesitar GPU
dedicada -- mismo perfil de despliegue que `facebook/dinov2-small`. Usa
`trust_remote_code=True` (ejecuta código Python del propio repo del
modelo, no solo pesos): es la forma estándar de usar Moondream con
`transformers`, pero es una superficie de confianza distinta a cargar un
modelo de arquitectura estándar como DINOv2 -- documentado aquí para que
quede explícito en la memoria, no por ser inseguro en la práctica (repo
oficial, ampliamente usado).

Trade-off aceptado: Moondream2 reconoce peor OBJETOS muy concretos que un
modelo grande como Pixtral (p. ej. puede no identificar que un vinilo es
de un artista concreto si la carátula no es muy reconocible), pero es
suficiente para escenas/actividades genéricas (deporte, instrumentos
musicales, mascotas, contexto romántico...) -- razonable para un indicio
de baja confianza, que es lo que se pide aquí.

Se le hace UNA sola pregunta combinada por foto (no una petición de JSON
estructurado con varias inferencias, como en la versión con Pixtral):
modelos locales pequeños son mucho menos fiables generando JSON complejo
bien formado que uno grande, así que se pide una respuesta en tres líneas
con un formato fijo y se parsea como texto -- más robusto ante
variaciones de formato que decodificar un JSON que podría no serlo.

LÍMITE ÉTICO/LEGAL DELIBERADO (léase antes de tocar el prompt): el modelo
NUNCA debe intentar identificar, nombrar, o describir físicamente a
NINGUNA persona que aparezca en la foto -- ni su sexo, ni edad, ni nada.
El consentimiento OAuth de este proyecto cubre únicamente a la cuenta
analizada; generar cualquier dato sobre la identidad de un tercero que
aparezca en su contenido público sería tratar datos personales de alguien
que nunca dio su consentimiento, fuera del alcance legal de esta
herramienta.

Matiz importante, detectado en revisión: el modelo NO tiene forma de saber
cuál de las personas que aparecen en una foto es la propia cuenta
analizada -- es una imagen suelta, sin ninguna referencia externa con la
que comparar. "Persona principal" es una simplificación razonable para
fotos con una sola persona (o ninguna), pero en una foto con VARIAS
personas de protagonismo similar (p. ej. una pareja) es una ambigüedad
real, no solo una cuestión de cómo esté redactado el prompt. Por eso el
prompt pide explícitamente que el modelo declare cuántas personas
identifica como protagonistas (campo PERSONAS): con una o ninguna, la
señal de afición/actividad se atribuye a la cuenta analizada con
normalidad (mismo supuesto que ya asume el resto del proyecto: que una
autodeclaración en primera persona es sobre quien la escribe). Con varias
personas, esa atribución dejaría de tener fundamento -- podría ser
perfectamente la otra persona quien toca la guitarra de la foto, no la
cuenta analizada -- así que la señal de afición se DESCARTA en ese caso
(ver `_parse_inferences`). La señal de PAREJA, en cambio, no necesita
resolver esta ambigüedad: da igual cuál de las dos personas sea la cuenta
analizada, el mero hecho de que la cuenta publique una foto con contexto
romántico ya es la señal, así que esa sí se mantiene con varias personas.

Degradación: si `torch`/`transformers` no están instalados (dependencias
opcionales, ver WITH_GEOLOCATION en el Dockerfile) o la inferencia falla
por cualquier motivo, esta foto simplemente no aporta nada -- nunca aborta
el análisis del resto de fotos ni del resto del pipeline (best-effort, ver
`analyze_image_content`, que nunca lanza).
"""
import re

from app.models.schemas import InferredAttribute

_MODEL_NAME = "vikhyatk/moondream2"
_MODEL_REVISION = "2025-06-21"  # fijado explícitamente, ver docstring del modelo en HuggingFace

_model = None

_QUERY = (
    "Analiza esta imagen. Responde EXACTAMENTE en este formato, tres líneas, sin nada más:\n"
    "PERSONAS: <'ninguna' si no aparece ninguna persona, 'una' si aparece exactamente una "
    "persona protagonista, o 'varias' si aparecen dos o más personas de protagonismo similar "
    "(p. ej. una pareja, un grupo)>\n"
    "AFICION: <una afición, actividad, deporte, instrumento musical o fandom concreto que "
    "sugiera la imagen, en pocas palabras, o la palabra 'ninguno' si no hay nada específico>\n"
    "PAREJA: <'si' si la imagen muestra a dos personas besándose, en un abrazo claramente "
    "romántico, o cogidas de la mano en un contexto de pareja, o 'no' en cualquier otro caso>\n"
    "No describas ni identifiques físicamente a ninguna persona que aparezca en la imagen -- "
    "ni su aspecto, ni su sexo, ni su edad -- más allá de contarlas y de si hay o no un "
    "contexto romántico entre ellas."
)

_PERSONAS_RE = re.compile(r"PERSONAS:[ \t]*(\S+)", re.IGNORECASE)
_AFICION_RE = re.compile(r"AFICION:[ \t]*(.+)", re.IGNORECASE)
_PAREJA_RE = re.compile(r"PAREJA:[ \t]*(\S+)", re.IGNORECASE)


def _scene_analysis_available() -> bool:
    """Comprobación barata (sin cargar el modelo) de si este módulo puede
    funcionar: dependencias opcionales instaladas. No hay ningún índice ni
    fichero que comprobar (a diferencia de geolocation.py), el modelo se
    descarga solo la primera vez vía el caché de HuggingFace."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _lazy_load():
    global _model
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        revision=_MODEL_REVISION,
        trust_remote_code=True,
        device_map=device,
    )


def analyze_image_content(image) -> tuple[list[InferredAttribute], bool]:
    """Devuelve (inferencias_visuales, indicio_pareja) para UNA foto ya
    decodificada (PIL.Image, la misma que usa geolocation.py para el
    embedding de DINOv2 -- no se descarga ni decodifica de nuevo). SÍNCRONA
    y con trabajo de CPU real (como `estimate_location_from_image`): quien
    llama debe envolverla en `asyncio.to_thread` para no bloquear el event
    loop, ver geolocation.py.

    `evidence` de cada InferredAttribute se deja vacío deliberadamente --
    quien llama (geolocation.py) rellena el permalink de la publicación,
    que esta función no conoce.

    Nunca lanza: cualquier fallo (dependencias no instaladas, modelo no
    descargable, respuesta con formato inesperado) se trata como "esta
    foto no aportó nada" y devuelve ([], False), sin abortar el análisis
    de las demás fotos."""
    if not _scene_analysis_available():
        return [], False

    try:
        _lazy_load()
        answer = _model.query(image, _QUERY)["answer"]
    except Exception:
        return [], False

    return _parse_inferences(answer), _parse_pareja(answer)


def _parse_personas(answer: str) -> str | None:
    match = _PERSONAS_RE.search(answer)
    if match is None:
        return None
    value = match.group(1).strip().lower().rstrip(".,;")
    return value if value in ("ninguna", "una", "varias") else None


def _parse_inferences(answer: str) -> list[InferredAttribute]:
    # Con varias personas de protagonismo similar en la foto, no hay forma
    # de saber si la afición/actividad detectada es de la cuenta analizada
    # o de la otra persona -- ver docstring del módulo. Se descarta la
    # señal en vez de arriesgarse a atribuirla a quien no toca. Si
    # PERSONAS no se pudo parsear (formato inesperado), se prefiere
    # también descartar por precaución antes que asumir que es seguro
    # atribuirla.
    if _parse_personas(answer) not in ("ninguna", "una"):
        return []

    match = _AFICION_RE.search(answer)
    if match is None:
        return []
    valor = match.group(1).strip().rstrip(".")
    if not valor or valor.lower() in ("ninguno", "ninguna", "none", "n/a"):
        return []
    return [
        InferredAttribute(
            category="aficion",
            value=f"Posible afición/interés detectado en una foto: {valor}",
            confidence=0.5,
            evidence=[],
        )
    ]


def _parse_pareja(answer: str) -> bool:
    # A diferencia de _parse_inferences, aquí SÍ es válida la señal con
    # "varias" personas (de hecho es el caso típico: hacen falta al menos
    # dos para un contexto romántico) -- no depende de resolver quién es
    # la cuenta analizada, ver docstring del módulo.
    match = _PAREJA_RE.search(answer)
    if match is None:
        return False
    return match.group(1).strip().lower().rstrip(".,;") in ("si", "sí", "yes", "true")
