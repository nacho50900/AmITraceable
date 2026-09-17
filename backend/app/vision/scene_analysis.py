"""
Análisis del CONTENIDO de cada foto (qué se ve: objetos, actividades,
aficiones, señales de relación de pareja) vía un modelo de
visión-lenguaje LOCAL (Moondream2) -- complementario y arquitectónicamente
distinto de geolocation.py (DINOv2, cargado con `transformers`), que solo
compara SIMILITUD VISUAL contra un índice para estimar dónde se tomó la
foto, sin "entender" qué hay en ella.

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
Hugging Face), diseñado específicamente para responder preguntas sobre
imágenes (VQA) de forma eficiente incluso en CPU, sin necesitar GPU
dedicada -- mismo perfil de despliegue que `facebook/dinov2-small`.

Backend de carga: `llama-cpp-python` sobre un GGUF
(`ggml-org/moondream2-20250414-GGUF`, ver `_GGUF_REPO_ID` para el porqué
y el historial completo de qué se probó antes de llegar aquí) -- NO
`transformers`. Se abandonó ese camino (y el de `torchao`) por
incompatibilidades de kernels/versión específicas de la GPU de despliegue
real de este proyecto (GTX 1650, Turing, sin tensor cores); ver ese mismo
historial para los detalles.

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

Degradación: si `llama_cpp` no está instalado (dependencia
opcional, ver WITH_GEOLOCATION en el Dockerfile) o la inferencia falla
por cualquier motivo, esta foto simplemente no aporta nada -- nunca aborta
el análisis del resto de fotos ni del resto del pipeline (best-effort, ver
`analyze_image_content`, que nunca lanza).
"""
import base64
import io
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import settings
from app.data.ine_reference import PLATE_PROVINCE_CODE_TO_PROVINCE
from app.log import visual_description_log
from app.models.schemas import InferredAttribute

logger = logging.getLogger(__name__)

_GGUF_REPO_ID = "ggml-org/moondream2-20250414-GGUF"
_GGUF_TEXT_MODEL_FILENAME = "*text-model*"  # match único en el repo: moondream2-text-model-f16_ct-vicuna.gguf
# Nombre EXACTO (no el glob de arriba): `huggingface_hub.hf_hub_download()`
# (usado por `_ensure_quantized_model()` para descargar el F16 y
# cuantizarlo) no admite comodines como sí hace `Llama.from_pretrained()`
# -- necesita el nombre de fichero literal.
_GGUF_TEXT_MODEL_FILENAME_EXACT = "moondream2-text-model-f16_ct-vicuna.gguf"

# Tipo de cuantización por defecto para _ensure_quantized_model() --
# sobreescribible con la variable de entorno MOONDREAM_QUANT_TYPE (ver
# esa función) sin tocar código ni reconstruir la imagen, p. ej. para
# probar Q4_K_M en vez de Q8_0. Q8_0 confirmado en producción (17/9):
# ~1.7x más rápido que F16, en línea con benchmarks públicos de
# llama.cpp -- buen punto de partida por defecto.
_DEFAULT_QUANT_TYPE = "Q8_0"
_GGUF_MMPROJ_FILENAME = "*mmproj*"  # match único en el repo: moondream2-mmproj-f16-20250414.gguf
# HISTORIAL DE ESTE CAMBIO (11-13/9, ver conversación con Claude -- se deja
# aquí porque cada intento anterior parecía razonable a priori y solo se
# descartó con evidencia real, no vale la pena repetirlos sin releer esto):
#
# 1. `vikhyatk/moondream2` bf16 vía `transformers` (revisión pinneada
#    2025-06-21): funcionaba, pero 27s/foto en esta GTX 1650 (Turing, SIN
#    tensor cores) -- todo el parcheo de dtype que había más abajo en este
#    fichero (`_upcast_bfloat16_tensors`, `_patch_vision_input_dtype`, ya
#    ELIMINADOS en este cambio) existía solo para ese modelo/backend.
# 2. `moondream/moondream-2b-2025-04-14-4bit` (int4 QAT oficial, vía
#    `torchao`): bloqueado en cadena -- primero `torchao` sin techo de
#    versión rompía el import (`int4_weight_only` eliminado en 0.16),
#    luego con el techo puesto cargaba pero SIN kernels CUDA compilados
#    (`torch 2.14.0+cu130` es más nuevo que cualquier build de torchao,
#    ni siquiera nightly), y aun así fallaba con
#    `AttributeError: 'QuantizedLinear' object has no attribute 'weight'`.
# 3. Runtime propio "Photon" de Moondream (`pip install moondream`):
#    descartado sin llegar a probarlo -- su documentación exige GPU Ampere
#    o más nueva, esta GTX 1650 es Turing.
# 4. Export ONNX comunitario (`Xenova/moondream2` / `onnx-community`):
#    descartado -- checkpoint de 2024 (más de un año más antiguo que el
#    que se usaba), etiquetado incluso como arquitectura `moondream1`, Y
#    la integración de Transformers.js con la que se probó la calidad
#    fallaba con "Number of tokens and features do not match" en la
#    versión instalada -- documentación desactualizada.
#
# ESTA es la quinta vía: GGUF oficial de `ggml-org` (el propio equipo de
# `llama.cpp`) del checkpoint `2025-04-14` de `vikhyatk/moondream2` --
# cercano al `2025-06-21` de antes. Vía `llama-cpp-python`, con un
# `MoondreamChatHandler` DEDICADO (no el handler MTMD genérico).
#
# PROBLEMAS REALES YA ENCONTRADOS Y ARREGLADOS EN PRODUCCIÓN, GTX 1650
# (no hipotéticos -- si algo de esto reaparece, empezar por aquí):
# - Wheel PRECOMPILADO con CUDA (índice cu130 de abetlen/llama-cpp-python):
#   cargaba pero moría con SIGILL (exit 132) en cuanto tocaba GPU --
#   confirmado que era específico de CUDA (en CPU pura cargaba bien). Se
#   compila ahora desde fuente en el Dockerfile con
#   CMAKE_CUDA_ARCHITECTURES=75 (compute capability real de esta tarjeta).
# - `_model.reset()` ANTES de cada `create_chat_completion()`: sin esto,
#   la caché KV interna no se limpiaba sola entre las dos llamadas
#   independientes de esta función -- "the tokens of sequence 0... have
#   inconsistent sequence positions", y el estado quedaba tan corrupto
#   que el proceso acababa muriendo con SIGSEGV (exit 139) unas fotos
#   después, no solo fallando esa foto.
# - `_model_lock` (threading.Lock, ver más abajo): el pipeline procesa
#   varias fotos a la vez (`asyncio.Semaphore` en geolocation.py, cada
#   una en su propio hilo real vía `asyncio.to_thread`) -- varios hilos
#   llamando a la vez sobre el MISMO objeto `Llama` (incluso con
#   `reset()`) volvía a corromper el estado y a matar el proceso con
#   SIGSEGV. `transformers` toleraba esto sin problema visible;
#   `llama.cpp` no.
# - `n_ctx=2048` (no 4096): el modelo se entrenó con contexto 2048,
#   pedir más generaba "possible training context overflow" en el log.
#
# SIGUE SIN VERIFICAR: si con todo lo anterior arreglado el tiempo real
# por foto mejora frente a los 27s de bf16 -- la primera medición en
# producción tras arreglar los crashes dio ~25.7s, es decir, SIN mejora
# clara todavía. Y el GGUF de `ggml-org` avisa en el log de carga
# ("GENERATION QUALITY WILL BE DEGRADED! CONSIDER REGENERATING THE
# MODEL", pre-tokenizador sin declarar) -- impacto real en la calidad de
# las respuestas (incluido el parseo de PERSONAS/AFICION/PAREJA/
# TEXTO_VISIBLE/MATRICULA) todavía sin confirmar contra el baseline bf16.


@dataclass(frozen=True)
class VisualDescriptionCodes:
    """Señales de este módulo ya parseadas pero SIN formatear a texto en
    español -- a diferencia de `descripcion_cruda` (la frase ya redactada
    que se sigue devolviendo tal cual, sin tocar, para no romper nada de
    lo que ya consume: la vista de detalle del frontend en español y el
    contexto que recibe Mistral en app/ai_analysis.py), esto es para
    internacionalización (ver ADR-30): el frontend traduce `personas`
    (vocabulario cerrado: "una"/"varias", nunca "ninguna" -- mismo filtro
    que ya aplica `_build_clean_summary`) él mismo, sin llamar al backend,
    y solo pide traducción real a `/analyze/translate-descriptions` para
    `aficion` (vocabulario semi-libre) cuando la UI está en un idioma
    distinto del español.

    `texto_visible` NUNCA se traduce (es texto literal leído de la foto --
    un cartel, una matrícula -- traducirlo falsearía la evidencia; el
    frontend lo muestra tal cual venga, en cualquier idioma de UI).

    `matricula` (añadido después, separado de `texto_visible` -- ver
    comentario junto al campo MATRICULA en `_STRUCTURED_QUERY`) TAMPOCO
    se traduce nunca por el mismo motivo, y además solo llega hasta aquí
    si ya pasó la validación de formato de `_parse_matricula` (formato
    español válido, antiguo o actual) -- si Moondream2 devuelve algo que
    no tiene forma de matrícula real, se descarta como probable error de
    lectura en vez de mostrarse.

    `indicio_pareja` es un booleano (vocabulario cerrado, como `personas`)
    -- se repite aquí aunque `analyze_image_content()` ya lo devuelve por
    separado como su propio valor de retorno (2º de la tupla, usado para
    `partner_signal_permalinks` a nivel de PUBLICACIÓN en geolocation.py),
    porque ese otro valor no llega intacto hasta el frontend por FOTO --
    aquí sí, con la misma clave (`photo_link`) que el resto de estas
    señales."""

    personas: str | None
    aficion: str | None
    texto_visible: str | None
    matricula: str | None
    indicio_pareja: bool


_model = None

# Lock REAL de threading (no asyncio.Lock) que serializa TODO acceso a
# `_model`. Bug real confirmado en producción (13/9, GTX 1650): el
# pipeline de geolocation.py procesa varias fotos a la vez
# (`asyncio.Semaphore(actual_concurrency)`, puede ser > 1 -- ver
# `Settings.photo_analysis_concurrency`), cada una en su propio hilo real
# vía `asyncio.to_thread(analyze_image_content, image)`. Sin este lock,
# varios hilos podían llamar a `_model.reset()` / `_model.create_chat_completion()`
# A LA VEZ sobre el MISMO objeto `Llama` -- que no está pensado para eso
# (a diferencia de cómo se comportaba el modelo de `transformers` de
# antes, que toleraba esto sin problema visible). El resultado no era un
# error limpio: era corrupción de estado que acababa tirando abajo el
# proceso ENTERO con SIGSEGV (exit 139) tras varias fotos, no solo
# fallando la foto en cuestión.
_model_lock = threading.Lock()

# Dispositivo en el que se pidió offload de capas a `_lazy_load()`
# ("cuda" o "cpu") -- a diferencia de la versión `transformers` de antes,
# `llama.cpp` no expone un `next(_model.parameters()).device` equivalente
# (el modelo no es un `nn.Module` de PyTorch), así que esto ya NO es un
# valor "confirmado" leído tras la carga, es el valor SOLICITADO vía
# `n_gpu_layers` -- ver `get_device()` para el matiz importante de que
# esto puede no reflejar si el offload a GPU realmente funcionó.
_actual_device: str | None = None

# Nombre del modelo TAL CUAL quedó cargado (copia de `_GGUF_REPO_ID` en el
# momento de `_lazy_load()`, no el valor actual del módulo) -- ver
# `get_model_variant()` sobre por qué es una copia y no una relectura.
_loaded_model_name: str | None = None

_CAPTION_QUERY = (
    # NOTA (ver registro de trabajo): la primera versión de este campo
    # (DESCRIPCION) vivía DENTRO del mismo prompt combinado de cuatro
    # líneas que PERSONAS/AFICION/PAREJA, con una línea de ejemplo tipo
    # "DESCRIPCION: varias personas charlando alrededor de una mesa" para
    # que el modelo "copiara la forma". Con Moondream2, eso resultó en el
    # MISMO fallo que ya habíamos visto con los placeholders <...>: el
    # modelo devolvía literalmente mi frase de ejemplo (o una muy
    # parecida) en vez de describir la imagen real, y en un caso incluso
    # la repitió una segunda vez tras terminar las otras tres líneas. Se
    # sacó a su propia pregunta de texto libre, sin plantilla que copiar
    # -- pero en producción (GTX 1650) seguía saliendo mal: frases con
    # gramática rota y palabras inventadas (p. ej. "comengan", que no
    # existe en español), tejiendo fragmentos sueltos del propio prompt en
    # español en vez de describir la imagen.
    #
    # Causa real, confirmada por el propio autor del modelo en la
    # discusión "OCR and multi-Language support?" de
    # huggingface.co/vikhyatk/moondream2/discussions/22: "The training
    # data is currently english-only" -- Moondream2 no tiene apenas datos
    # de entrenamiento en español. Los campos PERSONAS/AFICION/PAREJA
    # funcionan en español porque son respuestas CORTAS de un conjunto
    # cerrado (básicamente reproducir 2-3 palabras del propio ejemplo del
    # prompt); pedirle que GENERE una frase española libre y coherente lo
    # saca de su dominio de entrenamiento.
    #
    # Solución adoptada (decisión de producto, no solo técnica): preguntar
    # en INGLÉS, que es donde el modelo es fiable, y mostrar el caption
    # resultante tal cual, en inglés, en vez de traducirlo -- se descartó
    # traducir con Mistral (ya integrado en el resto del pipeline) por
    # reintroducir el mismo límite de 2 peticiones/minuto del tier
    # gratuito que motivó pasar a un modelo local en primer lugar (ver
    # docstring de cabecera del módulo), salvo que se batchee a UNA
    # llamada por informe en vez de por foto -- no implementado aquí.
    "Describe what's happening in this image in one short sentence (8-15 words): the "
    "activity, the setting, and the mood. Be specific about the real scene, not generic. "
    "Do not mention any person's physical appearance, race, ethnicity, skin tone, or age "
    "-- only the activity and the context."
)

_STRUCTURED_QUERY = (
    # NOTA (ver registro de trabajo): la primera versión de este prompt
    # metía la explicación de cada valor DENTRO de un placeholder <...> en
    # la misma línea que la etiqueta (p. ej. "AFICION: <una afición... o la
    # palabra 'ninguno' si no hay nada específico>"). Con Moondream2, eso
    # provocaba que el modelo devolviera el propio texto del placeholder
    # como si fuera la respuesta (copiaba literalmente la explicación en
    # vez de sustituirla por un valor real) -- fallo de instruction-
    # following típico de VQA pequeños cuando la etiqueta, la explicación
    # de las opciones y el valor esperado se mezclan en una sola línea.
    # Esto también explica por qué el análisis superaba el timeout
    # configurado (entonces 30s, ver Settings.scene_analysis_timeout_seconds
    # en config.py) incluso en GPU: al "confundirse", el modelo generaba varios cientos de tokens
    # de texto repetido en vez de líneas cortas. La solución: separar un
    # EJEMPLO literal de líneas (que el modelo solo tiene que "copiar la
    # forma de") de la explicación de qué valores son válidos, en párrafos
    # aparte -- y capar la generación con _STRUCTURED_SETTINGS más abajo
    # como red de seguridad adicional. A diferencia de _CAPTION_QUERY
    # (texto libre), aquí SÍ funciona dar un ejemplo literal porque las
    # cuatro respuestas son opciones fijas (o casi -- ver TEXTO_VISIBLE) --
    # no hay contenido "copiable" que pueda colarse como respuesta real,
    # solo forma.
    #
    # Campo TEXTO_VISIBLE añadido después: texto legible en la propia foto
    # (carteles, camisetas, pancartas, matrículas, nombres de lugares...).
    # Es el campo con más riesgo de fuga de privacidad de los cuatro --
    # texto real de la foto puede incluir nombres propios (una insignia,
    # una camiseta con un nombre bordado) -- de ahí la advertencia
    # explícita más abajo, aparte de la general del final del prompt.
    #
    # Campo MATRICULA añadido después, separado de TEXTO_VISIBLE aunque
    # ambos sean "texto legible en la foto": TEXTO_VISIBLE solo captura EL
    # texto "más relevante" de la imagen (uno solo, ver su explicación más
    # abajo), así que una matrícula que comparta foto con un cartel más
    # llamativo se perdería por completo si no tuviera su propia pregunta
    # dedicada. Es, con diferencia, el campo de más riesgo de identificar
    # de forma unívoca a la persona (o al menos su vehículo) de los cinco,
    # así que el resultado se valida por formato antes de mostrarse (ver
    # `_SPANISH_PLATE_OLD_FORMAT_RE`/`_SPANISH_PLATE_NEW_FORMAT_RE` más
    # abajo) -- un OCR de un VQA pequeño como Moondream2 sobre texto
    # diminuto y a menudo en ángulo es poco fiable, y una matrícula mal
    # leída pero con apariencia de matrícula real sería peor que no leer
    # ninguna. El aviso de "lectura automática, puede contener errores" se
    # añade explícitamente al mostrarla (ver _parse_inferences), nunca se
    # presenta como un dato cierto.
    "Analiza esta imagen y responde EXACTAMENTE en este formato de cinco líneas, sin nada más, "
    "como en este ejemplo (sustituyendo los valores por los reales de ESTA imagen):\n"
    # HISTORIAL DEL SESGO EN PERSONAS (ver test de regresión más abajo en
    # este módulo -- test_scene_analysis.py::TestParsePersonas -- y el
    # registro de trabajo): el ejemplo de este campo pasó por DOS
    # intentos previos, ambos con el mismo fallo de fondo.
    #
    # 1) Originalmente decía "PERSONAS: varias" (el único valor
    # "positivo" del bloque). En producción, Moondream2 lo marcaba de
    # forma sistemática incluso con una sola persona -- el fallo de
    # instruction-following ya documentado en este módulo para VQA
    # pequeños: copian el ejemplo en vez de razonar sobre la imagen real.
    #
    # 2) Se cambió a "PERSONAS: ninguna", asumiendo que el problema era
    # "varias" concretamente (el valor atípico entre los cuatro campos
    # del ejemplo). Error de diagnóstico: el problema no era QUÉ valor
    # se copiaba, sino que se copiaba CUALQUIERA que fuera un valor
    # VÁLIDO -- así que el sesgo simplemente se trasladó de "varias" a
    # "ninguna" (visto en producción: fotos con varias personas
    # mostrando "Personas en la foto: ninguna" de forma sistemática).
    #
    # Solución adoptada: el ejemplo usa "dos", que NO pertenece al
    # vocabulario válido de PERSONAS ('ninguna'/'una'/'varias', ver
    # `_parse_personas`). Si el modelo lo copia igualmente por
    # confusión, `_parse_personas` lo descarta (no es una de las tres
    # opciones) y el campo simplemente no se muestra -- degradación
    # segura (dato ausente) en vez de un dato incorrecto pero plausible
    # como antes. Mantiene el resto de campos del ejemplo con su valor
    # "negativo"/por defecto porque no hay evidencia de que sufran el
    # mismo problema (ver más abajo, "uno" en AFICION es un fallo
    # distinto, aún sin diagnosticar con datos reales).
    "PERSONAS: dos\n"
    "AFICION: ninguno\n"
    "PAREJA: no\n"
    "TEXTO_VISIBLE: ninguno\n"
    "MATRICULA: ninguna\n\n"
    "PERSONAS solo puede valer: 'ninguna' (no aparece ninguna persona), 'una' (aparece "
    "exactamente una persona protagonista), o 'varias' (dos o más personas de protagonismo "
    "similar, p. ej. una pareja o un grupo).\n"
    "AFICION solo puede valer: una afición, actividad, deporte, instrumento musical o fandom "
    "concreto que sugiera la imagen, en pocas palabras, o la palabra 'ninguno' si no hay nada "
    "específico. AFICION NUNCA es un número ni una cantidad de personas -- eso va en PERSONAS, "
    "no aquí.\n"
    "PAREJA solo puede valer: 'si' si la imagen muestra a dos personas besándose, en un abrazo "
    "claramente romántico, o cogidas de la mano en un contexto de pareja, o 'no' en cualquier "
    "otro caso.\n"
    "TEXTO_VISIBLE solo puede valer: el texto legible más relevante que aparezca en la imagen "
    "(cartel, escaparate, pancarta, nombre de una calle o un lugar -- NUNCA una matrícula, eso "
    "va en MATRICULA, no aquí), copiado tal cual, o la palabra 'ninguno' si no hay texto legible. "
    "TEXTO_VISIBLE NUNCA puede ser el nombre propio de una persona (en una camiseta, insignia, "
    "etiqueta con nombre, etc.), aunque se lea con claridad -- en ese caso responde 'ninguno' "
    "para ese texto en concreto.\n"
    "MATRICULA solo puede valer: el texto de una matrícula de vehículo español visible en la "
    "imagen, copiado tal cual, letra por letra y número por número, sin espacios ni guiones "
    "añadidos por ti -- o la palabra 'ninguna' si no hay ninguna matrícula visible o legible. "
    "Las matrículas españolas actuales tienen 4 números seguidos de 3 letras (p. ej. 1234BCD); "
    "las antiguas (antes de 2000) tienen 1-2 letras, 4 números y 1-2 letras (p. ej. M1234AB). Si "
    "hay una matrícula pero no puedes leerla con claridad, responde 'ninguna' -- NUNCA inventes "
    "o completes caracteres que no puedas distinguir.\n\n"
    "No describas ni identifiques físicamente a ninguna persona que aparezca en la imagen -- "
    "ni su aspecto, ni su sexo, ni su edad, ni su raza o etnia -- más allá de contarlas y de si "
    "hay o no un contexto romántico entre ellas.\n\n"
    "Responde ahora solo las cinco líneas, con los valores reales para esta imagen concreta."
)

# Settings por separado para cada llamada -- cada una necesita un límite
# de tokens distinto (la respuesta correcta es mucho más corta en la
# estructurada que en el caption) y capar cada una a su propio tamaño
# real reduce aún más el riesgo de que una respuesta confusa se alargue
# de más, además de acelerar cada llamada individualmente.
#
# _STRUCTURED_SETTINGS con más margen (45, antes 30) desde que se añadió
# TEXTO_VISIBLE -- ahora son CUATRO líneas en vez de tres, y ya no hace
# falta apurar el límite para evitar texto sobrante feo en pantalla: la
# descripción que se MUESTRA (`descripcion_cruda`, más abajo en
# analyze_image_content) ya no es el texto crudo del modelo, se
# RECONSTRUYE desde los valores YA PARSEADOS -- así que cualquier cola
# que el modelo genere de más (p. ej. empezar a copiar "PERSONAS solo
# puede val..." tras responder bien, visto en producción) se descarta
# automáticamente sin importar en qué punto exacto se corte.
#
# La clave "variant" YA NO SE USA (era un requisito de `encode_image()`
# en el código remoto de la versión `transformers` de antes, ver
# historial junto a `_GGUF_REPO_ID` -- ese backend revenaba con KeyError
# sin ella). Se deja en el dict por si algún día vuelve a hacer falta
# algo parecido, pero `analyze_image_content()` (que ahora llama a
# `_model.create_chat_completion()`) solo lee "max_tokens" y
# "temperature" de aquí.
#
# temperature: 0.2 en el caption (algo de margen para que la frase suene
# natural, ya que es texto libre) y 0.1 en la estructurada (ya probado
# fiable para mantener el formato de opciones fijas). max_tokens=55 en la
# estructurada (más que el caption): tiene CINCO líneas que generar
# (incluida MATRICULA), no cuatro.
_CAPTION_SETTINGS = {"max_tokens": 45, "temperature": 0.2, "variant": None}
_STRUCTURED_SETTINGS = {"max_tokens": 55, "temperature": 0.1, "variant": None}

# Redimensionado específico para Moondream2, aparte del que ya aplica
# geolocation.py para DINOv2 (_MAX_QUEUED_IMAGE_DIMENSION=1024, que ese
# modelo sí aprovecha para el matching de escenas) -- ver docstring de
# `analyze_image_content` para por qué se hace sobre una COPIA, no sobre
# la imagen recibida.
#
# El valor 378 viene directamente del código fuente real de esta revisión
# pinneada (`_MODEL_REVISION`), no de documentación pública genérica --
# confirmado en ejecución real (GTX 1650) leyendo config.py del propio
# modelo vía `inspect.getsource()`:
#
#   class VisionConfig:
#       crop_size: int = 378
#       max_crops: int = 12
#
# `encode_image()` trocea la imagen en un crop global (reescalado
# internamente a un tamaño fijo pequeño para dar contexto general) + N
# crops locales solapados de alta resolución (para detalle fino) -- TODOS
# pasan por el encoder de visión, así que el coste escala con el número
# de crops. `select_tiling()` (misma revisión, image_crops.py) solo
# devuelve un único crop (tiling=(1,1), rápido) si AMBAS dimensiones de
# la imagen ya caben dentro de crop_size; si no, trocea en varios.
#
# Medido en producción (GTX 1650, foto 612x408 ya redimensionada a los
# 1024px de _MAX_QUEUED_IMAGE_DIMENSION): sin este redimensionado
# adicional, select_tiling() devolvía (2, 4) -- 8 crops locales + 1
# global = 9 pasadas por el encoder de visión, y encode_image() tardaba
# ~33s. Redimensionando a que el lado mayor mida 378px (manteniendo
# aspect ratio, como aquí), select_tiling() pasa a (1, 1) -- 2 pasadas en
# vez de 9.
#
# Trade-off aceptado: se pierde la capacidad de leer detalle muy fino
# (texto pequeño, objetos lejanos) que solo aportarían los crops locales
# -- aceptable para las señales que este módulo extrae (descripción
# general, conteo aproximado de personas, aficiones visibles, indicio de
# pareja), que son deliberadamente de grano grueso, no para OCR ni
# detección de objetos pequeños. Si en el futuro se necesitara ese
# detalle fino, subir este valor (a costa de más crops y más tiempo).
_CAPTION_MAX_DIMENSION = 378

_DESCRIPCION_RE = re.compile(r"DESCRIPCION:[ \t]*(.+)", re.IGNORECASE)
_PERSONAS_RE = re.compile(r"PERSONAS:[ \t]*(\S+)", re.IGNORECASE)
_AFICION_RE = re.compile(r"AFICION:[ \t]*(.+)", re.IGNORECASE)
_PAREJA_RE = re.compile(r"PAREJA:[ \t]*(\S+)", re.IGNORECASE)
_TEXTO_VISIBLE_RE = re.compile(r"TEXTO_VISIBLE:[ \t]*(.+)", re.IGNORECASE)
_MATRICULA_RE = re.compile(r"MATRICULA:[ \t]*(.+)", re.IGNORECASE)

# Validación de FORMATO de matrícula española, aplicada al texto que
# devuelve Moondream2 antes de mostrarlo -- un VQA pequeño leyendo texto
# diminuto y a menudo en ángulo puede alucinar caracteres, y "algo con
# forma de matrícula" pero mal leído es peor que no detectar nada (ver
# comentario junto a MATRICULA en _STRUCTURED_QUERY). No confirma que la
# matrícula sea REAL ni que esté bien leída letra por letra, solo que
# tiene la FORMA correcta -- de ahí que el aviso de "lectura automática,
# puede contener errores" se mantenga siempre, incluso cuando el formato
# valida.
#
# Formato actual (desde el 18 de septiembre de 2000): 4 dígitos + 3
# consonantes, excluidas las vocales, la Ñ y la Q (para evitar
# confusiones). Ver PLATE_PROVINCE_CODE_TO_PROVINCE en ine_reference.py
# para el formato antiguo (antes de esa fecha), que sí codificaba
# provincia.
_SPANISH_PLATE_NEW_FORMAT_RE = re.compile(r"^(\d{4})[ -]?([BCDFGHJKLMNPRSTVWXYZ]{3})$")
# Formato antiguo (1971-2000): 1-2 letras (código de provincia) + 4
# dígitos + 1-2 letras (sufijo, sin Ñ/Q -- ver fuentes en el comentario
# de PLATE_PROVINCE_CODE_TO_PROVINCE). El sufijo se valida de forma
# permisiva (cualquier letra excepto Ñ/Q, incluidas vocales) en vez de
# replicar cada exclusión histórica exacta -- las reglas variaron por
# época (el sufijo de una sola letra excluía además la R; el de dos no
# usaba vocales salvo la U, pero eso no está confirmado con la misma
# certeza) y ser demasiado estricto aquí arriesga rechazar una matrícula
# antigua real más de lo que arriesga aceptar una inventada. Lo que de
# verdad importa para este caso de uso es el CÓDIGO DE PROVINCIA del
# principio, no la validación exhaustiva del sufijo.
_SPANISH_PLATE_OLD_FORMAT_RE = re.compile(
    r"^([A-Z]{1,2})[ -]?(\d{4})[ -]?([ABCDEFGHIJKLMNOPRSTUVWXYZ]{1,2})$"
)


def get_device() -> str | None:
    """Dispositivo en el que se PIDIÓ offload de capas a `_lazy_load()`
    ("cuda" o "cpu"), o `None` si `_lazy_load()` no se ha llamado todavía
    (modelo no cargado -- p. ej. `enable_scene_analysis` desactivado, o
    análisis sin ninguna foto procesada aún). Pensado para el logging de
    rendimiento (ver app/log/performance_log.py).

    IMPORTANTE, distinto a como funcionaba con `transformers` (ver
    docstring de `_actual_device` más arriba): esto es lo que se PIDIÓ vía
    `n_gpu_layers`, no una confirmación leída del modelo ya cargado --
    `llama.cpp` no lanza excepción si el offload a GPU falla parcialmente,
    solo lo indica en su log nativo, que con `verbose=False` (el valor por
    defecto ahora, ver `_lazy_load()`) no se ve. Ya se confirmó una vez en
    producción (12/9: "offloaded 25/25 layers to GPU") con `verbose=True`
    temporalmente -- si alguna vez hay que volver a confirmarlo (p. ej.
    tras cambiar de GPU), poner `verbose=True` otra vez ahí antes de
    fiarse solo de este campo."""
    return _actual_device


def get_model_variant() -> str | None:
    """`_GGUF_REPO_ID` tal cual, o `None` si el modelo no se ha cargado
    todavía en este proceso (mismo criterio que `get_device()`).

    Existe para el log de rendimiento (ver app/log/performance_log.py):
    sin este campo, entradas de distintos backends/modelos probados (bf16
    `transformers`, 4-bit `torchao`, ahora GGUF `llama.cpp`, ver
    historial junto a `_GGUF_REPO_ID`) quedarían mezcladas en el mismo
    `.jsonl` sin forma de separarlas para comparar.

    Devuelve el nombre TAL CUAL quedó cargado en `_lazy_load()`, no el
    valor actual del módulo `_GGUF_REPO_ID` -- mismo motivo que
    `get_device()` usa `_actual_device` y no una relectura en caliente: si
    el proceso lleva tiempo vivo y se cambia el código sin reiniciar (no
    debería pasar en producción, pero sí durante desarrollo local), el
    modelo ya cargado en memoria sigue siendo el de antes."""
    return _loaded_model_name


def _scene_analysis_available() -> bool:
    """Comprobación barata (sin cargar el modelo) de si este módulo puede
    funcionar: dependencia opcional instalada. No hay ningún índice ni
    fichero que comprobar (a diferencia de geolocation.py), el modelo se
    descarga solo la primera vez vía el caché de Hugging Face (a través de
    `Llama.from_pretrained()`, que usa `huggingface_hub` por debajo igual
    que `transformers`).

    Desde el cambio a `llama-cpp-python` (ver la nota junto a
    `_GGUF_REPO_ID`), la única dependencia propia de este módulo es
    `llama_cpp` -- ya NO se necesitan `timm`/`einops` (eran del código
    remoto `trust_remote_code=True` de la versión `transformers`,
    eliminada en este cambio) ni `torchao` (de la versión 4-bit intentada
    antes, también descartada). `torch`/`transformers` los sigue
    necesitando este proceso igualmente, pero solo para DINOv2
    (geolocation.py) -- este módulo ya no los importa para nada."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_quantized_model() -> tuple[str, str] | None:
    """Devuelve `(ruta_local, tipo)` de una versión cuantizada del modelo de
    TEXTO de Moondream2 (no del `mmproj`/vision encoder -- ese se queda
    en F16 siempre, este repo no tiene una variante cuantizada de esa
    parte, ver historial junto a `_GGUF_REPO_ID` sobre por qué el ahorro
    esperado NO es simplemente "la mitad de tiempo"), cuantizando una
    única vez POR TIPO y cacheando el resultado en disco -- o `None` si
    no se pudo (sin `llama-quantize` disponible, o cualquier fallo
    durante el proceso), en cuyo caso `_lazy_load()` cae a descargar/usar
    el F16 de siempre vía `Llama.from_pretrained()`. Nunca lanza --
    best-effort, igual que el resto de este módulo (ver
    `analyze_image_content`): que esta optimización falle no debe impedir
    que Moondream2 cargue en absoluto, solo que cargue sin cuantizar.

    Tipo configurable con la variable de entorno `MOONDREAM_QUANT_TYPE`
    (por defecto Q8_0, ver `_DEFAULT_QUANT_TYPE`) -- cualquier tipo que
    acepte `llama-quantize` vale (Q4_K_M, Q5_K_M, etc., ver su propio
    `--help`; Q4_K_M recomendado sobre Q4_0 a pelo si se prueba 4 bits,
    mejor calidad para un tamaño similar según la propia tabla de
    `--help`). El nombre del fichero cacheado incluye el tipo, así que
    cambiar de `MOONDREAM_QUANT_TYPE` entre reinicios no pisa ni obliga a
    borrar la cuantización anterior -- conviven varias a la vez en disco,
    cada una cuantizada solo una vez.

    SIN VERIFICAR TODAVÍA (mismo motivo que el resto de cambios de hoy:
    sin GPU en el entorno donde se escribió esto) -- en particular, que
    `llama-quantize` exista de verdad en el PATH depende de que la etapa
    `cuda-builder` del Dockerfile lo haya conseguido compilar, lo cual es
    en sí mismo best-effort ahí (ver ese fichero) por la misma razón: dos
    supuestos sin confirmar sobre el layout del código fuente vendorizado
    de `llama-cpp-python`."""
    import shutil
    import subprocess

    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        logger.info(
            "llama-quantize no está en el PATH -- Moondream2 cargará en F16 sin cuantizar "
            "(ver Dockerfile, etapa cuda-builder, sobre por qué esto puede faltar)"
        )
        return None

    quant_type = os.environ.get("MOONDREAM_QUANT_TYPE", _DEFAULT_QUANT_TYPE).strip().upper()

    # Bajo el mismo volumen persistente que ya montáis para la caché de
    # Hugging Face (ver docker-compose.yml, `./backend/data/hf_cache:/root/.cache/huggingface`)
    # -- así el resultado sobrevive a un reinicio del contenedor y esto
    # solo se paga una vez de verdad por tipo, no en cada arranque.
    quantized_dir = Path("/root/.cache/huggingface/moondream2-quantized")
    quantized_path = quantized_dir / f"moondream2-text-model-{quant_type.lower()}.gguf"
    if quantized_path.exists():
        return str(quantized_path), quant_type

    try:
        from huggingface_hub import hf_hub_download

        logger.info("Cuantizando Moondream2 a %s por primera vez (puede tardar varios minutos)...", quant_type)
        f16_path = hf_hub_download(repo_id=_GGUF_REPO_ID, filename=_GGUF_TEXT_MODEL_FILENAME_EXACT)

        quantized_dir.mkdir(parents=True, exist_ok=True)
        # Escribir a un fichero .tmp y renombrar al final SOLO si
        # `llama-quantize` termina bien: evita que una ejecución anterior
        # interrumpida a medias (p. ej. el contenedor parado sin querer
        # durante la cuantización) deje un .gguf incompleto que
        # `quantized_path.exists()` diera por bueno en el siguiente
        # arranque sin serlo.
        tmp_output = quantized_dir / f"moondream2-text-model-{quant_type.lower()}.gguf.tmp"
        result = subprocess.run(
            [quantize_bin, f16_path, str(tmp_output), quant_type],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            logger.warning(
                "llama-quantize terminó con código %s -- Moondream2 cargará en F16 sin cuantizar. stderr: %s",
                result.returncode,
                result.stderr[-2000:],
            )
            return None
        tmp_output.rename(quantized_path)
        logger.info("Moondream2 cuantizado a %s correctamente: %s", quant_type, quantized_path)
        return str(quantized_path), quant_type
    except Exception as exc:
        logger.warning(
            "Fallo cuantizando Moondream2 a %s (%s): %s -- cargará en F16 sin cuantizar",
            quant_type,
            type(exc).__name__,
            exc,
        )
        return None


def _lazy_load():
    """Carga perezosa de Moondream2 vía `llama-cpp-python` (GGUF, ver
    `_GGUF_REPO_ID` y su historial de por qué se llegó aquí) -- no hace
    nada si ya está cargado (`_model is not None`).

    A diferencia de la versión `transformers` anterior, aquí NO hay
    parcheo de dtype que hacer -- los pesos GGUF ya vienen en el formato
    final (F16 en este repo) y `llama.cpp` no tiene el problema de
    "el kwarg no llega a los pesos reales" que sí tenía el código remoto
    de `transformers` (ver el historial junto a `_GGUF_REPO_ID`, intento
    1). Por eso esta función es mucho más corta que antes -- no es que se
    haya simplificado de más, es que la mayoría de la complejidad de antes
    era específica de problemas de ESE backend concreto.

    `n_gpu_layers=-1` pide que TODAS las capas se offloadeen a GPU. Si no
    hay GPU disponible (`ENABLE_IGPU_OFFLOAD`/GPU no detectada, ver
    app/main.py), `llama.cpp` debería caer solo a CPU sin excepción -- si
    en la práctica no es así, ver la nota de `get_device()` sobre por qué
    este código no puede confirmarlo con certeza desde aquí."""
    global _model, _actual_device, _loaded_model_name
    if _model is not None:
        return

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler

    # Detección de GPU: MISMO criterio que ya usaba la versión anterior
    # (ver `app.main`, "GPU detectada" en el log de arranque) -- ese log
    # ya viene de comprobar `torch.cuda.is_available()` antes de llegar
    # aquí, así que no hace falta duplicar la comprobación con otra
    # librería; simplemente se pide offload total y se confía en que
    # `llama.cpp` decida bien si no hay GPU.
    import torch

    _requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    _n_gpu_layers = -1 if _requested_device == "cuda" else 0

    chat_handler = MoondreamChatHandler.from_pretrained(
        repo_id=_GGUF_REPO_ID,
        filename=_GGUF_MMPROJ_FILENAME,
    )

    # Cuantizar (tipo configurable con MOONDREAM_QUANT_TYPE, por defecto
    # Q8_0 -- ver _ensure_quantized_model() para el porqué y las
    # condiciones -- nunca lanza, best-effort). SOLO afecta al modelo de
    # TEXTO: el `mmproj` de arriba se carga igual en los dos casos, sigue
    # en F16 siempre.
    _quantized = _ensure_quantized_model()
    _common_kwargs = dict(
        chat_handler=chat_handler,
        n_gpu_layers=_n_gpu_layers,
        # 2048, no 4096: confirmado en producción (12/9) que `n_ctx_train`
        # de este modelo es 2048 -- pedir más (probado con 4096) generaba
        # el aviso "possible training context overflow" en el log. Con
        # los dos prompts de este módulo (image embedding, ~729 tokens,
        # + _CAPTION_QUERY/_STRUCTURED_QUERY) cabe de sobra dentro de
        # 2048, así que no hay motivo real para salirse del contexto de
        # entrenamiento solo por margen -- eso solo compraría degradar la
        # calidad sin necesitarlo.
        n_ctx=2048,
        # verbose=False: antes en True a propósito, para confirmar en el
        # log si el offload a GPU funcionaba de verdad (ver get_device())
        # -- ya confirmado en producción (12/9: "offloaded 25/25 layers
        # to GPU"), así que ya no compensa el ruido que mete por foto
        # (líneas de "create_tensor", "clip_model_loader", "CUDA Graph id
        # N reused" -- decenas por imagen). SIN VERIFICAR: `verbose=False`
        # no parece silenciar el logging nativo del componente
        # clip/multimodal (encoding image slice, clip_encode, add_media
        # -- confirmado que estas líneas siguen saliendo en producción,
        # 13/9), parece tener su propio control de verbosidad no atado a
        # este flag -- pendiente de investigar si molesta.
        verbose=False,
    )
    if _quantized is not None:
        _quantized_path, _quant_type = _quantized
        _model = Llama(model_path=_quantized_path, **_common_kwargs)
        _variant_suffix = f" ({_quant_type}, texto cuantizado)"
    else:
        _model = Llama.from_pretrained(
            repo_id=_GGUF_REPO_ID,
            filename=_GGUF_TEXT_MODEL_FILENAME,
            **_common_kwargs,
        )
        _variant_suffix = " (F16)"
    _actual_device = _requested_device
    # El sufijo (Q8_0 vs F16) queda en el propio nombre guardado -- así
    # `get_model_variant()` (y por tanto el log de rendimiento, ver
    # app/log/performance_log.py) separa las dos variantes sin tener que
    # añadir otro campo nuevo solo para esto.
    _loaded_model_name = _GGUF_REPO_ID + _variant_suffix
    logger.info(
        "Moondream2 cargado: model=%s n_gpu_layers=%s (dispositivo solicitado=%s; revisar el log nativo de llama.cpp arriba para confirmar si el offload a GPU funcionó de verdad, ver get_device())",
        _loaded_model_name,
        _n_gpu_layers,
        _requested_device,
    )



def analyze_image_content(
    image,
) -> tuple[list[InferredAttribute], bool, str | None, str | None, VisualDescriptionCodes | None]:
    """Devuelve (inferencias_visuales, indicio_pareja, descripcion_cruda,
    descripcion_general) para UNA foto ya decodificada (PIL.Image, la
    misma que usa geolocation.py para el embedding de DINOv2 -- no se
    descarga ni decodifica de nuevo). SÍNCRONA y con trabajo de CPU/GPU
    real (como `estimate_location_from_image`): quien llama debe
    envolverla en `asyncio.to_thread` para no bloquear el event loop, ver
    geolocation.py.

    Internamente, ANTES de nada, hace una copia de `image` y la
    redimensiona a `_CAPTION_MAX_DIMENSION` (ver esa constante para la
    medición real que respalda el valor) -- nunca se toca `image` en sí,
    porque geolocation.py ejecuta `estimate_location_from_image(image)` en
    paralelo sobre el MISMO objeto para DINOv2, que sí necesita la
    resolución mayor (`_MAX_QUEUED_IMAGE_DIMENSION` en geolocation.py).

    Internamente hace DOS llamadas a `_model.create_chat_completion()` --
    una con `_CAPTION_QUERY` (texto libre, sin plantilla que copiar) y
    otra con `_STRUCTURED_QUERY` (PERSONAS/AFICION/PAREJA/TEXTO_VISIBLE/
    MATRICULA, formato fijo) -- en vez de una sola combinada, porque
    mezclar un campo de texto libre con campos de opciones fijas en el
    mismo prompt hacía que Moondream2 copiara literalmente el ejemplo de
    texto libre en vez de describir la imagen real (ver nota en
    `_CAPTION_QUERY`). Desde el cambio a `llama-cpp-python` (ver
    historial junto a `_GGUF_REPO_ID`) ya NO hay un equivalente directo a
    "codificar la imagen una vez y reutilizarla" -- cada llamada manda la
    imagen (como data URI base64) otra vez.

    `descripcion_cruda` (ver `_build_clean_summary`) se RECONSTRUYE a
    partir de los valores YA PARSEADOS de `structured` -- ya NO es el
    texto crudo del modelo tal cual. Dos motivos: (a) el modelo, tras
    responder bien, a veces sigue generando y empieza a copiar fragmentos
    de la propia explicación del prompt (visto en producción); reconstruir
    desde valores parseados descarta esa cola sin depender de acertar el
    `max_tokens` exacto cada vez; (b) solo interesa mostrar señales
    POSITIVAS/informativas -- los valores negativos por defecto (personas
    ninguna, afición ninguna, sin pareja, sin texto visible) no aportan
    nada y solo acumulan líneas vacías si se muestran siempre. None si no
    hubo NADA informativo que mostrar.

    `descripcion_general` es directamente la respuesta de `_CAPTION_QUERY`
    (`caption`, ya limpia) -- pensada para mostrarse tal cual como pie de
    foto legible, a diferencia de `descripcion_cruda` (pensado para la
    vista "qué vio la IA" de detalle, con las señales estructuradas).
    EN INGLÉS, a diferencia del resto de este módulo y del resto del
    proyecto (español) -- decisión deliberada, no un descuido: Moondream2
    solo tiene datos de entrenamiento en inglés (confirmado por el autor
    del modelo, ver nota en `_CAPTION_QUERY`), y pedirle generar una frase
    libre en español producía gramática rota y palabras inventadas. Se
    decidió mostrar el caption en inglés tal cual antes que traducirlo
    (ver la misma nota para por qué se descartó traducir con Mistral).
    Mismo límite ético/legal que el resto del módulo: el prompt le
    prohíbe explícitamente mencionar raza, etnia, tono de piel, edad o
    aspecto físico -- pero como es texto libre (a diferencia de
    PERSONAS/PAREJA, que son una de tres opciones fijas), no hay garantía
    sintáctica de que el modelo lo respete siempre; se confía en el
    prompt, no en un filtro de post-procesado adicional (ver docstring de
    cabecera del módulo, mismo criterio que ya se aplica al resto de
    campos).

    None en cualquiera de los cuatro valores si el modelo no está
    disponible o la inferencia falla.

    `evidence` de cada InferredAttribute se deja vacío deliberadamente --
    quien llama (geolocation.py) rellena el permalink de la publicación,
    que esta función no conoce.

    Nunca lanza: cualquier fallo (dependencias no instaladas, modelo no
    descargable, respuesta con formato inesperado) se trata como "esta
    foto no aportó nada" y devuelve ([], False, None, None), sin abortar
    el análisis de las demás fotos."""
    if not _scene_analysis_available():
        logger.warning(
            "Análisis de contenido visual no disponible: falta llama-cpp-python "
            "(ver requirements-vision.txt)"
        )
        return [], False, None, None, None

    try:
        # Redimensionar una COPIA de la imagen antes de codificarla (ver
        # _CAPTION_MAX_DIMENSION más arriba para el porqué del valor 378
        # y la medición real que lo respalda). Nunca se llama .thumbnail()
        # sobre `image` directamente: geolocation.py ejecuta
        # estimate_location_from_image(image) y este análisis de forma
        # CONCURRENTE (asyncio.gather + to_thread) sobre el MISMO objeto
        # PIL.Image -- mutar `image` in-place aquí sería una condición de
        # carrera con el otro hilo, que sigue necesitando la resolución
        # de 1024px (_MAX_QUEUED_IMAGE_DIMENSION) para DINOv2.
        resized = image.copy()
        resized.thumbnail((_CAPTION_MAX_DIMENSION, _CAPTION_MAX_DIMENSION), Image.LANCZOS)
        # SIN VERIFICAR (ver historial junto a _GGUF_REPO_ID): el valor
        # 378 de _CAPTION_MAX_DIMENSION estaba medido contra el
        # tiling/crops del código remoto de `transformers`
        # (`VisionConfig.crop_size`, ver el comentario original de esa
        # constante más arriba) -- el vision encoder empaquetado en este
        # GGUF (`ggml-org`) puede preprocesar de otra forma. Se mantiene
        # el mismo redimensionado por ahora (como mínimo, reduce lo que
        # hay que codificar en base64 más abajo), pero si la calidad de
        # las descripciones cambia respecto a lo que dabas antes, este es
        # un sitio a revisar -- puede que ya no haga falta, o que el
        # tamaño óptimo sea distinto.
        buffer = io.BytesIO()
        resized.convert("RGB").save(buffer, format="JPEG", quality=90)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        image_content = {"type": "image_url", "image_url": {"url": data_uri}}

        # `with _model_lock:` (ver comentario junto a la declaración del
        # lock más arriba): SOLO cubre lo que de verdad toca `_model` --
        # el redimensionado/codificación de la imagen de arriba se queda
        # FUERA a propósito, es CPU/PIL puro y no comparte estado con
        # `llama.cpp`, así que no hay motivo para serializarlo también y
        # hacer la cola más larga de lo necesario.
        #
        # OJO, trade-off real y consciente: con `actual_concurrency` > 1
        # (varias fotos a la vez, ver geolocation.py), esto convierte el
        # análisis de CONTENIDO en efectivamente secuencial -- una foto
        # espera a que la anterior termine sus dos llamadas antes de
        # empezar las suyas, aunque el semáforo dejara varias "en vuelo"
        # a la vez. La geolocalización (DINOv2) NO se ve afectada, sigue
        # concurrente igual que antes -- este lock es solo para
        # Moondream2. Se acepta esto porque la alternativa confirmada en
        # producción (13/9) era un SIGSEGV que tiraba el proceso entero
        # cada pocas fotos -- coherente con la filosofía "nunca lanza,
        # best-effort" de esta función (ver docstring), pero para la
        # función SIGUIENTE, no para el backend ENTERO seguir vivo.
        with _model_lock:
            _lazy_load()  # no-op tras la primera carga (ver docstring) -- barato meterlo dentro del lock también, cierra el hueco teórico de doble carga concurrente
            # Dos llamadas independientes a create_chat_completion (no
            # hay equivalente directo al encode_image()+query()
            # reutilizable de `transformers`, ver historial junto a
            # _GGUF_REPO_ID).
            #
            # `_model.reset()` ANTES de cada una: bug real confirmado en
            # producción (12/9) -- sin esto, la caché KV interna del
            # objeto `Llama` no se limpia sola entre dos
            # `create_chat_completion()` independientes con imagen nueva.
            # La primera llamada deja la caché en una posición > 0; la
            # segunda intenta empezar de cero sin resetear, y
            # `llama.cpp` lo rechaza ("the tokens of sequence 0... have
            # inconsistent sequence positions") -- capturado por el
            # try/except de aquí abajo la primera vez que pasó, pero el
            # estado quedaba tan corrupto que el proceso entero acababa
            # muriendo con SIGSEGV (exit 139) unas pocas fotos después,
            # no solo fallando esa foto. `reset()` es el método propio de
            # `llama-cpp-python` para esto -- confirma que NO reaprovecha
            # el embedding de imagen entre las dos llamadas: cada
            # `create_chat_completion()` vuelve a calcularlo de cero, así
            # que el objetivo original de "codificar una vez, reutilizar
            # para las dos preguntas" (como sí hacía `transformers`) no
            # se ha conseguido con este backend -- es el coste de tener
            # esto funcionando sin crashear, no una optimización
            # pendiente de aprovechar.
            _model.reset()
            caption_response = _model.create_chat_completion(
                messages=[{"role": "user", "content": [image_content, {"type": "text", "text": _CAPTION_QUERY}]}],
                max_tokens=_CAPTION_SETTINGS["max_tokens"],
                temperature=_CAPTION_SETTINGS["temperature"],
            )
            _model.reset()
            structured_response = _model.create_chat_completion(
                messages=[{"role": "user", "content": [image_content, {"type": "text", "text": _STRUCTURED_QUERY}]}],
                max_tokens=_STRUCTURED_SETTINGS["max_tokens"],
                temperature=_STRUCTURED_SETTINGS["temperature"],
            )
        caption = caption_response["choices"][0]["message"]["content"].strip().rstrip(".")
        structured = structured_response["choices"][0]["message"]["content"].strip()

        # Log OPCIONAL de comparación entre variantes (ver
        # app/log/visual_description_log.py sobre el diseño y por qué
        # está desactivado por defecto) -- la comprobación de la flag va
        # PRIMERO a propósito, para no pagar el coste de hashear la
        # imagen (`image.tobytes()`) cuando está desactivado, que es el
        # caso normal.
        if settings.log_visual_descriptions:
            visual_description_log.log_visual_description(
                image_id=visual_description_log.image_content_id(image),
                model_variant=get_model_variant(),
                caption=caption,
                structured=structured,
            )
    except Exception as exc:
        # Motivo típico si `llama_cpp` SÍ está instalado (ver
        # _scene_analysis_available arriba): fallo de red al descargar el
        # modelo la primera vez, o el problema de offload a GPU descrito
        # en la nota SIN VERIFICAR junto a `_GGUF_REPO_ID`
        # (`libcudart.so` ausente con el wheel precompilado -- ya
        # resuelto compilando desde fuente, ver ese historial). Se
        # loguea para poder diagnosticarlo sin tener que quitar el
        # try/except (este módulo es best-effort y no debe abortar el
        # análisis de las demás fotos).
        logger.warning(
            "Análisis de contenido visual falló para una foto (%s): %s",
            type(exc).__name__,
            exc,
        )
        # NOTA histórica (ya no aplica al backend actual, `llama.cpp` vía
        # `llama-cpp-python` -- se deja por si ayuda a alguien buscando en
        # el historial de git): con el backend `transformers` de antes, un
        # `NotImplementedError` con el mensaje "Cannot copy out of meta
        # tensor" aquí fue en su momento un bug real de este módulo
        # (`device_map` pasado como string a `from_pretrained`), no un
        # problema de entorno -- afectaba al 100% de las fotos, en
        # cualquier máquina, con las dependencias bien instaladas.
        return [], False, None, None, None

    # `structured` (no un texto combinado con DESCRIPCION) es la única
    # fuente para PERSONAS/AFICION/PAREJA/TEXTO_VISIBLE -- ya no hace
    # falta reconstruir un texto "DESCRIPCION: {caption}\n{structured}"
    # como antes: `caption` ya ES directamente la descripción general (no
    # hace falta volver a parsearla de un texto reconstruido), y el bloque
    # que se MUESTRA en el frontend (`descripcion_cruda`) se reconstruye
    # desde los valores YA PARSEADOS, no desde texto crudo -- ver
    # `_build_clean_summary` para los dos motivos (descarta colas de
    # generación sobrantes Y solo muestra señales positivas).
    descripcion_general = caption or None
    personas = _parse_personas(structured)
    aficion_raw = _parse_aficion_raw(structured)
    indicio_pareja = _parse_pareja(structured)
    texto_visible = _parse_texto_visible(structured)
    matricula = _parse_matricula(structured)

    inferencias = _parse_inferences(structured)
    descripcion_cruda = _build_clean_summary(personas, aficion_raw, indicio_pareja, texto_visible, matricula)
    # Mismo filtro que _build_clean_summary aplica a `personas` para el
    # texto en español (solo "una"/"varias" son señal, no "ninguna") --
    # se repite aquí para que los códigos estructurados y el texto ya
    # redactado nunca se contradigan entre sí.
    codes = VisualDescriptionCodes(
        personas=personas if personas in ("una", "varias") else None,
        aficion=aficion_raw,
        texto_visible=texto_visible,
        matricula=matricula,
        indicio_pareja=indicio_pareja,
    )

    return inferencias, indicio_pareja, descripcion_cruda, descripcion_general, codes


def _parse_descripcion(answer: str) -> str | None:
    """Extrae el valor de la línea DESCRIPCION (caption general de la
    escena). None si no se pudo parsear, o si el modelo devolvió algo
    equivalente a "nada que describir" -- en la práctica esto último no
    debería pasar (DESCRIPCION siempre debería tener contenido si hay
    imagen), pero se trata igual que el resto de campos por consistencia
    y por si acaso el modelo se desvía del formato."""
    match = _DESCRIPCION_RE.search(answer)
    if match is None:
        return None
    valor = match.group(1).strip().rstrip(".")
    if not valor or valor.lower() in ("ninguna", "ninguno", "none", "n/a"):
        return None
    return valor


# Palabras que NUNCA son una afición real por sí solas -- si AFICION
# devuelve exactamente una de estas, es casi con toda seguridad un fallo
# de generación (p. ej. eco/confusión con el vocabulario de PERSONAS,
# como el ejemplo "PERSONAS: dos" de más arriba, o con el propio conteo
# de personas de la imagen) y no una afición genuina. Visto en producción
# (Comandante, agosto 2026): "Posible afición o interés: uno" en fotos
# donde no había ninguna pista de afición real -- se descarta por
# precaución en vez de mostrarlo como si fuera una señal fiable. Lista
# deliberadamente corta y literal (números en palabra hasta diez más sus
# formas más comunes) en vez de un intento de detectar "cualquier
# número" con regex, que podría descartar de más (p. ej. una afición
# real como "coleccionar cromos del Mundial 2010" contiene un número
# pero SÍ es una afición legítima).
_AFICION_INVALID_VALUES = frozenset({
    "cero", "uno", "una", "dos", "tres", "cuatro", "cinco", "seis",
    "siete", "ocho", "nueve", "diez",
    "ninguna", "varias", "persona", "personas",
})


def _parse_personas(answer: str) -> str | None:
    match = _PERSONAS_RE.search(answer)
    if match is None:
        return None
    value = match.group(1).strip().lower().rstrip(".,;")
    return value if value in ("ninguna", "una", "varias") else None


def _parse_aficion_raw(answer: str) -> str | None:
    """Extrae el valor CRUDO de la línea AFICION, SIN aplicar la cautela
    de atribución de `_parse_inferences` (que descarta la señal si
    PERSONAS no es 'ninguna'/'una', porque con varias personas no se
    puede saber de quién es la afición -- ver docstring de
    `_parse_inferences`). Se usa en dos sitios con necesidades distintas:
    `_parse_inferences` (que SÍ aplica esa cautela antes de convertirlo en
    un InferredAttribute atribuido a la cuenta analizada) y
    `_build_clean_summary` (que solo quiere mostrar "qué vio la IA" en el
    frontend, sin atribuírselo a nadie como rasgo personal -- ahí la
    cautela de atribución no aplica).

    Descarta también valores que son claramente un fallo de generación
    cruzada con PERSONAS en vez de una afición real -- ver
    `_AFICION_INVALID_VALUES`."""
    match = _AFICION_RE.search(answer)
    if match is None:
        return None
    valor = match.group(1).strip().rstrip(".")
    if not valor or valor.lower() in ("ninguno", "ninguna", "none", "n/a"):
        return None
    if valor.lower() in _AFICION_INVALID_VALUES:
        return None
    return valor


def _parse_texto_visible(answer: str) -> str | None:
    """Extrae el valor de la línea TEXTO_VISIBLE (texto legible en la
    propia foto: carteles, escaparates, nombres de lugares... NUNCA
    matrículas desde que existe el campo MATRICULA propio, ver
    `_parse_matricula`). None si no se pudo parsear o si el modelo
    respondió 'ninguno' (no hay texto legible). El prompt (ver
    _STRUCTURED_QUERY) ya le prohíbe explícitamente devolver el nombre
    propio de una persona aquí -- este parseo no repite ese filtro (no
    hay forma fiable de detectar "esto es un nombre propio" solo con
    regex), se confía en la instrucción del prompt, igual que ya se hace
    con el resto de restricciones éticas/legales del módulo (ver
    docstring de cabecera)."""
    match = _TEXTO_VISIBLE_RE.search(answer)
    if match is None:
        return None
    valor = match.group(1).strip().rstrip(".")
    if not valor or valor.lower() in ("ninguno", "ninguna", "none", "n/a"):
        return None
    return valor


def _parse_matricula(answer: str) -> str | None:
    """Extrae y VALIDA por formato la línea MATRICULA. None si no se pudo
    parsear, si el modelo respondió 'ninguna', o si el texto devuelto no
    tiene forma de matrícula española real (formato antiguo o actual, ver
    `_SPANISH_PLATE_OLD_FORMAT_RE`/`_SPANISH_PLATE_NEW_FORMAT_RE`) -- en
    ese último caso se descarta como probable alucinación/error de OCR de
    un VQA pequeño sobre texto diminuto, en vez de mostrarse igualmente.
    Para el formato antiguo, además, el código de provincia capturado
    debe ser uno de los 54 reales de PLATE_PROVINCE_CODE_TO_PROVINCE --
    "tiene la forma de 1-2 letras" no basta, "XZ-1234-BC" tendría la
    forma correcta pero "XZ" nunca fue un código de provincia real.

    Este filtro de formato/código sigue sin confirmar que los caracteres
    estén bien leídos, solo que la FORMA (y, para el antiguo, el código
    de provincia) es plausible -- de ahí que quien consuma este valor
    (`_parse_inferences`, `_build_clean_summary`) deba seguir avisando de
    que es una lectura automática que puede contener errores.

    Normaliza el separador a un guion (formato "XX-0000-XX"/"0000-XXX")
    para que se muestre de forma consistente sin importar cómo lo haya
    formateado el modelo (con espacios, sin separador, con guiones...)."""
    match = _MATRICULA_RE.search(answer)
    if match is None:
        return None
    valor = match.group(1).strip().rstrip(".")
    if not valor or valor.lower() in ("ninguno", "ninguna", "none", "n/a"):
        return None

    candidato = valor.upper().replace(" ", "").replace("-", "")

    new_match = _SPANISH_PLATE_NEW_FORMAT_RE.match(candidato)
    if new_match:
        digitos, letras = new_match.groups()
        return f"{digitos}-{letras}"

    old_match = _SPANISH_PLATE_OLD_FORMAT_RE.match(candidato)
    if old_match:
        provincia, digitos, sufijo = old_match.groups()
        if provincia not in PLATE_PROVINCE_CODE_TO_PROVINCE:
            return None
        return f"{provincia}-{digitos}-{sufijo}"

    return None


def _parse_inferences(answer: str) -> list[InferredAttribute]:
    inferences: list[InferredAttribute] = []

    # Con varias personas de protagonismo similar en la foto, no hay forma
    # de saber si la afición/actividad detectada es de la cuenta analizada
    # o de la otra persona -- ver docstring del módulo. Se descarta la
    # señal en vez de arriesgarse a atribuirla a quien no toca. Si
    # PERSONAS no se pudo parsear (formato inesperado), se prefiere
    # también descartar por precaución antes que asumir que es seguro
    # atribuirla.
    if _parse_personas(answer) in ("ninguna", "una"):
        aficion_raw = _parse_aficion_raw(answer)
        if aficion_raw is not None:
            inferences.append(
                InferredAttribute(
                    category="aficion",
                    value=f"Posible afición/interés detectado en una foto: {aficion_raw}",
                    confidence=0.5,
                    evidence=[],
                )
            )

    # TEXTO_VISIBLE NO necesita la misma cautela de atribución que AFICION:
    # un cartel o un nombre de lugar en la foto es verdad independientemente
    # de cuántas personas aparezcan -- no es un rasgo personal de "quién
    # sale en la foto", es evidencia sobre el LUGAR/CONTEXTO, más parecido
    # a la geolocalización por imagen que a un rasgo de la cuenta analizada.
    # Confianza más baja que AFICION (0.4 frente a 0.5): Moondream2 es un
    # VQA general, no un motor de OCR dedicado, más propenso a leer mal un
    # texto concreto que a describir mal una escena general.
    texto_visible = _parse_texto_visible(answer)
    if texto_visible is not None:
        inferences.append(
            InferredAttribute(
                category="texto_visible",
                value=f"Texto legible detectado en una foto: {texto_visible}",
                confidence=0.4,
                evidence=[],
            )
        )

    # MATRICULA es, de todos los campos de este módulo, el que más
    # directamente puede identificar a la persona (o al menos su
    # vehículo) -- confianza más baja todavía que TEXTO_VISIBLE (0.3) por
    # la misma razón de fondo (Moondream2 no es un OCR dedicado) más el
    # hecho de que una matrícula mal leída en un solo carácter sigue
    # aparentando ser una matrícula válida, a diferencia de un cartel mal
    # leído (que normalmente se nota como texto sin sentido). El aviso de
    # "lectura automática, puede contener errores" va DENTRO del propio
    # texto del valor, no solo en un campo de metadatos aparte, para que
    # sea imposible mostrarlo sin él en ningún punto del pipeline
    # (frontend, resumen de IA, exportación a JSON...).
    matricula = _parse_matricula(answer)
    if matricula is not None:
        inferences.append(
            InferredAttribute(
                category="matricula",
                value=f"Matrícula de vehículo visible en una foto: {matricula} "
                      "(lectura automática, puede contener errores -- verifica contra la foto original)",
                confidence=0.3,
                evidence=[],
            )
        )

    return inferences


def _parse_pareja(answer: str) -> bool:
    # A diferencia de _parse_inferences, aquí SÍ es válida la señal con
    # "varias" personas (de hecho es el caso típico: hacen falta al menos
    # dos para un contexto romántico) -- no depende de resolver quién es
    # la cuenta analizada, ver docstring del módulo.
    match = _PAREJA_RE.search(answer)
    if match is None:
        return False
    return match.group(1).strip().lower().rstrip(".,;") in ("si", "sí", "yes", "true")


def _build_clean_summary(
    personas: str | None,
    aficion_raw: str | None,
    indicio_pareja: bool,
    texto_visible: str | None,
    matricula: str | None,
) -> str | None:
    """Reconstruye el bloque 'qué vio la IA' que se muestra en el
    frontend (vista de detalle de cada foto) a partir de los valores YA
    PARSEADOS -- nunca a partir del texto crudo tal cual lo generó el
    modelo. Dos motivos:

    (1) El modelo, tras responder bien, a veces sigue generando y empieza
    a copiar fragmentos de la propia explicación del prompt (visto en
    producción: "PERSONAS solo puede val..." apareciendo tras las cuatro
    líneas correctas, cortado por `_STRUCTURED_SETTINGS['max_tokens']`).
    Reconstruir desde los valores ya parseados descarta esa cola
    automáticamente, sin depender de acertar el `max_tokens` exacto cada
    vez -- cada valor se extrajo con `.search()`, que coge la PRIMERA
    aparición de cada etiqueta y ya ignora cualquier eco posterior.

    (2) Solo interesa mostrar señales POSITIVAS/informativas (p. ej.
    "Indicio de contexto de pareja: sí", "Posible afición: guitarra") --
    los valores negativos por defecto (afición ninguna, sin pareja, sin
    texto visible) no aportan nada al usuario y solo acumulan líneas
    vacías si se muestran siempre. PERSONAS sigue el mismo criterio:
    'una'/'varias' sí se muestran (dan contexto básico sobre la foto),
    pero 'ninguna' se trata como el resto de valores negativos y se
    oculta. Antes 'ninguna' era la única excepción que sí se mostraba
    siempre -- se quitó esa excepción (Comandante, agosto 2026) tras el
    sesgo documentado en `_STRUCTURED_QUERY` (Moondream2 copiando el
    valor de ejemplo del prompt en vez de contar la imagen real, visto
    primero con "varias" y después con "ninguna"): aunque el ejemplo ya
    no usa un valor válido como placeholder (ver comentario en
    `_STRUCTURED_QUERY`), tratar 'ninguna' igual que el resto de valores
    negativos por defecto es más consistente con el resto del proyecto
    ("ante la duda, no mostrar") y limita el impacto visible si ese sesgo
    volviera a aparecer -- un 'ninguna' incorrecto oculto no confunde al
    usuario, uno mostrado sí.

    La descripción general (frase libre, `descripcion_general` en
    `analyze_image_content`) NO se repite aquí -- el frontend ya la
    muestra aparte, destacada, justo encima de este bloque.

    None si no hay NADA que mostrar (PERSONAS es 'ninguna' o no se pudo
    parsear, Y las demás son negativas) -- caso frecuente ahora que
    'ninguna' ya no se muestra por defecto, no solo un caso raro de
    formato inesperado."""
    lines = []
    if personas in ("una", "varias"):
        lines.append(f"Personas en la foto: {personas}")
    if aficion_raw:
        lines.append(f"Posible afición o interés: {aficion_raw}")
    if indicio_pareja:
        lines.append("Indicio de contexto de pareja: sí")
    if texto_visible:
        lines.append(f"Texto visible: {texto_visible}")
    if matricula:
        lines.append(f"Matrícula visible: {matricula} (lectura automática, puede contener errores)")
    return "\n".join(lines) if lines else None
