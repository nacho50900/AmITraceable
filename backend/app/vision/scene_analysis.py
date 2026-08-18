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
import logging
import re

from PIL import Image

from app.models.schemas import InferredAttribute

logger = logging.getLogger(__name__)

_MODEL_NAME = "vikhyatk/moondream2"
_MODEL_REVISION = "2025-06-21"  # fijado explícitamente, ver docstring del modelo en HuggingFace

_model = None

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
    "Analiza esta imagen y responde EXACTAMENTE en este formato de cuatro líneas, sin nada más, "
    "como en este ejemplo (sustituyendo los valores por los reales de ESTA imagen):\n"
    # PERSONAS usa aquí el mismo valor "nulo"/negativo que los otros tres
    # campos (ninguno/no/ninguno) -- antes decía "varias", el único valor
    # "positivo" de los cuatro del ejemplo. Visto en producción: Moondream2
    # marcaba "varias" de forma sistemática incluso en fotos con una sola
    # persona, un fallo de instruction-following ya documentado en este
    # mismo módulo para VQA pequeños (copian el ejemplo en vez de
    # sustituirlo, ver la nota sobre _STRUCTURED_QUERY más abajo) -- que
    # con "varias" como ejemplo sesga sistemáticamente hacia ese valor
    # cuando el modelo "se confunde" y copia en vez de razonar sobre esta
    # imagen en concreto. "ninguna" es coherente con el resto de ejemplos y
    # no favorece a priori ninguno de los otros dos valores reales (una
    # persona / varias).
    "PERSONAS: ninguna\n"
    "AFICION: ninguno\n"
    "PAREJA: no\n"
    "TEXTO_VISIBLE: ninguno\n\n"
    "PERSONAS solo puede valer: 'ninguna' (no aparece ninguna persona), 'una' (aparece "
    "exactamente una persona protagonista), o 'varias' (dos o más personas de protagonismo "
    "similar, p. ej. una pareja o un grupo).\n"
    "AFICION solo puede valer: una afición, actividad, deporte, instrumento musical o fandom "
    "concreto que sugiera la imagen, en pocas palabras, o la palabra 'ninguno' si no hay nada "
    "específico.\n"
    "PAREJA solo puede valer: 'si' si la imagen muestra a dos personas besándose, en un abrazo "
    "claramente romántico, o cogidas de la mano en un contexto de pareja, o 'no' en cualquier "
    "otro caso.\n"
    "TEXTO_VISIBLE solo puede valer: el texto legible más relevante que aparezca en la imagen "
    "(cartel, escaparate, pancarta, matrícula, nombre de una calle o un lugar), copiado tal cual, "
    "o la palabra 'ninguno' si no hay texto legible. TEXTO_VISIBLE NUNCA puede ser el nombre "
    "propio de una persona (en una camiseta, insignia, etiqueta con nombre, etc.), aunque se lea "
    "con claridad -- en ese caso responde 'ninguno' para ese texto en concreto.\n\n"
    "No describas ni identifiques físicamente a ninguna persona que aparezca en la imagen -- "
    "ni su aspecto, ni su sexo, ni su edad, ni su raza o etnia -- más allá de contarlas y de si "
    "hay o no un contexto romántico entre ellas.\n\n"
    "Responde ahora solo las cuatro líneas, con los valores reales para esta imagen concreta."
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
# "variant": None es obligatorio en AMBAS, no opcional -- descubierto en
# ejecución real (GTX 1650, revisión pinneada del modelo, ver
# _MODEL_REVISION): `encode_image()` en el código remoto de esta revisión
# hace `settings["variant"]` a pelo, SIN `.get()`, en cuanto `settings`
# no es None. Como no necesitamos usar variantes de encoder, cualquier
# `settings` que pasemos tiene que incluir esta clave con valor None o
# revienta con KeyError antes de generar nada -- no es un parámetro que
# hayamos elegido usar, es un requisito de esta versión concreta del
# modelo para poder usar settings en absoluto.
#
# temperature: 0.2 en el caption (algo de margen para que la frase suene
# natural, ya que es texto libre) y 0.1 en la estructurada (ya probado
# fiable para mantener el formato de opciones fijas).
_CAPTION_SETTINGS = {"max_tokens": 45, "temperature": 0.2, "variant": None}
_STRUCTURED_SETTINGS = {"max_tokens": 45, "temperature": 0.1, "variant": None}

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


def _scene_analysis_available() -> bool:
    """Comprobación barata (sin cargar el modelo) de si este módulo puede
    funcionar: dependencias opcionales instaladas. No hay ningún índice ni
    fichero que comprobar (a diferencia de geolocation.py), el modelo se
    descarga solo la primera vez vía el caché de HuggingFace.

    Comprueba también `timm` y `einops`: a diferencia de DINOv2
    (geolocation.py), que solo necesita torch/transformers/faiss, el
    código remoto de Moondream2 (`trust_remote_code=True`, ver
    requirements-vision.txt) los importa también. Antes de este chequeo,
    esta función devolvía True con solo torch/transformers instalados --
    suficiente para `estimate_locations_for_posts` (geolocalización), pero
    NO para Moondream2 -- así que un entorno que solo siguiera las
    instrucciones de `scripts/build_faiss_index.py` (que no menciona
    timm/einops) tenía la geolocalización funcionando con normalidad
    mientras esta función fallaba en silencio SIEMPRE dentro de
    `_lazy_load()` (capturado por el try/except de
    `analyze_image_content`, ver más abajo): la geolocalización parecía
    funcionar bien y el contenido visual nunca daba descripción para
    NINGUNA foto, sin ningún error visible más allá de un warning en el
    log del backend. `accelerate` y `pyvips` no se comprueban aquí a
    propósito: son igual de importables/baratos que timm/einops, pero
    `from_pretrained()` los necesita en combinaciones distintas según la
    revisión del modelo fijada en `_MODEL_REVISION` (a diferencia de
    timm/einops, que hacen falta siempre); comprobarlos aquí acoplaría
    este chequeo a la revisión concreta. Su ausencia sigue cayendo en el
    try/except de `analyze_image_content`, con el nombre de la excepción
    en el log para poder diagnosticarla."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import timm  # noqa: F401
        import einops  # noqa: F401
    except ImportError:
        return False
    return True


def _lazy_load():
    """Carga Moondream2 en memoria (una vez por proceso).

    Historial de bugs reales encontrados en producción con este modelo (no
    precauciones teóricas -- los cuatro, en orden, fueron intentos
    fallidos de solucionar el mismo síntoma: "Sin descripción visual
    disponible" en TODAS las fotos):

    1. `device_map` como STRING suelto ("cpu"/"cuda") a `from_pretrained()`
       -> `NotImplementedError: Cannot copy out of meta tensor; no data!`
    2. Quitar `device_map` pero añadir `torch_dtype` -> `RuntimeError:
       Tensor on device cpu is not on the expected device meta!`
    3. Forzar `low_cpu_mem_usage=False` junto con `torch_dtype` -> NO
       arregla nada, el error nº2 persiste igual.
    4. Quitar `torch_dtype` de `from_pretrained()` y hacer
       `.to(dtype=torch.bfloat16)` DESPUÉS, sobre el modelo ya cargado ->
       vuelve el error nº2 ("Tensor on device cpu is not on the expected
       device meta!"), esta vez disparado por el propio `.to()` posterior,
       no por `from_pretrained()`.

    Los cuatro intentos tocaban ALGO relacionado con dispositivo/dtype en
    algún punto de la carga. Investigando el código fuente real de este
    modelo (huggingface.co/vikhyatk/moondream2/blob/main/moondream.py) se
    confirma que el modelo NO usa meta device en su propio código (es un
    nn.Module normal, con pesos reales desde su propio __init__) -- así
    que el meta device tiene que estar viniendo de fuera, de cómo
    `transformers`/`accelerate` inicializan clases con
    `trust_remote_code=True` en general. Esto coincide con un patrón ya
    documentado por los propios mantenedores de Moondream2 (commit "Call
    post_init() en HfMoondream for Transformers 5 compatibility" en su
    sucesor Moondream 3) y con un issue público de compatibilidad
    (huggingface/transformers#31782, "Moondream breaks on transformers
    4.42+") -- el patrón de fondo es: la clase wrapper de HuggingFace de
    este modelo (`HfMoondream`) se escribió y probó contra una versión de
    `transformers` de la época de cada revisión, y versiones bastante más
    nuevas cambian cómo `PreTrainedModel` inicializa/coloca los parámetros
    internamente, rompiendo la compatibilidad con independencia de qué
    kwargs se le pasen desde nuestro lado.

    Por eso el intento nº5 (el actual) no toca NINGÚN kwarg de dispositivo/
    dtype -- es el ejemplo oficial exacto de la ficha del modelo para esta
    revisión, literal, con CERO añadidos nuestros -- y en su lugar ataca
    la causa por el otro lado: fijar `transformers` bastante por debajo de
    donde empiezan a aparecer estos problemas (ver requirements-vision.txt).
    Si esto tampoco funciona, el siguiente paso ya no es un kwarg distinto
    -- es probar una revisión distinta del modelo (`_MODEL_REVISION` más
    abajo), no seguir iterando aquí.

    Intento nº6, AÑADIDO DESPUÉS de que el nº5 cargase correctamente pero
    dejase el modelo inutilizable en la práctica en CPU: sin dtype/device
    explícitos, `from_pretrained()` carga los pesos con el dtype con el
    que están guardados en HuggingFace, que para este modelo es
    `bfloat16`. En GPU eso es lo correcto (los tensor cores modernos lo
    aceleran de forma nativa). En CPU de consumo -- sin instrucciones
    AVX-512 BF16, que solo traen CPUs de servidor recientes -- PyTorch
    EMULA bfloat16 por software: no "algo más lento", sino uno o dos
    órdenes de magnitud más lento (medido en producción: >3 horas para
    UNA sola foto, frente a segundos en float32 sobre la misma CPU).

    Se investigaron y descartaron dos alternativas antes de esta, con
    evidencia real, no solo teoría:
    - Cuantización dinámica int8 de PyTorch (`quantize_dynamic`):
      Moondream2 no usa `nn.Linear` para casi ninguna capa interna (qkv,
      proj, fc1, fc2 son una clase propia `LinearWeights`, con una función
      `linear()` que llama a `F.linear()` a mano) -- `quantize_dynamic`
      solo puede tocar `patch_emb`, que sí es un `nn.Linear` real. El
      resto de la red queda intacta: sin beneficio real.
    - Cuantización int4 nativa del propio modelo (`torchao`,
      `QuantizedLinear.unpack()` en el código remoto del modelo): termina
      en `torch.cuda.empty_cache()` -- pensada para GPU/CUDA, no aporta
      nada en CPU.

    La solución que sí funciona: dejar que `from_pretrained()` cargue en
    bfloat16 como siempre (intento nº5, sin tocar), y DESPUÉS convertir
    cada parámetro y buffer a float32 uno a uno (`param.data = param.data
    .float()`), nunca con `.to()` sobre el módulo completo -- eso es
    precisamente lo que disparaba el error nº2/nº4 de más arriba. La
    diferencia importa: `.to()` pasa por hooks internos de `accelerate`
    que asumen un modelo potencialmente repartido en meta device;
    reasignar `.data` en cada tensor por separado es una operación de más
    bajo nivel que nunca los toca. Solo se hace en CPU.

    Intento nº7, AÑADIDO cuando se pasó de correr esto en CPU a una GPU
    dedicada (NVIDIA GTX 1650, 4GB VRAM, arquitectura Turing/compute
    capability 7.5): en GPU no hace falta el parche de arriba tal cual
    (upcast A FLOAT32) -- ese parche existe solo por lo lenta que es la
    emulación software de bfloat16 en CPU de consumo. Pero tampoco vale
    con dejar bfloat16 tal cual (que sería lo ideal en una GPU con tensor
    cores bfloat16 nativos): esos tensor cores bfloat16 solo llegaron con
    Ampere (compute capability >= 8.0) -- Turing es 7.5, así que bfloat16
    en esta GPU también se emula por software (2-4x más lento que con
    tensor cores reales -- no tan grave como en CPU, pero tampoco la
    opción correcta aquí). float16 sí tiene aceleración nativa completa
    desde Volta (compute capability >= 7.0), así que es lo que se
    persigue en GPU.

    Para el `device_map`, se pasa la FORMA DE DICCIONARIO exacta del
    ejemplo oficial del modelo para GPU (comentada en el ejemplo oficial
    con "Uncomment to run on GPU"), NO la forma de STRING SUELTO
    (`device_map="cuda"`) que fue el intento nº1 fallido de más arriba
    (`NotImplementedError: Cannot copy out of meta tensor`) -- son dos
    invocaciones distintas de `accelerate` con comportamiento distinto: la
    de string dispara su lógica de reparto automático entre dispositivos
    (pensada para repartir un modelo grande entre varias GPUs), que es la
    que entra en conflicto con este wrapper concreto; la de diccionario
    con un solo destino ("" = todo el modelo) es una instrucción más
    simple ("todo aquí") que no necesita esa lógica de reparto. ESTA PARTE
    SÍ VERIFICADA EN PRODUCCIÓN: el modelo carga correctamente en
    `cuda:0` con esta forma.

    VERIFICADO EN PRODUCCIÓN, Y RESULTÓ SER FALSO: la idea original de
    este intento era que pasar `torch_dtype=torch.float16` junto al
    `device_map` bastaba para que el modelo quedara en float16. El log
    real mostró `dtype=torch.bfloat16` pese a ese kwarg -- ver "Intento
    nº8" (docstring de `_upcast_bfloat16_tensors`) para el porqué exacto
    y el arreglo real: `torch_dtype` de `from_pretrained()` no llega a la
    mayoría de los pesos de este modelo, así que forzar el dtype hace
    falta hacerlo después de cargar, con la misma máquina ya usada para
    el caso CPU.

    ACTUALIZACIÓN tras la primera ejecución real en esta GPU: la carga
    SÍ funciona (`device_map` en forma de diccionario, sin el error del
    intento nº1), y con el arreglo del intento nº8 el modelo queda de
    verdad en float16. Lo que SIGUE sin resolverse: el análisis de una
    foto individual sigue superando el timeout de 30s (el valor de
    entonces; ver Settings.scene_analysis_timeout_seconds en config.py,
    subido después a 60s por defecto y hecho configurable por variable de
    entorno tras confirmarse este mismo problema en producción) incluso en
    GPU -- no se sabe todavía si es porque bfloat16-a-float16 no basta para
    bajar de 30s en
    una GTX 1650, o si hay algo más de fondo (p. ej. la primera pasada de
    CUDA/cuDNN "calentando" kernels, que solo afecta a la primera foto,
    frente a un problema que afecte a todas). Si sigue pasando tras este
    arreglo, el siguiente paso es medir cuánto tarda realmente una foto
    sola (sin el timeout de por medio) para saber si hace falta subir el
    timeout, bajar `max_new_tokens`, o si hay otro cuello de botella.

    PRESUPUESTO DE VRAM (para tener a mano si esto peta por
    `torch.cuda.OutOfMemoryError` en vez de por un error de carga): los
    1.8B parámetros de Moondream2 en float16 son ~3.6GB SOLO en pesos --
    en una GPU de 4GB, eso deja muy poco margen para el contexto de
    CUDA/cuDNN (~300-500MB) y la caché KV de la generación. Concurrencia
    forzada a 1 en GPU (ver `_default_photo_analysis_concurrency` en
    config.py) para no multiplicar ese uso con varias fotos en vuelo a la
    vez. Cuantización (bitsandbytes/8-bit) NO se ha intentado a propósito:
    Moondream2 no usa `nn.Linear` para casi ninguna capa interna (ver la
    investigación de cuantización dinámica más arriba, mismo motivo) --
    `load_in_8bit` de `transformers` sustituye capas por CLASE
    (`nn.Linear` -> `bnb.nn.Linear8bitLt`), así que es muy probable que
    tampoco toque la mayoría de la red, igual que le pasó a
    `quantize_dynamic`. Si el presupuesto de VRAM de arriba no llega,
    antes que cuantización probar: bajar `max_new_tokens` de la
    generación (menos caché KV), o mantener `enable_scene_analysis` en
    CPU (`enable_scene_analysis=True` pero sin GPU visible para este
    proceso) mientras la geolocalización (DINOv2, mucho más ligera) sí usa
    la GPU -- DINOv2 ya selecciona GPU automáticamente si está disponible,
    con independencia de esto (ver geolocation.py)."""
    global _model
    if _model is not None:
        return

    from transformers import AutoModelForCausalLM
    import torch

    # Deliberadamente SIN device_map, SIN torch_dtype, SIN low_cpu_mem_usage,
    # SIN .to() posterior -- exactamente el ejemplo oficial para CPU de
    # huggingface.co/vikhyatk/moondream2 en la revisión _MODEL_REVISION
    # (ahí el device_map={"": "cuda"} aparece comentado con "# Uncomment
    # to run on GPU", es decir: para CPU, no se pasa nada de esto). Con
    # CUDA disponible, SÍ se pasa `device_map` -- ver "Intento nº7" en el
    # docstring de esta función para el porqué exacto de la forma concreta
    # que se usa (diccionario, no string).
    #
    # `torch_dtype=torch.float16` se deja puesto por si acaso (no hace
    # daño, y sí afecta a lo poco que `from_pretrained()` SÍ inicializa
    # como parámetro/buffer registrado -- p. ej. `patch_emb`), pero NO es
    # lo que deja el modelo en float16: eso lo hace
    # `_upcast_bfloat16_tensors()` más abajo. Ver "Intento nº8" en su
    # docstring -- este kwarg por sí solo no llega a la mayoría de los
    # pesos reales del modelo (viven en dataclasses propias del código
    # remoto, no en `nn.Parameter`), así que confiar solo en él dejaba el
    # modelo cargado en bfloat16 pese a pedir float16 explícitamente
    # (visto en producción: log "Moondream2 cargado: ... dtype=torch.bfloat16").
    #
    # local_files_only=True primero: aunque `revision` esté fijada a un
    # commit concreto (no "main"), `from_pretrained()` sigue haciendo por
    # defecto una petición HEAD a huggingface.co para verificar la caché
    # local -- innecesaria si ya está descargado y la revisión es fija, y
    # un punto de fallo real si la conexión es lenta/inestable (visto en
    # producción: `ReadTimeoutError` de 10s reintentando 5 veces,
    # bloqueando esta foto -- y por tanto, indirectamente, también DINOv2
    # para esa misma foto, ver `_maybe_analyze_content` en geolocation.py
    # -- durante minuto y medio o más antes de rendirse). Si ya está en
    # caché (backend/data/hf_cache/, ver docker-compose.yml), esto la usa
    # directamente sin ningún acceso a red. Si NO está en caché todavía
    # (primera vez), `local_files_only=True` falla rápido y limpio, y se
    # reintenta sin él para permitir la descarga inicial.
    _gpu_kwargs = {"torch_dtype": torch.float16, "device_map": {"": "cuda"}} if torch.cuda.is_available() else {}
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME, revision=_MODEL_REVISION, trust_remote_code=True, local_files_only=True, **_gpu_kwargs
        )
    except Exception:
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME, revision=_MODEL_REVISION, trust_remote_code=True, **_gpu_kwargs
        )

    # BUG ENCONTRADO en la sesión del "Intento nº7" (GPU): si el post-
    # procesado de abajo (upcast/parche de dtype) falla a medias, `_model`
    # ya está asignado (no None) -- la comprobación `if _model is not
    # None: return` del principio de esta función haría que un reintento
    # posterior (la siguiente foto) se quedara con este modelo roto en vez
    # de volver a intentar la carga completa. Se envuelve el post-
    # procesado y, si falla, se deja `_model = None` para que el próximo
    # intento arranque de cero.
    try:
        _upcast_bfloat16_tensors()
        _patch_vision_input_dtype()

        # Log explícito de dónde y en qué dtype ha quedado el modelo: sin
        # esto, no hay ninguna forma de confirmar desde los logs si cargó
        # de verdad en GPU o cayó en silencio a CPU --
        # `torch.cuda.is_available()` puede dar False por motivos que no
        # lanzan ninguna excepción (falta la reserva de GPU en
        # docker-compose.yml, el driver de Windows no tiene soporte WSL2,
        # "GPU support" desactivado en Docker Desktop...), y en ese caso
        # `_gpu_kwargs` queda vacío y todo el código de arriba sigue
        # funcionando igual, solo que en CPU -- sin ningún error, solo más
        # lento. Un timeout de 30s superado no distingue por sí solo entre
        # "la GPU no está siendo usada" y "la GPU sí se usa pero esta foto
        # en concreto ha tardado más de la cuenta" -- este log sí lo
        # distingue.
        _actual_device = next(_model.parameters()).device
        _actual_dtype = next(_model.parameters()).dtype
        logger.info(
            "Moondream2 cargado: device=%s dtype=%s (torch.cuda.is_available()=%s)",
            _actual_device,
            _actual_dtype,
            torch.cuda.is_available(),
        )
    except Exception:
        _model = None
        raise


def _patch_vision_input_dtype():
    """La imagen de entrada llega en bfloat16 pase lo que pase: la
    función `prepare_crops()` del código remoto de Moondream2
    (`vision.py`) fija `dtype=torch.bfloat16` a fuego, sin mirar en qué
    dtype está el modelo -- así que si el modelo se ha quedado en otro
    dtype (float32 tras `_upcast_bfloat16_tensors()` en CPU, o float16
    en GPU -- ver "Intento nº8" en `_lazy_load()`), la primera capa
    (`patch_emb`) recibiría una entrada en bfloat16 y fallaría con un
    `RuntimeError: expected scalar type X but found BFloat16` (X = Float
    en CPU, Half en GPU).

    Verificado en pruebas manuales que parchear `prepare_crops()` por
    nombre de módulo NO tiene efecto (no está claro por qué -- posible
    referencia ya vinculada de otra forma en el código remoto que no se
    ha investigado más a fondo); lo que sí funciona, comprobado, es
    interceptar directamente el `forward` del propio submódulo
    `patch_emb` -- ahí no importa cómo le haya llegado el dato, se fuerza
    el cast justo antes de usarlo. Localizado por búsqueda en
    `named_modules()` en vez de por ruta de atributo fija (p. ej.
    `_model.model.vision.patch_emb`): la clase wrapper de nivel superior
    (`HfMoondream`) no expone esa ruta como atributo estable, y esta
    forma es además más robusta ante cambios de revisión del modelo.

    CORREGIDO al añadir el "Intento nº7" (GPU): esta función comprobaba
    `torch.cuda.is_available()` para decidir si hacía falta el parche,
    asumiendo que "hay GPU" implicaba "el modelo se ha quedado en
    bfloat16" (que coincide con el dtype fijo que usa `prepare_crops()`,
    así que no habría descuadre). Eso dejó de ser cierto en cuanto
    `_lazy_load()` empezó a pedir `torch_dtype=torch.float16` en GPU
    (Turing no acelera bfloat16 por hardware, ver ahí) -- con el modelo en
    float16 y la entrada en bfloat16 fijo, el descuadre de dtype vuelve a
    aparecer, solo que ahora también en GPU. La condición correcta es
    comprobar el dtype REAL del modelo, no el dispositivo."""
    import torch

    if next(_model.parameters()).dtype == torch.bfloat16:
        return  # el modelo se ha quedado en bfloat16 (mismo dtype que la entrada fija de prepare_crops()) -- no hay descuadre que parchear

    patch_emb = None
    for name, module in _model.named_modules():
        if name.endswith("patch_emb"):
            patch_emb = module
            break

    if patch_emb is None:
        logger.warning(
            "No se encontro el submodulo patch_emb de Moondream2 -- no se puede "
            "aplicar el parche de dtype de entrada, el analisis de contenido "
            "probablemente fallara para todas las fotos."
        )
        return

    original_forward = patch_emb.forward
    target_dtype = next(_model.parameters()).dtype  # float32 en CPU (tras el upcast), float16 en GPU (ver "Intento nº7")

    def _forward_matching_dtype(x):
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        return original_forward(x)

    patch_emb.forward = _forward_matching_dtype


def _upcast_registered_tensors(model, target_dtype) -> None:
    """Reasigna a `target_dtype` los parámetros y buffers que PyTorch SÍ
    registra formalmente (`nn.Parameter`, buffers vía `register_buffer`) y
    que estén en bfloat16. Ver el docstring de `_upcast_bfloat16_tensors`
    para el porqué de hacerlo así (reasignar `.data`) en vez de
    `.to(dtype=...)`."""
    import torch

    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(target_dtype)
    for buf in model.buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(target_dtype)


def _upcast_dataclass_attr(attr_val, seen_ids: set, target_dtype) -> None:
    """Si `attr_val` es una dataclass no vista todavía (p.ej.
    `LinearWeights`/`LayerNormWeights`, ver docstring de
    `_upcast_bfloat16_tensors`), reasigna a `target_dtype` cualquiera de
    sus campos que sea un tensor en bfloat16."""
    import dataclasses
    import torch

    if not dataclasses.is_dataclass(attr_val) or id(attr_val) in seen_ids:
        return
    seen_ids.add(id(attr_val))
    for field in dataclasses.fields(attr_val):
        value = getattr(attr_val, field.name)
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            setattr(attr_val, field.name, value.to(target_dtype))


def _upcast_unregistered_dataclass_tensors(model, target_dtype) -> None:
    """Recorre a mano los atributos de cada submódulo buscando dataclasses
    colgadas como atributo normal -- no registradas por PyTorch, ver
    docstring de `_upcast_bfloat16_tensors` -- y sube sus tensores en
    bfloat16 a `target_dtype`."""
    seen_ids: set = set()
    for module in model.modules():
        for attr_name, attr_val in vars(module).items():
            if attr_name.startswith("_"):
                continue  # _parameters, _buffers, _modules, etc. -- ya cubiertos en _upcast_registered_tensors
            _upcast_dataclass_attr(attr_val, seen_ids, target_dtype)


def _upcast_bfloat16_tensors():
    """Convierte el modelo de bfloat16 (dtype con el que se descarga) a
    float32 en CPU, o a float16 en GPU. Ver el bloque "Intento nº6" en el
    docstring de `_lazy_load()` para el porqué del caso CPU -- en resumen:
    bfloat16 en CPU de consumo se emula por software y es órdenes de
    magnitud más lento que float32 nativo. Para el caso GPU, ver "Intento
    nº8" ahí -- en Turing (compute capability 7.5), bfloat16 tampoco tiene
    tensor cores nativos (solo desde Ampere), así que conviene float16 en
    su lugar (nativo desde Volta).

    BUG ENCONTRADO en la sesión del "Intento nº8" (renombrada esta
    función, antes `_upcast_to_float32_if_cpu`): esta función se saltaba
    ENTERA en GPU (`if torch.cuda.is_available(): return`), asumiendo que
    pasar `torch_dtype=torch.float16` a `from_pretrained()` en
    `_lazy_load()` (el "Intento nº7") ya dejaba el modelo en float16. En
    la práctica, el modelo cargó en GPU con
    `dtype=torch.bfloat16` (confirmado por el log de "Moondream2 cargado:
    ... dtype=torch.bfloat16" pese al `torch_dtype=torch.float16` pasado)
    -- `torch_dtype` de `from_pretrained()` solo afecta a lo que PyTorch
    registra formalmente como parámetro/buffer, y la mayoría de los pesos
    de este modelo NO lo son (ver el párrafo de más abajo sobre
    `LinearWeights`/`LayerNormWeights`) -- exactamente el mismo motivo por
    el que ya hacía falta `_upcast_unregistered_dataclass_tensors` para
    el caso CPU. Ahora esta función se ejecuta SIEMPRE (CPU o GPU), con
    el dtype destino según el dispositivo, reutilizando la misma
    maquinaria ya probada para CPU en vez de depender de un kwarg que no
    llega a la mayoría de los tensores.

    Deliberadamente NO usa `_model.to(dtype=...)`: esa llamada es la que
    causó el error "Tensor on device cpu is not on the expected device
    meta!" documentado como intento nº4. En su lugar, reasigna `.data` de
    cada parámetro y buffer por separado (ver `_upcast_registered_tensors`)
    -- una operación de más bajo nivel que no pasa por los hooks de
    `accelerate` que asumen un modelo potencialmente repartido en meta
    device.

    IMPORTANTE, descubierto investigando por qué la cuantización int8 de
    PyTorch fallaba a medias (ver commit que añade este bloque): la
    mayoría de las capas internas de Moondream2 (atención, MLP) NO son
    `nn.Linear`/`nn.LayerNorm` reales -- son una `@dataclass` propia del
    modelo (`LinearWeights`/`LayerNormWeights` en el código remoto) con
    tensores sueltos como atributos, colgada como atributo normal de un
    `nn.Module`. `model.parameters()`/`model.buffers()` SOLO recorren lo
    que PyTorch registra formalmente -- una dataclass colgada como
    atributo corriente NO aparece ahí. Por eso, además de
    `_upcast_registered_tensors`, hace falta
    `_upcast_unregistered_dataclass_tensors`: recorre a mano,
    recursivamente, los atributos de cada submódulo buscando cualquier
    objeto con campos de tipo `torch.Tensor` en bfloat16 (duck typing
    sobre `dataclass`, sin importar la clase concreta del código remoto
    -- más robusto ante cambios de revisión del modelo que importar
    `LinearWeights` directamente)."""
    import torch

    target_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16

    with torch.no_grad():
        _upcast_registered_tensors(_model, target_dtype)
        _upcast_unregistered_dataclass_tensors(_model, target_dtype)


def analyze_image_content(image) -> tuple[list[InferredAttribute], bool, str | None, str | None]:
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

    Internamente hace DOS llamadas a `_model.query()` -- una con
    `_CAPTION_QUERY` (texto libre, sin plantilla que copiar) y otra con
    `_STRUCTURED_QUERY` (PERSONAS/AFICION/PAREJA/TEXTO_VISIBLE, formato
    fijo) -- en vez de una sola combinada, porque mezclar un campo de
    texto libre con campos de opciones fijas en el mismo prompt hacía que
    Moondream2 copiara literalmente el ejemplo de texto libre en vez de
    describir la imagen real (ver nota en `_CAPTION_QUERY`). Ambas
    llamadas reutilizan el mismo `_model.encode_image()` (sobre la copia
    ya redimensionada) para no pagar el coste de codificar la foto dos
    veces.

    `descripcion_cruda` (ver `_build_clean_summary`) se RECONSTRUYE a
    partir de los valores YA PARSEADOS de `structured` -- ya NO es el
    texto crudo del modelo tal cual. Dos motivos: (a) el modelo, tras
    responder bien, a veces sigue generando y empieza a copiar fragmentos
    de la propia explicación del prompt (visto en producción); reconstruir
    desde valores parseados descarta esa cola sin depender de acertar el
    `max_tokens` exacto cada vez; (b) solo interesa mostrar señales
    POSITIVAS/informativas -- los valores negativos por defecto (afición
    ninguna, sin pareja, sin texto visible) no aportan nada y solo
    acumulan líneas vacías si se muestran siempre. None si no hubo NADA
    informativo que mostrar.

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
        # Causa más habitual: no se ha instalado requirements-vision.txt
        # completo. En particular, construir el índice FAISS (ver
        # scripts/build_faiss_index.py) NO instala timm/einops -- son
        # deps exclusivas de Moondream2, no de DINOv2 -- así que un
        # entorno con la geolocalización funcionando puede seguir sin
        # tener esto instalado.
        logger.warning(
            "Análisis de contenido visual no disponible: falta torch, transformers, "
            "timm o einops (ver requirements-vision.txt; 'pip install timm einops' "
            "si ya tienes torch/transformers instalados para la geolocalización)"
        )
        return [], False, None, None

    try:
        _lazy_load()
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
        # Codificar la imagen (ya redimensionada) UNA vez y reutilizarla
        # para las dos preguntas (ver docs.moondream.ai/advanced/transformers,
        # "If you're planning to run multiple inferences on the same
        # image, you can pre-encode it once and reuse the encoding") --
        # evita que el encoder de visión procese la misma foto dos veces,
        # coste que antes se pagaba por CADA llamada a .query() si
        # hiciéramos dos llamadas ingenuas con la imagen sin codificar.
        encoded = _model.encode_image(resized)
        caption = _model.query(encoded, _CAPTION_QUERY, settings=_CAPTION_SETTINGS)["answer"].strip().rstrip(".")
        structured = _model.query(encoded, _STRUCTURED_QUERY, settings=_STRUCTURED_SETTINGS)["answer"].strip()
    except Exception as exc:
        # Motivo típico si torch/transformers/timm/einops SÍ están
        # instalados (ver _scene_analysis_available arriba): falta
        # accelerate o pyvips (a diferencia de timm/einops, no se
        # comprueban en el chequeo barato de arriba porque from_pretrained
        # los necesita en combinaciones distintas según la revisión del
        # modelo -- ver requirements-vision.txt) o un fallo de red al
        # descargar el modelo la primera vez. Se loguea para poder
        # diagnosticarlo sin tener que quitar el try/except (este módulo
        # es best-effort y no debe abortar el análisis de las demás
        # fotos).
        logger.warning(
            "Análisis de contenido visual falló para una foto (%s): %s",
            type(exc).__name__,
            exc,
        )
        # NOTA histórica: un `NotImplementedError` con el mensaje "Cannot
        # copy out of meta tensor" aquí fue en su momento un bug real de
        # este módulo (device_map pasado como string a from_pretrained,
        # ver `_lazy_load`), no un problema de entorno -- se afectaba al
        # 100% de las fotos, en cualquier máquina, con las dependencias
        # bien instaladas. Ya está corregido; si reaparece tras cambiar
        # `_MODEL_REVISION` o la versión de `transformers`/`accelerate`,
        # revisar primero cómo se está pasando `device_map`.
        return [], False, None, None

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

    inferencias = _parse_inferences(structured)
    descripcion_cruda = _build_clean_summary(personas, aficion_raw, indicio_pareja, texto_visible)

    return inferencias, indicio_pareja, descripcion_cruda, descripcion_general


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
    cautela de atribución no aplica)."""
    match = _AFICION_RE.search(answer)
    if match is None:
        return None
    valor = match.group(1).strip().rstrip(".")
    if not valor or valor.lower() in ("ninguno", "ninguna", "none", "n/a"):
        return None
    return valor


def _parse_texto_visible(answer: str) -> str | None:
    """Extrae el valor de la línea TEXTO_VISIBLE (texto legible en la
    propia foto: carteles, escaparates, matrículas...). None si no se
    pudo parsear o si el modelo respondió 'ninguno' (no hay texto
    legible). El prompt (ver _STRUCTURED_QUERY) ya le prohíbe explícitamente
    devolver el nombre propio de una persona aquí -- este parseo no repite
    ese filtro (no hay forma fiable de detectar "esto es un nombre propio"
    solo con regex), se confía en la instrucción del prompt, igual que ya
    se hace con el resto de restricciones éticas/legales del módulo (ver
    docstring de cabecera)."""
    match = _TEXTO_VISIBLE_RE.search(answer)
    if match is None:
        return None
    valor = match.group(1).strip().rstrip(".")
    if not valor or valor.lower() in ("ninguno", "ninguna", "none", "n/a"):
        return None
    return valor


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
    vacías si se muestran siempre. PERSONAS es la excepción: se muestra
    siempre que se pudo parsear (incluido 'ninguna'), porque da contexto
    básico en una sola línea -- no es una "señal negativa" en el mismo
    sentido que las otras tres.

    La descripción general (frase libre, `descripcion_general` en
    `analyze_image_content`) NO se repite aquí -- el frontend ya la
    muestra aparte, destacada, justo encima de este bloque.

    None si no hay NADA que mostrar (PERSONAS no se pudo parsear Y las
    otras tres son negativas) -- caso raro pero posible si el modelo se
    desvió del formato por completo."""
    lines = []
    if personas is not None:
        lines.append(f"Personas en la foto: {personas}")
    if aficion_raw:
        lines.append(f"Posible afición o interés: {aficion_raw}")
    if indicio_pareja:
        lines.append("Indicio de contexto de pareja: sí")
    if texto_visible:
        lines.append(f"Texto visible: {texto_visible}")
    return "\n".join(lines) if lines else None
