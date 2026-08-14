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

from app.models.schemas import InferredAttribute

logger = logging.getLogger(__name__)

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
    capability 7.5): en GPU NO hace falta el parche de arriba (upcast a
    float32) -- ese parche existe solo por lo lenta que es la emulación
    software de bfloat16 en CPU de consumo. Pero tampoco vale con dejar
    bfloat16 tal cual (que sería lo ideal en una GPU con tensor cores
    bfloat16 nativos): esos tensor cores bfloat16 solo llegaron con Ampere
    (compute capability >= 8.0) -- Turing es 7.5, así que bfloat16 en esta
    GPU también se emula por software (2-4x más lento que con tensor cores
    reales -- no tan grave como en CPU, pero tampoco la opción correcta
    aquí). float16 sí tiene aceleración nativa completa desde Volta
    (compute capability >= 7.0), así que es lo que se usa en GPU: se pasa
    `torch_dtype=torch.float16` Y `device_map={"": "cuda"}` juntos a
    `from_pretrained()` -- la FORMA DE DICCIONARIO exacta del ejemplo
    oficial del modelo para GPU (comentada en el ejemplo oficial con
    "Uncomment to run on GPU"), NO la forma de STRING SUELTO
    (`device_map="cuda"`) que fue el intento nº1 fallido de más arriba
    (`NotImplementedError: Cannot copy out of meta tensor`) -- son dos
    invocaciones distintas de `accelerate` con comportamiento distinto: la
    de string dispara su lógica de reparto automático entre dispositivos
    (pensada para repartir un modelo grande entre varias GPUs), que es la
    que entra en conflicto con este wrapper concreto; la de diccionario
    con un solo destino ("" = todo el modelo) es una instrucción más
    simple ("todo aquí") que no necesita esa lógica de reparto.

    SIN VERIFICAR TODAVÍA contra el modelo real en esta GPU concreta (bien
    razonado por lo que documenta la propia ficha del modelo + lo que ya
    se sabe de intentos anteriores en este mismo fichero, no probado en
    producción) -- si esto falla, sigue el mismo patrón de depuración que
    los intentos nº1-6: pega aquí el traceback exacto y seguimos desde
    ahí, no se vuelve a intentar a ciegas.

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
    # CUDA disponible, SÍ se pasan esos dos kwargs -- ver "Intento nº7" en
    # el docstring de esta función para el porqué exacto de la forma
    # concreta que se usa (diccionario, no string) y del dtype elegido
    # (float16, no bfloat16) para esta GPU en particular.
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
        _upcast_to_float32_if_cpu()
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
    dtype (float32 tras `_upcast_to_float32_if_cpu()` en CPU, o float16
    en GPU -- ver "Intento nº7" en `_lazy_load()`), la primera capa
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


def _upcast_registered_tensors(model) -> None:
    """Reasigna a float32 los parámetros y buffers que PyTorch SÍ registra
    formalmente (`nn.Parameter`, buffers vía `register_buffer`) y que
    estén en bfloat16. Ver el docstring de `_upcast_to_float32_if_cpu`
    para el porqué de hacerlo así (reasignar `.data`) en vez de
    `.to(dtype=...)`."""
    import torch

    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.float()
    for buf in model.buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.float()


def _upcast_dataclass_attr(attr_val, seen_ids: set) -> None:
    """Si `attr_val` es una dataclass no vista todavía (p.ej.
    `LinearWeights`/`LayerNormWeights`, ver docstring de
    `_upcast_to_float32_if_cpu`), reasigna a float32 cualquiera de sus
    campos que sea un tensor en bfloat16."""
    import dataclasses
    import torch

    if not dataclasses.is_dataclass(attr_val) or id(attr_val) in seen_ids:
        return
    seen_ids.add(id(attr_val))
    for field in dataclasses.fields(attr_val):
        value = getattr(attr_val, field.name)
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            setattr(attr_val, field.name, value.float())


def _upcast_unregistered_dataclass_tensors(model) -> None:
    """Recorre a mano los atributos de cada submódulo buscando dataclasses
    colgadas como atributo normal -- no registradas por PyTorch, ver
    docstring de `_upcast_to_float32_if_cpu` -- y sube sus tensores en
    bfloat16 a float32."""
    seen_ids: set = set()
    for module in model.modules():
        for attr_name, attr_val in vars(module).items():
            if attr_name.startswith("_"):
                continue  # _parameters, _buffers, _modules, etc. -- ya cubiertos en _upcast_registered_tensors
            _upcast_dataclass_attr(attr_val, seen_ids)


def _upcast_to_float32_if_cpu():
    """Convierte el modelo de bfloat16 (dtype con el que se descarga) a
    float32, SOLO si no hay GPU disponible. Ver el bloque "Intento nº6"
    en el docstring de `_lazy_load()` para el porqué -- en resumen:
    bfloat16 en CPU de consumo se emula por software y es órdenes de
    magnitud más lento que float32 nativo. Con GPU disponible, esta
    función no actúa -- `_lazy_load()` ya evita bfloat16 desde el propio
    `from_pretrained()` (ver "Intento nº7" ahí: en Turing, compute
    capability 7.5, bfloat16 tampoco tiene tensor cores nativos -- solo
    desde Ampere -- así que se pide float16 en su lugar, que sí los
    tiene desde Volta).

    Deliberadamente NO usa `_model.to(dtype=torch.float32)`: esa llamada
    es la que causó el error "Tensor on device cpu is not on the expected
    device meta!" documentado como intento nº4. En su lugar, reasigna
    `.data` de cada parámetro y buffer por separado (ver
    `_upcast_registered_tensors`) -- una operación de más bajo nivel que
    no pasa por los hooks de `accelerate` que asumen un modelo
    potencialmente repartido en meta device.

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

    if torch.cuda.is_available():
        return

    with torch.no_grad():
        _upcast_registered_tensors(_model)
        _upcast_unregistered_dataclass_tensors(_model)


def analyze_image_content(image) -> tuple[list[InferredAttribute], bool, str | None]:
    """Devuelve (inferencias_visuales, indicio_pareja, descripcion) para UNA
    foto ya decodificada (PIL.Image, la misma que usa geolocation.py para el
    embedding de DINOv2 -- no se descarga ni decodifica de nuevo). SÍNCRONA
    y con trabajo de CPU real (como `estimate_location_from_image`): quien
    llama debe envolverla en `asyncio.to_thread` para no bloquear el event
    loop, ver geolocation.py.

    `descripcion` es la respuesta CRUDA del modelo (las tres líneas
    PERSONAS/AFICION/PAREJA), no una redacción libre -- se devuelve tal
    cual en vez de reformularla porque: (a) no hace falta un prompt ni un
    parseo nuevos, y (b) es más transparente para quien vea el informe
    (mismo criterio de "mostrar la evidencia real" que el resto del
    proyecto) que una paráfrasis que podría no ser fiel. None si el modelo
    no está disponible o la inferencia falla -- igual que los otros dos
    valores devueltos.

    `evidence` de cada InferredAttribute se deja vacío deliberadamente --
    quien llama (geolocation.py) rellena el permalink de la publicación,
    que esta función no conoce.

    Nunca lanza: cualquier fallo (dependencias no instaladas, modelo no
    descargable, respuesta con formato inesperado) se trata como "esta
    foto no aportó nada" y devuelve ([], False, None), sin abortar el
    análisis de las demás fotos."""
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
        return [], False, None

    try:
        _lazy_load()
        answer = _model.query(image, _QUERY)["answer"]
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
        return [], False, None

    return _parse_inferences(answer), _parse_pareja(answer), answer.strip()


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
