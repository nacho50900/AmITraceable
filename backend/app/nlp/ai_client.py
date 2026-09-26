"""
Cliente de IA LOCAL, exponiendo una única función `call_ai_json()` usada
por los tres módulos que necesitan razonamiento/extracción estructurada
con un LLM (ai_attribute_extraction.py, ai_analysis.py,
landmark_resolution.py) -- ninguno de los tres habla nunca directamente
con el modelo, igual que en el diseño anterior.

Por qué existe (sustituye al dispatch HTTP a Mistral/Google Gemini de este
mismo módulo -- ver `git log` de este fichero para el diseño anterior
completo, incluido el throttle por proveedor y el reintento en 429):
depender de un proveedor externo de pago/free-tier resultó frágil durante
el TFG (Mistral cambió su plan gratuito en septiembre de 2026 a mitad de
desarrollo, ver ADR-45) y, más importante, siempre implicó enviar a un
tercero datos personales inferidos de un usuario real (ubicación,
ocupación, edad...) -- exactamente lo que este proyecto pretende medir
como riesgo de exposición en OTROS. Un modelo local elimina el problema de
raíz: nunca hay una transferencia de datos, ni siquiera a un proveedor
europeo, porque nada sale del proceso.

Modelo elegido: Qwen3.5-4B (Apache-2.0), cuantizado a Q4_K_M -- se carga
DIRECTAMENTE ya cuantizado desde Hugging Face vía
`Llama.from_pretrained()` (descarga + cachea en
`~/.cache/huggingface/hub` automáticamente, sin pasos manuales), no se
cuantiza nada en este servidor: `unsloth/Qwen3.5-4B-GGUF` ya publica un
`Qwen3.5-4B-Q4_K_M.gguf` listo para usar (unsloth es una fuente de
confianza habitual para GGUF cuantizados, mismo criterio que si se usara
para Moondream2). Q4_K_M en vez de Q8_0 (el que usaba Moondream2) porque
aquí SÍ hay presión real de VRAM: 4.5B parámetros en Q8_0 (~4.5GB) no
caben en los 4GB de la GTX 1650 de despliegue; en Q4_K_M (~2.7GB, tamaño
real del fichero de unsloth) el modelo deja margen para el KV cache.

DESDE ADR-50: este módulo TAMBIÉN carga el proyector de visión (`mmproj`,
ver `settings.qwen_mmproj_filename`) y es quien de verdad hace el
análisis visual de fotos -- `app/vision/scene_analysis.py` (antes
Moondream2) ya NO carga ningún modelo propio, reutiliza este mismo
`_model` a través de `get_model()`/`get_lock()`/`ensure_loaded()` más
abajo. Un solo modelo multimodal (~2.7GB) sustituye a los dos que había
antes (Moondream2 ~1.8GB + este módulo solo-texto ~2.7GB) -- el ahorro de
VRAM que motivó el cambio. Backend: `MTMDChatHandler`, el handler
GENÉRICO de `llama-cpp-python` (lee la plantilla de chat embebida en el
propio GGUF, no hace falta un handler específico por modelo) -- a
diferencia de la generación anterior de modelos de visión de Qwen
(Qwen3-VL), que sí necesitaba un handler dedicado (`Qwen3VLChatHandler`,
solo en forks no oficiales, ver ADR-49 histórico): Qwen3.5 usa una
arquitectura "early fusion" nativa distinta, soportada en `llama.cpp`
mainline vía `mtmd` sin necesitar ese fork. SIN VERIFICAR EN GPU REAL
todavía (mismo motivo de siempre en este historial: sin GPU en el entorno
donde se escribió esto) -- si `MTMDChatHandler` no carga bien este GGUF
en la práctica, `qwen_mmproj_filename` se puede dejar vacío para volver
al comportamiento anterior (solo texto, sin visión unificada) sin tocar
código, y `app/vision/scene_analysis.py` degrada solo a "no disponible".

(Versión anterior de este módulo: cuantizaba un GGUF F16/BF16 propio con
`llama-quantize`, mismo patrón que usaba `scene_analysis.py` para
Moondream2 -- se simplificó porque un Q4_K_M ya publicado hace ese paso
innecesario: menos dependencias, menos descarga -2.7GB en vez de ~9GB en
F16- y menos puntos de fallo. Si `unsloth/Qwen3.5-4B-GGUF` dejase de estar
disponible o cambiase de nombre de fichero, hay alternativas equivalentes
en `bartowski/Qwen_Qwen3.5-4B-GGUF` y `lmstudio-community/Qwen3.5-4B-GGUF`.)

Repo/fichero GGUF configurables vía `settings.qwen_gguf_repo_id`/
`qwen_gguf_filename`/`qwen_mmproj_filename` (ver app/config.py, valores
por defecto ya apuntando a `unsloth/Qwen3.5-4B-GGUF`) -- por si en el
futuro conviene cambiar de cuantización (Q5_K_M/Q6_K si sobra VRAM,
Q3_K_M si hace falta apretar más) sin tocar código. Con `qwen_gguf_repo_id`
vacío, `_qwen_available()` devuelve False y este módulo se comporta
exactamente igual que si `llama_cpp` no estuviera instalado -- "no
disponible", sin excepción que rompa el resto del pipeline (mismo
criterio best-effort que scene_analysis.py).
"""
import asyncio
import json
import logging
import os
import threading

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_loaded_model_name: str | None = None
_actual_device: str | None = None


class AIHTTPError(Exception):
    """El modelo local SÍ generó una respuesta, pero no se pudo
    interpretar como el JSON esperado -- nombre y forma (status_code,
    body) heredados del diseño anterior por HTTP (ver docstring del
    módulo) para que ai_attribute_extraction.py, ai_analysis.py y
    landmark_resolution.py no necesiten ningún cambio: `status_code`
    siempre vale 200 aquí (nunca hubo una petición HTTP real que pudiera
    fallar con otro código), `body` lleva el detalle del fallo de parseo."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:300]}")


class AIRequestError(Exception):
    """El modelo local no se pudo cargar o la inferencia falló antes de
    producir ninguna respuesta (dependencia no instalada, GGUF no
    configurado/descargable, excepción de llama.cpp durante la
    generación...) -- nombre heredado del diseño anterior ("fallo de red
    antes de recibir respuesta"), ahora "fallo de inferencia antes de
    recibir respuesta"."""


def get_model_variant() -> str | None:
    """Igual que antes de ADR-50 (`scene_analysis.get_model_variant()`
    delega en esto ahora, en vez de tener su propio Moondream2 cargado) --
    para el log de rendimiento (ver app/log/performance_log.py). `None` si
    el modelo todavía no se ha cargado en este proceso."""
    return _loaded_model_name


def get_device() -> str | None:
    """Dispositivo en el que se PIDIÓ offload de capas (`n_gpu_layers`) en
    la última carga -- "cuda" o "cpu", `None` si no se ha cargado
    todavía. Mismo matiz que tenía `scene_analysis.get_device()`: esto es
    lo que se PIDIÓ, no una confirmación real de que `llama.cpp` lo
    consiguiera -- ver el historial de ese módulo (ya obsoleto para
    visión, mantenido por su valor histórico) sobre por qué no se puede
    confirmar con certeza desde aquí sin `verbose=True`."""
    return _actual_device


def get_model():
    """El objeto `Llama` ya cargado (o `None` si aún no se cargó) --
    expuesto para que `app/vision/scene_analysis.py` pueda hacer sus
    propias llamadas a `create_chat_completion()` con contenido de imagen
    (ver `ensure_loaded()` y `get_lock()`, los tres se usan siempre
    juntos: cargar, obtener el lock, obtener el modelo)."""
    return _model


def get_lock() -> threading.Lock:
    """El mismo `Lock` que usa `call_ai_json()` internamente -- `Llama` no
    tolera llamadas concurrentes desde varios hilos sobre el mismo objeto
    (confirmado en producción con Moondream2, mismo motivo aplica aquí:
    un único modelo compartido significa una única cola real, tanto para
    llamadas de texto (ai_attribute_extraction.py, ai_analysis.py,
    landmark_resolution.py) como de imagen (scene_analysis.py) -- ANTES
    de ADR-50 cada modelo tenía su propio lock independiente y podían
    correr en paralelo; ahora comparten uno solo, así que una llamada de
    imagen y una de texto NUNCA se solapan entre sí. Trade-off aceptado a
    cambio del ahorro de VRAM de un solo modelo -- SIN MEDIR el impacto
    real en tiempo total del pipeline."""
    return _model_lock


def ensure_loaded() -> None:
    """Wrapper público de `_lazy_load()` para que otros módulos
    (`scene_analysis.py`) puedan forzar la carga sin depender de un
    nombre "privado" -- no hace nada si ya está cargado, igual que
    `_lazy_load()`."""
    _lazy_load()


def vision_available() -> bool:
    """Como `_qwen_available()`, pero exige ADEMÁS que
    `qwen_mmproj_filename` esté configurado -- usado por
    `scene_analysis._scene_analysis_available()`. Con `qwen_mmproj_filename`
    vacío (ver app/config.py), este módulo sigue sirviendo texto con
    normalidad, solo el análisis visual queda "no disponible"."""
    return _qwen_available() and bool(settings.qwen_mmproj_filename)


def _qwen_available() -> bool:
    """Comprobación barata (sin cargar el modelo): dependencia opcional
    instalada Y hay un repo GGUF configurado."""
    if not settings.qwen_gguf_repo_id or not settings.qwen_gguf_filename:
        return False
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _lazy_load():
    """Carga perezosa de Qwen3.5-4B vía llama-cpp-python -- no hace nada
    si ya está cargado. `Llama.from_pretrained()` descarga el GGUF de
    `settings.qwen_gguf_repo_id`/`qwen_gguf_filename` (ya cuantizado, ver
    docstring del módulo) y lo cachea en `~/.cache/huggingface/hub` la
    primera vez -- las siguientes cargas son solo lectura de disco.

    Desde ADR-50, si `settings.qwen_mmproj_filename` está configurado (por
    defecto sí, ver app/config.py) también se carga el proyector de
    visión (`MTMDChatHandler`, handler GENÉRICO de llama-cpp-python que
    lee la plantilla de chat embebida en el GGUF -- no hace falta un
    handler específico para Qwen3.5, ver docstring del módulo) y el
    modelo resultante sirve TANTO llamadas de texto (`call_ai_json`) como
    de imagen (`app/vision/scene_analysis.py`, vía `get_model()`). Con
    `qwen_mmproj_filename` vacío, se carga solo texto (comportamiento
    anterior a ADR-50)."""
    global _model, _loaded_model_name, _actual_device
    if _model is not None:
        return

    from llama_cpp import Llama

    import torch

    n_gpu_layers = -1 if torch.cuda.is_available() else 0
    _actual_device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {
        "repo_id": settings.qwen_gguf_repo_id,
        "filename": settings.qwen_gguf_filename,
        "n_gpu_layers": n_gpu_layers,
        # BUG real, confirmado en producción (24/9): 4096 (el valor
        # original de este módulo) es demasiado pequeño -- un informe
        # completo (ai_analysis.py le manda el JSON entero del informe,
        # con population_narrowing/recommendations/attributes incluidos)
        # ya ronda 8000 tokens él solo, sin contar el prompt de sistema ni
        # el margen para la respuesta. Asunción incorrecta de partida:
        # "informes/perfiles de un único usuario, no documentos largos"
        # (ver versión anterior de este comentario) -- el informe
        # SERIALIZADO A JSON no es corto ni de lejos. Qwen3.5 soporta
        # hasta 262144 de contexto (ver la ficha del modelo en Hugging
        # Face), así que hay margen de sobra -- 32768 por defecto,
        # sobreescribible con QWEN_N_CTX si algún informe muy grande
        # (muchas fotos analizadas) lo siguiera superando, o si hay que
        # bajarlo por presión de VRAM (el KV cache crece con n_ctx; con
        # GQA -- que Qwen3.5 usa -- el coste por token es bajo, pero SIN
        # MEDIR en la GTX 1650 real cuánto ocupa en la práctica a este
        # tamaño de contexto).
        "n_ctx": int(os.environ.get("QWEN_N_CTX", "32768")),
        "verbose": False,
    }
    variant_suffix = f" ({settings.qwen_gguf_filename})"
    if settings.qwen_mmproj_filename:
        from llama_cpp.llama_chat_format import MTMDChatHandler

        kwargs["chat_handler"] = MTMDChatHandler.from_pretrained(
            repo_id=settings.qwen_gguf_repo_id,
            filename=settings.qwen_mmproj_filename,
        )
        variant_suffix = f" ({settings.qwen_gguf_filename}, con visión)"

    _model = Llama.from_pretrained(**kwargs)
    _loaded_model_name = "Qwen3.5-4B" + variant_suffix
    logger.info(
        "Qwen3.5-4B cargado: model=%s n_gpu_layers=%s (revisar el log nativo de llama.cpp "
        "arriba para confirmar si el offload a GPU funcionó de verdad)",
        _loaded_model_name,
        n_gpu_layers,
    )


def _call_ai_json_sync(system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> dict:
    with _model_lock:
        _lazy_load()
        # reset() antes de cada llamada: mismo motivo que Moondream2 en
        # scene_analysis.py -- evita que el KV cache de una llamada
        # anterior contamine esta, con el lock ya protegiendo contra
        # accesos concurrentes desde varias corrutinas (ver
        # `call_ai_json`, envuelto en asyncio.to_thread).
        _model.reset()
        response = _model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)


async def call_ai_json(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict:
    """Pide al modelo local respuesta JSON y devuelve el contenido ya
    parseado (dict).

    No comprueba `settings.ai_key_configured` -- eso sigue siendo
    responsabilidad de cada llamador, igual que antes.

    Es una llamada SÍNCRONA y con trabajo real de CPU/GPU por debajo
    (`Llama.create_chat_completion`), a diferencia del diseño anterior por
    HTTP (que ya era async de forma nativa vía httpx) -- por eso se
    envuelve en `asyncio.to_thread` aquí dentro, para no bloquear el event
    loop mientras el modelo genera; los tres llamadores no necesitan
    saber este detalle, ya reciben una función `async def` como antes.

    Lanza AIRequestError si el modelo no está disponible o la inferencia
    falla, o AIHTTPError si respondió pero no se pudo interpretar como
    JSON -- nunca devuelve un dict vacío ni None, así el llamador decide
    explícitamente cómo degradar."""
    if not _qwen_available():
        raise AIRequestError(
            "El modelo de IA local no está disponible (llama-cpp-python no instalado, o "
            "settings.qwen_gguf_repo_id/qwen_gguf_filename sin configurar -- ver app/config.py)"
        )

    try:
        return await asyncio.to_thread(_call_ai_json_sync, system_prompt, user_prompt, max_tokens, temperature)
    except (AIHTTPError, AIRequestError):
        raise
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AIHTTPError(200, f"respuesta con forma inesperada: {exc}") from exc
    except Exception as exc:
        raise AIRequestError(str(exc)) from exc
