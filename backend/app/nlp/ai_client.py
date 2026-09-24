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

Modelo elegido: Qwen3.5-4B (Apache-2.0), cuantizado a Q4_K_M vía
llama-cpp-python -- mismo backend que ya usa app/vision/scene_analysis.py
para Moondream2, y mismo patrón de carga perezosa + cuantización cacheada
en disco (ver `_ensure_quantized_model()` más abajo, calcado de esa
función). Solo se usa la parte de TEXTO del modelo (Qwen3.5-4B es
multimodal, pero la carga de la parte de visión requiere un chat handler
de llama-cpp-python que, a fecha de este cambio, solo existe en forks no
oficiales -- ver ADR pendiente de escribir sobre por qué NO se ha tocado
app/vision/scene_analysis.py/Moondream2 en este mismo cambio) -- Q4_K_M
en vez de Q8_0 (el que usa Moondream2) porque aquí SÍ hay presión real de
VRAM: 4.5B parámetros en Q8_0 (~4.5GB) no caben en los 4GB de la GTX 1650
de despliegue; en Q4_K_M (~4-5 bits efectivos/parámetro con la variante
K_M) el modelo ronda 2.5-3GB, dejando margen para el KV cache.

RIESGO SIN VERIFICAR, IMPORTANTE: este modelo y Moondream2
(scene_analysis.py) pueden necesitar estar cargados en GPU
SIMULTÁNEAMENTE -- landmark_resolution.py llama a este módulo durante el
mismo pase de análisis de fotos en el que scene_analysis.py ya tiene
Moondream2 cargado. Moondream2 (Q8_0) + este modelo (Q4_K_M) sumados
rondan ya 4-5GB solo en pesos, sin contar KV cache de ninguno de los dos
-- muy probablemente NO quepan a la vez en 4GB de VRAM. Sin GPU en el
entorno donde se escribió este cambio, no se ha podido medir el
comportamiento real (¿OOM directo? ¿llama.cpp degrada solo a CPU para uno
de los dos sin avisar?). Antes de dar esto por cerrado, medir en la GTX
1650 real si ambos modelos conviven o hace falta serializar su uso
(cargar/descargar según se necesite, con el coste de latencia que eso
añade) o mover uno de los dos a CPU explícitamente.

Repo/fichero GGUF del modelo BASE (sin cuantizar) configurables vía
`settings.qwen_gguf_repo_id`/`qwen_gguf_filename` (ver app/config.py) --
DEJADOS VACÍOS A PROPÓSITO: no se ha podido confirmar desde este entorno
(sin acceso a huggingface.co) cuál es el repo/fichero GGUF correcto para
Qwen3.5-4B ahora mismo. Con `qwen_gguf_repo_id` vacío, `_qwen_available()`
devuelve False y este módulo se comporta exactamente igual que si
`llama_cpp` no estuviera instalado -- "no disponible", sin excepción que
rompa el resto del pipeline (mismo criterio best-effort que
scene_analysis.py).
"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Tipo de cuantización por defecto -- sobreescribible con la variable de
# entorno QWEN_QUANT_TYPE (mismo mecanismo que MOONDREAM_QUANT_TYPE en
# scene_analysis.py) sin tocar código ni reconstruir la imagen. Q4_K_M por
# defecto (no Q8_0 como Moondream2): ver docstring del módulo para el
# cálculo de VRAM que justifica bajar a 4 bits aquí.
_DEFAULT_QUANT_TYPE = "Q4_K_M"

_model = None
_model_lock = threading.Lock()
_loaded_model_name: str | None = None


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
    """Igual que `scene_analysis.get_model_variant()` -- para el log de
    rendimiento (ver app/log/performance_log.py). `None` si el modelo
    todavía no se ha cargado en este proceso."""
    return _loaded_model_name


def _qwen_available() -> bool:
    """Comprobación barata (sin cargar el modelo): dependencia opcional
    instalada Y hay un repo GGUF configurado (ver docstring del módulo
    sobre por qué `qwen_gguf_repo_id` empieza vacío)."""
    if not settings.qwen_gguf_repo_id or not settings.qwen_gguf_filename:
        return False
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_quantized_model() -> tuple[str, str] | None:
    """Calcado de `scene_analysis._ensure_quantized_model()` -- mismo
    razonamiento completo ahí (cacheo en disco por tipo, escritura
    atómica vía .tmp + rename, nunca lanza). Única diferencia real: el
    modelo BASE a cuantizar aquí es bastante más grande (Qwen3.5-4B, ~9GB
    en F16/BF16, frente a los ~3.5GB del modelo de texto de Moondream2) --
    la primera cuantización va a tardar sensiblemente más y a consumir
    más disco/red la primera vez; SIN MEDIR EN CONCRETO (mismo motivo de
    siempre: sin GPU en el entorno donde se escribió esto)."""
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        logger.info(
            "llama-quantize no está en el PATH -- Qwen3.5-4B cargaría en F16/BF16 sin "
            "cuantizar si se intentase (ver Dockerfile, etapa cuda-builder)"
        )
        return None

    quant_type = os.environ.get("QWEN_QUANT_TYPE", _DEFAULT_QUANT_TYPE).strip().upper()

    quantized_dir = Path("/root/.cache/huggingface/qwen3.5-4b-quantized")
    quantized_path = quantized_dir / f"qwen3.5-4b-{quant_type.lower()}.gguf"
    if quantized_path.exists():
        return str(quantized_path), quant_type

    try:
        from huggingface_hub import hf_hub_download

        logger.info("Cuantizando Qwen3.5-4B a %s por primera vez (puede tardar varios minutos)...", quant_type)
        base_path = hf_hub_download(repo_id=settings.qwen_gguf_repo_id, filename=settings.qwen_gguf_filename)

        quantized_dir.mkdir(parents=True, exist_ok=True)
        tmp_output = quantized_dir / f"qwen3.5-4b-{quant_type.lower()}.gguf.tmp"
        result = subprocess.run(
            [quantize_bin, base_path, str(tmp_output), quant_type],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            logger.warning(
                "llama-quantize terminó con código %s -- no se pudo cuantizar Qwen3.5-4B. stderr: %s",
                result.returncode,
                result.stderr[-2000:],
            )
            return None
        tmp_output.rename(quantized_path)
        logger.info("Qwen3.5-4B cuantizado a %s correctamente: %s", quant_type, quantized_path)
        return str(quantized_path), quant_type
    except Exception as exc:
        logger.warning(
            "Fallo cuantizando Qwen3.5-4B a %s (%s): %s",
            quant_type,
            type(exc).__name__,
            exc,
        )
        return None


def _lazy_load():
    """Carga perezosa de Qwen3.5-4B (solo texto) vía llama-cpp-python --
    no hace nada si ya está cargado. Mismo criterio de detección de GPU
    que scene_analysis._lazy_load() (n_gpu_layers=-1 si hay CUDA, si no
    0) -- ver el RIESGO SIN VERIFICAR del docstring del módulo sobre la
    contención de VRAM con Moondream2 si ambos piden GPU a la vez.

    A diferencia de Moondream2, NO se pasa `chat_handler` -- este cliente
    solo hace llamadas de texto (sin imagen), así que no hace falta cargar
    ningún `mmproj`/vision encoder."""
    global _model, _loaded_model_name
    if _model is not None:
        return

    from llama_cpp import Llama

    import torch

    n_gpu_layers = -1 if torch.cuda.is_available() else 0

    quantized = _ensure_quantized_model()
    common_kwargs = dict(
        n_gpu_layers=n_gpu_layers,
        # Contexto generoso frente a los prompts cortos de este proyecto
        # (informes/perfiles de un único usuario, no documentos largos) --
        # sin la restricción de 2048 de Moondream2 (ese límite viene de
        # `n_ctx_train` de ESE modelo en concreto, no aplica aquí). SIN
        # VERIFICAR contra el `n_ctx_train` real de Qwen3.5-4B.
        n_ctx=4096,
        verbose=False,
    )
    if quantized is not None:
        quantized_path, quant_type = quantized
        _model = Llama(model_path=quantized_path, **common_kwargs)
        variant_suffix = f" ({quant_type})"
    else:
        _model = Llama.from_pretrained(
            repo_id=settings.qwen_gguf_repo_id,
            filename=settings.qwen_gguf_filename,
            **common_kwargs,
        )
        variant_suffix = " (sin cuantizar)"
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
    model: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict:
    """Pide al modelo local respuesta JSON y devuelve el contenido ya
    parseado (dict). `model` se conserva en la firma por compatibilidad
    con el diseño anterior (permitía pisar el modelo del proveedor activo
    por llamada) pero se IGNORA -- solo hay un modelo local cargado por
    proceso, ver docstring del módulo.

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
