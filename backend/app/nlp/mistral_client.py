"""
Cliente HTTP compartido para llamadas a la API de Mistral (La Plateforme),
usado por app/nlp/ai_attribute_extraction.py, app/ai_analysis.py y
app/vision/landmark_resolution.py.

Se creó para arreglar un fallo real observado en producción (Docker,
build de producción): el free tier de Mistral permite 30 peticiones/
minuto, pero SOLO 1 petición/segundo de ráfaga (límite aparte del de
minuto, confirmado en el propio panel de Mistral). Un análisis dispara,
de forma automática y sin botón, la llamada de autodeclaraciones
(ai_attribute_extraction.py) seguida casi inmediatamente por la del
veredicto/conclusiones (ai_analysis.py) -- con un análisis rápido, el
hueco entre ambas puede ser bien inferior a 1s, suficiente para topar el
límite de ráfaga aunque se esté muy lejos de las 30/min. Con las nuevas
llamadas de atributos-por-foto (ai_attribute_extraction.extract_soft_inferences_from_photos)
y de resolución de edificios emblemáticos (vision/landmark_resolution.py),
un mismo análisis puede llegar a disparar 4 llamadas seguidas -- razón de
más para centralizar el throttle aquí en vez de confiar en que la
latencia real las separe lo suficiente.

Dos mecanismos, independientes y complementarios:
1. Throttle: antes de cada llamada, espera lo que haga falta para que
   pase al menos _MIN_INTERVAL_SECONDS desde el INICIO de la última
   llamada disparada por este proceso (cualquier módulo llamador, no
   solo el mismo) -- protegido por un asyncio.Lock para que dos llamadas
   concurrentes del mismo análisis no lean el mismo timestamp a la vez.
2. Reintento único ante 429: si el throttle no bastó (margen de
   seguridad insuficiente, o varios workers del mismo proceso
   compartiendo la key), se reintenta UNA vez tras un margen extra antes
   de rendirse. Sigue sin reintentar en NINGÚN otro código de error
   (401, 5xx, fallo de red) -- mismo criterio de "nunca gastar cuota
   extra a ciegas" que ya tenía este proyecto antes de este cambio. Un
   429 real por CUOTA MENSUAL agotada (no ráfaga) simplemente vuelve a
   fallar en el reintento, con el único coste de _RETRY_BACKOFF_SECONDS
   de más antes de rendirse -- no hay forma barata de distinguir los dos
   motivos sin parsear el cuerpo de cada proveedor de forma frágil, así
   que se loguea el cuerpo completo (ver más abajo) para que quien
   revise los logs sí pueda distinguirlo a mano.
"""
import asyncio
import json
import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# 1.1s de margen de seguridad sobre el segundo exacto del límite de
# ráfaga medido (1 req/s).
_MIN_INTERVAL_SECONDS = 1.1
# Mayor que _MIN_INTERVAL_SECONDS a propósito: si YA hubo un 429 pese al
# throttle normal, el margen normal no bastó esta vez -- se espera más
# antes de reintentar, en vez de repetir exactamente el mismo margen que
# ya ha fallado una vez.
_RETRY_BACKOFF_SECONDS = 1.5

_throttle_lock = asyncio.Lock()
_last_call_started_at: float | None = None


class MistralHTTPError(Exception):
    """Fallo de la API de Mistral con el status code y el cuerpo crudo de
    la respuesta -- a diferencia de las excepciones específicas de cada
    módulo llamador (AiExtractionUnavailable, AiAnalysisUnavailable), esta
    lleva el detalle completo para que cada llamador decida su propio
    mensaje de cara al usuario, y para que el log sí distinga el motivo
    real (antes solo se logueaba "HTTP {status_code}", nunca el cuerpo,
    que normalmente distingue cuota mensual agotada de rate limit de
    ráfaga/minuto de API key inválida)."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:300]}")


class MistralRequestError(Exception):
    """Fallo de red (timeout, DNS, conexión rechazada...) antes de recibir
    ninguna respuesta HTTP."""


async def _throttle() -> None:
    global _last_call_started_at
    async with _throttle_lock:
        now = time.monotonic()
        if _last_call_started_at is not None:
            wait = _MIN_INTERVAL_SECONDS - (now - _last_call_started_at)
            if wait > 0:
                await asyncio.sleep(wait)
        _last_call_started_at = time.monotonic()


async def _post(payload: dict) -> httpx.Response:
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            return await client.post(MISTRAL_API_URL, json=payload, headers=headers)
    except httpx.RequestError as exc:
        raise MistralRequestError(str(exc)) from exc


async def call_mistral_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict:
    """Llama a Mistral pidiendo respuesta JSON y devuelve `content` ya
    parseado (dict). Aplica el throttle de _MIN_INTERVAL_SECONDS antes de
    la petición; si el servidor responde 429 de todos modos, reintenta UNA
    vez tras _RETRY_BACKOFF_SECONDS (pensado para el límite de RÁFAGA, no
    para la cuota mensual -- ver docstring del módulo).

    No comprueba `settings.mistral_api_key` -- eso es responsabilidad de
    cada llamador (igual que antes de este cambio), para que cada uno
    pueda decidir su propio mensaje/degradación cuando no está
    configurada, en vez de forzar aquí un único criterio para los tres
    módulos que usan este cliente.

    Lanza MistralHTTPError (con status_code/body) o MistralRequestError
    ante cualquier fallo -- nunca devuelve un dict vacío ni None, así el
    llamador decide explícitamente cómo degradar."""
    payload = {
        "model": model or settings.mistral_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }

    await _throttle()
    response = await _post(payload)

    if response.status_code == 429:
        logger.warning(
            "Mistral devolvió 429 pese al throttle de %.1fs -- reintentando una vez "
            "tras %.1fs. Cuerpo: %s",
            _MIN_INTERVAL_SECONDS,
            _RETRY_BACKOFF_SECONDS,
            response.text[:300],
        )
        await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        response = await _post(payload)

    if response.status_code != 200:
        # Se loguea el cuerpo completo (antes solo se logueaba el status
        # code) -- normalmente distingue "rate limit de ráfaga/minuto" de
        # "cuota mensual agotada" de "key inválida", información que
        # antes se perdía por completo.
        logger.warning("Mistral devolvió %s: %s", response.status_code, response.text[:500])
        raise MistralHTTPError(response.status_code, response.text)

    try:
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise MistralHTTPError(response.status_code, f"respuesta con forma inesperada: {exc}") from exc
