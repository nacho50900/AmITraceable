"""
Cliente de IA genérico (no atado a un proveedor): dispatcha a Mistral o a
Google Gemini según `settings.ai_provider`, exponiendo una única función
`call_ai_json()` usada por los tres módulos que hablan con un LLM externo
(ai_attribute_extraction.py, ai_analysis.py, landmark_resolution.py) --
ninguno de los tres habla nunca directamente con la API de un proveedor
concreto.

Por qué existe (sustituye a app/nlp/mistral_client.py, que solo sabía
hablar con Mistral): Mistral cambió su plan gratuito en septiembre de
2026 -- el free tier de rate-limit por API key (el que documentaba ADR-45)
dejó de existir, sustituido por un modelo de crédito de pago que exige
activar pay-as-you-go para tener CUALQUIER límite usable, ni siquiera 1
peticion/minuto. Para no depender de la política de un único proveedor a
mitad de TFG, `AI_PROVIDER` (env, ver app/config.py) elige entre
"mistral" y "gemini" -- cambiar de proveedor es una variable de entorno,
nunca un cambio de código en los tres módulos que llaman a esto.

Google Gemini (`gemini-2.5-flash` por defecto) es el proveedor por
defecto: su free tier (AI Studio) es PERMANENTE y sin tarjeta -- no un
crédito que se agota, a diferencia del de Mistral ahora mismo -- con 10
peticiones/minuto y ~250K tokens/minuto en Flash, mucho margen para las
~4 llamadas por análisis que hace este proyecto (ver el desglose de
llamadas en el docstring de cada función llamadora). No es un proveedor
europeo (Google Cloud, EE.UU., expuesto al Cloud Act) -- ver
config.py para la justificación completa de por qué se eligió así de
todos modos, y cómo volver a Mistral (u otro proveedor europeo) sin
tocar código.
"""
import asyncio
import json
import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Margen mínimo entre llamadas SALIENTES a cada proveedor, por proveedor
# (cada uno con su propio throttle -- no tiene sentido frenar Gemini por
# el límite de ráfaga histórico de Mistral, ni viceversa).
#
# Mistral: 1.1s, medido -- el free tier ANTIGUO limitaba a 1 petición/s de
# ráfaga (ver ADR-45); se mantiene por si se vuelve a usar Mistral, o para
# cualquier plan de pago que mantenga ese límite.
#
# Gemini: sin throttle artificial -- el free tier de AI Studio da 10
# peticiones/minuto en Flash, muchísimo margen sobre las ~4 llamadas que
# hace un análisis completo (nunca simultáneas, todas secuenciales dentro
# del mismo request). El único límite real que se puede agotar en la
# práctica es el diario (RPD) -- y contra ESE no hay throttle que valga,
# solo el reintento ante 429 de más abajo.
_MIN_INTERVAL_SECONDS = {"mistral": 1.1, "gemini": 0.0}
# Margen extra antes de reintentar un 429 puntual -- mayor que el
# throttle normal porque, si YA hubo un 429 pese a él, el margen normal
# no bastó esta vez.
_RETRY_BACKOFF_SECONDS = 1.5

_throttle_locks: dict[str, asyncio.Lock] = {}
_last_call_started_at: dict[str, float] = {}


class AIHTTPError(Exception):
    """Fallo de la API del proveedor de IA (el que esté activo en
    `settings.ai_provider`) con el status code y el cuerpo crudo de la
    respuesta -- a diferencia de las excepciones específicas de cada
    módulo llamador (AiExtractionUnavailable, AiAnalysisUnavailable), esta
    lleva el detalle completo para que cada llamador decida su propio
    mensaje de cara al usuario, y para que el log distinga el motivo real
    (cuota agotada / rate limit / key inválida / modelo no encontrado...)."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:300]}")


class AIRequestError(Exception):
    """Fallo de red (timeout, DNS, conexión rechazada...) antes de recibir
    ninguna respuesta HTTP, contra cualquier proveedor."""


async def _throttle(provider: str) -> None:
    min_interval = _MIN_INTERVAL_SECONDS.get(provider, 0.0)
    if min_interval <= 0:
        return
    lock = _throttle_locks.setdefault(provider, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        last = _last_call_started_at.get(provider)
        if last is not None:
            wait = min_interval - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
        _last_call_started_at[provider] = time.monotonic()


async def _post_mistral(system_prompt: str, user_prompt: str, model: str, max_tokens: int, temperature: float) -> httpx.Response:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        return await client.post(MISTRAL_API_URL, json=payload, headers=headers)


def _parse_mistral_content(response: httpx.Response) -> dict:
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


async def _post_gemini(system_prompt: str, user_prompt: str, model: str, max_tokens: int, temperature: float) -> httpx.Response:
    url = GEMINI_API_URL_TEMPLATE.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            # Gemini 2.5 Flash razona ("thinking") por defecto, consumiendo
            # parte de maxOutputTokens en tokens internos antes de llegar a
            # la respuesta -- con presupuestos pequeños (200-1000 tokens,
            # lo que usan estas llamadas) eso puede dejar la respuesta
            # final truncada o vacía. Estas llamadas son extracción/
            # clasificación estructurada, no razonamiento complejo, así
            # que se desactiva explícitamente para que TODO el presupuesto
            # vaya a la respuesta JSON real.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    # Header, no query param -- evita que la key quede en la URL (y por
    # tanto en cualquier log de proxy/servidor que registre la URL
    # completa de la petición saliente).
    headers = {"x-goog-api-key": settings.gemini_api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        return await client.post(url, json=payload, headers=headers)


def _parse_gemini_content(response: httpx.Response) -> dict:
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


async def call_ai_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict:
    """Llama al proveedor de IA activo (`settings.ai_provider`) pidiendo
    respuesta JSON y devuelve el contenido ya parseado (dict). Aplica el
    throttle del proveedor activo antes de la petición (ver
    `_MIN_INTERVAL_SECONDS`); si el servidor responde 429 de todos modos,
    reintenta UNA vez tras `_RETRY_BACKOFF_SECONDS` -- pensado para un
    límite de RÁFAGA puntual, no para una cuota diaria/mensual realmente
    agotada (ver docstring del módulo).

    No comprueba que la API key del proveedor activo esté configurada --
    eso es responsabilidad de cada llamador, igual que antes de este
    cambio, para que cada uno decida su propio mensaje/degradación.

    Lanza AIHTTPError (con status_code/body) o AIRequestError ante
    cualquier fallo -- nunca devuelve un dict vacío ni None, así el
    llamador decide explícitamente cómo degradar."""
    provider = settings.ai_provider

    if provider == "gemini":
        resolved_model = model or settings.gemini_model
        post = lambda: _post_gemini(system_prompt, user_prompt, resolved_model, max_tokens, temperature)
        parse = _parse_gemini_content
    elif provider == "mistral":
        resolved_model = model or settings.mistral_model
        post = lambda: _post_mistral(system_prompt, user_prompt, resolved_model, max_tokens, temperature)
        parse = _parse_mistral_content
    else:
        raise AIHTTPError(0, f"AI_PROVIDER desconocido: '{provider}' (valores válidos: 'mistral', 'gemini')")

    await _throttle(provider)
    try:
        response = await post()
    except httpx.RequestError as exc:
        raise AIRequestError(str(exc)) from exc

    if response.status_code == 429:
        logger.warning(
            "%s devolvió 429 -- reintentando una vez tras %.1fs. Cuerpo: %s",
            provider, _RETRY_BACKOFF_SECONDS, response.text[:300],
        )
        await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        try:
            response = await post()
        except httpx.RequestError as exc:
            raise AIRequestError(str(exc)) from exc

    if response.status_code != 200:
        # Se loguea el cuerpo completo (no solo el status code) --
        # normalmente distingue "rate limit" de "cuota agotada" de "key
        # inválida" de "modelo no encontrado", información que si no se
        # pierde por completo.
        logger.warning("%s devolvió %s: %s", provider, response.status_code, response.text[:500])
        raise AIHTTPError(response.status_code, response.text)

    try:
        return parse(response)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AIHTTPError(response.status_code, f"respuesta con forma inesperada: {exc}") from exc
