"""
Módulo 8 (nuevo, opcional): pide a un LLM (Mistral AI) que lea el informe
de exposición YA GENERADO y devuelva conclusiones priorizadas en lenguaje
natural. No es parte del pipeline de análisis principal -- se dispara bajo
demanda desde el frontend (botón "Analizar con IA"), con el informe que ya
está en memoria tras el análisis normal.

Decisiones de diseño (para la memoria):

1. Proveedor: Mistral AI (La Plateforme), empresa francesa. Se eligió
   frente a alternativas más baratas fuera de la UE (p. ej. DeepSeek) para
   evitar transferencias internacionales de datos personales fuera del
   Espacio Económico Europeo (RGPD, Capítulo V, Art. 44-49) -- aquí se
   están enviando datos personales inferidos de un usuario real (ubicación,
   ocupación, edad...), así que la jurisdicción del proveedor es relevante,
   no solo el precio.

2. Sin entrenamiento ni fine-tuning: es una tarea de razonamiento en
   contexto (in-context learning) sobre datos ya estructurados, no una
   tarea de dominio tan especializada como para justificar el coste de
   entrenar o hacer fine-tuning de un modelo propio. El informe completo en
   JSON se envía como contexto en cada llamada; no hay estado entre
   llamadas ni memoria del modelo entre usuarios.

3. Tier gratuito, sin gasto: se usa el plan gratuito de Mistral (límite de
   peticiones/minuto + tope mensual de tokens). Si la cuota se agota
   (respuesta 429) o la API key no está configurada, este módulo NO
   reintenta ni degrada a otro proveedor de pago -- simplemente devuelve
   "no disponible ahora mismo", para que nunca se genere gasto no
   presupuestado ni se rompa el resto de la app.

4. Minimización: se envía el informe ya generado (agregados, no el texto
   crudo de los posts), y solo cuando el usuario pulsa el botón
   explícitamente -- no ocurre automáticamente en cada análisis.
"""
import json

import httpx

from app.config import settings
from app.models.schemas import ExposureReport

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

_SYSTEM_PROMPT = (
    "Eres un asistente que ayuda a personas no técnicas a entender su nivel de "
    "exposición de privacidad en redes sociales, a partir de un informe ya generado "
    "por una herramienta de análisis. Responde SIEMPRE en español, con un tono claro, "
    "directo y sin alarmismo innecesario. No repitas los datos del informe tal cual "
    "aparecen (el usuario ya los ha visto en el dashboard); en su lugar, sintetiza qué "
    "significan en conjunto y qué debería priorizar. Da entre 3 y 5 conclusiones, "
    "ordenadas de mayor a menor riesgo. Cada conclusión: 1-2 frases, concreta y "
    "accionable. No inventes datos que no estén en el informe."
)


class AiAnalysisUnavailable(Exception):
    """Se lanza cuando el análisis con IA no se puede realizar (sin API key
    configurada, cuota agotada, o error del proveedor). El llamador (router)
    la traduce a una respuesta clara para el frontend, nunca a un 500 opaco."""


async def analyze_report_with_ai(report: ExposureReport) -> list[str]:
    if not settings.mistral_api_key:
        raise AiAnalysisUnavailable(
            "El análisis con IA no está configurado en este servidor (falta MISTRAL_API_KEY)."
        )

    # Se manda el informe ya generado (agregados/conclusiones propias de la
    # herramienta), no los posts originales -- minimización de datos.
    report_json = report.model_dump_json(indent=2)

    payload = {
        "model": settings.mistral_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Aquí tienes el informe de exposición de privacidad:\n"
                    f"<informe>\n{report_json}\n</informe>\n\n"
                    "Dame tus conclusiones priorizadas."
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(MISTRAL_API_URL, json=payload, headers=headers)
    except httpx.RequestError as exc:
        raise AiAnalysisUnavailable(f"No se pudo contactar con el servicio de IA: {exc}") from exc

    if response.status_code == 429:
        # Cuota del tier gratuito agotada (peticiones/minuto o tope mensual).
        # NO se reintenta -- eso podría seguir gastando cuota o, en un plan
        # de pago, generar coste no deseado.
        raise AiAnalysisUnavailable(
            "Se ha alcanzado el límite del plan gratuito de IA por ahora. Inténtalo de nuevo más tarde."
        )
    if response.status_code == 401:
        raise AiAnalysisUnavailable("La clave de API de Mistral no es válida.")
    if response.status_code >= 400:
        raise AiAnalysisUnavailable(f"El servicio de IA devolvió un error ({response.status_code}).")

    data = response.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise AiAnalysisUnavailable("Respuesta inesperada del servicio de IA.") from exc

    # El prompt pide de 3 a 5 conclusiones; se devuelven como lista de
    # líneas no vacías (el modelo normalmente las numera o las pone en
    # líneas separadas), recortando numeración/viñetas iniciales.
    lines = [line.strip().lstrip("-•0123456789. ").strip() for line in text.strip().splitlines()]
    conclusions = [line for line in lines if line]

    return conclusions or [text.strip()]
