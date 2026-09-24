"""
Módulo 8 (nuevo, opcional): pide a un LLM que lea el informe de exposición
YA GENERADO y devuelva conclusiones priorizadas en lenguaje natural. No es
parte del pipeline de análisis principal, en el sentido de que es una
llamada aislada y opcional aparte -- pero SÍ se dispara automáticamente en
cuanto el informe principal está listo (ver punto 4 más abajo): ya no hay
ningún botón "Analizar con IA" en el frontend.

Decisiones de diseño (para la memoria):

1. Proveedor: Qwen3.5-4B, LOCAL (vía llama-cpp-python, ver
   app/nlp/ai_client.py) -- sustituye al diseño anterior de Mistral AI /
   Google Gemini por HTTP (ver historial de ese módulo). El motivo
   original de elegir un proveedor EUROPEO (Mistral, para evitar
   transferencias internacionales de datos personales fuera del EEE,
   RGPD Cap. V Art. 44-49) queda resuelto de raíz con un modelo local: no
   hay transferencia a NINGÚN tercero, ni europeo ni de fuera, porque el
   informe (con datos personales inferidos de un usuario real: ubicación,
   ocupación, edad...) nunca sale del servidor. Trade-off aceptado a
   cambio: un modelo de 4B parámetros cuantizado razona peor que Mistral
   Small o Gemini Flash -- sin verificar todavía con casos reales de
   producción si la calidad del veredicto/conclusiones se resiente (ver
   app/nlp/ai_client.py para el estado de esa verificación).

2. Sin entrenamiento ni fine-tuning: es una tarea de razonamiento en
   contexto (in-context learning) sobre datos ya estructurados, no una
   tarea de dominio tan especializada como para justificar el coste de
   entrenar o hacer fine-tuning de un modelo propio. El informe completo en
   JSON se envía como contexto en cada llamada; no hay estado entre
   llamadas ni memoria del modelo entre usuarios.

3. Sin coste ni cuota: al ser local, no hay tier gratuito que agotar ni
   límite de peticiones por minuto que gestionar (ver el historial de
   app/nlp/ai_client.py sobre por qué esto sustituyó al diseño anterior
   de throttle + reintento en 429 contra Mistral/Gemini) -- el único
   límite real es el hardware de despliegue (VRAM/tiempo de inferencia).
   Si el modelo no puede cargarse o ejecutarse (dependencia no instalada,
   fallo de inferencia), este módulo degrada a "no disponible ahora
   mismo" sin reintentos ni fallback a un proveedor de pago.

4. Minimización: se envía el informe ya generado (agregados, no el texto
   crudo de los posts). Se dispara automáticamente en cuanto el informe
   principal está listo (no hay botón ni confirmación explícita adicional
   -- el usuario ya dio su consentimiento OAuth para todo el análisis al
   principio), pero sigue siendo una llamada AISLADA y opcional: si falla o
   no está configurada, el resto del informe no se ve afectado.

5. `report.recommendations` (las reglas fijas basadas en umbrales, ver
   `_build_recommendations` en report/generator.py) ya no se muestra como
   sección propia en el dashboard -- se le pasa al LLM como INSUMO
   explícito, para que las use de base y decida cuáles de verdad merecen
   la pena mencionar (con el mismo criterio de "nada obvio" del resto del
   prompt), en vez de mostrarlas todas sin filtrar. Así no se pierde esa
   señal (barata, determinista, sin depender de que la IA esté disponible
   ese día) ni se duplica con las conclusiones de la IA.
"""
from app.config import settings
from app.models.schemas import ExposureReport
from app.nlp.ai_client import AIHTTPError, AIRequestError, call_ai_json

_SYSTEM_PROMPT = (
    "Eres un asistente que ayuda a personas no técnicas a entender su nivel de "
    "exposición de privacidad en redes sociales, a partir de un informe ya generado "
    "por una herramienta de análisis. Responde SIEMPRE en español, con un tono claro, "
    "objetivo y sin alarmismo innecesario. No repitas los datos del informe tal cual "
    "aparecen (el usuario ya los ha visto en el dashboard); en su lugar, sintetiza qué "
    "significan en conjunto.\n\n"
    "Calibra el riesgo por CAPACIDAD DE IDENTIFICAR/LOCALIZAR A LA PERSONA REAL, no por "
    "cantidad de datos inferidos. Un dato demográfico genérico (p. ej. sexo, si tiene "
    "pareja) o una tendencia horaria leve NO identifica a nadie por sí solo ni combinado "
    "con otro dato igual de genérico -- hacen falta miles o millones de personas que "
    "encajen en esa combinación. Si el conjunto de señales del informe es de ese tipo "
    "(genéricas, de baja confianza, sin nada que reduzca la población real a un grupo "
    "pequeño o identificable), el riesgo real es BAJO de verdad, no 'bajo pero hay que "
    "tener cuidado' -- dilo así de claro, sin añadir una advertencia de relleno solo "
    "por norma. Cuando el conjunto de señales sea tan escaso o genérico que la persona "
    "sea, en la práctica, ilocalizable a partir de este perfil, dilo explícitamente en el "
    "veredicto (p. ej. \"Con la información visible en este perfil no es posible "
    "identificarte ni localizarte razonablemente\") y, si es sincero hacerlo, reconoce "
    "positivamente que el perfil mantiene buen anonimato -- no hace falta inventar una "
    "recomendación de mejora donde no hay ningún riesgo real que mitigar. Reserva un "
    "tono de alerta genuino para cuando el informe SÍ combine señales que en conjunto "
    "reducen la población real a un grupo pequeño o concreto (p. ej. ciudad o barrio "
    "concreto + ocupación poco común + edad estrecha, o cualquier dato directamente "
    "identificable). La regla sigue siendo la misma en ambas direcciones: si el riesgo "
    "es bajo, dilo así de claro; no dramatices ni suavices un riesgo alto.\n\n"
    "El informe incluye un campo `recommendations`: son recomendaciones fijas generadas "
    "por reglas simples (umbrales de puntuación), no por ti. Úsalas como INSUMO de "
    "partida -- no las copies literalmente, pero tenlas en cuenta al decidir tus "
    "conclusiones y no ignores un riesgo real solo porque ya esté ahí listado; si de "
    "verdad aportan algo, intégralas de forma sintetizada en tus propias conclusiones. Si "
    "ninguna aporta nada real dado lo escaso del perfil, no las fuerces en tus "
    "conclusiones solo porque estén en la lista.\n\n"
    "Responde ÚNICAMENTE con un JSON con esta forma exacta, sin texto adicional ni "
    "backticks:\n"
    '{"veredicto": "<una frase>", "conclusiones": ["<frase 1>", "<frase 2>", ...]}\n\n'
    "El campo 'veredicto' es una valoración general de una sola frase, el titular del "
    "informe: p. ej. \"Este perfil no comparte directamente información que permita "
    "identificarte con facilidad\", o \"La línea general es buena, pero la publicación "
    "sobre [tema] revela [dato concreto]\". Debe reflejar fielmente el nivel de riesgo "
    "real del informe (si el riesgo es bajo, dilo así de claro; no dramatices ni "
    "suavices un riesgo alto).\n\n"
    "El campo 'conclusiones' es una lista de hallazgos MÁS ESPECÍFICOS que el veredicto. "
    "Criterio de selección, muy importante: NO listes una conclusión solo por rellenar. "
    "Descarta cualquier cosa obvia, genérica o que cualquiera adivinaría sin leer el "
    "informe (p. ej. \"comparte menos información personal\", \"ten cuidado con lo que "
    "publicas\", o repetir un solo dato aislado sin más contexto). Incluye SOLO "
    "conclusiones que combinen varios datos del informe de una forma que no sea obvia a "
    "simple vista, o que señalen un riesgo concreto y accionable que el usuario "
    "probablemente no había considerado. Sé objetiva: basa cada conclusión en datos "
    "concretos del informe, no en suposiciones. Da como máximo 5 conclusiones -- pero si "
    "de verdad no hay más de 1 o 2 que merezcan la pena (o ninguna), da solo esas, no "
    "rellenes hasta llegar a un mínimo; una lista vacía es una respuesta válida y "
    "esperable en un perfil con poca información inferible -- en ese caso, la ausencia de "
    "conclusiones ES el resultado correcto, no una respuesta incompleta que haya que "
    "arreglar inventando algo. Cada conclusión: 1-2 frases, concreta y accionable. No "
    "inventes datos que no estén en el informe."
)


class AiAnalysisUnavailable(Exception):
    """Se lanza cuando el análisis con IA no se puede realizar (sin API key
    configurada, cuota agotada, o error del proveedor). El llamador (router)
    la traduce a una respuesta clara para el frontend, nunca a un 500 opaco."""


# Instrucción de idioma añadida al prompt de sistema cuando `lang != "es"`.
# Se genera directamente en el idioma pedido en la MISMA llamada, en vez de
# traducir el veredicto/conclusiones después con una segunda llamada: el
# prompt ya está calibrado en español (ver docstring del módulo y ADR-17),
# y añadir una llamada de traducción aparte solo suma latencia, un segundo
# punto de fallo, y cuota extra del tier gratuito, sin ganar fiabilidad
# frente a pedirle directamente al modelo que responda en otro idioma
# (algo que los LLM instruction-tuned actuales hacen de forma fiable).
_LANGUAGE_INSTRUCTIONS = {
    "en": (
        "\n\nIMPORTANTE: a pesar de que el resto de estas instrucciones esté en "
        "español, responde en INGLÉS -- tanto 'veredicto' como cada elemento de "
        "'conclusiones' deben estar en inglés. Las claves del JSON siguen siendo "
        "literalmente 'veredicto' y 'conclusiones' (no las traduzcas), solo su "
        "contenido de texto va en inglés."
    ),
}

# Idiomas soportados por el selector de la webapp (ver webapp/src/i18n) --
# cualquier otro valor de `lang` se ignora silenciosamente y se sirve en
# español, el idioma por defecto/fallback en todo el proyecto.
SUPPORTED_LANGUAGES = frozenset({"es", *_LANGUAGE_INSTRUCTIONS.keys()})


async def _call_ai_chat(system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    """Delegación fina sobre app.nlp.ai_client.call_ai_json (modelo local
    Qwen3.5-4B, ver ese módulo). Conserva la excepción
    `AiAnalysisUnavailable` de siempre -- el llamador (analyze_report_with_ai)
    no cambia."""
    if not settings.ai_key_configured:
        raise AiAnalysisUnavailable(
            "El análisis con IA no está disponible en este servidor "
            "(el modelo local no está instalado o configurado)."
        )

    try:
        return await call_ai_json(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except AIRequestError as exc:
        raise AiAnalysisUnavailable(f"No se pudo ejecutar el modelo de IA local: {exc}") from exc
    except AIHTTPError as exc:
        # A diferencia del diseño anterior (Mistral/Gemini por HTTP), un
        # modelo local nunca devuelve un status_code real -- AIHTTPError
        # solo se lanza ahora cuando el modelo SÍ respondió pero su salida
        # no se pudo interpretar como JSON (ver app/nlp/ai_client.py). Los
        # antiguos casos 429 (cuota agotada del proveedor) y 401 (key
        # inválida) ya no existen: no hay proveedor de terceros ni API key
        # que pueda fallar de esas formas.
        raise AiAnalysisUnavailable(f"Respuesta inesperada del modelo de IA local: {exc.body}") from exc


async def analyze_report_with_ai(report: ExposureReport, lang: str = "es") -> dict:
    if not settings.ai_key_configured:
        raise AiAnalysisUnavailable(
            "El análisis con IA no está disponible en este servidor "
            "(el modelo local no está instalado o configurado)."
        )

    # Se manda el informe ya generado (agregados/conclusiones propias de la
    # herramienta, incluido `recommendations`) -- minimización de datos, no
    # se reenvían los posts originales.
    report_json = report.model_dump_json(indent=2)

    system_prompt = _SYSTEM_PROMPT + _LANGUAGE_INSTRUCTIONS.get(lang, "")
    user_prompt = (
        "Aquí tienes el informe de exposición de privacidad:\n"
        f"<informe>\n{report_json}\n</informe>\n\n"
        "Dame el veredicto general y tus conclusiones priorizadas."
    )

    # call_ai_json ya devuelve el `content` parseado como dict (no el
    # payload crudo de la API) -- a diferencia de antes, aquí ya no hace
    # falta volver a indexar la respuesta ni llamar json.loads() otra vez;
    # ai_client.py ya lanza AIHTTPError/AiAnalysisUnavailable si esa forma
    # no se cumplió.
    parsed = await _call_ai_chat(system_prompt, user_prompt, max_tokens=600)
    if not isinstance(parsed, dict):
        raise AiAnalysisUnavailable("Respuesta inesperada del servicio de IA.")

    verdict = parsed.get("veredicto")
    verdict = verdict.strip() if isinstance(verdict, str) else ""

    raw_conclusions = parsed.get("conclusiones")
    conclusions = (
        [c.strip() for c in raw_conclusions if isinstance(c, str) and c.strip()]
        if isinstance(raw_conclusions, list)
        else []
    )

    return {"verdict": verdict, "conclusions": conclusions}


# NOTA: la traducción de descripciones de fotos (aficion/caption de
# Moondream2) usaba antes Mistral desde aquí (translate_texts()) -- ver
# ADR-30. Se sustituyó por traducción LOCAL con modelos MarianMT vía
# CTranslate2 (ver ADR-31 y app/nlp/translation.py::translate_texts_local()),
# muchísimo más ligera en RAM/CPU para este caso concreto (frases cortas)
# y sin depender de red ni cuota. Este módulo (ai_analysis.py) sigue
# existiendo solo para el veredicto/conclusiones del informe
# (analyze_report_with_ai, arriba), donde SÍ hace falta un LLM de
# propósito general.
