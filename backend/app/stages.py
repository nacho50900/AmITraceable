"""
Códigos de fase del pipeline de análisis, usados como valor de `stage` en
los eventos de progreso SSE (ver `progress.py` y `analysis_router.py`).

Antes de esto, `stage` llevaba directamente el texto ya renderizado en
español ("Leyendo publicaciones...", "Analizando tu forma de escribir...").
Eso funcionaba bien mientras la webapp solo existía en español, pero con la
internacionalización del frontend (ver webapp/src/i18n) un texto libre en
español no se puede traducir de forma fiable en el cliente sin parsear
strings -- así que ahora se emite un CÓDIGO estable (independiente de
idioma) y es el frontend quien lo traduce, con el mismo patrón ya usado
para `risk_level`, `source`, etc. en `PopulationNarrowingStep`.

Es un conjunto CERRADO y pequeño (a diferencia de `attribute_label` o las
conclusiones de la IA, que llevan texto libre o nombres propios y no
encajan en este patrón) -- por eso una tabla de constantes simple es
suficiente, sin necesitar tocar los prompts de Mistral ni reestructurar
ningún modelo de datos.
"""

CONNECTING = "connecting"
READING_POSTS = "reading_posts"
ANALYZING_WRITING_STYLE = "analyzing_writing_style"
DETECTING_ATTRIBUTES = "detecting_attributes"
COMPUTING_SCORE = "computing_score"
SEARCHING_AI_SELF_DISCLOSURES = "searching_ai_self_disclosures"
GENERATING_REPORT = "generating_report"
