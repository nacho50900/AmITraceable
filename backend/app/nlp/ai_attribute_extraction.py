"""
Extracción de datos demográficos autodeclarados usando un LLM (Mistral AI),
como complemento de las regex de `demographic_extraction.py`.

Motivación: las regex cubren un vocabulario fijo ("estudio X", "estudiante
de X", "vivo en X"...). Cualquier redacción real que no encaje en esas
plantillas (p. ej. "voy a 2º de Enfermería", "curso el grado en
Enfermería", "trabajo de comercial en una aseguradora") se pierde en
silencio. Este módulo pide a un LLM que lea el texto igual que lo haría una
persona y detecte el mismo tipo de autodeclaraciones explícitas -- no
inferencias ni suposiciones -- devolviendo un JSON estructurado.

Decisión de diseño importante: el LLM NO decide a qué categoría del INE
pertenece un valor. Solo propone el texto literal más cercano (p. ej.
"enfermeria"); la normalización final contra las tablas de
`ine_reference.py` (mismo criterio de coincidencia por subcadena que ya
usan las regex, ver `_set_normalized`) la sigue haciendo este módulo, no el
LLM. Así el cálculo de k-anonimato sigue anclado a categorías auditables
del INE: si el LLM alucinase una categoría inventada, simplemente no
coincide con ninguna clave conocida y no se estima nada (igual que ya pasa
hoy con las regex cuando no hay coincidencia), en vez de colar un número de
población falso.

Cuándo se ejecuta: automáticamente dentro del pipeline principal
(`report/generator.py`), en cada análisis -- a diferencia de
`ai_analysis.py` (conclusiones priorizadas en lenguaje natural), que sigue
siendo un botón aparte que el usuario pulsa bajo demanda.

Nota RGPD (para la memoria): esto envía el TEXTO CRUDO de las
publicaciones a un proveedor externo (Mistral AI, UE) en cada análisis, no
solo agregados como hace `ai_analysis.py`. Es una excepción consciente al
principio de minimización que se sigue en el resto del proyecto,
justificada porque es indispensable para la propia función (detectar
autodeclaraciones en lenguaje natural libre) y porque el usuario ya ha
dado consentimiento OAuth explícito sobre su propio contenido.

También se envían la biografía y el nombre público de la cuenta (si la
plataforma los expone), por el mismo motivo de minimización justificada.
El nombre público NO es una autodeclaración -- es una convención cultural
del nombre (p. ej. "Ana" sugiere sexo femenino en español), así que es una
señal mucho más débil y con más falsos positivos (nombres unisex,
transliteraciones, apodos) que una frase explícita como "soy mujer". Por
eso se pide al modelo en un campo JSON APARTE ("sexo_por_nombre") y se
marca con una procedencia distinta ("ia_nombre") en vez de mezclarla sin
más con `sexo` -- ver `_to_findings` y `k_anonymity.py`, donde esa
procedencia añade una nota de menor fiabilidad al informe.

Degradación: si no hay `MISTRAL_API_KEY` configurada, o la llamada falla
por cualquier motivo (cuota agotada, timeout, respuesta con forma
inesperada...), se devuelven unos `DemographicFindings` vacíos y el
pipeline sigue con normalidad solo con lo que hayan encontrado las regex --
mismo principio de "módulo opcional que nunca rompe el resto" que
`vision/geolocation.py` y `ai_analysis.py`.

También se pide un campo aparte "fotos_de_viaje": publicaciones cuyo texto
indica que la persona está de viaje/vacaciones/de paso por un sitio. No es
una autodeclaración de atributo (no rellena provincia/municipio), sino una
lista de EXCLUSIÓN que usa report/generator.py para no confundir "dónde fue
tomada esta foto" con "dónde vive esta persona" al combinar geolocalización
de imagen con inferencia de residencia -- ver
app/nlp/travel_detection.py (su equivalente por regex, que actúa como red
de seguridad cuando no hay IA disponible) y `DemographicFindings.travel_permalinks`.

Un tercer tipo de campo, "inferencias_blandas", es cualitativamente
distinto de los dos anteriores: aquí SÍ se le pide al modelo que razone
sobre contenido SIMBÓLICO O INDIRECTO (emojis, fechas sueltas, estilo de
escritura) en vez de solo detectar declaraciones literales -- p. ej. una
biografía como "18/05/20🧡👸✨" no dice "tengo pareja" en ningún sitio, pero
un lector humano razonablemente infiere que es un aniversario y que
probablemente la persona tiene pareja. Cada inferencia lleva su propia
`confianza` (0-1, deliberadamente moderada: es una suposición, no un
hecho) y se guarda en `DemographicFindings.soft_inferences` como
`InferredAttribute`, NO como un campo más de autodeclaración -- no
participa en `merge_findings` ni en el estimador de k-anonimato
(`scoring/k_anonymity.py`, que exige una autodeclaración explícita para
narrowear población): se añade directamente a la lista general de
"atributos inferidos" del informe (ver `report/generator.py`), igual que
las inferencias indirectas por hashtags/comunidades de
`attribute_inference.py`, pero basada en razonamiento del LLM en vez de
coincidencia de palabras clave. Se pide en la MISMA llamada a Mistral que
el resto de este módulo (un único mensaje, no una segunda petición aparte)
para no duplicar coste ni latencia.
"""
import json
import logging

import httpx

from app.config import settings
from app.data.ine_reference import (
    AUTONOMOUS_COMMUNITY_PROVINCES,
    MUNICIPALITY_POPULATION,
    OCCUPATION_DISTRIBUTION,
    PROVINCE_POPULATION,
    RELIGION_DISTRIBUTION,
    SEXUAL_ORIENTATION_DISTRIBUTION,
    SPORT_PRACTICE_DISTRIBUTION,
    STUDIES_DISTRIBUTION,
    resolve_autonomous_community,
)
from app.models.schemas import InferredAttribute, SocialPost
from app.nlp.demographic_extraction import DemographicFindings, _strip_accents, _ZODIAC_TEXT_MAP

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Campos "simples" (valor libre, sin normalizar contra una tabla del INE).
_FREE_TEXT_FIELDS = ("universidad", "empresa")
# Campos de enumeración EXACTA (el prompt ya le pide al modelo uno de estos
# valores concretos, a diferencia de estudios/ocupacion -- ahí el modelo
# devuelve texto libre que luego se normaliza por subcadena contra
# STUDIES_DISTRIBUTION/OCCUPATION_DISTRIBUTION, ver `_set_normalized`).
_NATIONALITY_VALUES = ("espanola", "extranjera")
_EMPLOYMENT_VALUES = ("activo", "parado", "jubilado", "estudiante", "otro_inactivo")
_LANGUAGE_VALUES = ("catalan", "euskera", "gallego", "valenciano")
_HOUSEHOLD_VALUES = ("unipersonal", "pareja_sin_hijos", "pareja_con_hijos", "monoparental")
# Igual que las cuatro de arriba: valores exactos, no texto libre a
# normalizar por subcadena -- se validan con `_set_exact_enum` contra las
# claves REALES de las tablas de distribución de ine_reference.py, en vez
# de mantener una lista duplicada a mano que podría desincronizarse.
_SEXUAL_ORIENTATION_VALUES = tuple(SEXUAL_ORIENTATION_DISTRIBUTION.keys())
_RELIGION_VALUES = tuple(RELIGION_DISTRIBUTION.keys())
_SPORT_PRACTICE_VALUES = tuple(SPORT_PRACTICE_DISTRIBUTION.keys())
# Tope defensivo de inferencias blandas aceptadas por respuesta, aunque el
# prompt ya pide un máximo de 5 -- por si el modelo no lo respeta al pie de
# la letra, no se toma "gratis" lo que devuelva de más.
_MAX_SOFT_INFERENCES = 5
# Confianza mínima (0-1) para aceptar el RANGO de edad estimado
# INDIRECTAMENTE ("edad_estimada", ver _set_edad_rango). Por debajo de
# este umbral no se añade nada -- ni edad ni rango -- en vez de mostrar
# un dato de baja fiabilidad como si fuera sólido. Deliberadamente más
# alto que el rango 0.3-0.7 que se acepta sin filtrar para
# `inferencias_blandas` genéricas, porque esto SÍ participa en el
# cálculo de k-anonimato (afecta al número de personas que se muestra en
# el informe), no solo en la lista informativa de atributos inferidos.
#
# Historial (Comandante, agosto 2026): primero se subió de 0.5 a 0.7
# tras detectar en producción estimaciones erróneas coladas con
# confianza "moderada" sobre un valor PUNTUAL (p. ej. 30 años estimados
# para alguien de 21). Pero subir solo el umbral tiene un límite: obliga
# a elegir entre descartar la estimación del todo o aceptar un único
# valor que el modelo no puede justificar con precisión. La solución de
# fondo fue rediseñar 'edad_estimada' para pedir un RANGO en vez de un
# valor puntual (ver el prompt más abajo y `_set_edad_rango`): ahora,
# ante una pista débil, el modelo debe ENSANCHAR el rango hasta tener
# una confianza alta genuina, en vez de arriesgarse con un año concreto.
# Esto traslada la responsabilidad de expresar la incertidumbre del
# umbral (una decisión binaria de aceptar/descartar) al propio ancho del
# rango (una escala continua) -- un rango de 20 años con confianza alta
# es preferible a uno estrecho con confianza baja, y matemáticamente no
# hace daño: `age_range_proportion` en ine_reference.py simplemente
# devuelve una proporción de población más alta (narrowea menos) cuanto
# más ancho es el rango, así que un rango deliberadamente amplio nunca
# produce una falsa precisión, solo aporta menos información. El umbral
# de 0.7 se mantiene como red de seguridad adicional, ya no como
# mecanismo principal para absorber la incertidumbre.
_AGE_RANGE_MIN_CONFIDENCE = 0.7
# Todos los campos que puede rellenar este módulo, en el mismo orden que
# `DemographicFindings`, usado por `merge_findings`.
_ALL_FIELDS = (
    "sexo", "edad", "provincia", "municipio", "comunidad_autonoma",
    "estudios", "ocupacion", "universidad", "empresa",
    "nacionalidad", "situacion_laboral", "tipo_hogar", "lengua_materna",
    "practica_deportiva",
    # Rango de edad estimado INDIRECTAMENTE (ver docstring del campo en
    # DemographicFindings) -- solo lo rellena la IA, nunca la regex, pero
    # reutiliza el mismo mecanismo de merge (copiar si regex_findings.edad
    # sigue en None) para no duplicar lógica.
    "edad_rango_min", "edad_rango_max",
    # A diferencia de los anteriores (autodeclaraciones explícitas que la
    # regex TAMBIÉN podría detectar), estos SOLO los rellena la IA -- pero
    # reutilizan el mismo mecanismo de merge (copiar si regex_findings lo
    # tiene a None) para no duplicar lógica.
    "estado_civil",
    "orientacion_sexual",
    "signo_zodiacal",
    "religion",
)

_SYSTEM_PROMPT = (
    "Eres un extractor de datos. Se te da: (1) el nombre público y la biografía de una "
    "cuenta, y (2) una lista de sus publicaciones, cada una precedida por su "
    "identificador entre corchetes, p. ej. [abc123] texto de la publicación. En la "
    "biografía y las publicaciones, busca AUTOD ECLARACIONES EXPLÍCITAS en "
    "primera persona sobre la propia persona -- nunca sobre otras personas mencionadas, y "
    "nunca inferencias o suposiciones tuyas, EXCEPTO en los campos marcados como "
    "'simbólico/indirecto' más abajo. Responde EXCLUSIVAMENTE con un JSON con esta "
    "forma exacta, sin texto adicional ni backticks:\n"
    '{"sexo": "hombre"|"mujer"|null, "edad": <entero>|null, '
    '"edad_estimada": {"edad_min": <entero>|null, "edad_max": <entero>|null, "confianza": <0-1>}|null, '
    '"provincia": <string>|null, '
    '"municipio": <string>|null, "comunidad_autonoma": <string>|null, "estudios": <string>|null, '
    '"ocupacion": <string>|null, "universidad": <string>|null, "empresa": <string>|null, '
    '"nacionalidad": "espanola"|"extranjera"|null, '
    '"situacion_laboral": "activo"|"parado"|"jubilado"|"estudiante"|"otro_inactivo"|null, '
    '"tipo_hogar": "unipersonal"|"pareja_sin_hijos"|"pareja_con_hijos"|"monoparental"|null, '
    '"lengua_materna": "catalan"|"euskera"|"gallego"|"valenciano"|null, '
    '"practica_deportiva": "senderismo"|"ciclismo"|"gimnasia_intensa"|"natacion"|"yoga_pilates"|'
    '"running"|"musculacion"|"padel"|"futbol"|"futbol_sala"|"baloncesto"|"tenis"|"golf"|null, '
    '"sexo_por_nombre": "hombre"|"mujer"|null, '
    '"fotos_de_viaje": [<identificador_de_publicacion>, ...], '
    '"estado_civil": "soltero"|"con_pareja"|"casado"|"divorciado"|"viudo"|null, '
    '"orientacion_sexual": <string>|null, '
    '"signo_zodiacal": <string>|null, '
    '"religion": <string>|null, '
    '"inferencias_blandas": [{"categoria": <string>, "valor": <string>, "confianza": <0-1>, '
    '"evidencia": <identificador_de_publicacion_o_bio>}, ...], '
    '"evidence": {"<nombre_de_campo>": "<identificador_de_publicacion_o_bio>"}}\n'
    "Usa null si no hay una declaración explícita y clara para ese campo. No inventes "
    "datos que no estén literalmente en el texto. El campo 'evidence' debe indicar, para "
    "cada campo que no sea null, el identificador exacto (el que va entre corchetes, o la "
    "palabra 'bio' si viene de la biografía) que lo prueba. 'comunidad_autonoma' es DISTINTO "
    "de 'provincia': úsalo SOLO cuando la persona diga que vive en una comunidad autónoma "
    "española COMPLETA sin especificar la provincia concreta (p. ej. 'vivo en Canarias', 'soy "
    "de Andalucía', 'vivo en el País Vasco'); si además especifica la provincia o ciudad "
    "(p. ej. 'vivo en Las Palmas', 'vivo en Sevilla'), usa 'provincia' o 'municipio' en su "
    "lugar y deja 'comunidad_autonoma' en null. 'sexo_por_nombre' es distinto "
    "de 'sexo': aquí NO busques una autodeclaración, sino tu mejor estimación de qué sexo "
    "sugiere culturalmente el NOMBRE PÚBLICO de la cuenta en español (p. ej. 'Ana' -> "
    "'mujer'); usa null si el nombre es ambiguo, es un alias/apodo sin relación con un "
    "nombre real, o no se te ha proporcionado nombre. 'nacionalidad', 'situacion_laboral', "
    "'tipo_hogar' y 'lengua_materna' son autodeclaraciones EXPLÍCITAS igual que sexo/edad/"
    "ubicación -- NO son inferencias ni razonamiento simbólico (eso es 'inferencias_blandas' "
    "o 'estado_civil', ver más abajo). 'situacion_laboral' distingue si la persona TRABAJA "
    "actualmente ('activo'), busca trabajo ('parado'), está jubilada/pensionista "
    "('jubilado'), estudia ('estudiante') o ninguna de las anteriores, p. ej. labores del "
    "hogar ('otro_inactivo') -- distinto del SECTOR profesional, que va en 'ocupacion'. "
    "'tipo_hogar' es SOLO si la persona dice explícitamente con quién vive o si menciona "
    "vivir sola: 'unipersonal' (vive sola), 'pareja_sin_hijos'/'pareja_con_hijos' (vive con "
    "su pareja, con o sin hijos en el mismo hogar) o 'monoparental' (un solo progenitor con "
    "hijos, sin pareja). 'lengua_materna' es SOLO una de las 4 lenguas cooficiales listadas "
    "(catalán, euskera, gallego, valenciano) si la persona dice explícitamente que es su "
    "lengua materna o habitual -- no infieras esto del lugar donde vive ni del idioma en que "
    "está escrito el texto. "
    "'practica_deportiva' detecta si la persona declara EXPLÍCITAMENTE que practica un "
    "deporte con regularidad -- verbos de práctica como 'juego al...', 'hago...', 'salgo a "
    "correr', 'voy al gimnasio', 'practico...', 'soy runner/nadador/futbolista'. NO uses "
    "este campo si el texto solo MENCIONA el deporte sin indicar que la persona lo practica "
    "-- p. ej. 'vi el partido de fútbol', 'me encanta el fútbol' (como afición de "
    "espectador), un resultado deportivo, o una noticia sobre un equipo NO cuentan; solo "
    "cuenta una autodeclaración de PRÁCTICA real. Los únicos valores válidos son "
    "'musculacion' (gimnasio, pesas, musculación, halterofilia, crossfit), 'senderismo' "
    "(senderismo, montañismo, rutas de montaña), 'running' (running, atletismo, correr con "
    "regularidad), 'natacion' (natación, nadar), 'futbol' (fútbol 11 o 7, jugador/futbolista), "
    "'futbol_sala' (fútbol sala, futbito, fútbol 7 o fútbol playa -- DISTINTO de 'futbol'), "
    "'ciclismo' (ciclismo, ir en bici con regularidad, ciclista), 'padel' (pádel), 'tenis' "
    "(tenis), 'baloncesto' (baloncesto, jugador/jugadora de baloncesto), 'golf' (golf, "
    "golfista), 'yoga_pilates' (yoga, pilates, tai-chi) y 'gimnasia_intensa' (aerobic, step, "
    "zumba, spinning) -- si "
    "practica otro deporte no listado aquí, usa null (no inventes una categoría nueva ni "
    "fuerces la más parecida). Usa null si no hay ninguna autodeclaración de práctica. "
    "'fotos_de_viaje' es una lista aparte, "
    "sin relación con las autodeclaraciones anteriores: incluye ahí el identificador de "
    "CUALQUIER publicación cuyo texto indique que la persona está de viaje, de vacaciones, "
    "de paso, o visitando temporalmente un sitio que NO es necesariamente donde vive (p. ej. "
    "'De viaje en Roma', 'Unos días en la playa', 'Visitando a mi prima en Londres'). El "
    "objetivo es señalar publicaciones que NO deben usarse para deducir dónde vive la "
    "persona habitualmente, aunque la foto en sí esté geolocalizada con confianza. Si un "
    "texto no da ninguna pista de viaje/vacaciones, NO lo incluyas en esa lista.\n"
    "'orientacion_sexual': detecta si la persona DECLARA EXPLICITAMENTE su orientación "
    "sexual (p. ej. 'soy heterosexual', 'heterosexual', 'soy gay', 'soy lesbiana', "
    "'soy bisexual', 'soy pansexual', 'soy asexual'). Devuelve el valor en minúscula "
    "('heterosexual', 'gay', 'lesbiana', 'bisexual', 'pansexual', 'asexual', 'homosexual'). "
    "NO lo infieras del estilo de escritura ni de emojis ambiguos -- solo autodeclaración "
    "literal. Usa null si no hay declaración.\n"
    "'signo_zodiacal': detecta si el texto o la biografía contiene un emoji de signo "
    "zodiacal o una mención explícita del signo (p. ej. \"soy aries\", \"soy escorpio\"). "
    "Los emojis zodiacales y sus rangos de nacimiento son: "
    "♈ Aries (21 mar - 19 abr), ♉ Tauro (20 abr - 20 may), ♊ Géminis (21 may - 20 jun), "
    "♋ Cáncer (21 jun - 22 jul), ♌ Leo (23 jul - 22 ago), ♍ Virgo (23 ago - 22 sep), "
    "♎ Libra (23 sep - 22 oct), ♏ Escorpio (23 oct - 21 nov), ♐ Sagitario (22 nov - 21 dic), "
    "♑ Capricornio (22 dic - 19 ene), ♒ Acuario (20 ene - 18 feb), ♓ Piscis (19 feb - 20 mar). "
    "Si detectas un emoji o mención, devuelve el nombre del signo y su rango, "
    "p. ej. 'aries (21 mar - 19 abr)'. Usa null si no hay ninguna señal.\n"
    "'religion': detecta si la persona indica explícitamente (texto o emoji) su creencia "
    "religiosa. Los principales símbolos: 🔯 o ✡️ = judaísmo; ☪️ o ☪ = islam; ✝️ o ✝ o ☦ = "
    "cristianismo; ☸️ o ☸ = budismo; 🕉️ o ॐ = hinduismo; ☪ = paínismo/Wicca; "
    "🧹 = brujeria/magia; un rosario 📿 puede indicar catolicismo o islam. Además, "
    "menciones literales como 'soy judío/a', 'soy musulmán/a', 'soy católico/a', "
    "'soy cristiano/a', 'soy budista', 'soy ateo/a', 'soy agnóstico/a', etc. "
    "Devuelve el nombre de la religión en minúscula ('judaismo', 'islam', 'catolicismo', "
    "'cristianismo', 'budismo', 'hinduismo', 'ateismo', 'agnosticismo'...). "
    "Usa null si no hay ninguna señal clara.\n"
    "'edad_estimada' es DISTINTO de 'edad': úsalo SOLO cuando 'edad' se haya quedado en null "
    "porque la persona NO ha declarado su edad literalmente, pero el texto SÍ da alguna pista "
    "INDIRECTA de la que se pueda deducir razonablemente un RANGO de edad plausible -- por "
    "ejemplo, menciona un año de graduación o de inicio de estudios/trabajo, en qué curso "
    "está, referencias a hitos vitales (jubilación próxima, hijos adultos, empezar la "
    "universidad este año), o jerga/referencias claramente generacionales. En vez de acertar "
    "un año exacto, da un RANGO ('edad_min', 'edad_max', ambos incluidos): SI LA PISTA ES "
    "DÉBIL O AMBIGUA, ENSANCHA EL RANGO en vez de arriesgarte con un valor puntual que no "
    "puedas justificar -- no hay ninguna penalización por dar un rango amplio (p. ej. 20 años "
    "de ancho: 'edad_min': 20, 'edad_max': 40), y es MUCHO PREFERIBLE un rango amplio con "
    "confianza alta a uno estrecho con confianza baja o, peor, a un año concreto que resulte "
    "estar equivocado. Usa un rango estrecho SOLO cuando la pista sea casi tan clara como una "
    "declaración explícita (p. ej. dice el año exacto en que nació, o un cálculo de años sin "
    "ninguna ambigüedad posible). En 'confianza' pon un número entre 0 y 1 que refleje "
    "honestamente cuánta seguridad tienes de que la edad REAL esté DENTRO del rango que has "
    "dado -- como el rango ya absorbe la incertidumbre de la pista, reserva 0.7 o más para "
    "cuando estés genuinamente convencido/a de que la edad real cae ahí dentro (ensanchando el "
    "rango todo lo que haga falta para llegar a esa confianza), no para acertar un valor "
    "concreto. Si no hay ninguna pista razonable, usa null en 'edad_min'/'edad_max' (o "
    "directamente null en todo el objeto) antes que inventar un rango sin fundamento. NUNCA "
    "reutilices esto para razonar sobre la edad de otra persona mencionada en el texto, solo "
    "sobre la propia persona.\\n"
    "'inferencias_blandas' es DISTINTO de todo lo anterior: aquí SÍ debes razonar, como lo "
    "haría una persona observadora, sobre contenido SIMBÓLICO O INDIRECTO -- combinaciones "
    "de emojis, fechas sueltas, estilo de escritura, jerga -- que sugieran algo sobre la vida "
    "de la persona sin decirlo explícitamente. Ejemplo: una biografía que sea solo "
    "'18/05/20🧡👸✨' sugiere que esa fecha es un aniversario (de pareja, o de otro tipo "
    "importante para la persona) y que probablemente tiene pareja -- razónalo igual que lo "
    "haría un humano leyendo el perfil. Cada elemento debe tener: 'categoria' (una etiqueta "
    "corta en español, p. ej. 'relacion_sentimental', 'tiene_mascota', 'afición'), 'valor' "
    "(una frase breve explicando la inferencia y en qué te basas, p. ej. 'Posible relación de "
    "pareja: la biografía solo contiene una fecha con emojis de corazón y corona, un patrón "
    "típico de aniversario'), 'confianza' (un número entre 0 y 1 que refleje que esto es una "
    "SUPOSICIÓN, no un hecho -- usa valores moderados, entre 0.3 y 0.7, casi nunca por encima "
    "de 0.7 salvo que la señal sea muy clara) y 'evidencia' (el identificador de la "
    "publicación o 'bio' en que te basas). Incluye como máximo 5 inferencias, solo las que "
    "tengan un anclaje textual/simbólico real -- ante la duda, no la incluyas. Devuelve una "
    "lista vacía si no encuentras ninguna señal de este tipo.\n"
    "'estado_civil' usa el MISMO tipo de razonamiento simbólico que "
    "'inferencias_blandas' (no busques una autodeclaración explícita como 'estoy casado', "
    "sino indicios indirectos), pero distingue varias categorías -- no te quedes en un simple "
    "sí/no:\n"
    "- 'casado': indicios de matrimonio o convivencia formal como pareja estable -- "
    "menciones a 'mi marido/esposa/mujer', anillo de boda visible en fotos, fecha "
    "acompañada de palabras como 'boda' o 'aniversario de boda'.\n"
    "- 'con_pareja': indicios de una relación de pareja SIN que quede claro que estén "
    "casados -- menciones a 'mi novio/novia/pareja', fecha con emojis románticos (corazón, "
    "corona, anillo) sin mención de boda, fotos recurrentes con la misma persona en "
    "contexto romántico.\n"
    "- 'soltero': indicios de estar SIN pareja actualmente -- declaraciones tipo 'soltera y "
    "feliz', menciones a estar buscando pareja, o similar.\n"
    "- 'divorciado': indicios de una ruptura matrimonial o separación legal pasada -- "
    "menciones a 'mi ex marido/esposa', 'desde mi divorcio', 'estoy separado/a', SIN indicios "
    "de una nueva pareja actual (si además hay indicios de nueva pareja, usa 'con_pareja' o "
    "'casado' en su lugar: refleja la situación ACTUAL, no el historial).\n"
    "- 'viudo': indicios de que la pareja/cónyuge ha fallecido -- menciones a 'mi difunto "
    "marido/esposa', 'en memoria de', luto explícito por una pareja.\n"
    "- null: no hay ninguna señal en ningún sentido. Ante la duda entre dos categorías, o "
    "si la señal es débil, usa null antes que adivinar."
)


class AiExtractionUnavailable(Exception):
    """Excepción interna: se captura siempre dentro de este módulo, nunca se
    propaga al llamador -- el pipeline principal no debe romperse porque la
    IA falle o no esté configurada."""


def _posts_prompt(posts: list[SocialPost]) -> str:
    lines = []
    for post in posts:
        text = (post.text or "").strip()
        if not text:
            continue
        # Recorte defensivo por publicación: acota el tamaño del prompt
        # (y por tanto el coste/cuota) sin depender de que el texto de
        # origen ya venga acotado.
        snippet = text[:600]
        lines.append(f"[{post.permalink}] {snippet}")
    return "\n".join(lines)


def _profile_prompt(username: str, full_name: str | None, bio: str | None) -> str:
    lines = [f"Nombre de usuario/handle: {username}"]
    if full_name:
        lines.append(f"Nombre público mostrado por la cuenta: {full_name}")
    if bio:
        lines.append(f"Biografía: {bio.strip()[:600]}")
    return "\n".join(lines)


async def extract_demographics_with_ai(
    posts: list[SocialPost],
    username: str,
    full_name: str | None = None,
    bio: str | None = None,
) -> DemographicFindings:
    """Punto de entrada del módulo. Nunca lanza excepciones: cualquier fallo
    (sin API key, red, cuota, forma de respuesta inesperada) se traduce en
    unos `DemographicFindings` vacíos para que el pipeline siga solo con lo
    que hayan encontrado las regex.

    `bio` se pasa aquí también por separado (además de ya poder venir
    inyectada como pseudo-post desde report/generator.py para que las
    regex la analicen igual que un post) porque aquí sirve además de
    contexto para que el modelo entienda mejor el resto del perfil, no solo
    como una fuente más de autodeclaraciones."""
    if not settings.mistral_api_key:
        return DemographicFindings()

    posts_text = _posts_prompt(posts)
    profile_text = _profile_prompt(username, full_name, bio)
    if not posts_text and not profile_text:
        return DemographicFindings()

    prompt = f"{profile_text}\n\nPublicaciones:\n{posts_text}" if posts_text else profile_text

    try:
        parsed = await _call_mistral(prompt)
    except AiExtractionUnavailable as exc:
        logger.warning("Extracción de atributos con IA no disponible: %s", exc)
        return DemographicFindings()

    return _to_findings(parsed)


async def _call_mistral(prompt_text: str) -> dict:
    payload = {
        "model": settings.mistral_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        # Subido de 800 a 1000: los nuevos campos orientacion_sexual,
        # signo_zodiacal y religion añaden más espacio en la respuesta JSON.
        "max_tokens": 1000,
    }
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(MISTRAL_API_URL, json=payload, headers=headers)
    except httpx.RequestError as exc:
        raise AiExtractionUnavailable(f"error de red: {exc}") from exc

    if response.status_code != 200:
        raise AiExtractionUnavailable(f"HTTP {response.status_code}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AiExtractionUnavailable(f"respuesta con forma inesperada: {exc}") from exc


def _set_evidence(findings: DemographicFindings, field: str, evidence_map: dict) -> None:
    permalink = evidence_map.get(field) if isinstance(evidence_map, dict) else None
    findings.evidence.setdefault(field, [])
    if isinstance(permalink, str) and permalink:
        findings.evidence[field].append(permalink)
    findings.source[field] = "ia"


def _set_normalized(
    findings: DemographicFindings, parsed: dict, field: str, distribution: dict, evidence_map: dict
) -> None:
    raw = parsed.get(field)
    if not isinstance(raw, str) or not raw.strip():
        return
    candidate = _strip_accents(raw.strip().lower())
    matched = next((k for k in distribution if k in candidate), None)
    if matched:
        setattr(findings, field, matched)
        _set_evidence(findings, field, evidence_map)


def _set_signo_zodiacal(findings: DemographicFindings, parsed: dict, evidence_map: dict) -> None:
    """Como `_set_exact_enum`, pero para 'signo_zodiacal': el modelo no
    devuelve un valor exacto de un enum cerrado, sino texto libre tipo
    'aries (21 mar - 19 abr)' (el prompt le da ese formato como ejemplo).
    Aquí se valida que el NOMBRE del signo (antes del primer paréntesis,
    sin acentos/mayúsculas) sea uno de los 12 reales y se normaliza al
    formato canónico de `_ZODIAC_TEXT_MAP` -- el mismo que produce la
    detección por regex/emoji -- para que ambas rutas de detección
    terminen SIEMPRE en uno de los 12 valores exactos que espera
    `_step_signo_zodiacal` (k_anonymity.py) y las traducciones del
    frontend (`attributeValue.signo_zodiacal` en i18n), en vez de dejar
    pasar el rango de fechas que el modelo haya escrito con su propio
    formato (que podría no coincidir con ninguna traducción)."""
    raw = parsed.get("signo_zodiacal")
    if not isinstance(raw, str) or not raw.strip():
        return
    signo = _strip_accents(raw.strip().lower().split(" (")[0].split("(")[0].strip())
    canonical = _ZODIAC_TEXT_MAP.get(signo)
    if canonical is None:
        return
    findings.signo_zodiacal = canonical
    _set_evidence(findings, "signo_zodiacal", evidence_map)


def _set_exact_enum(
    findings: DemographicFindings, parsed: dict, field: str, valid_values: tuple[str, ...], evidence_map: dict
) -> None:
    """Como `_set_normalized`, pero para campos donde el prompt ya le pide
    al modelo uno de un puñado de valores EXACTOS (nacionalidad,
    situacion_laboral, tipo_hogar, lengua_materna) -- no hace falta
    normalizar por subcadena contra una tabla, solo validar que el modelo
    no se haya inventado un valor fuera del enum."""
    raw = parsed.get(field)
    if raw in valid_values:
        setattr(findings, field, raw)
        _set_evidence(findings, field, evidence_map)


def _match_place(raw: object, distribution: dict) -> str | None:
    """Busca `raw` (texto libre, p.ej. "Vivo en Las Palmas") como subcadena
    de alguna clave de `distribution` (municipio o provincia). None si
    `raw` no es un string usable o no hay coincidencia."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = _strip_accents(raw.strip().lower())
    return next((k for k in distribution if k in candidate), None)


def _set_comunidad_autonoma(findings: DemographicFindings, ccaa_raw: object, evidence_map: dict) -> None:
    """Procesa el campo 'comunidad_autonoma' del modelo (ver _SYSTEM_PROMPT):
    solo se usa cuando el texto no dio ya municipio/provincia concreta. Si
    la comunidad tiene una sola provincia, no hay ambigüedad y se guarda
    como provincia (más específico); si tiene varias, se guarda a nivel de
    comunidad autónoma."""
    if not isinstance(ccaa_raw, str) or not ccaa_raw.strip():
        return

    ccaa = resolve_autonomous_community(ccaa_raw)
    if ccaa is None:
        return

    provinces = AUTONOMOUS_COMMUNITY_PROVINCES[ccaa]
    if len(provinces) == 1:
        findings.provincia = provinces[0]
        _set_evidence(findings, "provincia", evidence_map)
    else:
        findings.comunidad_autonoma = ccaa
        _set_evidence(findings, "comunidad_autonoma", evidence_map)


def _set_location(findings: DemographicFindings, parsed: dict, evidence_map: dict) -> None:
    # Municipio primero (más específico), igual que en demographic_extraction.py.
    municipio = _match_place(parsed.get("municipio"), MUNICIPALITY_POPULATION)
    if municipio:
        findings.municipio = municipio
        _set_evidence(findings, "municipio", evidence_map)
        return

    provincia = _match_place(parsed.get("provincia"), PROVINCE_POPULATION)
    if provincia:
        findings.provincia = provincia
        _set_evidence(findings, "provincia", evidence_map)
        return

    # Ninguna coincidencia de municipio/provincia: puede que el modelo haya
    # devuelto una comunidad autónoma COMPLETA en su propio campo.
    _set_comunidad_autonoma(findings, parsed.get("comunidad_autonoma"), evidence_map)


def _parse_soft_inference_confidence(raw: object) -> float:
    """Convierte el campo 'confianza' de un item de 'inferencias_blandas'
    a un float entre 0 y 1, con un valor moderado por defecto si el
    modelo no dio un número usable."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return max(0.0, min(1.0, float(raw)))
    return 0.5


def _parse_soft_inference_evidence(raw: object) -> list[str]:
    if isinstance(raw, str) and raw.strip():
        return [raw]
    return []


def _valid_soft_inference_fields(item: dict) -> tuple[str, str] | None:
    """Extrae y valida (categoria, valor) de un item de
    'inferencias_blandas'. Devuelve None si al item le falta alguna de las
    dos cadenas no vacías que hacen falta para construir un
    InferredAttribute con sentido."""
    categoria = item.get("categoria")
    valor = item.get("valor")
    if not isinstance(categoria, str) or not categoria.strip():
        return None
    if not isinstance(valor, str) or not valor.strip():
        return None
    return categoria.strip(), valor.strip()


def _parse_soft_inferences(parsed: dict) -> list[InferredAttribute]:
    """Valida y convierte el campo 'inferencias_blandas' del modelo (ver
    _SYSTEM_PROMPT) en InferredAttribute. A diferencia del resto de este
    módulo, aquí NO se normaliza contra ninguna tabla del INE -- son
    categorías libres, no alimentan k_anonymity.py, así que no hay riesgo
    de que una alucinación del modelo cuele un número de población falso
    (el peor caso es una fila de más en la lista de "atributos inferidos"
    del informe, no un cálculo erróneo)."""
    raw = parsed.get("inferencias_blandas")
    if not isinstance(raw, list):
        return []

    result = []
    for item in raw[:_MAX_SOFT_INFERENCES]:
        if not isinstance(item, dict):
            continue
        fields = _valid_soft_inference_fields(item)
        if fields is None:
            continue
        categoria, valor = fields

        result.append(
            InferredAttribute(
                category=categoria,
                value=valor,
                confidence=_parse_soft_inference_confidence(item.get("confianza")),
                evidence=_parse_soft_inference_evidence(item.get("evidencia")),
            )
        )
    return result


def _set_sexo(findings: DemographicFindings, parsed: dict, evidence_map: dict) -> None:
    sexo = parsed.get("sexo")
    if sexo in ("hombre", "mujer"):
        findings.sexo = sexo
        _set_evidence(findings, "sexo", evidence_map)
        return

    # Solo se usa la estimación por nombre si no hay autodeclaración
    # explícita -- es una señal más débil (ver docstring del módulo) y
    # nunca debe pisar una frase literal tipo "soy mujer".
    sexo_por_nombre = parsed.get("sexo_por_nombre")
    if sexo_por_nombre in ("hombre", "mujer"):
        findings.sexo = sexo_por_nombre
        findings.evidence.setdefault("sexo", []).append("nombre público de la cuenta")
        findings.source["sexo"] = "ia_nombre"


def _set_edad(findings: DemographicFindings, parsed: dict, evidence_map: dict) -> None:
    edad = parsed.get("edad")
    if not isinstance(edad, int) or isinstance(edad, bool):
        return
    if not (12 <= edad <= 100):
        return
    findings.edad = edad
    _set_evidence(findings, "edad", evidence_map)


def _set_edad_rango(findings: DemographicFindings, parsed: dict) -> None:
    """Procesa el campo 'edad_estimada' del modelo (ver _SYSTEM_PROMPT):
    RANGO de edad estimado por razonamiento INDIRECTO, distinto de una
    autodeclaración explícita. Solo se acepta si (1) no hay ya una edad
    EXACTA (más precisa, ver docstring de `edad_rango_min`/`edad_rango_max`
    en DemographicFindings) y (2) la confianza declarada alcanza
    `_AGE_RANGE_MIN_CONFIDENCE`.

    A diferencia de la primera versión (que pedía una única edad puntual
    y la encajaba en un tramo quinquenal fijo), aquí el ANCHO del rango
    lo decide el propio modelo: el prompt le pide EXPLÍCITAMENTE
    ensanchar el rango, no la confianza, cuando la pista sea débil -- así
    una estimación con poca certeza sigue pudiendo aportar algo (un rango
    amplio con confianza alta) en vez de descartarse sin más o, peor,
    colarse con una confianza moderada sobre un valor puntual que puede
    estar muy equivocado (caso real detectado en producción, ver el
    comentario en DemographicFindings). El umbral de confianza sigue
    existiendo como red de seguridad adicional, no como mecanismo
    principal para absorber la incertidumbre."""
    if findings.edad is not None:
        return  # ya hay edad exacta, siempre más precisa: no se sustituye por un rango

    raw = parsed.get("edad_estimada")
    if not isinstance(raw, dict):
        return

    confidence = _parse_soft_inference_confidence(raw.get("confianza"))
    if confidence < _AGE_RANGE_MIN_CONFIDENCE:
        return

    edad_min = raw.get("edad_min")
    edad_max = raw.get("edad_max")
    if not isinstance(edad_min, int) or isinstance(edad_min, bool):
        return
    if not isinstance(edad_max, int) or isinstance(edad_max, bool):
        return
    if not (12 <= edad_min <= edad_max <= 100):
        return

    findings.edad_rango_min = edad_min
    findings.edad_rango_max = edad_max
    # Mismo valor duplicado bajo las dos claves para que el traspaso
    # genérico por nombre de campo en `merge_findings` (que copia
    # `ai_findings.confidence[field]` para cada `field` de `_ALL_FIELDS`)
    # funcione sin caso especial -- ver comentario en
    # DemographicFindings.confidence.
    findings.confidence["edad_rango_min"] = confidence
    findings.confidence["edad_rango_max"] = confidence
    findings.evidence.setdefault("edad_rango_min", [])
    findings.evidence.setdefault("edad_rango_max", [])
    # No se usa `evidence_map`/`_set_evidence` aquí porque este campo va
    # en un objeto JSON aparte ("edad_estimada"), sin entrada propia en el
    # mapa 'evidence' genérico del prompt -- la evidencia real de este
    # tipo de estimación son las pistas dispersas por varios posts, no un
    # único permalink localizable, así que se deja vacía a propósito.
    # NUNCA "ia" a secas: es una estimación indirecta por rango, no una
    # autodeclaración -- misma distinción de fiabilidad que
    # `estado_civil` ("ia_simbolica"); k_anonymity.py usa esto para la
    # nota de menor fiabilidad en el informe.
    findings.source["edad_rango_min"] = "ia_estimada"
    findings.source["edad_rango_max"] = "ia_estimada"


def _set_free_text_fields(findings: DemographicFindings, parsed: dict, evidence_map: dict) -> None:
    for field in _FREE_TEXT_FIELDS:
        value = parsed.get(field)
        if isinstance(value, str) and value.strip():
            setattr(findings, field, value.strip())
            _set_evidence(findings, field, evidence_map)


def _set_travel_permalinks(findings: DemographicFindings, parsed: dict) -> None:
    fotos_de_viaje = parsed.get("fotos_de_viaje")
    if isinstance(fotos_de_viaje, list):
        findings.travel_permalinks = {p for p in fotos_de_viaje if isinstance(p, str) and p.strip()}


_ESTADO_CIVIL_VALUES = ("soltero", "con_pareja", "casado", "divorciado", "viudo")


def _set_estado_civil(findings: DemographicFindings, parsed: dict, evidence_map: dict) -> None:
    estado_civil_raw = parsed.get("estado_civil")
    if not isinstance(estado_civil_raw, str) or estado_civil_raw not in _ESTADO_CIVIL_VALUES:
        return

    findings.estado_civil = estado_civil_raw
    permalink = evidence_map.get("estado_civil") if isinstance(evidence_map, dict) else None
    findings.evidence.setdefault("estado_civil", [])
    if isinstance(permalink, str) and permalink:
        findings.evidence["estado_civil"].append(permalink)
    # NUNCA "ia" a secas: es una inferencia simbólica/indirecta (ver
    # docstring del campo en DemographicFindings), categóricamente menos
    # fiable que una autodeclaración explícita detectada por IA en el
    # resto de este módulo -- k_anonymity.py usa esta distinción para
    # añadir una nota de fiabilidad menor en el informe.
    findings.source["estado_civil"] = "ia_simbolica"


def _to_findings(parsed: dict) -> DemographicFindings:
    if not isinstance(parsed, dict):
        return DemographicFindings()

    findings = DemographicFindings()
    evidence_map = parsed.get("evidence") or {}

    _set_sexo(findings, parsed, evidence_map)
    _set_edad(findings, parsed, evidence_map)
    _set_edad_rango(findings, parsed)
    _set_normalized(findings, parsed, "estudios", STUDIES_DISTRIBUTION, evidence_map)
    _set_normalized(findings, parsed, "ocupacion", OCCUPATION_DISTRIBUTION, evidence_map)
    _set_location(findings, parsed, evidence_map)
    _set_free_text_fields(findings, parsed, evidence_map)
    _set_exact_enum(findings, parsed, "nacionalidad", _NATIONALITY_VALUES, evidence_map)
    _set_exact_enum(findings, parsed, "situacion_laboral", _EMPLOYMENT_VALUES, evidence_map)
    _set_exact_enum(findings, parsed, "tipo_hogar", _HOUSEHOLD_VALUES, evidence_map)
    _set_exact_enum(findings, parsed, "lengua_materna", _LANGUAGE_VALUES, evidence_map)
    # orientacion_sexual/religion: valores exactos, validados contra las
    # claves reales de sus tablas de distribución (ver comentario junto a
    # _SEXUAL_ORIENTATION_VALUES/_RELIGION_VALUES más arriba) -- antes se
    # aceptaba CUALQUIER string que devolviera el modelo sin comprobar
    # nada, así que un valor inventado se guardaba igual y luego
    # _step_orientacion_sexual/_step_religion en k_anonymity.py
    # simplemente no encontraban proporción (se comportaba como "no
    # estimable" en vez de fallar, pero sin avisar de que el dato era
    # basura). signo_zodiacal usa su propio validador porque no es un
    # enum exacto, sino texto con un rango de fechas -- ver
    # _set_signo_zodiacal.
    _set_exact_enum(findings, parsed, "orientacion_sexual", _SEXUAL_ORIENTATION_VALUES, evidence_map)
    _set_exact_enum(findings, parsed, "religion", _RELIGION_VALUES, evidence_map)
    _set_signo_zodiacal(findings, parsed, evidence_map)
    # practica_deportiva: enum exacto igual que nacionalidad/situacion_laboral
    # -- requiere que el prompt distinga PRÁCTICA (autodeclaración de hacer
    # el deporte con regularidad) de una simple MENCIÓN/afición como
    # espectador (ver instrucción explícita en _SYSTEM_PROMPT), mismo
    # motivo por el que _SPORT_PRACTICE_RE en demographic_extraction.py usa
    # frases-ancla de práctica en vez de una simple mención por subcadena.
    _set_exact_enum(findings, parsed, "practica_deportiva", _SPORT_PRACTICE_VALUES, evidence_map)
    _set_travel_permalinks(findings, parsed)
    _set_estado_civil(findings, parsed, evidence_map)

    findings.soft_inferences = _parse_soft_inferences(parsed)

    return findings


def merge_findings(regex_findings: DemographicFindings, ai_findings: DemographicFindings) -> DemographicFindings:
    """Combina lo detectado por regex (determinista y gratuito) con lo
    detectado por IA. Las regex tienen prioridad: la IA solo rellena los
    campos que las regex NO encontraron -- nunca sobrescribe una
    autodeclaración ya confirmada por coincidencia exacta de patrón."""
    for field in _ALL_FIELDS:
        if getattr(regex_findings, field) is None and getattr(ai_findings, field) is not None:
            setattr(regex_findings, field, getattr(ai_findings, field))
            if field in ai_findings.evidence:
                regex_findings.evidence[field] = ai_findings.evidence[field]
            # Se preserva la procedencia tal cual la puso _to_findings
            # ("ia" para autodeclaraciones en texto, "ia_nombre" para la
            # estimación por nombre público) -- nunca se aplana a "ia" a
            # secas, o se perdería la distinción de fiabilidad.
            regex_findings.source[field] = ai_findings.source.get(field, "ia")
            if field in ai_findings.confidence:
                regex_findings.confidence[field] = ai_findings.confidence[field]
    return regex_findings
