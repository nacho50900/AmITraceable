"""
Script de mantenimiento MANUAL (no se ejecuta en cada análisis, ni
programado -- lo corre un desarrollador de vez en cuando) para comprobar
si el INE ha publicado cifras más recientes que las de
`app/data/ine_reference.py`.

CÓMO USARLO:
    python scripts/update_ine_reference.py            # solo compara, no escribe nada
    python scripts/update_ine_reference.py --apply     # además, aplica PROVINCE_POPULATION
    python scripts/update_ine_reference.py --apply --yes  # sin pedir confirmación por teclado
    python scripts/update_ine_reference.py --apply --force-tasa-paro   # aplica SITUACION_LABORAL_DISTRIBUTION
                                                                        # aunque la tasa de paro parezca implausible
    python scripts/update_ine_reference.py --apply --force-ocupacion   # aplica OCCUPATION_DISTRIBUTION aunque la
                                                                        # suma de categorías mapeadas parezca implausible
    python scripts/update_ine_reference.py --apply --force-hogar       # aplica HOUSEHOLD_TYPE_DISTRIBUTION aunque la
                                                                        # suma de categorías parezca implausible

NOTA (encontrada en esta misma sesión, no corregida a propósito -- fuera
del alcance de lo pedido): el párrafo siguiente, sobre qué toca `--apply`,
está desactualizado frente al comportamiento real de `main()` más abajo
-- NATIONALITY_DISTRIBUTION, MARITAL_STATUS_DISTRIBUTION/BY_SEX y
SITUACION_LABORAL_DISTRIBUTION SÍ se aplican con `--apply` hoy (esta
última con la guarda de `--force-tasa-paro`), no solo PROVINCE_POPULATION.
Revisar y corregir este docstring es trabajo aparte.

Por defecto (sin `--apply`) imprime, por cada tabla soportada, el valor
actual en el código frente al valor recién descargado del INE, y si
difieren. NO SOBREESCRIBE `ine_reference.py`.

`--apply` SÍ escribe en `ine_reference.py`, pero ÚNICAMENTE la tabla
PROVINCE_POPULATION (la única de las cuatro ya verificada de extremo a
extremo contra la API real, con las 49 provincias casando correctamente
-- ver el histórico de ejecuciones más abajo). Actualiza también su fecha
en `_LAST_VERIFIED` a hoy. Antes de escribir, imprime cada cambio
concreto (clave, valor antiguo, valor nuevo) y pide confirmación por
teclado, salvo que se pase `--yes` (pensado para automatizarlo en un cron
o GitHub Action en el futuro, no para el uso normal).

MARITAL_STATUS_DISTRIBUTION, NATIONALITY_DISTRIBUTION y
SITUACION_LABORAL_DISTRIBUTION quedan DELIBERADAMENTE fuera de `--apply`,
aunque sus IDs de tabla ya están confirmados: sus valores en
`ine_reference.py` no son un volcado directo del INE, tienen razonamiento
a mano en los comentarios (ver p. ej. MARITAL_STATUS_DISTRIBUTION, que
combina dos encuestas distintas, o SITUACION_LABORAL_DISTRIBUTION, que
usa una base de cálculo distinta a la tasa cruda de la EPA) que este
script todavía no sabe recalcular -- aplicarlas sin ese paso de
normalización antes introduciría un dato mal derivado en una herramienta
que depende precisamente de la precisión de estos números. Automatizar
esas tres es el siguiente paso pendiente, no algo ya resuelto aquí.

LIMITACIÓN IMPORTANTE, para que quede documentada y no se asuma más
cobertura de la que hay: la API del INE (Tempus3, servicios.ine.es) exige
conocer el ID numérico exacto de cada tabla (o, en el caso de tablas
PC-Axis, su ruta exacta -- ver más abajo). Se han localizado por
búsqueda web IDs/rutas candidatas para las siete tablas con fuente
periódica conocida (población por provincia, estado civil, nacionalidad,
tasas EPA, ocupación CNO-11, tipo de hogar ECH).

BUG CORREGIDO (sesión posterior, a petición explícita de Nacho: "resuelve
y busca por el bug de SITUACION_LABORAL_DISTRIBUTION"): la tasa de paro
de 26,03% que devolvía _TABLA_TASAS_EPA (en vez del ~9,93% real,
confirmado contra la nota de prensa oficial EPA T4 2025) no era un
problema de metodología vigente/no vigente como se había hipotetizado
sin poder confirmarlo -- esa dualidad, comprobada con la propia página
del INE, es sobre la clasificación CNAE de rama de actividad, no sobre
esta tabla de tasas. La causa real, respaldada por ejemplos externos
documentados del formato JSON de esta misma API, es que el array `Data`
de cada serie viene ordenado DESCENDENTEMENTE por fecha (el dato más
reciente primero), y las 6 funciones fetch_* de este script cogían
`datos[-1]` -- el ÚLTIMO elemento, el más ANTIGUO, no el más reciente --
cuando `Data` traía más de un periodo. Un 26,03% encaja con un trimestre
real del pico de la crisis de 2013. Corregido en las 6 funciones fetch_*
a la vez (no solo en la de tasas EPA) con `_latest_valor_por_nombre` /
`_valor_mas_reciente`, que comparan la fecha real de cada punto en vez de
fiarse de la posición en el array o del orden de iteración. Ver el bloque
de comentario junto a esas dos funciones (más abajo en este fichero) para
el detalle completo, la evidencia externa concreta, y el aviso de que
sigue sin poder confirmarse al 100% sin ejecutar esto contra la API real.

INVESTIGACIÓN de STUDIES_DISTRIBUTION y OCCUPATION_DISTRIBUTION (sesión
posterior a la del párrafo anterior, a petición explícita de Nacho tras
descartar en una pasada previa que fuera un "quick win"):

- OCCUPATION_DISTRIBUTION SÍ tiene tabla INE anual/trimestral con
  granularidad suficiente: tabla 65134, "Ocupados por sexo y ocupación"
  (EPA, subgrupos principales -2 dígitos- de la CNO-11), confirmada por
  su ficha en datos.gob.es (da explícitamente la URL JSON de Tempus3).
  El trabajo de diseño real -- mapear cada uno de los ~40 subgrupos
  CNO-11 a una de las 10 categorías coloquiales de
  OCCUPATION_DISTRIBUTION -- se ha hecho a mano usando los nombres
  oficiales de subgrupo (ver `_CNO11_SUBGRUPO_TO_APP_CATEGORY`, más
  abajo en este fichero): es DELIBERADAMENTE un mapeo incompleto (la
  mayoría de subgrupos CNO-11 -agricultura, industria pesada, fuerzas
  armadas...- no tienen equivalente en las 10 categorías actuales y se
  dejan sin mapear a propósito). Como con estado civil/nacionalidad/tasas
  EPA, el ID de tabla y el mapeo NO se han podido ejecutar contra la API
  real desde este entorno de trabajo (sin acceso de red a
  servicios.ine.es) -- `fetch_occupation`/`_normalize_occupation` hacen
  además un supuesto sin verificar sobre el formato exacto de `Nombre`
  (ver AVISO DE FORMATO en el docstring de `fetch_occupation`), así que
  esta tabla necesita más revisión que ninguna otra la primera vez que
  se corra de verdad.

- STUDIES_DISTRIBUTION NO tiene un camino equivalente. Sus claves son
  titulaciones/carreras concretas (medicina, derecho, ingenieria
  informatica...), no niveles de formación (primaria/secundaria/
  universitaria, que sí tiene el INE con esa granularidad amplia) ni
  ramas de conocimiento amplias (que es todo lo que ofrece la fuente más
  cercana encontrada). Esa fuente más cercana es la Encuesta de
  Inserción Laboral de Titulados Universitarios (EILU) del propio INE
  (https://ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176991),
  que sí desglosa por "ámbito de estudio" con una granularidad parecida
  a la de este diccionario -- pero con tres problemas que la hacen no
  automatizable de la misma forma que las demás tablas de este script:
  (1) es una encuesta PUNTUAL, no anual/trimestral (ediciones conocidas:
  2014 y 2019 -- sin edición más reciente localizada al escribir esto);
  (2) mide TITULADOS de una cohorte concreta (p. ej. "curso 2013-2014"),
  no la población total viva con esa titulación, que es lo que necesita
  STUDIES_DISTRIBUTION; y (3) para convertir eso en una proporción sobre
  población total haría falta combinarlo con la tabla de nivel de
  formación alcanzado (ID candidato 66017/4194, sin verificar tampoco)
  para saber qué fracción de la población tiene estudios universitarios,
  y aun así seguiría siendo una aproximación (titulados de una cohorte
  puntual como proxy de la población total con cada titulación). Si se
  quiere abordar STUDIES_DISTRIBUTION con datos reales, ese es el camino
  -- pero es una tarea de diseño y verificación en sí misma, no una
  automatización directa como las demás tablas de este script; se deja
  documentado aquí en vez de intentarlo a ciegas en esta misma pasada.

INVESTIGACIÓN de HOUSEHOLD_TYPE_DISTRIBUTION (sesión posterior, a
petición explícita de Nacho): esta SÍ es -tal como ya se sospechaba antes
de investigarla- el caso genuino de tabla PC-Axis (identificador en forma
de ruta, no ID numérico) que las demás tablas de este fichero no son. Se
han resuelto dos cosas, no solo el ID/ruta:

1. Cómo se llama una tabla PC-Axis desde la misma API Tempus3 que ya usa
   el resto de este script (mismo endpoint DATOS_TABLA, pasando la ruta
   completa en vez de un ID numérico -- confirmado con un ejemplo literal
   de la documentación oficial del INE para otra tabla PC-Axis). Ver el
   comentario de `_TABLA_HOGAR_TIPO` más abajo para el detalle y la URL
   de referencia.
2. Qué ruta concreta usar: "t20/p274/serie/prov/p01/l0/01013.px"
   ("Número de hogares según el tipo de hogar y el tipo de edificio
   donde se encuentra la vivienda", ECH, nacional), confirmada por sus
   etiquetas en datos.gob.es, que incluyen literalmente varias de las
   categorías de tipo de hogar que necesitamos.

Sigue habiendo un supuesto SIN VERIFICAR sobre el formato exacto de
`Nombre` en esta tabla (no se ha podido ejecutar contra la API real desde
este entorno de trabajo) -- ver `_normalize_household_type` para el
supuesto concreto y cómo corregirlo con el ejemplo real la primera vez
que se corra.

BÚSQUEDA FUERA DEL INE de STUDIES_DISTRIBUTION y LANGUAGE_BY_CCAA
(sesión posterior, a petición explícita de Nacho: "no tenemos una manera
de sacar las otras dos, aunque no sea por el INE, busca por internet").
Resultado desigual entre las dos -- una tiene un candidato nuevo y real,
la otra sigue sin un camino que valga la pena automatizar:

- STUDIES_DISTRIBUTION -- CANDIDATO NUEVO Y PROMETEDOR, pero sin
  mecanismo de acceso confirmado: el Ministerio de Ciencia, Innovación y
  Universidades (NO el INE -- fuente distinta, mismo Estado) publica a
  través de su Sistema Integrado de Información Universitaria (SIIU,
  https://www.ciencia.gob.es/Ministerio/Estadisticas/SIIU.html) una
  tabla ANUAL (última edición confirmada: curso 2024-2025) de
  "Matriculados por nivel académico, sexo, grupo de edad y CAMPO de
  estudio" -- "campo de estudio" es justo la granularidad de carrera
  concreta que necesita STUDIES_DISTRIBUTION (ISCED-F: Medicina,
  Derecho, Informática... como categorías individuales, no ramas
  amplias), confirmada por su propio catálogo de clasificaciones
  ("Titulaciones según rama, ISCED 2013, ámbito de estudio y campo de
  estudio. Curso 2024-2025"). Es un candidato claramente MEJOR que la
  EILU mencionada arriba: es anual (no puntual) y viene del mismo
  sistema estadístico oficial. Su limitación real, distinta de la de la
  EILU: mide MATRICULADOS actuales, no la población total que ya tiene
  cada titulación -- una aproximación razonable si la popularidad
  relativa de cada campo es estable en el tiempo (asunción ya implícita
  en que STUDIES_DISTRIBUTION se documenta a sí misma como "MUY
  aproximado"), pero sigue siendo una aproximación, no una medición
  directa de población total como si tiene OCCUPATION_DISTRIBUTION.
  LO QUE FALTA CONFIRMAR, y por lo que NO se ha implementado nada
  todavía en este script: el portal que sirve estos datos
  (estadisticas.ciencia.gob.es, con URLs "dynPx/inebase" muy similares a
  las del propio INE) bloquea el acceso automatizado desde este entorno
  de trabajo (robots.txt), y no se ha podido confirmar si expone una API
  JSON tipo Tempus3 como la del INE (con URL y parámetros documentados)
  o si solo distribuye XLS/CSV para descarga manual -- a diferencia de
  HOUSEHOLD_TYPE_DISTRIBUTION, aquí NO hay una URL de API confirmada por
  ninguna documentación oficial encontrada, así que escribir un
  fetch_studies() ahora mismo sería inventar un mecanismo de acceso sin
  respaldo, algo que este script ha evitado deliberadamente en todo lo
  demás. Trabajo pendiente concreto para una sesión dedicada: entrar a
  https://www.ciencia.gob.es/Ministerio/Estadisticas/SIIU/Estudiantes.html
  desde un navegador normal (sin el bloqueo de robots.txt de este
  entorno) y comprobar si el explorador de tablas PC-Axis de ese portal
  ofrece la misma opción de descarga JSON/API que sí tiene INEbase.

- LANGUAGE_BY_CCAA -- sigue sin un camino que compense el esfuerzo: los
  institutos estadísticos autonómicos de las CCAA con lengua cooficial
  SÍ hacen sus propias encuestas sociolingüísticas, y al menos una
  (IDESCAT, Cataluña, "Encuesta de usos lingüísticos de la población",
  https://www.idescat.cat/pub/?id=eulp&lang=es) tiene mejor periodicidad
  que la ECEPOV del INE (quinquenal -2003/2008/2013/2018/2023- frente a
  los ~7-8 años entre ediciones de la ECEPOV) -- pero sigue sin ser
  anual, y automatizar esto de verdad exigiría repetir esta misma
  investigación por separado para CADA comunidad con lengua cooficial
  (Cataluña, Baleares, C. Valenciana, País Vasco, Galicia, Navarra),
  cada una con su propio instituto, su propio diseño de encuesta, su
  propia periodicidad y sus propias categorías de respuesta -- un
  trabajo de fragmentación en 6 fuentes distintas, no una tabla única
  como el resto de este script. No se ha considerado que compense frente
  al beneficio (pasar de "cada 7-8 años" a "cada 5 años" en el mejor de
  los casos) en esta pasada; se deja documentado como posible tarea
  aparte si en algún momento se decide abordarla región por región.

Estado tras la primera ejecución real (por Nacho, en su máquina): las
tres tablas nuevas (estado civil, nacionalidad, tasas EPA) funcionaron a
la primera, con nombres de serie legibles y valores -- sus IDs quedan
confirmados en la práctica, aunque la normalización final a las claves
de `ine_reference.py` sigue pendiente de revisión humana (ver más
arriba). La de población por provincia (entonces t=2917) devolvió una
lista vacía para las 49 provincias, sin ningún error HTTP.

Estado tras la SEGUNDA ejecución (con el ID corregido a 67988): la tabla
sí devolvió series (216), pero el AVISO de `_warn_if_empty` reveló que el
parseo de `fetch_population_by_province` estaba invertido -- asumía que
el nombre de provincia era el ÚLTIMO segmento del campo `Nombre` cuando
en realidad va PRIMERO. Reescrito para usar el campo estructurado
`MetaData` en vez de trocear ese texto.

Estado tras la TERCERA ejecución (con el parseo por MetaData ya en su
sitio): las 49 provincias casaron -- salvo 13 con nombre oficial bilingüe
actual del INE (p. ej. "Bizkaia", "Girona", "Rioja, La") distinto del
nombre tradicional castellano usado como clave en PROVINCE_POPULATION
("vizcaya", "gerona", "la rioja"). Resuelto con `_INE_TO_CANONICAL_PROVINCE`.

Estado tras la CUARTA ejecución: las 49 provincias casan ya sin ninguna
diferencia (solo quedan, como se espera, los agregados de comunidad
autónoma multiprovincial marcados como "no existe en el código" -- no
son provincias, no deben tener clave en esta tabla). A partir de aquí se
añade el modo `--apply` para esta tabla."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data import ine_reference  # noqa: E402

_INE_API_BASE = "https://servicios.ine.es/wstempus/js/ES"

# ID=67988, "Población según comunidad autónoma y provincia y sexo" --
# confirmado por la ficha oficial del conjunto de datos en datos.gob.es
# (https://datos.gob.es/es/catalogo/ea0042823-poblacion-segun-comunidad-autonoma-y-provincia-y-sexo-identificador-api-67988),
# que da explícitamente la URL de la API:
# https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/67988?tip=AM
# Anual, cobertura 2021-2025, actualizado por última vez en dic. 2025.
#
# El "t=2917" usado antes en este mismo comentario era la tabla
# EQUIVOCADA: existe y es de Tempus3 nativo (por eso la petición no daba
# ningún error), pero es "Población por provincias y TAMAÑO DE LOS
# MUNICIPIOS" (desglose muy distinto), no "... y sexo" -- de ahí que
# ninguna serie encajara con el parseo esperado y el resultado saliera
# vacío. Localizado el ID correcto buscando el título exacto de la tabla
# que SÍ queremos en el catálogo de datos.gob.es, en vez de asumir que el
# "t=" visible en la URL de jaxiT3 de una tabla con nombre similar era la
# correcta -- no verificado todavía contra la API real desde este
# entorno de trabajo (sin acceso, ver LIMITACIÓN más arriba); revisa la
# salida de este script con cuidado la primera vez que lo corras con
# este ID nuevo.
_TABLA_POBLACION_PROVINCIAS = 67988

# Los tres siguientes SÍ son IDs nuevos, localizados por búsqueda web
# dirigida (una búsqueda por tabla, por nombre y contenido esperado) al
# escribir este script -- pero NO se han podido ejecutar contra la API
# real desde este entorno de trabajo (sin acceso de red a
# servicios.ine.es), así que el nombre y la URL de la tabla encajan con
# lo que se busca, pero no está confirmado el desglose exacto de valores
# que devuelve cada una. Revisa la salida de este script con más cuidado
# que la de PROVINCE_POPULATION la primera vez que lo corras.
_TABLA_ESTADO_CIVIL = 76288  # "Población de 16 y más años por sexo y estado civil"
_TABLA_NACIONALIDAD = 59587  # "Población residente por fecha, sexo, grupo de edad y nacionalidad (española/extranjera)"

# RESUELTO tras confirmar que 1113 estaba descontinuada (ver historial
# completo justo abajo): _TABLA_TASAS_EPA (1113, "Tasas de actividad,
# paro y empleo, por sexo y distintos grupos de edad", una sola tabla
# combinada) se sustituye por DOS tablas separadas, encontradas y
# CONFIRMADAS por Nacho -- descargó el Excel real de la 65219 desde
# https://www.ine.es/jaxiT3/Tabla.htm?t=65219 ("consultar todo") y sus
# datos llegan hasta 2026T2 (9,87%) y 2025T4 (9,93%, coincide EXACTO con
# la cifra ya confirmada contra la nota de prensa EPA al principio de
# esta investigación) -- esta SÍ es la tabla vigente, no descontinuada.
# La hermana de "tasa de actividad" (65081) se localizó por búsqueda
# igual que el resto de IDs de este fichero (mismo catálogo de
# datos.gob.es, tema "empleo", misma familia EPA) pero NO se ha
# descargado ni confirmado un Excel real para ella todavía -- candidato
# de alta confianza (mismo catálogo, mismo patrón de nombre/ID que
# 65219), pero no al mismo nivel de certeza que 65219.
_TABLA_TASA_ACTIVIDAD_EPA = 65081  # "Tasas de actividad por sexo y grupo de edad" -- CANDIDATO, sin Excel de verificación descargado
_TABLA_TASA_PARO_EPA = 65219  # "Tasas de paro por sexo y grupo de edad" -- CONFIRMADO con Excel real (datos hasta 2026T2)

# HISTORIAL COMPLETO de 1113 (ya NO se usa, se deja documentado para que
# quede el rastro de por qué se cambió a las dos tablas de arriba):
# CONFIRMADO DESCONTINUADA (primera ejecución real de este script,
# sesión con Nacho): el histórico completo (nult=20) y la descarga
# oficial "consultar todo" desde la propia web del INE
# (https://www.ine.es/jaxiT3/Tabla.htm?t=1113) muestran ambos que esta
# tabla se congela en 2013T4 -- 108 trimestres de datos reales y
# coherentes terminando ahí, no un fallo de `nult` ni de este script.
# Investigado en la misma sesión: las tablas hermanas de esta misma
# operación EPA (3996, 4086, 4247, 4942, 4966) están TODAS marcadas
# explícitamente "histórica" por el INE, con enlace a "resultados
# actuales" -- 1113 casi seguro tiene el mismo problema aunque su propia
# página no mostrara el aviso "histórica" de forma visible. La tabla
# vigente parece vivir bajo un sistema de carpetas por "metodología"
# (PC-Axis, path tipo /t22/e308/meto_XX/pae/px/l0/NNNNN.px, con
# meto_02/meto_05/meto_05_bis como versiones archivadas encontradas) --
# no se ha podido confirmar el ID/ruta de la versión vigente por
# búsqueda web; pendiente de que Nacho lo localice a mano en la propia
# web del INE (mismo método que ya usó para descargar 1113.xlsx) y lo
# pase para terminar de cablear esto.

# ID=65134, "Ocupados por sexo y ocupación. Valores absolutos y
# porcentajes respecto del total de cada sexo" (EPA, trimestral,
# nacional, desglose por subgrupo principal -2 dígitos- de la CNO-11) --
# confirmado por la ficha oficial del conjunto de datos en datos.gob.es
# (https://datos.gob.es/es/catalogo/ea0042823-ocupados-por-sexo-y-ocupacion-valores-absolutos-y-porcentajes-respecto-del-total-de-cada-sexo-epa-identificador-api-4143),
# cuyo título muestra el ID real (65134) aunque la URL conserve en el
# slug un ID antiguo/redirigido (4143) -- la propia ficha da la URL JSON
# de Tempus3 usada por este script:
# https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/65134?tip=AM
# Cobertura 2011-2026, identificador Tempus3 urn:ine:es:TABLA:T3:330:4143
# (el "4143" que sí aparece aquí es el ID interno T3:330:xxxx, DISTINTO
# del ID de tabla 65134 usado en la URL de descarga -- confusión real de
# la propia ficha de datos.gob.es, no un error de este script; se usa
# 65134 porque es el que da la URL JSON explícita).
#
# NO verificado todavía contra la API real desde este entorno de trabajo
# (mismo motivo que _TABLA_ESTADO_CIVIL/_TABLA_NACIONALIDAD/_TABLA_TASAS_EPA:
# sin acceso de red a servicios.ine.es aquí) -- revisa la salida de este
# script con cuidado la primera vez que lo corras con este ID.
_TABLA_OCUPACION_CNO11 = 65134

# "t20/p274/serie/prov/p01/l0/01013.px" -- Encuesta Continua de Hogares
# (ECH), "Número de hogares según el tipo de hogar y el tipo de edificio
# donde se encuentra la vivienda", NACIONAL. A diferencia de las tablas
# anteriores, esta NO tiene un ID numérico Tempus3 -- es una tabla
# PC-Axis, identificada por una RUTA (path + fichero .px), tal como ya
# se sospechaba en el registro de trabajo antes de esta sesión ("aquí sí
# que es el caso genuino de tabla PC-Axis"). Dos cosas se han resuelto
# en esta sesión, no solo una:
#
# 1. CÓMO SE LLAMA una tabla PC-Axis desde la misma API Tempus3 que ya
#    usa el resto de este script: la documentación oficial del INE
#    (https://www.ine.es/dyngs/DAB/index.htm?cid=1102) da un ejemplo
#    literal para OTRA tabla PC-Axis con el mismo formato de ruta:
#    https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/t20/e245/p08/l0/01001.px?nult=2&tv=...
#    -- es decir, el mismo endpoint DATOS_TABLA de `_fetch_series`,
#    pasando la ruta completa (path + fichero) en vez de un ID numérico.
#    `_fetch_series` ya funciona para esto sin cambios: construye la URL
#    con un f-string, así que un `table_id` de tipo `str` con la ruta
#    completa encaja directamente en `{_INE_API_BASE}/DATOS_TABLA/{table_id}`.
#
# 2. QUÉ RUTA CONCRETA usar: localizada por búsqueda del título exacto
#    de la tabla (mismo método que el resto de IDs de este fichero), y
#    confirmada -- a diferencia de estado civil/nacionalidad/tasas
#    EPA/ocupación, aquí SÍ hay evidencia directa de que las categorías
#    de "tipo de hogar" de esta tabla concreta son las que necesitamos:
#    la ficha de datos.gob.es
#    (https://datos.gob.es/es/catalogo/ea0010587-numero-de-hogares-segun-el-tipo-de-hogar-y-el-tipo-de-edificio-donde-se-encuentra-la-vivienda-identificador-api-t20-p274-serie-prov-p01-l0-01013-px1)
#    incluye entre sus etiquetas, tal cual: "Hogar unipersonal", "Hogar
#    monoparental", "Núcleo familiar con otras personas que no forman
#    núcleo familiar", "Dos o más núcleos familiares" -- coinciden
#    exactamente con las categorías reales de la ECH ya confirmadas por
#    fuentes espejo de institutos autonómicos (p. ej. ICANE) para esta
#    misma variable "tipo de hogar", que además incluye "Pareja sin
#    hijos que convivan en el hogar" y "Pareja con hijos que convivan en
#    el hogar" (estas dos no aparecían truncadas en las etiquetas
#    encontradas, pero son parte de la misma clasificación oficial de la
#    ECH -- ver `_TIPO_HOGAR_TO_APP_CATEGORY` más abajo).
#
# Alternativa NO usada pero igual de válida si esta diera problemas al
# ejecutarla de verdad: "t20/p274/serie/def/p01/l0/01003.px" (mismo
# desglose de tipo de hogar, cruzado con número de habitaciones de la
# vivienda en vez de con tipo de edificio) -- misma variable de interés,
# solo cambia el segundo eje de la tabla.
#
# NO verificado todavía contra la API real desde este entorno de trabajo
# (sin acceso de red a servicios.ine.es) -- revisa la salida de este
# script con cuidado la primera vez que lo corras con esta tabla.
_TABLA_HOGAR_TIPO = "t20/p274/serie/prov/p01/l0/01013.px"


# El INE usa el nombre OFICIAL bilingüe actual para las provincias con
# lengua cooficial (y alfabetiza poniendo el artículo detrás de una coma,
# p. ej. "Rioja, La"), mientras que PROVINCE_POPULATION (ine_reference.py)
# usa desde siempre el nombre tradicional monolingüe en castellano como
# clave canónica -- y el resto de la app ya depende de esas claves para
# el matching (hashtags, ubicaciones detectadas, etc.), así que NO se
# tocan. Este mapa traduce lo que devuelve el INE a la clave canónica que
# ya existe, solo dentro de este script de comprobación/actualización.
# Confirmado con la segunda ejecución real de este script (ver output):
# el parseo por MetaData ya funcionaba bien -- lo único que fallaba para
# estas provincias concretas era la nomenclatura, no el parseo.
_INE_TO_CANONICAL_PROVINCE = {
    "alicante/alacant": "alicante",
    "araba/alava": "alava",
    "balears, illes": "baleares",
    "bizkaia": "vizcaya",
    "castellon/castello": "castellon",
    "coruna, a": "a coruna",
    "girona": "gerona",
    "gipuzkoa": "guipuzcoa",
    "lleida": "lerida",
    "ourense": "orense",
    "palmas, las": "las palmas",
    "rioja, la": "la rioja",
    "valencia/valencia": "valencia",
}


def _fetch_series(table_id: int | str) -> list[dict]:
    """Helper compartido: pide el último dato de cada serie de una tabla
    Tempus3. Devuelve la lista cruda de series tal como la da el INE
    (cada una con al menos "Nombre" y "Data") -- cada función fetch_*
    decide cómo interpretar los nombres de serie de SU tabla concreta,
    porque el formato del campo "Nombre" varía de una tabla a otra.

    `tip=AM` (amigable + metadatos) no es estrictamente necesario para las
    tablas que ya funcionan (MARITAL_STATUS/NATIONALITY/SITUACION_LABORAL
    ya trajeron "Nombre" legible sin él, en la primera ejecución real de
    este script) -- se añade de todos modos porque no hace daño y da más
    contexto para depurar si hiciera falta."""
    response = httpx.get(
        f"{_INE_API_BASE}/DATOS_TABLA/{table_id}",
        params={"nult": 1, "tip": "AM"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _warn_if_empty(table_id: int | str, series: list[dict], result: dict) -> None:
    """Diagnóstico defensivo. Cubre los dos síntomas distintos que puede
    dar una tabla mal identificada:

    1. `series` YA viene vacía (el INE no devolvió ninguna serie para ese
       ID). CAUSA REAL confirmada para _TABLA_POBLACION_PROVINCIAS (ver
       comentario junto a esa constante): no era un problema de formato
       de identificador (la hipótesis inicial de "quizá es una tabla
       PC-Axis con ruta de varios segmentos" quedó descartada al
       confirmar el ID correcto por su título exacto en el catálogo de
       datos.gob.es) -- simplemente el ID usado (2917) era el de OTRA
       tabla con un nombre parecido ("... y tamaño de los municipios" en
       vez de "... y sexo"), que sí existe y no da ningún error, pero
       nunca iba a tener series con el desglose que se buscaba. Primer
       paso ante este síntoma: verificar el TÍTULO EXACTO de la tabla
       usando ese ID (p. ej. en https://www.ine.es/jaxiT3/Tabla.htm?t=<ID>
       o buscando "Identificador API: <ID>" en datos.gob.es) antes de
       asumir un problema de formato PC-Axis vs Tempus3 -- ese sigue
       siendo un motivo posible en general (ver
       https://www.ine.es/dyngs/DAB/index.htm?cid=1102), pero compruébese
       el título primero, es más probable y más rápido de confirmar.
    2. `series` SÍ trae datos pero ninguno se pudo interpretar -- el
       formato de "Nombre" de esa tabla concreta no encaja con el parseo
       de la función fetch_* correspondiente (orden de campos distinto,
       idioma, etc.)."""
    if not series:
        print(
            f"  AVISO: la tabla {table_id} no devolvió NINGUNA serie (lista "
            "vacía, sin error HTTP) -- antes de nada, comprueba que el ID "
            "corresponde de verdad al título de tabla esperado (ver "
            "docstring de _warn_if_empty para el motivo real ya encontrado "
            "una vez con este mismo script)."
        )
        return
    if not result:
        print(
            f"  AVISO: la tabla {table_id} devolvió {len(series)} series, pero "
            "ninguna se pudo interpretar (revisa el formato real de 'Nombre' "
            "más abajo y ajusta el parseo de la función fetch_* correspondiente):"
        )
        for serie in series[:2]:
            print(f"    ejemplo crudo: {serie}")


def _mostrar_ejemplos_nombres(raw: dict, *, filtro_subcadenas: list[str] | None = None, max_n: int = 20) -> None:
    """Imprime hasta `max_n` claves `Nombre` reales de `raw` -- a
    diferencia de `_warn_if_empty` (que solo dispara cuando `raw`/`series`
    quedan del todo vacíos), esto se llama cuando el fetch SÍ trajo datos
    pero la normalización posterior no encontró las filas que esperaba
    (p. ej. SITUACION_LABORAL_DISTRIBUTION con una tasa de paro
    implausible, u OCCUPATION_DISTRIBUTION/HOUSEHOLD_TYPE_DISTRIBUTION
    sin ninguna fila reconocible) -- en esos casos `_warn_if_empty` no
    imprime nada útil porque `raw` no está vacío, solo mal interpretado.
    Ver esta salida real es lo único que permite ajustar el parseo de
    forma definitiva en vez de seguir hipotetizando sin verlo."""
    claves = list(raw.keys())
    if filtro_subcadenas:
        claves_filtradas = [
            c for c in claves
            if any(sub.lower() in c.lower() for sub in filtro_subcadenas)
        ]
    else:
        claves_filtradas = claves

    print(f"  Ejemplos reales de 'Nombre' ({len(claves_filtradas)} de {len(claves)} totales, mostrando hasta {max_n}):")
    for clave in claves_filtradas[:max_n]:
        print(f"    {clave!r} -> {raw[clave]}")
    if not claves_filtradas:
        print(f"    (ningún 'Nombre' contiene {filtro_subcadenas} -- aquí van {min(max_n, len(claves))} claves sin filtrar en su lugar:)")
        for clave in claves[:max_n]:
            print(f"    {clave!r} -> {raw[clave]}")


def _diagnosticar_series_duplicadas(table_id: int | str, nombre_exacto: str) -> None:
    """DIAGNÓSTICO EXTRA, más caro que `_mostrar_ejemplos_nombres` (vuelve
    a pedir la tabla entera a la API) -- se llama solo cuando ya se sabe
    qué `Nombre` exacto da un valor implausible (p. ej. la tasa de paro),
    para ver TODAS las series que comparten ese texto EXACTO con su
    array `Data` COMPLETO (no solo el punto que `_latest_valor_por_nombre`
    ya eligió como "más reciente").

    Por qué hace falta esto y no basta con `_mostrar_ejemplos_nombres`:
    esa otra función opera sobre `raw`, que ya está COLAPSADO por
    `_latest_valor_por_nombre` -- por construcción, solo puede enseñar
    UN valor por Nombre (el que ganó), nunca si había varias series
    compitiendo por la misma clave ni qué pinta tenían sus campos de
    fecha reales. Si de verdad hay una serie antigua/descontinuada
    compartiendo Nombre con la vigente (la hipótesis original, nunca
    confirmada con datos reales), esto es lo único que puede probarlo o
    descartarlo de una vez.

    RESULTADO YA CONOCIDO (primera vez que se corrió esto de verdad,
    tabla 1113): NO hay ninguna serie duplicada -- solo existe UNA serie
    (COD='EPAH796') para el Nombre exacto de la tasa de paro nacional
    total, y su único dato (con nult=1) es de 2013-T4. Esto descarta la
    hipótesis de "dos series con el mismo Nombre" -- lo que queda por
    determinar es si esa serie está genuinamente descontinuada desde
    2013, o si `nult=1` no está devolviendo de verdad el periodo más
    reciente para esta tabla (ver `_diagnosticar_historial_completo`,
    añadida después de ver este resultado, para esa segunda pregunta)."""
    print(f"\n  DIAGNÓSTICO EXTRA: repitiendo la petición de la tabla {table_id} (sin colapsar) para buscar Nombre exacto = {nombre_exacto!r}...")
    series = _fetch_series(table_id)
    coincidencias = [s for s in series if s.get("Nombre") == nombre_exacto]
    print(f"  {len(coincidencias)} serie(s) encontradas con ESE Nombre EXACTO (si aquí sale más de 1, ESA es la causa del problema):")
    for i, serie in enumerate(coincidencias):
        print(f"    Serie #{i}: COD={serie.get('COD')!r}")
        for punto in serie.get("Data", []):
            print(f"      {punto}")


def _diagnosticar_historial_completo(table_id: int | str, nombre_exacto: str, *, nult: int = 20) -> None:
    """Repite la petición con `nult` más alto (en vez del `nult=1` fijo
    de `_fetch_series`) para ver el HISTORIAL COMPLETO reciente de la
    serie exacta, no solo el único punto que devuelve nult=1. Se llama
    DESPUÉS de `_diagnosticar_series_duplicadas`, cuando esa ya descartó
    que haya dos series compitiendo -- esto responde una pregunta
    distinta: ¿de verdad la serie no tiene datos más recientes que 2013,
    o es `nult=1` el que no está devolviendo el periodo correcto para
    esta tabla en concreto?

    Si esto muestra puntos posteriores a 2013, `nult=1` tiene un
    problema real para esta tabla y hay que dejar de usarlo aquí (pedir
    con nult más alto y quedarse con el más reciente en el propio código,
    en vez de fiarse de que el servidor ya lo hace). Si esto NO muestra
    nada más reciente que 2013 tampoco, la serie está genuinamente
    descontinuada y hay que buscar un ID de tabla distinto para la tasa
    de paro actual (el título "Tasas de actividad, paro y empleo, por
    sexo y distintos grupos de edad" puede corresponder a más de un ID a
    lo largo del tiempo, si el INE reestructuró la tabla)."""
    print(f"\n  DIAGNÓSTICO EXTRA 2: repitiendo la petición de la tabla {table_id} con nult={nult} (en vez de 1) para Nombre = {nombre_exacto!r}...")
    response = httpx.get(
        f"{_INE_API_BASE}/DATOS_TABLA/{table_id}",
        params={"nult": nult, "tip": "AM"},
        timeout=30,
    )
    response.raise_for_status()
    series = response.json()
    coincidencias = [s for s in series if s.get("Nombre") == nombre_exacto]
    print(f"  {len(coincidencias)} serie(s) encontradas -- Data completo (hasta {nult} puntos por serie):")
    for i, serie in enumerate(coincidencias):
        print(f"    Serie #{i}: COD={serie.get('COD')!r}")
        for punto in serie.get("Data", []):
            print(f"      {punto}")


# ============================================================================
# BUG ENCONTRADO Y CORREGIDO en esta sesión, al investigar por qué
# _TABLA_TASAS_EPA (ID 1113) devolvía una tasa de paro de 26,03% en vez
# del ~9,93% real confirmado contra la nota de prensa oficial (ver
# comentario extenso de _TASA_PARO_RECIENTE_CONOCIDA más abajo, que
# documentaba la hipótesis ANTES de esta sesión -- se deja sin borrar
# para que quede el rastro de qué se pensaba antes y por qué se corrigió).
#
# EVIDENCIA ENCONTRADA (esta vez sí externa y concreta, no solo
# hipótesis): varios ejemplos reales y documentados públicamente de la
# respuesta JSON de esta misma API
# (https://www.lapaginadefinitiva.com/2016/02/14/tutorial-para-leer-con-json-las-estadisticas-del-ine/,
# serie EPA87 "Ocupados"; también replicado en
# https://github.com/es-ine/ineapir) muestran el array `Data` de cada
# serie ORDENADO DESCENDENTEMENTE por fecha: el dato más reciente
# aparece PRIMERO (`Data[0]`), los más antiguos después. TODAS las
# funciones fetch_* de este script (hasta esta sesión) leían
# `datos[-1]["Valor"]` -- el ÚLTIMO elemento del array-- asumiendo que
# era el más reciente. Si `Data` trae más de un periodo para una serie
# (algo que `nult=1` debería evitar, pero sin garantía confirmada para
# todas las tablas -- _TABLA_TASAS_EPA es de las series con más
# historial de este fichero, con datos que arrancan en 2002 o antes),
# `datos[-1]` coge el dato MÁS ANTIGUO, no el más reciente. Un 26,03% de
# tasa de paro encaja perfectamemte con un trimestre real del pico de la
# crisis de 2013 (España rondó el 26-27% ese año) apareciendo por error
# en vez del dato actual.
#
# Esto también resuelve, de paso, el otro mecanismo que se había
# hipotetizado (series distintas -p. ej. metodología vigente/no
# vigente- compartiendo el mismo texto `Nombre` y sobrescribiéndose
# según el orden de iteración del diccionario): en vez de fiarse de CUÁL
# serie se procesó última para decidir qué valor gana, se compara la
# fecha real (`Fecha`, o `Anyo`+`FK_Periodo` si `Fecha` no viniera) de
# TODOS los puntos de TODAS las series que comparten `Nombre`, y se elige
# el genuinamente más reciente -- sea cual sea la razón por la que había
# más de un candidato para la misma clave.
#
# Aplicado a las 5 funciones fetch_* que indexaban por `Nombre` de texto
# (fetch_marital_status, fetch_nationality, fetch_situacion_laboral,
# fetch_occupation, fetch_household_type) vía `_latest_valor_por_nombre`,
# y a fetch_population_by_province (que indexa por MetaData, no por
# Nombre, pero tenía el mismo `datos[-1]` dentro de CADA serie) vía
# `_valor_mas_reciente`. Es decir: el bug que se pidió investigar para
# SITUACION_LABORAL_DISTRIBUTION en realidad afectaba potencialmente a
# las 6 tablas de este script por igual, no solo a esa -- se corrige en
# las 6 a la vez en vez de dejar el mismo fallo latente en las demás.
#
# SIGUE sin poder ejecutarse contra la API real desde este entorno de
# trabajo para confirmar el 100% del mecanismo -- pero a diferencia de la
# hipótesis anterior (sin ningún respaldo externo), esta corrección está
# basada en el formato de `Data` documentado con ejemplos reales fuera de
# este proyecto, no solo en conjetura. Si al ejecutar esto de verdad la
# tasa de paro sigue sin acercarse al ~9-11% esperado, el problema NO es
# este (revisar entonces si de verdad hay dos series con `Nombre`
# IDÉNTICO pero fechas ambas "recientes" -- ambigüedad genuina que ni
# esta corrección puede resolver sola, ver AVISO impreso en main()).
# ============================================================================

def _valor_mas_reciente(datos: list[dict]) -> float | None:
    """Devuelve el `Valor` del punto con la fecha más reciente dentro de
    UN array `Data` de una sola serie (no compara entre series distintas
    -- para eso, ver `_latest_valor_por_nombre`). Usa `Fecha` (timestamp
    Unix en ms, confirmado en la documentación/ejemplos reales de esta
    API) y cae a `Anyo`*100 + `FK_Periodo` si `Fecha` no viniera en algún
    punto -- ambas formas son comparables como número creciente en el
    tiempo, sin necesidad de parsear una fecha real."""
    mejor_fecha: float | None = None
    mejor_valor: float | None = None
    for punto in datos:
        valor = punto.get("Valor")
        if valor is None:
            continue
        fecha = punto.get("Fecha")
        if fecha is None:
            fecha = punto.get("Anyo", 0) * 100 + punto.get("FK_Periodo", 0)
        if mejor_fecha is None or fecha > mejor_fecha:
            mejor_fecha, mejor_valor = fecha, valor
    return mejor_valor


def _latest_valor_por_nombre(series: list[dict]) -> dict[str, float]:
    """Sustituye el patrón `{serie["Nombre"]: serie["Data"][-1]["Valor"]}`
    que usaban (hasta esta sesión) fetch_marital_status/fetch_nationality/
    fetch_situacion_laboral/fetch_occupation/fetch_household_type -- ver
    el bloque de comentario justo encima de `_valor_mas_reciente` para el
    porqué. Considera TODOS los puntos de TODAS las series (no solo el
    último elemento de cada una), y para cada texto de `Nombre` se queda
    con el valor del punto genuinamente más reciente por fecha real, sin
    depender del orden de iteración de `series` ni de que `Data` venga
    ordenado de una forma concreta."""
    mejor: dict[str, tuple[float, float]] = {}  # nombre -> (fecha_ordenable, valor)
    for serie in series:
        nombre = serie.get("Nombre", "")
        for punto in serie.get("Data", []):
            valor = punto.get("Valor")
            if valor is None:
                continue
            fecha = punto.get("Fecha")
            if fecha is None:
                fecha = punto.get("Anyo", 0) * 100 + punto.get("FK_Periodo", 0)
            anterior = mejor.get(nombre)
            if anterior is None or fecha > anterior[0]:
                mejor[nombre] = (fecha, valor)
    return {nombre: valor for nombre, (_fecha, valor) in mejor.items()}


# Las 4 categorías que NO son la variable geográfica en esta tabla, según
# el MetaData real capturado en la ejecución que reveló el formato (ver
# docstring de fetch_population_by_province) -- cualquier entrada de
# MetaData que no sea una de estas 4 es, por descarte, la variable
# geográfica (nacional/CCAA/provincia), sin necesidad de asumir un nombre
# fijo para ella (ver por qué en el docstring de la función).
_NON_GEO_METADATA_VARS = {"Sexo", "Nacionalidad", "Totales de edad", "Tipo de dato"}


def fetch_population_by_province() -> dict[str, int]:
    """Descarga la tabla 67988 del INE (población por comunidad autónoma
    y provincia y sexo) y la deja en el mismo formato de claves que
    PROVINCE_POPULATION (nombre de provincia en minúsculas, sin tildes --
    ver `_strip_accents` en ine_reference.py). nult=1 pide solo el dato
    más reciente disponible de cada serie, no todo el histórico.

    Usa el campo `MetaData` de cada serie (una lista de
    {"T3_Variable": ..., "Nombre": ..., "Codigo": ...}), NO el campo
    `Nombre` de nivel superior -- la primera versión de esta función
    trataba `Nombre` como texto libre y asumía que la provincia era el
    ÚLTIMO segmento tras trocear por ". "; el ejemplo real capturado en la
    primera ejecución que sí llegó a traer datos (ver AVISO impreso por
    `_warn_if_empty`, tabla 67988) demostró que es justo lo CONTRARIO: el
    territorio va PRIMERO ("Total Nacional. Total. Total. Todas las
    edades. Dato base. "), no al final. `MetaData` evita depender del
    orden textual en absoluto: da la categoría y el valor por separado
    para cada dimensión de la serie.

    La variable geográfica en el `MetaData` NO tiene un nombre fijo
    predecible -- en el ejemplo capturado, su propio `T3_Variable` es
    literalmente "Total Nacional" (coincide con su propio valor cuando la
    serie es el agregado nacional), así que asumir ese nombre de variable
    rompería en cuanto una fila fuera de una provincia o CCAA concreta en
    vez del total nacional. En su lugar, se identifica por DESCARTE: es la
    única entrada de `MetaData` que no es Sexo/Nacionalidad/Totales de
    edad/Tipo de dato (ver `_NON_GEO_METADATA_VARS`).

    CONFIRMADO en la segunda ejecución real de este script (ver output
    completo): el parseo por `MetaData` funciona -- la mayoría de
    provincias (Albacete, Almería, Ávila, Badajoz, Barcelona, Burgos,
    Madrid, Zaragoza...) casaron directamente con sus claves ya
    existentes en PROVINCE_POPULATION. Dos síntomas quedaron, distintos
    entre sí:

    1. Tal como se esperaba (ver el título de la tabla, "... por
       COMUNIDAD AUTÓNOMA Y provincia..."), la variable geográfica trae
       TANTO filas de provincia COMO de comunidad autónoma agregada (p.
       ej. "Andalucía", "Cataluña"). Estas últimas no tienen clave en
       PROVINCE_POPULATION a propósito (esa tabla es de POBLACIÓN POR
       PROVINCIA, no por CCAA) y se ven en el `_compare()` de `main()`
       como "(no existe en el código)" -- una señal visible para
       descartarlas a mano, que es justo lo que se ve en el output real:
       ninguna se coló como si fuera una provincia real.
    2. El INE usa el nombre OFICIAL BILINGÜE ACTUAL para las provincias
       con lengua cooficial (p. ej. "Araba/Álava", "Bizkaia", "Girona",
       "Lleida", "Ourense") y alfabetiza poniendo el artículo detrás de
       una coma (p. ej. "Rioja, La", "Palmas, Las", "Coruña, A"), mientras
       que PROVINCE_POPULATION usa el nombre tradicional monolingüe
       castellano de siempre como clave ("vizcaya", "gerona", "lerida",
       "orense", "la rioja", "las palmas", "a coruna") -- no es un fallo
       de parseo, es una diferencia real de nomenclatura entre dos fuentes
       válidas. Resuelto con `_INE_TO_CANONICAL_PROVINCE`: traduce el
       nombre del INE a la clave tradicional ya existente, SOLO dentro de
       este script -- no se toca PROVINCE_POPULATION ni su convención de
       nombres, de la que depende el resto de la app (matching de
       hashtags/ubicaciones detectadas)."""
    result: dict[str, int] = {}
    series = _fetch_series(_TABLA_POBLACION_PROVINCIAS)
    for serie in series:
        metadata = serie.get("MetaData", [])
        datos = serie.get("Data", [])
        if not datos or not metadata:
            continue

        by_variable = {m.get("T3_Variable"): m.get("Nombre") for m in metadata}
        # Filtra al TOTAL simple -- mismo agregado que usa PROVINCE_POPULATION
        # hoy: todas las edades, ambos sexos, todas las nacionalidades,
        # dato base (no una tasa ni una proyección).
        if by_variable.get("Sexo") != "Total":
            continue
        if by_variable.get("Nacionalidad") != "Total":
            continue
        if by_variable.get("Totales de edad") != "Todas las edades":
            continue
        if by_variable.get("Tipo de dato") != "Dato base":
            continue

        geo_entries = [m for m in metadata if m.get("T3_Variable") not in _NON_GEO_METADATA_VARS]
        if len(geo_entries) != 1:
            continue  # formato inesperado para esta serie -- se omite en vez de arriesgar un emparejamiento erróneo

        territorio = (geo_entries[0].get("Nombre") or "").strip()
        if not territorio or territorio == "Total Nacional":
            continue  # agregado nacional, no una provincia ni CCAA

        valor = _valor_mas_reciente(datos)
        if valor is not None:
            clave = ine_reference._strip_accents(territorio).lower()
            # Traduce el nombre oficial bilingüe del INE a la clave
            # canónica tradicional que ya usa PROVINCE_POPULATION (ver
            # _INE_TO_CANONICAL_PROVINCE) -- si no hay alias para esta
            # clave, se deja tal cual (es el caso normal para la mayoría
            # de provincias, que no tienen nombre bilingüe distinto, y
            # también el caso de los agregados de CCAA multiprovincial,
            # que se dejan sin traducir a propósito: ver LIMITACIÓN en el
            # docstring de esta función).
            clave = _INE_TO_CANONICAL_PROVINCE.get(clave, clave)
            result[clave] = int(valor)

    _warn_if_empty(_TABLA_POBLACION_PROVINCIAS, series, result)
    return result


def fetch_marital_status() -> dict[str, float]:
    """CANDIDATO SIN VERIFICAR (ver _TABLA_ESTADO_CIVIL más arriba):
    localizado por búsqueda web dirigida, no ejecutado contra la API
    real. Devuelve valores en BRUTO (nombre de serie tal cual, sin
    normalizar a las claves soltero/casado/con_pareja/divorciado/viudo de
    MARITAL_STATUS_DISTRIBUTION) precisamente porque no está confirmado
    que las categorías de esta tabla EPA encajen 1:1 con las del Censo
    que se usó para derivar esa distribución (ver comentario extenso en
    ine_reference.py sobre cómo se combinaron dos fuentes distintas) --
    la normalización final es una decisión humana que requiere mirar los
    nombres de serie reales primero, no algo que este script deba asumir."""
    series = _fetch_series(_TABLA_ESTADO_CIVIL)
    result = _latest_valor_por_nombre(series)
    _warn_if_empty(_TABLA_ESTADO_CIVIL, series, result)
    return result


def fetch_nationality() -> dict[str, float]:
    """CANDIDATO SIN VERIFICAR (ver _TABLA_NACIONALIDAD más arriba) --
    mismo aviso que fetch_marital_status: valores en bruto, sin normalizar."""
    series = _fetch_series(_TABLA_NACIONALIDAD)
    result = _latest_valor_por_nombre(series)
    _warn_if_empty(_TABLA_NACIONALIDAD, series, result)
    return result


def fetch_situacion_laboral() -> tuple[dict[str, float], dict[str, float]]:
    """Devuelve (actividad_raw, paro_raw) -- DOS tablas separadas desde
    esta sesión (ver _TABLA_TASA_ACTIVIDAD_EPA/_TABLA_TASA_PARO_EPA más
    arriba para el porqué: 1113, la tabla combinada original, resultó
    estar descontinuada desde 2013, confirmado con datos reales).

    SUPUESTO SIN VERIFICAR sobre el formato de `Nombre` de estas DOS
    tablas nuevas: no se ha visto un ejemplo real de la respuesta JSON
    (solo el Excel de "consultar todo" de la web, que no expone los
    nombres de serie de la API). Al ser tablas de un solo concepto cada
    una (no "actividad, paro y empleo" combinadas como 1113), lo más
    probable es que `Nombre` sea más simple -- posiblemente sin el
    prefijo "Tasa de paro. Nacional." de antes, algo más parecido a
    "Total. Ambos sexos. Valor absoluto" o similar. `_normalize_situacion_laboral`
    intenta varias formas plausibles; si ninguna encaja, hace falta ver
    un ejemplo real con `_mostrar_ejemplos_nombres` (ya se llama
    automáticamente si el resultado sale sospechoso)."""
    series_actividad = _fetch_series(_TABLA_TASA_ACTIVIDAD_EPA)
    actividad_raw = _latest_valor_por_nombre(series_actividad)
    _warn_if_empty(_TABLA_TASA_ACTIVIDAD_EPA, series_actividad, actividad_raw)

    series_paro = _fetch_series(_TABLA_TASA_PARO_EPA)
    paro_raw = _latest_valor_por_nombre(series_paro)
    _warn_if_empty(_TABLA_TASA_PARO_EPA, series_paro, paro_raw)

    return actividad_raw, paro_raw


def fetch_occupation() -> dict[str, float]:
    """CANDIDATO SIN VERIFICAR (ver _TABLA_OCUPACION_CNO11 más arriba) --
    mismo aviso que fetch_marital_status/fetch_situacion_laboral: valores
    en bruto tal como los da `Nombre`, sin normalizar ni filtrar
    sexo/unidad todavía (eso lo hace `_normalize_occupation`).

    AVISO DE FORMATO, distinto del resto de tablas de este script: el
    título de la tabla ("... Valores absolutos Y porcentajes respecto
    del total de cada sexo") indica que probablemente trae, para cada
    subgrupo CNO-11, tanto la cifra absoluta como el porcentaje -- y
    encima desglosado por sexo (Total/Hombres/Mujeres), igual que
    _TABLA_ESTADO_CIVIL. Eso significa que lo más probable es que varias
    series compartan el mismo texto de `Nombre` si ese texto no incluye
    ambas dimensiones (sexo Y tipo de dato) de forma explícita, con el
    mismo riesgo de sobrescritura silenciosa ya documentado en
    `_warn_if_empty` y en el comentario de `_TASA_PARO_RECIENTE_CONOCIDA`
    para _TABLA_TASAS_EPA. NO se ha podido confirmar el formato real de
    `Nombre` para esta tabla concreta (sin acceso a la API desde aquí) --
    `_normalize_occupation` documenta el supuesto que hace mientras tanto
    y cómo verificarlo en la primera ejecución real.

    Usa `_latest_valor_por_nombre` desde esta sesión -- ver el bug real
    que corrige junto a esa función."""
    series = _fetch_series(_TABLA_OCUPACION_CNO11)
    result = _latest_valor_por_nombre(series)
    _warn_if_empty(_TABLA_OCUPACION_CNO11, series, result)
    return result


# Mapeo de subgrupo principal (2 dígitos) de la CNO-11 -- Clasificación
# Nacional de Ocupaciones 2011, ver notas explicativas del INE
# (https://www.ine.es/daco/daco42/clasificaciones/cno11_notas.pdf) -- a
# las 10 categorías, mucho más amplias y coloquiales, que ya usa
# OCCUPATION_DISTRIBUTION en ine_reference.py. Es EL trabajo de diseño
# que el registro de trabajo (sesión anterior) dejó pendiente a
# propósito en vez de resolverlo a ciegas: cada clave de la izquierda es
# el nombre real de un subgrupo principal CNO-11 (confirmado por las
# etiquetas de la ficha de datos.gob.es de _TABLA_OCUPACION_CNO11 y por
# la estructura oficial de 9 grandes grupos / subgrupos principales de la
# CNO-11), cada valor de la derecha es la clave de OCCUPATION_DISTRIBUTION
# a la que ese subgrupo contribuye.
#
# ES UN MAPEO MUCHOS-A-UNO Y DELIBERADAMENTE INCOMPLETO: la CNO-11 tiene
# unos 40 subgrupos principales y OCCUPATION_DISTRIBUTION solo 10
# categorías -- la mayoría de subgrupos (agricultura, industria pesada,
# fuerzas armadas, servicios domésticos, artes y espectáculo...) no
# encajan en ninguna de las 10 categorías actuales y se dejan SIN
# mapear a propósito (no contribuyen a ninguna clave, no es un olvido).
# Si en el futuro se amplía OCCUPATION_DISTRIBUTION con más categorías,
# hay que revisar también qué subgrupos sin mapear deberían entrar.
#
# Cada subgrupo se cuenta en UNA sola categoría de la app (no hay reparto
# fraccionario entre dos claves), incluso cuando el encaje es parcial --
# p. ej. "Dependientes en tiendas y almacenes" se cuenta entero en
# "comercial" aunque parte de ese subgrupo podría razonablemente
# describirse como "hosteleria" en algún caso concreto; es una
# aproximación deliberada, coherente con el resto de este fichero (ver
# p. ej. el reparto de inactivos en _normalize_situacion_laboral).
_CNO11_SUBGRUPO_TO_APP_CATEGORY: dict[str, str] = {
    # docente
    "Profesionales de la enseñanza infantil, primaria, secundaria y postsecundaria": "docente",
    "Otros profesionales de la enseñanza": "docente",
    # sanitario
    "Profesionales de la salud": "sanitario",
    "Técnicos sanitarios y profesionales de las terapias alternativas": "sanitario",
    "Trabajadores de los cuidados a las personas en servicios de salud": "sanitario",
    # desarrollador de software
    "Profesionales de las tecnologías de la información": "desarrollador de software",
    "Técnicos de las tecnologías de la información y las comunicaciones (TIC)": "desarrollador de software",
    # ingeniero
    "Profesionales de la ciencias físicas, químicas, matemáticas y de las ingenierías": "ingeniero",
    "Técnicos de las ciencias y de las ingenierías": "ingeniero",
    # abogado
    "Profesionales en derecho": "abogado",
    # comercial
    "Representantes, agentes comerciales y afines": "comercial",
    "Comerciantes propietarios de tiendas": "comercial",
    "Vendedores (excepto en tiendas y almacenes)": "comercial",
    "Dependientes en tiendas y almacenes": "comercial",
    # hosteleria
    "Camareros y cocineros propietarios": "hosteleria",
    "Trabajadores asalariados de los servicios de restauración": "hosteleria",
    # administracion publica
    #
    # OJO: el texto de la primera clave se corrigió tras ver la lista real
    # de 84 ocupaciones únicas (primera ejecución real) -- el texto
    # anterior estaba TRUNCADO ("...Administración Pública" a secas) y no
    # coincidía con el real, así que esta fila nunca se sumaba. El texto
    # completo real es más largo (incluye "...y organizaciones de interés
    # social; directores ejecutivos").
    "Miembros del poder ejecutivo y de los cuerpos legislativos; directivos de la Administración Pública y organizaciones de interés social; directores ejecutivos": "administracion publica",
    "Especialistas en organización de la Administración Pública y de las empresas y en la comercialización": "administracion publica",
    "Empleados administrativos con tareas de atención al público no clasificados bajo otros epígrafes": "administracion publica",
    "Otros empleados administrativos sin tareas de atención al público": "administracion publica",
    # construccion
    #
    # OJO: se ha quitado "Trabajadores cualificados de la construcción,
    # excepto operadores de máquinas" de aquí -- confirmado con la lista
    # real de 84 ocupaciones (primera ejecución real) que es el PADRE
    # directo de las dos filas de abajo (obras estructurales + acabados),
    # así que tenerlo también aquí sumaba el mismo colectivo dos veces
    # (la suma total salía en 66.4% en vez de max. 55%). Se dejan solo
    # los hijos, más específicos y sin solapamiento entre sí.
    "Trabajadores en obras estructurales de construcción y afines": "construccion",
    "Trabajadores de acabado de construcciones e instalaciones (excepto electricistas), pintores y afines": "construccion",
    "Peones de la construcción y de la minería": "construccion",
    # transporte
    #
    # OJO: mismo problema que construccion -- "Conductores y operadores
    # de maquinaria móvil" es el PADRE directo de las dos filas de abajo
    # (conductores urbanos + maquinistas), quitado de aquí por el mismo
    # motivo (doble conteo confirmado con datos reales).
    "Conductores de vehículos para el transporte urbano o por carretera": "transporte",
    "Peones del transporte, descargadores y reponedores": "transporte",
    "Maquinistas de locomotoras, operadores de maquinaria agrícola y de equipos pesados móviles, y marineros": "transporte",
}

# Rango de plausibilidad para la SUMA de las 10 categorías mapeadas --
# guarda de seguridad, mismo espíritu que _TASA_PARO_PLAUSIBLE. Como el
# mapeo es DELIBERADAMENTE incompleto (ver comentario de
# _CNO11_SUBGRUPO_TO_APP_CATEGORY), la suma nunca debe acercarse al 100%
# -- si se acerca o lo supera, lo más probable es que `Nombre` esté
# repitiendo la misma serie varias veces por sexo/tipo de dato sin
# filtrar (ver AVISO DE FORMATO en fetch_occupation), no que de repente
# el mapeo cubra toda la población ocupada.
#
# LÍMITE SUPERIOR AJUSTADO a 62 (era 55) tras verificar contra datos
# reales, dos veces: la primera ejecución real dio 66.4% -- SÍ era un bug
# real (dos categorías tenían un grupo "padre" y sus propios "hijos" del
# CNO-11 sumados a la vez, ver historial en el propio diccionario). Tras
# arreglarlo, la SEGUNDA ejecución real dio 57.2% -- revisado a mano
# entrada por entrada contra la jerarquía completa del CNO-11 (84
# ocupaciones únicas vistas), SIN ningún padre marcado junto a sus
# hijos esta vez. 57.2% es, por tanto, el resultado correcto para estas
# 10 categorías (bastante amplias: comercial+construcción+transporte+
# sanitario+docente ya son, cada una, una fracción considerable de la
# población ocupada) -- el límite de 55 era una estimación a mano
# demasiado ajustada, no una señal real de bug. Se deja algo de margen
# por encima de 57.2 (hasta 62) para seguir detectando una regresión real
# si el mapeo se rompe de nuevo, sin disparar una alarma falsa por este
# valor ya confirmado correcto.
_OCUPACION_SUMA_PLAUSIBLE = (15.0, 62.0)


def _normalize_occupation(raw: dict[str, float]) -> dict[str, float] | None:
    """Filtra `raw` (salida cruda de fetch_occupation) a las filas de
    ÁMBITO NACIONAL, AMBOS SEXOS y PORCENTAJE (no cifra absoluta ni
    desglose por sexo) y sea cual sea el subgrupo CNO-11 de cada fila, lo
    suma dentro de la categoría de OCCUPATION_DISTRIBUTION a la que
    apunte `_CNO11_SUBGRUPO_TO_APP_CATEGORY` (ninguna, una entrada, o
    varias entradas sumadas en la misma categoría).

    FORMATO REAL confirmado contra la API en vivo (ya no es un supuesto
    sin verificar): "Ocupados. Total Nacional. <Sexo>. <Ocupación>.
    <TipoDato>. " -- con un PREFIJO FIJO de 2 segmentos ("Ocupados",
    "Total Nacional") antes del sexo, que no se había visto en ningún
    ejemplo real hasta la primera ejecución contra la API en vivo. El
    supuesto anterior (sexo en segments[0]) cogía literalmente
    "Ocupados" como si fuera el sexo, así que ninguna fila pasaba nunca
    el filtro `sexo != "Ambos sexos"`.

    Nivel de jerarquía CNO-11: esta tabla mezcla en la misma lista plana
    filas de varios niveles de la clasificación a la vez (gran grupo de
    1 dígito, subgrupo de 2, sub-subgrupo de 3...) -- p. ej. "Técnicos y
    profesionales científicos e intelectuales" (1 dígito) Y, aparte,
    "Profesionales de la salud" (más granular, dentro del anterior)
    aparecen ambas como filas independientes. `_CNO11_SUBGRUPO_TO_APP_CATEGORY`
    se diseñó a propósito usando solo nombres granular-específicos (nunca
    los nombres de gran grupo de 1 dígito), así que no debería haber
    doble conteo por mezclar padre e hijo -- pero si `OCCUPATION_DISTRIBUTION`
    sale con una suma sospechosamente alta, revisar primero si alguna
    clave del diccionario coincide sin querer con un nombre de nivel
    superior en vez de nivel específico."""
    por_categoria: dict[str, float] = {}
    alguna_fila_nacional_valida = False
    for nombre, valor in raw.items():
        segments = [s.strip() for s in nombre.split(". ") if s.strip()]
        if len(segments) < 4 or segments[0] != "Ocupados":
            continue
        sexo = segments[2]
        tipo_dato = segments[-1]
        ocupacion = ". ".join(segments[3:-1])
        if sexo != "Ambos sexos":
            continue
        if "orcentaje" not in tipo_dato and "%" not in tipo_dato:
            continue
        categoria = _CNO11_SUBGRUPO_TO_APP_CATEGORY.get(ocupacion)
        alguna_fila_nacional_valida = True
        if categoria is None:
            continue  # subgrupo sin mapear a propósito, ver comentario del diccionario
        por_categoria[categoria] = por_categoria.get(categoria, 0.0) + valor

    if not alguna_fila_nacional_valida or not por_categoria:
        return None

    return {clave: round(valor / 100, 3) for clave, valor in por_categoria.items()}


def fetch_household_type() -> dict[str, float]:
    """CANDIDATO SIN VERIFICAR (ver _TABLA_HOGAR_TIPO más arriba) --
    misma llamada que el resto de fetch_*, pero con un identificador de
    RUTA en vez de un ID numérico (ver comentario de la constante para
    por qué `_fetch_series` funciona igual para ambos casos).

    A diferencia de las tablas EPA de este script (que ya dan
    porcentajes), esta tabla da CIFRAS ABSOLUTAS ("Número de hogares"),
    no proporciones -- `_normalize_household_type` calcula la proporción
    dividiendo por la fila "Total" de la propia tabla, no asume un
    total de hogares aparte.

    Usa `_latest_valor_por_nombre` desde esta sesión -- ver el bug real
    que corrige junto a esa función."""
    series = _fetch_series(_TABLA_HOGAR_TIPO)
    result = _latest_valor_por_nombre(series)
    _warn_if_empty(_TABLA_HOGAR_TIPO, series, result)
    return result


# Categorías reales de "tipo de hogar" de la ECH (confirmadas por fuentes
# espejo de institutos autonómicos que replican esta misma tabla del INE,
# ver comentario de _TABLA_HOGAR_TIPO) mapeadas a las 5 claves que ya usa
# HOUSEHOLD_TYPE_DISTRIBUTION. A diferencia de
# _CNO11_SUBGRUPO_TO_APP_CATEGORY, este mapeo SÍ es exhaustivo -- la ECH
# solo tiene estas categorías de tipo de hogar (más su fila "Total"), así
# que las 5 claves de la app deberían cubrir el 100% de los hogares, no
# un subconjunto. "otro" agrupa las 3 categorías residuales de hogares
# complejos/multi-núcleo, igual que ya documentaba el comentario de
# HOUSEHOLD_TYPE_DISTRIBUTION en ine_reference.py antes de esta sesión.
_TIPO_HOGAR_TO_APP_CATEGORY: dict[str, str] = {
    "Hogar unipersonal": "unipersonal",
    "Pareja sin hijos que convivan en el hogar": "pareja_sin_hijos",
    "Pareja con hijos que convivan en el hogar": "pareja_con_hijos",
    "Hogar monoparental": "monoparental",
    "Núcleo familiar con otras personas que no forman núcleo familiar": "otro",
    "Personas que no forman ningún núcleo familiar entre sí": "otro",
    "Dos o más núcleos familiares": "otro",
}

# Rango de plausibilidad para la SUMA de las 5 categorías -- a diferencia
# de _OCUPACION_SUMA_PLAUSIBLE (mapeo deliberadamente incompleto), aquí
# el mapeo SÍ es exhaustivo (ver comentario de _TIPO_HOGAR_TO_APP_CATEGORY),
# así que la suma debería acercarse a 1.0 (100% de los hogares). Un valor
# lejos de 1.0 indica más probablemente un fallo de parseo (p. ej. no
# haber filtrado bien la fila "Total" del segundo eje de la tabla, con lo
# que se estarían sumando varias filas del mismo hogar cruzadas con cada
# tipo de edificio) que una categoría real sin cubrir.
_HOGAR_SUMA_PLAUSIBLE = (0.85, 1.15)


def _categorizar_tipo_hogar(tipo_hogar: str) -> str | None:
    """Wrapper sobre `_TIPO_HOGAR_TO_APP_CATEGORY` que además reconoce
    "Pareja con hijos que convivan en el hogar" -- CONFIRMADO contra la
    API real (primera ejecución real) que esta categoría tiene DOS
    formas de aparecer en la misma tabla: una fila resumen
    "...: Total" (ya es la suma de todo) Y, ADEMÁS, filas desglosadas por
    número de hijos ("...: 1 hijo", "...: 2 hijos", "...: 3 o más
    hijos"...). El primer intento de reconocer esto usaba `startswith`
    sobre CUALQUIER sufijo -- eso sumaba la fila ": Total" JUNTO CON sus
    propias partes desglosadas, duplicando el conteo (la suma total de
    las 5 categorías salió en 133% en vez de ~100%). Ahora se usa
    ÚNICAMENTE la fila ": Total" (ya correcta de por sí, sin tener que
    sumar nada a mano) y se ignoran a propósito las filas desglosadas por
    número de hijos."""
    categoria = _TIPO_HOGAR_TO_APP_CATEGORY.get(tipo_hogar)
    if categoria is not None:
        return categoria
    if tipo_hogar == "Pareja con hijos que convivan en el hogar: Total":
        return "pareja_con_hijos"
    return None


def _normalize_household_type(raw: dict[str, float]) -> dict[str, float] | None:
    """FORMATO REAL confirmado contra la API en vivo (ya no es un
    supuesto sin verificar): "<TipoDeHogar>, <TipoDeEdificio>" -- con
    COMA+espacio como separador, NO punto+espacio como se asumía antes
    de ver un ejemplo real (p. ej. 'Hogar unipersonal, Total',
    'Total (tipo de hogar), Vivienda unifamiliar independiente'). Ese
    supuesto equivocado explicaba exactamente el fallo visto en la
    primera ejecución real: el `split('. ')` no partía nada (no hay
    ningún '. ' en el string), así que ninguna fila pasaba nunca el
    `len(segments) < 2`.

    La fila de gran total tampoco se llama "Total" a secas en el primer
    eje, se llama literalmente "Total (tipo de hogar)" -- de ahí el
    `{"total", "total (tipo de hogar)"}` de abajo en vez de una sola
    cadena fija.

    Ver `_categorizar_tipo_hogar` para el caso especial de "Pareja con
    hijos", partido en sub-filas por número de hijos."""
    total: float | None = None
    por_categoria: dict[str, float] = {}
    etiquetas_total = {"total", "total (tipo de hogar)"}
    for nombre, valor in raw.items():
        segments = [s.strip() for s in nombre.split(", ") if s.strip()]
        if len(segments) < 2:
            continue
        tipo_hogar = segments[0]
        resto = segments[1:]
        if not any(s.lower() == "total" for s in resto):
            continue  # fila de un tipo de edificio concreto, no la marginal

        if tipo_hogar.lower() in etiquetas_total:
            total = valor
            continue
        categoria = _categorizar_tipo_hogar(tipo_hogar)
        if categoria is None:
            continue  # categoría de tipo de hogar no reconocida -- revisar el mapeo si aparece en el AVISO
        por_categoria[categoria] = por_categoria.get(categoria, 0.0) + valor

    if not total or not por_categoria:
        return None

    return {clave: round(valor / total, 3) for clave, valor in por_categoria.items()}


def _diagnosticar_ocupaciones_unicas(raw: dict) -> None:
    """Para OCCUPATION_DISTRIBUTION cuando la suma de categorías mapeadas
    sale implausible (visto en la primera ejecución real: 66.4%, muy por
    encima del rango (15,55) esperado) -- extrae y muestra el nombre de
    OCUPACIÓN único de cada fila (sin repetir por sexo/tipo de dato),
    usando el mismo parseo que `_normalize_occupation`. La sospecha a
    confirmar/descartar con esto: la tabla mezcla varios NIVELES de la
    jerarquía CNO-11 en la misma lista plana (gran grupo de 1 dígito,
    subgrupo de 2, sub-subgrupo de 3...), y si dos nombres de
    `_CNO11_SUBGRUPO_TO_APP_CATEGORY` resultan ser padre e hijo el uno
    del otro, se estaría sumando el mismo grupo de ocupados dos veces."""
    ocupaciones_unicas: set[str] = set()
    for nombre in raw:
        segments = [s.strip() for s in nombre.split(". ") if s.strip()]
        if len(segments) < 4 or segments[0] != "Ocupados":
            continue
        ocupaciones_unicas.add(". ".join(segments[3:-1]))
    print(f"\n  DIAGNÓSTICO EXTRA: {len(ocupaciones_unicas)} nombres de ocupación ÚNICOS encontrados (sin repetir por sexo/tipo de dato):")
    for ocupacion in sorted(ocupaciones_unicas):
        marcado = " <-- EN _CNO11_SUBGRUPO_TO_APP_CATEGORY" if ocupacion in _CNO11_SUBGRUPO_TO_APP_CATEGORY else ""
        print(f"    {ocupacion!r}{marcado}")


def _diagnosticar_tipos_hogar_unicos(raw: dict) -> None:
    """Para HOUSEHOLD_TYPE_DISTRIBUTION cuando falta alguna categoría del
    todo (visto en la primera ejecución real: 'pareja_con_hijos' no
    aparece en absoluto, y la suma total queda en 67% en vez de ~100%)
    -- extrae y muestra el nombre de TIPO DE HOGAR único de cada fila
    (primer segmento antes de la coma), para ver la redacción EXACTA que
    usa la API y compararla con las claves de
    `_TIPO_HOGAR_TO_APP_CATEGORY`."""
    tipos_unicos: set[str] = set()
    for nombre in raw:
        segments = [s.strip() for s in nombre.split(", ") if s.strip()]
        if segments:
            tipos_unicos.add(segments[0])
    print(f"\n  DIAGNÓSTICO EXTRA: {len(tipos_unicos)} nombres de tipo de hogar ÚNICOS encontrados:")
    for tipo in sorted(tipos_unicos):
        marcado = " <-- reconocido" if _categorizar_tipo_hogar(tipo) is not None else ""
        print(f"    {tipo!r}{marcado}")


def _compare(name: str, current: dict, fetched: dict) -> None:
    print(f"\n=== {name} ===")
    keys = sorted(set(current) | set(fetched))
    any_diff = False
    for key in keys:
        old = current.get(key, "(no existe en el código)")
        new = fetched.get(key, "(no vino en la respuesta del INE)")
        if old != new:
            any_diff = True
            print(f"  {key}: código={old}  ine={new}")
    if not any_diff:
        print("  Sin diferencias -- el código ya coincide con el INE.")


# Identificadores Python usados como clave dentro del literal de
# PROVINCE_POPULATION que NO son directamente un string entre comillas
# (ver ine_reference.py, línea junto a "_CCAA_LA_RIOJA: 320_000,") --
# hace falta esta tabla para poder emparejar esa línea con la clave
# "la rioja" que devuelve `fetch_population_by_province`. Si en el futuro
# se usa otra constante como clave dentro de ese mismo diccionario, hay
# que añadirla aquí o esa línea concreta se dejará sin actualizar (con un
# aviso, no en silencio -- ver `_apply_province_population`).
_KEY_IDENTIFIER_ALIASES = {"_CCAA_LA_RIOJA": "la rioja"}

_INE_REFERENCE_PATH = Path(__file__).parent.parent / "app" / "data" / "ine_reference.py"

_PROVINCE_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<key>"[^"]+"|_[A-Z_]+):\s*(?P<value>[\d_]+),(?P<rest>.*)$'
)


def _format_int_literal(value: int) -> str:
    """Formatea un entero con guiones bajos cada 3 cifras (7_113_886), el
    mismo estilo que ya usa PROVINCE_POPULATION -- para que el diff en git
    de aplicar esto muestre solo el número que cambia, no un cambio de
    estilo de por medio."""
    return f"{value:,}".replace(",", "_")


def _apply_province_population(fetched: dict[str, int], *, auto_confirm: bool) -> bool:
    """Reescribe SOLO los valores dentro del bloque `PROVINCE_POPULATION = {...}`
    de ine_reference.py que tengan un valor distinto en `fetched`, y
    actualiza su fecha en `_LAST_VERIFIED` a hoy. Deliberadamente NO usa
    un serializador de diccionarios genérico (como `pprint` o
    `json.dumps`) -- eso reescribiría el bloque entero perdiendo el orden
    original (de mayor a menor población, no alfabético), los comentarios
    inline y el uso de `_CCAA_LA_RIOJA` como clave en vez de un string
    literal. En su lugar, se editan solo las líneas cuyo VALOR cambia,
    letra a letra, dejando todo lo demás del fichero exactamente igual.

    Devuelve True si se escribió algún cambio, False si no había nada que
    cambiar o el usuario no confirmó."""
    lines = _INE_REFERENCE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)

    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("PROVINCE_POPULATION = {"))
        end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip("\n") == "}")
    except StopIteration:
        print("ERROR: no se encontró el bloque 'PROVINCE_POPULATION = { ... }' en el fichero -- "
              "¿ha cambiado el formato? Revisa a mano, no se ha tocado nada.")
        return False

    changes: list[tuple[int, str, str, str]] = []  # (índice de línea, clave, valor viejo, valor nuevo)
    unmatched_identifiers: list[str] = []

    for i in range(start + 1, end):
        m = _PROVINCE_LINE_RE.match(lines[i].rstrip("\n"))
        if not m:
            continue  # línea que no es "clave: valor," (comentario, línea en blanco...) -- se deja tal cual

        raw_key = m.group("key")
        if raw_key.startswith('"'):
            clave = raw_key.strip('"')
        else:
            clave = _KEY_IDENTIFIER_ALIASES.get(raw_key)
            if clave is None:
                unmatched_identifiers.append(raw_key)
                continue

        nuevo_valor = fetched.get(clave)
        if nuevo_valor is None:
            continue  # el INE no trajo esta clave en esta ejecución -- se deja el valor actual, no se borra

        valor_actual = int(m.group("value").replace("_", ""))
        if valor_actual == int(nuevo_valor):
            continue  # ya coincide

        nueva_linea = (
            f"{m.group('indent')}{raw_key}: {_format_int_literal(int(nuevo_valor))},{m.group('rest')}\n"
        )
        changes.append((i, clave, m.group("value"), _format_int_literal(int(nuevo_valor))))
        lines[i] = nueva_linea

    if unmatched_identifiers:
        print(
            f"  AVISO: {len(unmatched_identifiers)} línea(s) usan un identificador de clave no "
            f"reconocido ({', '.join(sorted(set(unmatched_identifiers)))}) -- añádelo a "
            "_KEY_IDENTIFIER_ALIASES si corresponde a una provincia real; no se han tocado."
        )

    if not changes:
        print("PROVINCE_POPULATION: sin cambios que aplicar (ya coincide con el INE).")
        return False

    print(f"\nSe van a aplicar {len(changes)} cambios en PROVINCE_POPULATION:")
    for _i, clave, viejo, nuevo in changes:
        print(f"  {clave}: {viejo} -> {nuevo}")

    if not auto_confirm:
        respuesta = input("\n¿Aplicar estos cambios a ine_reference.py? [s/N]: ").strip().lower()
        if respuesta not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado -- no se ha escrito nada.")
            return False

    # Fecha del dato de origen a hoy, no la fecha en que se ejecuta esto en
    # otro sentido -- ver docstring de `_LAST_VERIFIED` en ine_reference.py:
    # es la fecha del dato del INE, y `nult=1` siempre trae el más reciente
    # publicado, así que "hoy" es una aproximación razonable a "la fecha en
    # que se confirmó que este es el dato más reciente disponible".
    today = date.today()
    last_verified_re = re.compile(r'^(\s*)"PROVINCE_POPULATION":\s*date\([^)]*\),(.*)$')
    for i, line in enumerate(lines):
        m = last_verified_re.match(line.rstrip("\n"))
        if m:
            lines[i] = f'{m.group(1)}"PROVINCE_POPULATION": date({today.year}, {today.month}, {today.day}),{m.group(2)}\n'
            break

    _INE_REFERENCE_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nEscrito en {_INE_REFERENCE_PATH} -- revisa el diff con git antes de hacer commit.")
    return True


# ============================================================================
# NORMALIZACIÓN de las 3 tablas restantes: convertir el diccionario BRUTO
# que ya devuelven fetch_marital_status/fetch_nationality/
# fetch_situacion_laboral (nombre de serie -> valor, tal cual el INE) en
# las proporciones que de verdad usa ine_reference.py. Aquí es donde se
# aplican las fórmulas ya documentadas a mano en los comentarios de esas
# constantes -- no es un volcado directo como PROVINCE_POPULATION.
# ============================================================================

def _normalize_nationality(raw: dict[str, float]) -> dict[str, float] | None:
    """NATIONALITY_DISTRIBUTION es la única de las tres sin ninguna
    fórmula manual detrás -- es un volcado directo español/extranjero
    sobre el total, filtrado a la fila nacional/todas las
    edades/ambos sexos (ver ejemplo real capturado en la tercera
    ejecución de este script). Se automatiza sin ningún supuesto
    intermedio, a diferencia de las otras dos."""
    total = raw.get("Total Nacional. Total. Todas las edades. Total. Población. Número. ")
    espanola = raw.get("Total Nacional. Española. Todas las edades. Total. Población. Número. ")
    extranjera = raw.get("Total Nacional. Extranjera. Todas las edades. Total. Población. Número. ")
    if not total or espanola is None or extranjera is None:
        return None
    return {
        "espanola": round(espanola / total, 3),
        "extranjera": round(extranjera / total, 3),
    }


# El INE alfabetiza "Divorciado/a o separado/a" y compañía con un punto y
# espacio detrás de cada categoría (ver ejemplo real capturado). Estas
# etiquetas son las que aparecen tal cual en el nombre de serie -- si el
# INE cambia la redacción exacta en una futura edición, hay que
# actualizar este diccionario (se avisará solo con un `None` en el
# resultado si ninguna etiqueta encaja, no con un cálculo silenciosamente
# erróneo, porque `_normalize_marital_status` exige las 4 claves).
_ESTADO_CIVIL_LABELS = {
    "Soltero/a": "soltero_bruto",
    "Casado/a": "casado",
    "Viudo/a": "viudo",
    "Divorciado/a o separado/a": "divorciado",
}


def _parse_marital_status_by_sex(raw: dict[str, float]) -> dict[str, dict[str, float]]:
    """Del diccionario BRUTO de fetch_marital_status, se queda solo con
    las filas de ámbito NACIONAL (no CCAA ni provincia) y las reparte por
    sexo. Formato real del nombre de serie, confirmado con la primera
    ejecución que trajo esta tabla: "<Territorio>. <Sexo>. <EstadoCivil>.
    Dato base. <Ámbito>. " -- las filas nacionales tienen Territorio=
    "Total" (el propio agregado nacional, mismo patrón self-referencial
    que ya se vio en PROVINCE_POPULATION) Y Ámbito conteniendo "Nacional";
    las de CCAA/provincia tienen el nombre real (p. ej. "Andalucía") como
    Territorio y Ámbito="Total" sin más. Se exige AMBAS condiciones a la
    vez, no solo una, para no confundir una futura provincia que por lo
    que sea se llamara "Total" (no existe ninguna, pero por seguridad)."""
    result: dict[str, dict[str, float]] = {"total": {}, "hombre": {}, "mujer": {}}
    sexo_key = {"Total": "total", "Hombres": "hombre", "Mujeres": "mujer"}
    for nombre, valor in raw.items():
        segments = [s.strip() for s in nombre.split(". ") if s.strip()]
        if len(segments) != 5:
            continue
        territorio, sexo, estado_civil, _tipo_dato, ambito = segments
        if territorio != "Total" or "Nacional" not in ambito:
            continue  # fila de CCAA/provincia, no la nacional
        if sexo not in sexo_key:
            continue
        if estado_civil == "Total":
            result[sexo_key[sexo]]["total"] = valor
        elif estado_civil in _ESTADO_CIVIL_LABELS:
            result[sexo_key[sexo]][_ESTADO_CIVIL_LABELS[estado_civil]] = valor
    return result


# ECEPOV 2021 (INE): ~70% de la población de 16+ años "tiene pareja" en
# sentido amplio. Ver el comentario junto a MARITAL_STATUS_DISTRIBUTION en
# ine_reference.py para la fórmula completa -- este dato NO viene de
# _TABLA_ESTADO_CIVIL (es una encuesta DISTINTA, la ECEPOV, no el Censo),
# así que no hay ninguna tabla Tempus3 identificada para automatizarlo:
# si el INE publica una edición más reciente de la ECEPOV, este número
# hay que actualizarlo A MANO.
_ECEPOV_TIENE_PAREJA_SENTIDO_AMPLIO = 0.70


def _normalize_marital_status(raw_nacional: dict[str, float]) -> dict[str, float] | None:
    """Aplica la fórmula ya documentada en el comentario de
    MARITAL_STATUS_DISTRIBUTION: 'casado', 'viudo' y 'divorciado' se
    toman directamente del Censo (aquí, ya real -- `raw_nacional` viene de
    `_parse_marital_status_by_sex(...)['total']`); 'con_pareja' se deriva
    de _ECEPOV_TIENE_PAREJA_SENTIDO_AMPLIO menos 'casado'; 'soltero' es el
    complementario de las otras cuatro."""
    total = raw_nacional.get("total")
    casado = raw_nacional.get("casado")
    viudo = raw_nacional.get("viudo")
    divorciado = raw_nacional.get("divorciado")
    if not total or casado is None or viudo is None or divorciado is None:
        return None
    p_casado = casado / total
    p_viudo = viudo / total
    p_divorciado = divorciado / total
    p_con_pareja = _ECEPOV_TIENE_PAREJA_SENTIDO_AMPLIO - p_casado
    p_soltero = 1 - p_casado - p_con_pareja - p_divorciado - p_viudo
    return {
        "casado": round(p_casado, 3),
        "con_pareja": round(p_con_pareja, 3),
        "divorciado": round(p_divorciado, 3),
        "soltero": round(p_soltero, 3),
        "viudo": round(p_viudo, 3),
    }


def _normalize_marital_status_by_sex(
    raw_hombre: dict[str, float], raw_mujer: dict[str, float], nacional: dict[str, float]
) -> dict[str, dict[str, float]] | None:
    """MARITAL_STATUS_BY_SEX: MEJORA sobre la versión anterior de esta
    tabla -- su comentario en ine_reference.py decía que 'divorciado' por
    sexo NO venía de ninguna tabla cruzada real y se aproximaba
    repartiendo el dato nacional por igual entre hombres y mujeres. Con
    esta tabla del INE (`_TABLA_ESTADO_CIVIL`) SÍ hay 'casado', 'viudo' Y
    'divorciado' reales por sexo (ver `_parse_marital_status_by_sex`), así
    que ya no hace falta esa aproximación para 'divorciado'.

    'con_pareja' y 'soltero' SIGUEN sin tener fuente cruzada por sexo (la
    ECEPOV no se desagrega así en lo que se ha localizado): se reparte la
    probabilidad que queda tras casado/viudo/divorciado entre los dos,
    manteniendo la misma proporción con_pareja:soltero que ya salió para
    `nacional` -- mismo criterio de reparto que documentaba la versión
    anterior de esta tabla, aplicado ahora sobre una base más precisa."""
    ratio_total = nacional["con_pareja"] + nacional["soltero"]
    if ratio_total <= 0:
        return None
    frac_con_pareja = nacional["con_pareja"] / ratio_total

    def _por_sexo(raw: dict[str, float]) -> dict[str, float] | None:
        total = raw.get("total")
        casado = raw.get("casado")
        viudo = raw.get("viudo")
        divorciado = raw.get("divorciado")
        if not total or casado is None or viudo is None or divorciado is None:
            return None
        p_casado = casado / total
        p_viudo = viudo / total
        p_divorciado = divorciado / total
        resto = max(0.0, 1 - p_casado - p_viudo - p_divorciado)
        return {
            "casado": round(p_casado, 3),
            "viudo": round(p_viudo, 3),
            "divorciado": round(p_divorciado, 3),
            "con_pareja": round(resto * frac_con_pareja, 3),
            "soltero": round(resto * (1 - frac_con_pareja), 3),
        }

    hombre = _por_sexo(raw_hombre)
    mujer = _por_sexo(raw_mujer)
    if hombre is None or mujer is None:
        return None
    return {"hombre": hombre, "mujer": mujer}


# Reparto de los INACTIVOS entre sus subcategorías -- mismo criterio ya
# documentado en el comentario de SITUACION_LABORAL_DISTRIBUTION en
# ine_reference.py (jubilados/pensionistas ~57% de los inactivos,
# estudiantes ~20%, resto -labores del hogar, incapacidad permanente,
# otras situaciones- ~23%). Estas proporciones NO vienen de
# _TABLA_TASAS_EPA (esa tabla solo da tasas agregadas de
# actividad/paro, no el desglose de los inactivos) -- son un supuesto
# razonado heredado de la versión anterior de esta tabla, no algo que
# este script pueda derivar de la API.
_INACTIVOS_JUBILADO_FRAC = 0.57
_INACTIVOS_ESTUDIANTE_FRAC = 0.20
_INACTIVOS_OTRO_FRAC = 0.23

# Rango de plausibilidad para la tasa de paro NACIONAL española -- guarda
# de seguridad para no aplicar en silencio un dato claramente erróneo si
# el ID de tabla o el filtro de categoría estuvieran mal (como ya ocurrió
# una vez con PROVINCE_POPULATION, tabla equivocada sin ningún error
# HTTP). España no ha bajado de ~8% ni ha subido de ~27% (pico de la
# crisis de 2013) en las últimas dos décadas -- pero el valor real
# esperado en 2025-2026 es de ~10-11%, así que aunque un valor dentro de
# este rango técnicamente no dispara la guarda, sigue mereciendo una
# revisión humana si se aleja mucho de esa cifra reciente conocida (ver
# el aviso explícito impreso en main() antes de aplicar esta tabla).
#
# CONFIRMADO contra la nota de prensa oficial del INE (EPA 4º trimestre
# 2025, https://www.ine.es/dyngs/Prensa/EPA4T25.htm): tasa de paro real
# 9,93%, tasa de actividad real 58,94% -- exactamente las cifras que ya
# había en el comentario de SITUACION_LABORAL_DISTRIBUTION. El 26,03% que
# trae _TABLA_TASAS_EPA (ID 1113, "Tasas de actividad, paro y empleo, por
# sexo y distintos grupos de edad") es, por tanto, CONFIRMADO erróneo -- y
# el ID de tabla en sí SÍ es el correcto por nombre (no es el mismo tipo
# de error que tuvo PROVINCE_POPULATION con el t=2917 equivocado).
#
# Hipótesis más probable, sin confirmar (no se ha podido inspeccionar el
# `MetaData` crudo de esta tabla, solo el `Nombre` en texto): la propia
# INE avisa en su página de la EPA de que conviven, bajo la misma
# operación estadística, resultados con la "metodología vigente" (2021)
# y resultados con "metodologías no vigentes" de trimestres anteriores a
# ese cambio. Si esta tabla concreta incluye AMBAS series bajo un
# `Nombre` de texto IDÉNTICO (distinguibles solo por un campo de
# `MetaData` que `fetch_situacion_laboral` no usa -- guarda por `Nombre`
# tal cual, con riesgo de que una serie sobrescriba a la otra en el
# diccionario si comparten la misma clave de texto), el valor que quede
# en el diccionario final dependería del orden de iteración, no de cuál
# es la vigente. Es EXACTAMENTE el mismo tipo de fallo que tuvo
# PROVINCE_POPULATION al fiarse del texto `Nombre` en vez de `MetaData`
# estructurado -- pero aquí no se ha podido verificar contra la API real
# para confirmarlo. Antes de forzar esta tabla con --force-tasa-paro,
# valdría la pena volcar `serie` completa (no solo `Nombre`/`Data`) para
# las entradas "Tasa de paro. Nacional. Ambos sexos. Total." y comprobar
# si de verdad hay más de una con el mismo texto pero MetaData distinto.
_TASA_PARO_PLAUSIBLE = (5.0, 28.0)
_TASA_PARO_RECIENTE_CONOCIDA = 9.93  # EPA T4 2025, la que ya había en el comentario de esta constante


def _buscar_por_segmentos(raw: dict[str, float], requeridos: list[str]) -> float | None:
    """Busca en `raw` la primera clave cuyos segmentos (separados por
    ". ") contengan TODOS los textos de `requeridos`, sin importar el
    ORDEN en que aparezcan -- más robusto que adivinar una concatenación
    exacta como "Total. Ambos sexos. Valor absoluto" cuando no se ha
    visto un ejemplo real del formato (caso de _TABLA_TASA_ACTIVIDAD_EPA/
    _TABLA_TASA_PARO_EPA en esta sesión: solo se vio la rejilla del Excel
    de "consultar todo", no el JSON real de la API, así que no se sabe el
    orden exacto de los segmentos de `Nombre`)."""
    for nombre, valor in raw.items():
        segments = {s.strip() for s in nombre.split(". ") if s.strip()}
        if all(req in segments for req in requeridos):
            return valor
    return None


def _normalize_situacion_laboral(actividad_raw: dict[str, float], paro_raw: dict[str, float]) -> dict[str, float] | None:
    """Recibe las DOS tablas por separado (ver fetch_situacion_laboral)
    -- antes era una sola tabla combinada (1113, descontinuada, ver
    historial en el comentario de _TABLA_TASA_ACTIVIDAD_EPA). Busca
    "Total" + "Ambos sexos" en cada una con `_buscar_por_segmentos`, sin
    asumir un orden concreto de los segmentos de `Nombre` (formato sin
    confirmar todavía contra la API real de estas dos tablas nuevas)."""
    tasa_actividad = _buscar_por_segmentos(actividad_raw, ["Total", "Ambos sexos"])
    tasa_paro = _buscar_por_segmentos(paro_raw, ["Total", "Ambos sexos"])
    if tasa_actividad is None or tasa_paro is None:
        return None
    activo = tasa_actividad / 100 * (1 - tasa_paro / 100)
    parado = tasa_actividad / 100 * (tasa_paro / 100)
    inactivos = 1 - tasa_actividad / 100
    return {
        "activo": round(activo, 3),
        "parado": round(parado, 3),
        "jubilado": round(inactivos * _INACTIVOS_JUBILADO_FRAC, 3),
        "estudiante": round(inactivos * _INACTIVOS_ESTUDIANTE_FRAC, 3),
        "otro_inactivo": round(inactivos * _INACTIVOS_OTRO_FRAC, 3),
    }


# ============================================================================
# Escritura genérica para bloques "clave": float, -- reutilizable para
# MARITAL_STATUS_DISTRIBUTION, NATIONALITY_DISTRIBUTION,
# SITUACION_LABORAL_DISTRIBUTION y los dos sub-bloques de
# MARITAL_STATUS_BY_SEX. A diferencia de PROVINCE_POPULATION (enteros con
# guiones bajos), aquí los valores son proporciones con 3 decimales.
# ============================================================================

_FLOAT_LINE_RE = re.compile(r'^(?P<indent>\s*)"(?P<key>[^"]+)":\s*(?P<value>[\d.]+),(?P<rest>.*)$')


def _locate_block(lines: list[str], start_predicate) -> tuple[int, int] | None:
    """Devuelve (índice de la línea de apertura, índice de la línea de
    cierre) del primer bloque `{ ... }` cuya línea de apertura cumple
    `start_predicate`. El cierre se identifica por tener la MISMA
    indentación que la apertura y ser solo "}" o "},"  -- así funciona
    igual para un diccionario de nivel superior (indentación "") y para
    un sub-diccionario anidado como "hombre": {...} (indentación de 4
    espacios), sin necesitar dos funciones distintas."""
    for i, line in enumerate(lines):
        if start_predicate(line):
            indent = re.match(r"^(\s*)", line).group(1)
            for j in range(i + 1, len(lines)):
                stripped = lines[j].rstrip("\n")
                if stripped[len(indent):] in ("}", "},") and stripped.startswith(indent):
                    return i, j
            return None
    return None


def _apply_float_block(
    lines: list[str], start: int, end: int, new_values: dict[str, float], ndigits: int = 3
) -> list[tuple[str, str, str]]:
    """Sustituye, dentro de lines[start+1:end], el valor numérico de cada
    línea `"clave": 0.123,` cuya clave esté en `new_values` Y cuyo valor
    actual sea distinto (con `ndigits` decimales) -- deja todo lo demás
    (indentación, comentarios, claves no reconocidas) exactamente igual.
    Devuelve la lista de cambios aplicados (clave, valor viejo, valor
    nuevo) para poder mostrarlos antes de pedir confirmación."""
    changes: list[tuple[str, str, str]] = []
    for i in range(start + 1, end):
        m = _FLOAT_LINE_RE.match(lines[i].rstrip("\n"))
        if not m:
            continue
        clave = m.group("key")
        if clave not in new_values:
            continue
        nuevo = round(float(new_values[clave]), ndigits)
        if abs(float(m.group("value")) - nuevo) < 10 ** (-ndigits) / 2:
            continue  # ya coincide (dentro de la precisión de ndigits decimales)
        nuevo_str = f"{nuevo:.{ndigits}f}"
        changes.append((clave, m.group("value"), nuevo_str))
        lines[i] = f"{m.group('indent')}\"{clave}\": {nuevo_str},{m.group('rest')}\n"
    return changes


def _update_last_verified(lines: list[str], table_name: str) -> None:
    """Actualiza `_LAST_VERIFIED["<table_name>"]` a la fecha de hoy --
    misma fecha del dato de origen que ya explica `_apply_province_population`
    (nult=1 siempre trae el más reciente publicado)."""
    today = date.today()
    pattern = re.compile(rf'^(\s*)"{re.escape(table_name)}":\s*date\([^)]*\),(.*)$')
    for i, line in enumerate(lines):
        m = pattern.match(line.rstrip("\n"))
        if m:
            lines[i] = f'{m.group(1)}"{table_name}": date({today.year}, {today.month}, {today.day}),{m.group(2)}\n'
            return


def _confirm_and_write(changes: list[tuple[str, str, str]], table_label: str, lines: list[str], *, auto_confirm: bool) -> bool:
    """Paso común a todos los `_apply_*`: si no hay cambios, no hace nada;
    si hay, los muestra y pide confirmación (salvo `auto_confirm`) antes
    de escribir de verdad el fichero completo."""
    if not changes:
        print(f"{table_label}: sin cambios que aplicar (ya coincide con el INE).")
        return False

    print(f"\nSe van a aplicar {len(changes)} cambios en {table_label}:")
    for clave, viejo, nuevo in changes:
        print(f"  {clave}: {viejo} -> {nuevo}")

    if not auto_confirm:
        respuesta = input(f"\n¿Aplicar estos cambios de {table_label} a ine_reference.py? [s/N]: ").strip().lower()
        if respuesta not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado -- no se ha escrito nada.")
            return False

    _INE_REFERENCE_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Escrito en {_INE_REFERENCE_PATH} -- revisa el diff con git antes de hacer commit.")
    return True


def _apply_nationality(normalized: dict[str, float], *, auto_confirm: bool) -> bool:
    lines = _INE_REFERENCE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    block = _locate_block(lines, lambda line: line.startswith("NATIONALITY_DISTRIBUTION = {"))
    if block is None:
        print("ERROR: no se encontró 'NATIONALITY_DISTRIBUTION = { ... }' -- no se ha tocado nada.")
        return False
    start, end = block
    changes = _apply_float_block(lines, start, end, normalized)
    if changes:
        _update_last_verified(lines, "NATIONALITY_DISTRIBUTION")
    return _confirm_and_write(changes, "NATIONALITY_DISTRIBUTION", lines, auto_confirm=auto_confirm)


def _apply_marital_status(normalized: dict[str, float], *, auto_confirm: bool) -> bool:
    lines = _INE_REFERENCE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    block = _locate_block(lines, lambda line: line.startswith("MARITAL_STATUS_DISTRIBUTION = {"))
    if block is None:
        print("ERROR: no se encontró 'MARITAL_STATUS_DISTRIBUTION = { ... }' -- no se ha tocado nada.")
        return False
    start, end = block
    changes = _apply_float_block(lines, start, end, normalized)
    if changes:
        _update_last_verified(lines, "MARITAL_STATUS_DISTRIBUTION")
    return _confirm_and_write(changes, "MARITAL_STATUS_DISTRIBUTION", lines, auto_confirm=auto_confirm)


def _apply_marital_status_by_sex(normalized: dict[str, dict[str, float]], *, auto_confirm: bool) -> bool:
    """A diferencia de las demás, MARITAL_STATUS_BY_SEX tiene DOS
    sub-bloques anidados ("hombre": {...} y "mujer": {...}) dentro del
    bloque principal -- se localizan y actualizan por separado, pero se
    escriben juntos en una sola confirmación (son la misma tabla lógica;
    pedir confirmación dos veces para "hombre" y "mujer" por separado
    sería más confuso que útil)."""
    lines = _INE_REFERENCE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    outer = _locate_block(lines, lambda line: line.startswith("MARITAL_STATUS_BY_SEX = {"))
    if outer is None:
        print("ERROR: no se encontró 'MARITAL_STATUS_BY_SEX = { ... }' -- no se ha tocado nada.")
        return False
    outer_start, outer_end = outer

    changes: list[tuple[str, str, str]] = []
    for sexo in ("hombre", "mujer"):
        block = _locate_block(
            lines[: outer_end + 1],
            lambda line, sexo=sexo: line.strip() == f'"{sexo}": {{',
        )
        if block is None or block[0] <= outer_start or block[1] > outer_end:
            print(f"  AVISO: no se encontró el sub-bloque \"{sexo}\": {{ ... }} dentro de MARITAL_STATUS_BY_SEX.")
            continue
        start, end = block
        changes.extend(_apply_float_block(lines, start, end, normalized.get(sexo, {})))

    if changes:
        _update_last_verified(lines, "MARITAL_STATUS_BY_SEX")
    return _confirm_and_write(changes, "MARITAL_STATUS_BY_SEX", lines, auto_confirm=auto_confirm)


def _apply_situacion_laboral(normalized: dict[str, float], *, auto_confirm: bool) -> bool:
    lines = _INE_REFERENCE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    block = _locate_block(lines, lambda line: line.startswith("SITUACION_LABORAL_DISTRIBUTION = {"))
    if block is None:
        print("ERROR: no se encontró 'SITUACION_LABORAL_DISTRIBUTION = { ... }' -- no se ha tocado nada.")
        return False
    start, end = block
    changes = _apply_float_block(lines, start, end, normalized)
    if changes:
        _update_last_verified(lines, "SITUACION_LABORAL_DISTRIBUTION")
    return _confirm_and_write(changes, "SITUACION_LABORAL_DISTRIBUTION", lines, auto_confirm=auto_confirm)


def _apply_occupation(normalized: dict[str, float], *, auto_confirm: bool) -> bool:
    """Solo toca las claves de OCCUPATION_DISTRIBUTION presentes en
    `normalized` -- las que no tengan ningún subgrupo CNO-11 mapeado
    (ver _CNO11_SUBGRUPO_TO_APP_CATEGORY) se quedan con su valor actual
    sin tocar, no se ponen a 0 ni se borran."""
    lines = _INE_REFERENCE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    block = _locate_block(lines, lambda line: line.startswith("OCCUPATION_DISTRIBUTION = {"))
    if block is None:
        print("ERROR: no se encontró 'OCCUPATION_DISTRIBUTION = { ... }' -- no se ha tocado nada.")
        return False
    start, end = block
    changes = _apply_float_block(lines, start, end, normalized)
    if changes:
        _update_last_verified(lines, "OCCUPATION_DISTRIBUTION")
    return _confirm_and_write(changes, "OCCUPATION_DISTRIBUTION", lines, auto_confirm=auto_confirm)


def _apply_household_type(normalized: dict[str, float], *, auto_confirm: bool) -> bool:
    lines = _INE_REFERENCE_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    block = _locate_block(lines, lambda line: line.startswith("HOUSEHOLD_TYPE_DISTRIBUTION = {"))
    if block is None:
        print("ERROR: no se encontró 'HOUSEHOLD_TYPE_DISTRIBUTION = { ... }' -- no se ha tocado nada.")
        return False
    start, end = block
    changes = _apply_float_block(lines, start, end, normalized)
    if changes:
        _update_last_verified(lines, "HOUSEHOLD_TYPE_DISTRIBUTION")
    return _confirm_and_write(changes, "HOUSEHOLD_TYPE_DISTRIBUTION", lines, auto_confirm=auto_confirm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--apply", action="store_true",
        help="Además de comparar, escribe los cambios en ine_reference.py (las 4 tablas)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Con --apply, no pedir confirmación por teclado antes de escribir",
    )
    parser.add_argument(
        "--force-tasa-paro", action="store_true",
        help="Con --apply, aplicar SITUACION_LABORAL_DISTRIBUTION aunque la tasa de paro "
        "se aleje mucho del valor reciente conocido (ver aviso más abajo)",
    )
    parser.add_argument(
        "--force-ocupacion", action="store_true",
        help="Con --apply, aplicar OCCUPATION_DISTRIBUTION aunque la suma de las categorías "
        "mapeadas se salga del rango de plausibilidad esperado (ver aviso más abajo)",
    )
    parser.add_argument(
        "--force-hogar", action="store_true",
        help="Con --apply, aplicar HOUSEHOLD_TYPE_DISTRIBUTION aunque la suma de las 5 "
        "categorías se salga del rango de plausibilidad esperado (ver aviso más abajo)",
    )
    parser.add_argument(
        "--no-studies", action="store_true",
        help="No llamar a update_studies_distribution.py al final (por si solo se quieren "
        "comprobar las tablas del INE de este script)",
    )
    parser.add_argument(
        "--insecure", action="store_true",
        help="Pasado tal cual a update_studies_distribution.py -- desactiva verificación SSL "
        "al descargar (mismo uso que en ese script: diagnóstico/último recurso)",
    )
    args = parser.parse_args()

    print("Comprobando tablas de app/data/ine_reference.py contra el INE...")
    if args.apply:
        print("(--apply: SÍ se escribirá en ine_reference.py lo que tenga cambios y se confirme)\n")
    else:
        print("(esto NO modifica ine_reference.py -- solo muestra diferencias; usa --apply para escribir)\n")

    # Resumen final -- cada entrada es (tabla, estado, detalle). Se
    # rellena en cada punto de salida de cada bloque try/except de abajo,
    # para poder imprimir al final una tabla de un vistazo con qué se
    # actualizó, qué no, y por qué (pedido explícito de Nacho: "que ponga
    # una tabla con las que se han podido actualizar y las que no y por
    # qué -- url no encontrado, fallo de descarga, fallo de mapeo...").
    resultados: list[tuple[str, str, str]] = []

    def _registrar(tabla: str, estado: str, detalle: str) -> None:
        resultados.append((tabla, estado, " ".join(detalle.split())))  # aplana saltos de línea (p. ej. de httpx.HTTPError) para que la tabla resumen no se rompa

    try:
        province_data = fetch_population_by_province()
        _compare("PROVINCE_POPULATION", ine_reference.PROVINCE_POPULATION, province_data)
        if args.apply:
            escrito = _apply_province_population(province_data, auto_confirm=args.yes)
            _registrar("PROVINCE_POPULATION", "actualizada" if escrito else "sin cambios", "descarga y mapeo OK")
        else:
            _registrar("PROVINCE_POPULATION", "comparada (sin --apply)", "descarga y mapeo OK")
    except httpx.HTTPError as e:
        print(f"ERROR al descargar población por provincia: {e}")
        _registrar("PROVINCE_POPULATION", "FALLO", f"error de descarga (HTTP): {e}")

    try:
        marital_raw = fetch_marital_status()
        by_sex_raw = _parse_marital_status_by_sex(marital_raw)
        marital_normalized = _normalize_marital_status(by_sex_raw["total"])
        if marital_normalized is None:
            print(
                "\n=== MARITAL_STATUS_DISTRIBUTION ===\n"
                "  No se pudo aislar la fila nacional (Total/Total/Dato base/Total "
                "Nacional) en la respuesta -- revisa el formato real con "
                "fetch_marital_status() antes de forzar nada."
            )
            _registrar("MARITAL_STATUS_DISTRIBUTION", "FALLO", "fallo de mapeo -- formato de 'Nombre' no reconocido (¿estructura del INE cambiada?)")
            _registrar("MARITAL_STATUS_BY_SEX", "FALLO", "depende de MARITAL_STATUS_DISTRIBUTION, no calculada")
        else:
            _compare("MARITAL_STATUS_DISTRIBUTION", ine_reference.MARITAL_STATUS_DISTRIBUTION, marital_normalized)
            by_sex_normalized = _normalize_marital_status_by_sex(
                by_sex_raw["hombre"], by_sex_raw["mujer"], marital_normalized
            )
            if by_sex_normalized is not None:
                print("\n=== MARITAL_STATUS_BY_SEX ===")
                for sexo in ("hombre", "mujer"):
                    _compare(f"  {sexo}", ine_reference.MARITAL_STATUS_BY_SEX[sexo], by_sex_normalized[sexo])
            if args.apply:
                escrito = _apply_marital_status(marital_normalized, auto_confirm=args.yes)
                _registrar("MARITAL_STATUS_DISTRIBUTION", "actualizada" if escrito else "sin cambios", "descarga y mapeo OK")
                if by_sex_normalized is not None:
                    escrito_sexo = _apply_marital_status_by_sex(by_sex_normalized, auto_confirm=args.yes)
                    _registrar("MARITAL_STATUS_BY_SEX", "actualizada" if escrito_sexo else "sin cambios", "descarga y mapeo OK")
                else:
                    _registrar("MARITAL_STATUS_BY_SEX", "FALLO", "fallo de mapeo del desglose por sexo")
            else:
                _registrar("MARITAL_STATUS_DISTRIBUTION", "comparada (sin --apply)", "descarga y mapeo OK")
                _registrar("MARITAL_STATUS_BY_SEX", "comparada (sin --apply)" if by_sex_normalized is not None else "FALLO", "descarga y mapeo OK" if by_sex_normalized is not None else "fallo de mapeo del desglose por sexo")
    except httpx.HTTPError as e:
        print(f"ERROR al descargar estado civil: {e}")
        _registrar("MARITAL_STATUS_DISTRIBUTION", "FALLO", f"error de descarga (HTTP): {e}")
        _registrar("MARITAL_STATUS_BY_SEX", "FALLO", "depende de MARITAL_STATUS_DISTRIBUTION, no calculada")

    try:
        nationality_raw = fetch_nationality()
        nationality_normalized = _normalize_nationality(nationality_raw)
        if nationality_normalized is None:
            print(
                "\n=== NATIONALITY_DISTRIBUTION ===\n"
                "  No se encontraron las filas nacionales esperadas -- revisa el "
                "formato real con fetch_nationality() antes de forzar nada."
            )
            _registrar("NATIONALITY_DISTRIBUTION", "FALLO", "fallo de mapeo -- formato de 'Nombre' no reconocido (¿estructura del INE cambiada?)")
        else:
            _compare("NATIONALITY_DISTRIBUTION", ine_reference.NATIONALITY_DISTRIBUTION, nationality_normalized)
            if args.apply:
                escrito = _apply_nationality(nationality_normalized, auto_confirm=args.yes)
                _registrar("NATIONALITY_DISTRIBUTION", "actualizada" if escrito else "sin cambios", "descarga y mapeo OK")
            else:
                _registrar("NATIONALITY_DISTRIBUTION", "comparada (sin --apply)", "descarga y mapeo OK")
    except httpx.HTTPError as e:
        print(f"ERROR al descargar nacionalidad: {e}")
        _registrar("NATIONALITY_DISTRIBUTION", "FALLO", f"error de descarga (HTTP): {e}")

    try:
        actividad_raw, paro_raw = fetch_situacion_laboral()
        laboral_normalized = _normalize_situacion_laboral(actividad_raw, paro_raw)
        if laboral_normalized is None:
            print(
                "\n=== SITUACION_LABORAL_DISTRIBUTION ===\n"
                "  No se encontraron 'Total'+'Ambos sexos' en alguna de las dos tablas "
                "(actividad 65081 / paro 65219) -- revisa el formato real con "
                "fetch_situacion_laboral()."
            )
            _mostrar_ejemplos_nombres(actividad_raw, max_n=15)
            _mostrar_ejemplos_nombres(paro_raw, max_n=15)
            _registrar("SITUACION_LABORAL_DISTRIBUTION", "FALLO", "fallo de mapeo -- formato de 'Nombre' no reconocido en 65081/65219 (¿estructura del INE cambiada?)")
        else:
            _compare("SITUACION_LABORAL_DISTRIBUTION", ine_reference.SITUACION_LABORAL_DISTRIBUTION, laboral_normalized)
            tasa_paro = _buscar_por_segmentos(paro_raw, ["Total", "Ambos sexos"])
            tasa_sospechosa = tasa_paro is not None and (
                not (_TASA_PARO_PLAUSIBLE[0] <= tasa_paro <= _TASA_PARO_PLAUSIBLE[1])
                or abs(tasa_paro - _TASA_PARO_RECIENTE_CONOCIDA) > _TASA_PARO_RECIENTE_CONOCIDA
            )
            if tasa_sospechosa:
                print(
                    f"\n  AVISO IMPORTANTE: la tasa de paro nacional que trae esta tabla "
                    f"es {tasa_paro}%, muy distinta del ~{_TASA_PARO_RECIENTE_CONOCIDA}% "
                    "conocido de la EPA más reciente. La tabla 65219 se confirmó vigente "
                    "con un Excel real (datos hasta 2026T2) -- si esto sigue saliendo mal, "
                    "sospecha primero de _buscar_por_segmentos (¿hay más de una fila que "
                    "matchee 'Total'+'Ambos sexos', p. ej. por edad 'Total' Y sexo 'Ambos "
                    "sexos' combinados con otra dimensión?) antes que del ID de tabla."
                )
                _mostrar_ejemplos_nombres(paro_raw, filtro_subcadenas=["Total"], max_n=25)
            if args.apply:
                if tasa_sospechosa and not args.force_tasa_paro:
                    print(
                        "  SITUACION_LABORAL_DISTRIBUTION: NO se aplica por el aviso de "
                        "arriba -- usa --force-tasa-paro si quieres aplicarlo igualmente."
                    )
                    _registrar("SITUACION_LABORAL_DISTRIBUTION", "no aplicada", f"valor sospechoso (tasa de paro {tasa_paro}%) -- usa --force-tasa-paro")
                else:
                    escrito = _apply_situacion_laboral(laboral_normalized, auto_confirm=args.yes)
                    _registrar("SITUACION_LABORAL_DISTRIBUTION", "actualizada" if escrito else "sin cambios", "descarga y mapeo OK")
            else:
                _registrar("SITUACION_LABORAL_DISTRIBUTION", "comparada (sin --apply)" if not tasa_sospechosa else "comparada, valor sospechoso", "descarga y mapeo OK")
    except httpx.HTTPError as e:
        print(f"ERROR al descargar situación laboral: {e}")
        _registrar("SITUACION_LABORAL_DISTRIBUTION", "FALLO", f"error de descarga (HTTP): {e}")

    try:
        ocupacion_raw = fetch_occupation()
        ocupacion_normalized = _normalize_occupation(ocupacion_raw)
        if ocupacion_normalized is None:
            print(
                "\n=== OCCUPATION_DISTRIBUTION ===\n"
                "  No se encontró ninguna fila 'Ambos sexos'/porcentaje reconocible -- "
                "revisa el formato real con fetch_occupation() antes de ajustar el "
                "parseo de _normalize_occupation()."
            )
            _mostrar_ejemplos_nombres(ocupacion_raw)
            _registrar("OCCUPATION_DISTRIBUTION", "FALLO", "fallo de mapeo -- formato de 'Nombre' no reconocido (¿estructura del INE cambiada?)")
        else:
            _compare("OCCUPATION_DISTRIBUTION", ine_reference.OCCUPATION_DISTRIBUTION, ocupacion_normalized)
            suma = sum(ocupacion_normalized.values()) * 100
            suma_sospechosa = not (_OCUPACION_SUMA_PLAUSIBLE[0] <= suma <= _OCUPACION_SUMA_PLAUSIBLE[1])
            if suma_sospechosa:
                print(
                    f"\n  AVISO IMPORTANTE: la suma de las categorías mapeadas de "
                    f"OCCUPATION_DISTRIBUTION es {suma:.1f}%, fuera del rango "
                    f"{_OCUPACION_SUMA_PLAUSIBLE} esperado para un mapeo deliberadamente "
                    "incompleto (ver _CNO11_SUBGRUPO_TO_APP_CATEGORY). Puede que "
                    "_normalize_occupation esté sumando series repetidas por sexo o tipo "
                    "de dato en vez de filtrarlas -- revisa el formato real de 'Nombre' "
                    "antes de confiar en estos números."
                )
                _diagnosticar_ocupaciones_unicas(ocupacion_raw)
            if args.apply:
                if suma_sospechosa and not args.force_ocupacion:
                    print(
                        "  OCCUPATION_DISTRIBUTION: NO se aplica por el aviso de arriba -- "
                        "usa --force-ocupacion si quieres aplicarlo igualmente."
                    )
                    _registrar("OCCUPATION_DISTRIBUTION", "no aplicada", f"suma sospechosa ({suma:.1f}%) -- usa --force-ocupacion")
                else:
                    escrito = _apply_occupation(ocupacion_normalized, auto_confirm=args.yes)
                    _registrar("OCCUPATION_DISTRIBUTION", "actualizada" if escrito else "sin cambios", "descarga y mapeo OK")
            else:
                _registrar("OCCUPATION_DISTRIBUTION", "comparada (sin --apply)" if not suma_sospechosa else "comparada, valor sospechoso", "descarga y mapeo OK")
    except httpx.HTTPError as e:
        print(f"ERROR al descargar ocupación (CNO-11): {e}")
        _registrar("OCCUPATION_DISTRIBUTION", "FALLO", f"error de descarga (HTTP): {e}")

    try:
        hogar_raw = fetch_household_type()
        hogar_normalized = _normalize_household_type(hogar_raw)
        if hogar_normalized is None:
            print(
                "\n=== HOUSEHOLD_TYPE_DISTRIBUTION ===\n"
                "  No se encontró ninguna fila 'Total' (tipo de edificio) reconocible -- "
                "revisa el formato real con fetch_household_type() antes de ajustar el "
                "parseo de _normalize_household_type()."
            )
            _mostrar_ejemplos_nombres(hogar_raw)
            _registrar("HOUSEHOLD_TYPE_DISTRIBUTION", "FALLO", "fallo de mapeo -- formato de 'Nombre' no reconocido (¿estructura del INE cambiada?)")
        else:
            _compare("HOUSEHOLD_TYPE_DISTRIBUTION", ine_reference.HOUSEHOLD_TYPE_DISTRIBUTION, hogar_normalized)
            suma = sum(hogar_normalized.values())
            suma_sospechosa = not (_HOGAR_SUMA_PLAUSIBLE[0] <= suma <= _HOGAR_SUMA_PLAUSIBLE[1])
            if suma_sospechosa:
                print(
                    f"\n  AVISO IMPORTANTE: la suma de las 5 categorías de "
                    f"HOUSEHOLD_TYPE_DISTRIBUTION es {suma:.3f} (se espera ~1.0, mapeo "
                    "exhaustivo -- ver _TIPO_HOGAR_TO_APP_CATEGORY). Puede que el filtro "
                    "por fila 'Total' de tipo de edificio no esté funcionando como se "
                    "espera (p. ej. sumando varias filas del mismo tipo de hogar cruzadas "
                    "con cada tipo de edificio en vez de solo la marginal) -- revisa el "
                    "formato real de 'Nombre' antes de confiar en estos números."
                )
                _diagnosticar_tipos_hogar_unicos(hogar_raw)
            if args.apply:
                if suma_sospechosa and not args.force_hogar:
                    print(
                        "  HOUSEHOLD_TYPE_DISTRIBUTION: NO se aplica por el aviso de arriba -- "
                        "usa --force-hogar si quieres aplicarlo igualmente."
                    )
                    _registrar("HOUSEHOLD_TYPE_DISTRIBUTION", "no aplicada", f"suma sospechosa ({suma:.3f}) -- usa --force-hogar")
                else:
                    escrito = _apply_household_type(hogar_normalized, auto_confirm=args.yes)
                    _registrar("HOUSEHOLD_TYPE_DISTRIBUTION", "actualizada" if escrito else "sin cambios", "descarga y mapeo OK")
            else:
                _registrar("HOUSEHOLD_TYPE_DISTRIBUTION", "comparada (sin --apply)" if not suma_sospechosa else "comparada, valor sospechoso", "descarga y mapeo OK")
    except httpx.HTTPError as e:
        print(f"ERROR al descargar tipo de hogar (ECH): {e}")
        _registrar("HOUSEHOLD_TYPE_DISTRIBUTION", "FALLO", f"error de descarga (HTTP): {e}")

    print(
        "\nSTUDIES_DISTRIBUTION: fuente distinta del INE, sin API en vivo -- "
        "delegando en scripts/update_studies_distribution.py (intenta "
        "descargar los ficheros del Ministerio solo, con instrucciones "
        "manuales si falla)."
    )
    if args.no_studies:
        print("(--no-studies: NO se llama a update_studies_distribution.py)")
        _registrar("STUDIES_DISTRIBUTION", "omitida", "--no-studies")
    else:
        comando = [sys.executable, str(Path(__file__).parent / "update_studies_distribution.py")]
        if args.apply:
            comando.append("--apply")
        if args.yes:
            comando.append("--yes")
        if args.insecure:
            comando.append("--insecure")
        print(f"  Ejecutando: {' '.join(comando)}\n")
        resultado = subprocess.run(comando)
        if resultado.returncode != 0:
            print(
                f"\n  AVISO: update_studies_distribution.py terminó con código "
                f"{resultado.returncode} (fallo) -- revisa su salida de arriba. "
                "No es un fallo de las tablas del INE de este script, es "
                "independiente (usa --no-studies para saltártelo si hace falta)."
            )
            _registrar("STUDIES_DISTRIBUTION", "FALLO", f"update_studies_distribution.py devolvió código {resultado.returncode} -- ver su salida arriba (fallo de descarga o de mapeo del Ministerio)")
        else:
            _registrar("STUDIES_DISTRIBUTION", "actualizada" if args.apply else "comparada (sin --apply)", "delegado a update_studies_distribution.py -- ver su salida arriba para el detalle")

    print(
        "\nLANGUAGE_BY_CCAA: encuesta puntual (ECEPOV), sin tabla anual "
        "equivalente que comparar -- revisar a mano si el INE ha "
        "publicado una edición más reciente."
    )
    _registrar("LANGUAGE_BY_CCAA", "sin fuente automática", "ECEPOV es puntual, sin equivalente anual conocido -- ver README")

    # --- Tabla resumen final ---
    ancho_tabla = max(len(t) for t, _, _ in resultados)
    ancho_estado = max(len(e) for _, e, _ in resultados)
    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    for tabla, estado, detalle in resultados:
        marca = "✅" if estado in ("actualizada", "sin cambios") else "⚠️ " if estado in ("no aplicada", "omitida", "sin fuente automática") or estado.startswith("comparada") else "❌"
        print(f"{marca} {tabla.ljust(ancho_tabla)}  {estado.ljust(ancho_estado)}  {detalle}")
    print("=" * 78)
    n_fallo = sum(1 for _, e, _ in resultados if e == "FALLO")
    n_ok = sum(1 for _, e, _ in resultados if e in ("actualizada", "sin cambios"))
    print(f"{n_ok}/{len(resultados)} tablas OK, {n_fallo} con fallo real, el resto sin --apply/sin fuente/bloqueadas por un aviso de plausibilidad.")


if __name__ == "__main__":
    main()
