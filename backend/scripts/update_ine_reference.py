"""
Script de mantenimiento MANUAL (no se ejecuta en cada análisis, ni
programado -- lo corre un desarrollador de vez en cuando) para comprobar
si el INE ha publicado cifras más recientes que las de
`app/data/ine_reference.py`.

CÓMO USARLO:
    python scripts/update_ine_reference.py            # solo compara, no escribe nada
    python scripts/update_ine_reference.py --apply     # además, aplica PROVINCE_POPULATION
    python scripts/update_ine_reference.py --apply --yes  # sin pedir confirmación por teclado

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
conocer el ID numérico exacto de cada tabla. Se han localizado por
búsqueda web IDs candidatos para las cinco tablas con fuente periódica
conocida (población por provincia, estado civil, nacionalidad, tasas EPA).

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
import sys
from datetime import date
from pathlib import Path

import httpx

sys.path.insert(0, "..")
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
_TABLA_TASAS_EPA = 1113  # "Tasas de actividad, paro y empleo, por sexo y distintos grupos de edad"


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

        valor = datos[-1].get("Valor")
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
    result: dict[str, float] = {}
    series = _fetch_series(_TABLA_ESTADO_CIVIL)
    for serie in series:
        nombre = serie.get("Nombre", "")
        datos = serie.get("Data", [])
        if datos and datos[-1].get("Valor") is not None:
            result[nombre] = datos[-1]["Valor"]
    _warn_if_empty(_TABLA_ESTADO_CIVIL, series, result)
    return result


def fetch_nationality() -> dict[str, float]:
    """CANDIDATO SIN VERIFICAR (ver _TABLA_NACIONALIDAD más arriba) --
    mismo aviso que fetch_marital_status: valores en bruto, sin normalizar."""
    result: dict[str, float] = {}
    series = _fetch_series(_TABLA_NACIONALIDAD)
    for serie in series:
        nombre = serie.get("Nombre", "")
        datos = serie.get("Data", [])
        if datos and datos[-1].get("Valor") is not None:
            result[nombre] = datos[-1]["Valor"]
    _warn_if_empty(_TABLA_NACIONALIDAD, series, result)
    return result


def fetch_situacion_laboral() -> dict[str, float]:
    """CANDIDATO SIN VERIFICAR (ver _TABLA_TASAS_EPA más arriba) -- mismo
    aviso que fetch_marital_status: valores en bruto, sin normalizar.
    Nota adicional: esta tabla da TASAS (porcentajes sobre población
    activa/de 16+), no la misma base que SITUACION_LABORAL_DISTRIBUTION
    (proporción sobre población de 16+ total, incluyendo inactivos
    desglosados) -- hace falta el mismo cálculo que ya está documentado
    en el comentario de esa constante en ine_reference.py, esta función
    solo trae el dato crudo de la EPA."""
    result: dict[str, float] = {}
    series = _fetch_series(_TABLA_TASAS_EPA)
    for serie in series:
        nombre = serie.get("Nombre", "")
        datos = serie.get("Data", [])
        if datos and datos[-1].get("Valor") is not None:
            result[nombre] = datos[-1]["Valor"]
    _warn_if_empty(_TABLA_TASAS_EPA, series, result)
    return result


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


def _normalize_situacion_laboral(raw: dict[str, float]) -> dict[str, float] | None:
    tasa_actividad = raw.get("Tasa de actividad. Nacional. Ambos sexos. Total. Valor absoluto")
    tasa_paro = raw.get("Tasa de paro. Nacional. Ambos sexos. Total. Valor absoluto")
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
    args = parser.parse_args()

    print("Comprobando tablas de app/data/ine_reference.py contra el INE...")
    if args.apply:
        print("(--apply: SÍ se escribirá en ine_reference.py lo que tenga cambios y se confirme)\n")
    else:
        print("(esto NO modifica ine_reference.py -- solo muestra diferencias; usa --apply para escribir)\n")

    try:
        province_data = fetch_population_by_province()
        _compare("PROVINCE_POPULATION", ine_reference.PROVINCE_POPULATION, province_data)
        if args.apply:
            _apply_province_population(province_data, auto_confirm=args.yes)
    except httpx.HTTPError as e:
        print(f"ERROR al descargar población por provincia: {e}")

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
                _apply_marital_status(marital_normalized, auto_confirm=args.yes)
                if by_sex_normalized is not None:
                    _apply_marital_status_by_sex(by_sex_normalized, auto_confirm=args.yes)
    except httpx.HTTPError as e:
        print(f"ERROR al descargar estado civil: {e}")

    try:
        nationality_raw = fetch_nationality()
        nationality_normalized = _normalize_nationality(nationality_raw)
        if nationality_normalized is None:
            print(
                "\n=== NATIONALITY_DISTRIBUTION ===\n"
                "  No se encontraron las filas nacionales esperadas -- revisa el "
                "formato real con fetch_nationality() antes de forzar nada."
            )
        else:
            _compare("NATIONALITY_DISTRIBUTION", ine_reference.NATIONALITY_DISTRIBUTION, nationality_normalized)
            if args.apply:
                _apply_nationality(nationality_normalized, auto_confirm=args.yes)
    except httpx.HTTPError as e:
        print(f"ERROR al descargar nacionalidad: {e}")

    try:
        laboral_raw = fetch_situacion_laboral()
        laboral_normalized = _normalize_situacion_laboral(laboral_raw)
        if laboral_normalized is None:
            print(
                "\n=== SITUACION_LABORAL_DISTRIBUTION ===\n"
                "  No se encontraron 'Tasa de actividad'/'Tasa de paro' nacionales "
                "esperadas -- revisa el formato real con fetch_situacion_laboral()."
            )
        else:
            _compare("SITUACION_LABORAL_DISTRIBUTION", ine_reference.SITUACION_LABORAL_DISTRIBUTION, laboral_normalized)
            tasa_paro = laboral_raw.get("Tasa de paro. Nacional. Ambos sexos. Total. Valor absoluto")
            tasa_sospechosa = tasa_paro is not None and (
                not (_TASA_PARO_PLAUSIBLE[0] <= tasa_paro <= _TASA_PARO_PLAUSIBLE[1])
                or abs(tasa_paro - _TASA_PARO_RECIENTE_CONOCIDA) > _TASA_PARO_RECIENTE_CONOCIDA
            )
            if tasa_sospechosa:
                print(
                    f"\n  AVISO IMPORTANTE: la tasa de paro nacional que trae esta tabla "
                    f"es {tasa_paro}%, muy distinta del ~{_TASA_PARO_RECIENTE_CONOCIDA}% "
                    "conocido de la EPA más reciente. Puede que _TABLA_TASAS_EPA tenga el "
                    "mismo problema que tuvo _TABLA_POBLACION_PROVINCIAS al principio (ID o "
                    "filtro de categoría no del todo correcto, sin dar ningún error HTTP). "
                    "Verifica el ID de tabla antes de confiar en este número."
                )
            if args.apply:
                if tasa_sospechosa and not args.force_tasa_paro:
                    print(
                        "  SITUACION_LABORAL_DISTRIBUTION: NO se aplica por el aviso de "
                        "arriba -- usa --force-tasa-paro si quieres aplicarlo igualmente."
                    )
                else:
                    _apply_situacion_laboral(laboral_normalized, auto_confirm=args.yes)
    except httpx.HTTPError as e:
        print(f"ERROR al descargar situación laboral: {e}")

    print(
        "\nSTUDIES_DISTRIBUTION y OCCUPATION_DISTRIBUTION no tienen tabla "
        "INE concreta citada (ver _LAST_VERIFIED en ine_reference.py) -- "
        "nada que comprobar automáticamente."
    )
    print(
        "LANGUAGE_BY_CCAA: encuesta puntual (ECEPOV), sin tabla anual "
        "equivalente que comparar -- revisar a mano si el INE ha "
        "publicado una edición más reciente."
    )


if __name__ == "__main__":
    main()
