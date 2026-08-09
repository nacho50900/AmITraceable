"""
Script de mantenimiento MANUAL (no se ejecuta en cada análisis, ni
programado -- lo corre un desarrollador de vez en cuando) para comprobar
si el INE ha publicado cifras más recientes que las de
`app/data/ine_reference.py`.

CÓMO USARLO:
    python scripts/update_ine_reference.py

Imprime, por cada tabla soportada, el valor actual en el código frente al
valor recién descargado del INE, y si difieren. NO SOBREESCRIBE
`ine_reference.py` automáticamente: los valores derivados de ese fichero
tienen razonamiento a mano en los comentarios (ver p. ej.
MARITAL_STATUS_DISTRIBUTION, que combina dos encuestas distintas) que un
script no puede recalcular con seguridad -- aplicar el cambio, revisar el
razonamiento derivado y actualizar `_LAST_VERIFIED` es una decisión
humana, no automatizable sin riesgo de introducir un dato mal derivado en
una herramienta que depende precisamente de la precisión de estos números.

LIMITACIÓN IMPORTANTE, para que quede documentada y no se asuma más
cobertura de la que hay: la API del INE (Tempus3, servicios.ine.es) exige
conocer el ID numérico exacto de cada tabla. Se han localizado por
búsqueda web IDs candidatos para las cinco tablas con fuente periódica
conocida (población por provincia, estado civil, nacionalidad, tasas EPA)
-- pero solo la de población por provincia (t=2917) estaba ya
referenciada de antes en este mismo fichero; las otras tres son IDs
nuevos, localizados al escribir este script, y **no ha sido posible
ejecutarlos contra la API real** desde el entorno de trabajo usado para
esto (sin acceso de red a servicios.ine.es) -- así que el nombre y
contenido esperado de la tabla encajan con lo que se busca, pero no está
confirmado el formato exacto de la respuesta ni que las categorías
encajen 1:1 con las de `ine_reference.py`. Revisa con cuidado la salida
de esas tres la primera vez que corras esto, antes de aplicar ningún
cambio a mano.
"""

from __future__ import annotations

import sys

import httpx

sys.path.insert(0, "..")
from app.data import ine_reference  # noqa: E402

_INE_API_BASE = "https://servicios.ine.es/wstempus/js/ES"

# t=2917 en https://www.ine.es/jaxiT3/Tabla.htm?t=2917 -- "Población por
# provincias y sexo", ya referenciado en el docstring de ine_reference.py
# antes de este script (no es un ID nuevo sin verificar).
_TABLA_POBLACION_PROVINCIAS = 2917

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


def _fetch_series(table_id: int) -> list[dict]:
    """Helper compartido: pide el último dato de cada serie de una tabla
    Tempus3. Devuelve la lista cruda de series tal como la da el INE
    (cada una con al menos "Nombre" y "Data") -- cada función fetch_*
    decide cómo interpretar los nombres de serie de SU tabla concreta,
    porque el formato del campo "Nombre" varía de una tabla a otra."""
    response = httpx.get(
        f"{_INE_API_BASE}/DATOS_TABLA/{table_id}",
        params={"nult": 1},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_population_by_province() -> dict[str, int]:
    """Descarga la tabla 2917 del INE (población por provincia) y la deja
    en el mismo formato de claves que PROVINCE_POPULATION (nombre de
    provincia en minúsculas, sin tildes -- ver `_strip_accents` en
    ine_reference.py). nult=1 pide solo el dato más reciente disponible
    de cada serie, no todo el histórico."""
    result: dict[str, int] = {}
    for serie in _fetch_series(_TABLA_POBLACION_PROVINCIAS):
        nombre = serie.get("Nombre", "")
        datos = serie.get("Data", [])
        if not datos:
            continue
        # El nombre de la serie trae ruido (p. ej. "Ambos sexos. ",
        # "Total. ") delante del nombre real de la provincia -- se separa
        # por el último ". " del nombre, patrón habitual de Tempus3 para
        # estas tablas. Verifícalo contra la respuesta real antes de
        # confiar en este script a ciegas: el formato exacto puede variar
        # entre tablas y no se ha podido probar contra la API real desde
        # el entorno de trabajo usado para escribir esto.
        provincia = nombre.split(". ")[-1].strip()
        valor = datos[-1].get("Valor")
        if provincia and valor is not None:
            result[ine_reference._strip_accents(provincia).lower()] = int(valor)
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
    for serie in _fetch_series(_TABLA_ESTADO_CIVIL):
        nombre = serie.get("Nombre", "")
        datos = serie.get("Data", [])
        if datos and datos[-1].get("Valor") is not None:
            result[nombre] = datos[-1]["Valor"]
    return result


def fetch_nationality() -> dict[str, float]:
    """CANDIDATO SIN VERIFICAR (ver _TABLA_NACIONALIDAD más arriba) --
    mismo aviso que fetch_marital_status: valores en bruto, sin normalizar."""
    result: dict[str, float] = {}
    for serie in _fetch_series(_TABLA_NACIONALIDAD):
        nombre = serie.get("Nombre", "")
        datos = serie.get("Data", [])
        if datos and datos[-1].get("Valor") is not None:
            result[nombre] = datos[-1]["Valor"]
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
    for serie in _fetch_series(_TABLA_TASAS_EPA):
        nombre = serie.get("Nombre", "")
        datos = serie.get("Data", [])
        if datos and datos[-1].get("Valor") is not None:
            result[nombre] = datos[-1]["Valor"]
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


def main() -> None:
    print("Comprobando tablas de app/data/ine_reference.py contra el INE...")
    print("(esto NO modifica ine_reference.py -- solo muestra diferencias)\n")

    try:
        _compare(
            "PROVINCE_POPULATION",
            ine_reference.PROVINCE_POPULATION,
            fetch_population_by_province(),
        )
    except httpx.HTTPError as e:
        print(f"ERROR al descargar población por provincia: {e}")

    print(
        "\n--- A partir de aquí, tablas candidatas SIN VERIFICAR contra la "
        "API real (ver docstrings de cada fetch_*) -- revisa que los "
        "nombres de serie tengan sentido antes de usar estos números ---"
    )
    for name, fetch_fn in [
        ("MARITAL_STATUS_DISTRIBUTION (bruto, sin normalizar)", fetch_marital_status),
        ("NATIONALITY_DISTRIBUTION (bruto, sin normalizar)", fetch_nationality),
        ("SITUACION_LABORAL_DISTRIBUTION (bruto, tasas EPA)", fetch_situacion_laboral),
    ]:
        print(f"\n=== {name} ===")
        try:
            for serie_nombre, valor in fetch_fn().items():
                print(f"  {serie_nombre}: {valor}")
        except httpx.HTTPError as e:
            print(f"  ERROR al descargar: {e}")

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
