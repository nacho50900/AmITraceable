"""
Construye (o amplía) un índice FAISS a partir de fotos de Wikimedia
Commons con coordenadas dentro de España, tomadas a pie por personas
(a diferencia de OSV-5M / Mapillary / KartaView, que son fotos desde
vehículo) -- mismo objetivo que build_flickr_index.py: reducir el domain
gap frente a las fotos de Instagram que analiza app/vision/geolocation.py.

POR QUÉ COMMONS EN VEZ DE FLICKR (decisión para la memoria del TFG):
Flickr cambió su política en 2025/2026 -- crear una API key ahora requiere
una suscripción Flickr Pro de pago (antes era gratuita). El API de
Wikimedia Commons (action=query&list=geosearch) es público, sin API key,
sin coste, y sin límite de tasa agresivo -- solo pide un User-Agent
identificable (ver _USER_AGENT más abajo, exigido por la política de la
Wikimedia Foundation: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_User-Agent_policy).

MISMO diseño de muestreo que build_flickr_index.py (ver ese docstring para
el razonamiento completo): sin mínimo forzado por celda, techo
(--cap-per-cell) para que un monumento no entierre al resto del índice,
dedup por perceptual hash dentro de cada celda, blur de caras SOLO (sin
persistir ninguna imagen a disco).

Diferencia práctica con Flickr: Commons no expone un campo de "accuracy"
del geotag como Flickr -- en cambio, casi toda foto en Commons con
coordenadas las tiene puestas a mano por quien subió el archivo (via la
plantilla {{Location}}), lo cual en la práctica suele ser más fiable que
un geotag EXIF automático de Flickr, pero no hay un número que filtrar.
Como filtro de calidad indirecto, aquí se descartan resultados que no son
fotografías reales (mapas, logos, diagramas, escaneos) por tipo MIME y
tamaño mínimo -- no es perfecto (algunos mapas/carteles se cuelan), pero
es la señal disponible sin analizar el contenido de cada imagen.

Uso:
    pip install httpx opencv-python imagehash pandas numpy faiss-cpu \
        pillow tqdm torch transformers huggingface_hub
    python build_commons_index.py --output ../../data/commons_spain --cell-km 10 \
        --cap-per-cell 400

Salida (en --output): igual formato que build_flickr_index.py
    (embeddings.npy, index.faiss, index_meta.csv, _completed_cells.txt,
    _cell_stats.csv) -- combínalo con las demás fuentes usando
    scripts/merge_faiss_indices.py.

Resumible con Ctrl+C igual que build_flickr_index.py.
"""
import argparse
import io
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from build_faiss_index import MODEL_NAME, embed_image, load_model  # noqa: E402
from flickr_grid import GridCell, generate_spain_grid  # noqa: E402
from image_ingest_common import (  # noqa: E402
    blur_faces,
    ensure_yunet_model,
    is_near_duplicate,
    nearest_province,
    sort_cells_by_proximity,
)

import faiss  # noqa: E402
import torch  # noqa: E402

_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
# Wikimedia exige un User-Agent identificable en todas las peticiones al
# API (no es opcional -- peticiones con User-Agent genérico/por defecto de
# la librería HTTP reciben más 429 y pueden acabar bloqueadas sin aviso).
# Ver https://foundation.wikimedia.org/wiki/Policy:Wikimedia_User-Agent_policy
_USER_AGENT = "AmITraceable-TFG-ImageIngest/1.0 (https://github.com/nacho50900/AmITraceable)"

# Radio máximo que admite list=geosearch (metros). Si --cell-km implica un
# radio mayor, se recorta aquí -- ver _radius_for_cell().
_MAX_GEOSEARCH_RADIUS_M = 10000

_ACCEPTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MIN_DIMENSION_PX = 400  # descarta iconos/miniaturas/escaneos muy pequeños
# Descarta imágenes anormalmente grandes (escaneos de alta resolución,
# panorámicas gigantes) ANTES de descargarlas -- ya tenemos width/height
# en los metadatos de imageinfo, no hace falta bajar el fichero para
# saberlo. En una calibración real aparecieron imágenes de 136M y 94M
# píxeles que dispararon el tiempo de esa celda muy por encima de la
# media (decodificar+reescalar una imagen así es carísimo para un solo
# elemento del índice que, además, DINOv2 va a reescalar igual que
# cualquier otra). 50 megapíxeles es generoso para cualquier foto normal
# (una réflex de 45MP moderna produce ~45M píxeles) y excluye estos casos
# patológicos.
_MAX_PIXELS = 50_000_000


def _radius_for_cell(cell: GridCell) -> int:
    """Radio en metros para cubrir la celda desde su centro -- se usa la
    semidiagonal (algo más que medio lado) para no dejar huecos sin cubrir
    en las esquinas de la celda, recortado al máximo que admite la API."""
    lat_km = (cell.max_lat - cell.min_lat) * 111.0
    lon_km = (cell.max_lon - cell.min_lon) * 111.320
    half_diagonal_km = ((lat_km ** 2 + lon_km ** 2) ** 0.5) / 2
    return min(_MAX_GEOSEARCH_RADIUS_M, int(half_diagonal_km * 1000) + 100)


def _commons_request(client: httpx.Client, params: dict, max_retries: int = 4) -> dict:
    params = {**params, "format": "json"}
    # Códigos de error del propio API de Wikimedia que son transitorios
    # (el backend de búsqueda está saturado momentáneamente) -- merece la
    # pena reintentar con backoff. Otros códigos de error (parámetro mal
    # formado, etc.) no se arreglan reintentando, así que se relanzan tal
    # cual. Antes de este fix, "cirrussearch-too-busy-error" (visto en una
    # calibración real, 2 veces en 30 celdas) tiraba la celda entera a la
    # primera sin reintentar -- y, peor aún, esa celda se marcaba como
    # completada para siempre en _load_existing_state/main() aunque nunca
    # se hubiera llegado a consultar de verdad. Ver también el flag
    # `is_transient_error` que main() usa para NO marcar la celda como
    # completada si el fallo persiste tras agotar los reintentos.
    _TRANSIENT_ERROR_CODES = {"cirrussearch-too-busy-error", "ratelimited", "maxlag"}

    for attempt in range(max_retries):
        try:
            response = client.get(_COMMONS_API_URL, params=params, headers={"User-Agent": _USER_AGENT}, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                error_code = data["error"].get("code", "")
                if error_code in _TRANSIENT_ERROR_CODES and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Wikimedia API error: {data['error']}")
            return data
        except httpx.HTTPError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Wikimedia API: agotados los reintentos (posible rate limit sostenido)")


def _geosearch_cell(client: httpx.Client, cell: GridCell, limit: int, gscontinue: str | None) -> tuple[list[dict], str | None]:
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{cell.center_lat}|{cell.center_lon}",
        "gsradius": _radius_for_cell(cell),
        "gsnamespace": 6,  # namespace File:
        "gslimit": limit,
    }
    if gscontinue:
        params["gscontinue"] = gscontinue

    data = _commons_request(client, params)
    results = data.get("query", {}).get("geosearch", [])
    next_continue = data.get("continue", {}).get("gscontinue")
    return results, next_continue


def _fetch_imageinfo(client: httpx.Client, titles: list[str]) -> dict[str, dict]:
    """Batch de hasta 50 títulos (límite del API sin permisos elevados).
    Devuelve {title: {url, width, height, mime, license}}."""
    if not titles:
        return {}
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
    }
    data = _commons_request(client, params)
    pages = data.get("query", {}).get("pages", {})

    result = {}
    for page in pages.values():
        title = page.get("title")
        infos = page.get("imageinfo")
        if not title or not infos:
            continue
        info = infos[0]
        license_name = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "desconocida")
        result[title] = {
            "url": info.get("url"),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "mime": info.get("mime"),
            "license": license_name,
        }
    return result


def _download_bytes(client: httpx.Client, url: str, max_retries: int = 5) -> tuple[bytes | None, str | None]:
    """Devuelve (bytes, None) si va bien, o (None, motivo) si falla -- el
    motivo se usa solo para diagnóstico agregado (ver reject_reasons en
    _process_cell).

    Con reintentos y backoff, a diferencia de la primera versión: en la
    calibración real sobre datos reales, el 100% de los fallos de
    descarga (3260/3260) eran HTTP 429 -- 8 descargas en paralelo sin
    ningún reintento saturaban el rate limit de upload.wikimedia.org
    (el CDN de medios, que tiene su propio límite, distinto del de
    commons.wikimedia.org/w/api.php). Se respeta la cabecera
    Retry-After si el servidor la manda; si no, backoff exponencial.
    """
    for attempt in range(max_retries):
        try:
            response = client.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20)
            if response.status_code == 429:
                if attempt == max_retries - 1:
                    return None, "http_429_reintentos_agotados"
                retry_after = response.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else (2 ** attempt)
                time.sleep(wait_s)
                continue
            response.raise_for_status()
            return response.content, None
        except httpx.HTTPStatusError as e:
            return None, f"http_{e.response.status_code}"
        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                return None, "timeout"
            time.sleep(2 ** attempt)
        except httpx.ConnectError as e:
            if attempt == max_retries - 1:
                return None, f"connect_error:{e}"
            time.sleep(2 ** attempt)
        except Exception as e:
            return None, f"otro:{type(e).__name__}:{e}"
    return None, "http_429_reintentos_agotados"


def _process_cell(
    cell: GridCell,
    client: httpx.Client,
    detector,
    processor,
    model,
    device: str,
    cap: int,
    phash_threshold: int,
    max_pages: int,
) -> tuple[list[np.ndarray], list[dict], dict]:
    accepted_hashes: list[imagehash.ImageHash] = []
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []
    # Diagnóstico: en qué paso se pierde cada candidato, y qué mime types
    # dominan los rechazos por tipo -- para poder distinguir "Commons no
    # tiene fotos aquí" (n_vistas bajo) de "hay muchos ficheros geotagged
    # que no son fotos modernas -- mapas, escaneos, etc." (n_vistas alto,
    # rechazo alto por mime), que son situaciones muy distintas de cara a
    # decidir si hay que ajustar el filtro.
    rejected_mime_counter: dict[str, int] = {}
    download_error_counter: dict[str, int] = {}
    reject_reasons = {
        "sin_imageinfo": 0,
        "mime_no_valido": 0,
        "demasiado_pequena": 0,
        "demasiado_grande": 0,
        "fallo_descarga": 0,
        "casi_duplicada": 0,
    }

    # FASE 1: recopilar candidatos (solo metadatos vía imageinfo, sin
    # descargar ninguna imagen todavía) de hasta max_pages páginas.
    #
    # IMPORTANTE: per_page es un valor FIJO (el máximo que admite gslimit),
    # NO depende de --cap-per-cell. La primera versión de este script usaba
    # per_page = min(500, cap*2) -- con un cap pequeño (p.ej. 10, como en
    # un test rápido) eso limitaba la propia BÚSQUEDA a los 20 resultados
    # más cercanos al centro de la celda, así que el cap se llenaba
    # siempre con el mismo punto hiper-local (un monumento, una plaza)
    # antes de que la búsqueda llegara a cualquier otro rincón de la
    # celda -- visto en un test real sobre Madrid: 10/10 fotos en un radio
    # de ~250m, dos de ellas con coordenadas idénticas. Desacoplar
    # per_page de cap evita que el TAMAÑO del cap decida, de rebote, cómo
    # de amplia es la búsqueda.
    per_page = 500
    n_seen = 0
    gscontinue = None
    page = 0
    candidates_pool: list[tuple[dict, dict]] = []
    had_unrecovered_error = False

    while page < max_pages:
        page += 1
        try:
            results, gscontinue = _geosearch_cell(client, cell, per_page, gscontinue)
        except Exception as e:
            print(f"  Aviso: fallo en geosearch celda {cell.id}, página {page}: {e}")
            had_unrecovered_error = True
            break

        if not results:
            break
        n_seen += len(results)

        titles = [r["title"] for r in results]
        # imageinfo solo admite 50 títulos por petición -- se trocea.
        infos: dict[str, dict] = {}
        for i in range(0, len(titles), 50):
            infos.update(_fetch_imageinfo(client, titles[i:i + 50]))

        for r in results:
            info = infos.get(r["title"])
            if not info or not info["url"]:
                reject_reasons["sin_imageinfo"] += 1
                continue
            if info["mime"] not in _ACCEPTED_MIME_TYPES:
                reject_reasons["mime_no_valido"] += 1
                rejected_mime_counter[info["mime"]] = rejected_mime_counter.get(info["mime"], 0) + 1
                continue
            if info["width"] < _MIN_DIMENSION_PX or info["height"] < _MIN_DIMENSION_PX:
                reject_reasons["demasiado_pequena"] += 1
                continue
            if info["width"] * info["height"] > _MAX_PIXELS:
                reject_reasons["demasiado_grande"] += 1
                continue
            candidates_pool.append((r, info))

        if gscontinue is None:
            break
        time.sleep(0.2)  # cortesía con el API, aunque no publique un límite estricto

    # FASE 2: barajar ANTES de descargar nada -- así el cap se llena con
    # una muestra aleatoria de toda la celda, no con los resultados más
    # cercanos al centro (que es como los devuelve geosearch por defecto).
    random.shuffle(candidates_pool)

    # FASE 3: descargar/dedup/blur/embed en lotes, parando en cuanto se
    # llena el cap -- no se descarga todo el pool si el cap se llena antes.
    #
    # _DOWNLOAD_BATCH bajado de 8 a 3 tras la calibración real: con 8
    # descargas en paralelo SIN reintentos, el 100% de los fallos eran
    # HTTP 429 -- saturábamos el rate limit de upload.wikimedia.org de
    # entrada. Ahora _download_bytes ya reintenta con backoff, pero
    # mantener menos hilos en paralelo reduce cuántas peticiones chocan
    # contra el límite A LA VEZ (y por tanto cuántas necesitan reintentar).
    _DOWNLOAD_BATCH = 3
    i = 0
    while len(meta_rows) < cap and i < len(candidates_pool):
        batch = candidates_pool[i : i + _DOWNLOAD_BATCH]
        i += _DOWNLOAD_BATCH

        with ThreadPoolExecutor(max_workers=3) as pool:
            downloaded = list(pool.map(lambda c: _download_bytes(client, c[1]["url"]), batch))

        for (r, info), (image_bytes, error) in zip(batch, downloaded):
            if len(meta_rows) >= cap:
                break
            if image_bytes is None:
                reject_reasons["fallo_descarga"] += 1
                download_error_counter[error] = download_error_counter.get(error, 0) + 1
                continue
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                reject_reasons["fallo_descarga"] += 1
                key = f"pil_no_decodifica:{type(e).__name__}"
                download_error_counter[key] = download_error_counter.get(key, 0) + 1
                continue

            phash = imagehash.phash(pil_image)
            if is_near_duplicate(phash, accepted_hashes, phash_threshold):
                reject_reasons["casi_duplicada"] += 1
                continue

            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            image_bgr = blur_faces(image_bgr, detector)
            blurred_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            try:
                embedding = embed_image(blurred_pil, processor, model, device)
            except Exception as e:
                print(f"  Aviso: fallo extrayendo embedding de {r['title']}: {e}")
                continue

            lat, lon = float(r["lat"]), float(r["lon"])
            embeddings.append(embedding)
            meta_rows.append(
                {
                    "id": f"commons_{r['pageid']}",
                    "lat": lat,
                    "lon": lon,
                    "region": nearest_province(lat, lon),
                    "source": "wikimedia_commons",
                    "license": info["license"],
                }
            )
            accepted_hashes.append(phash)

    diagnostics = {
        "n_seen": n_seen,
        "n_pool": len(candidates_pool),
        **reject_reasons,
        "rejected_mime_counter": rejected_mime_counter,
        "download_error_counter": download_error_counter,
        "had_unrecovered_error": had_unrecovered_error,
    }
    return embeddings, meta_rows, diagnostics


def _load_existing_state(output_dir: Path):
    """Carga solo lo que hace falta mantener en memoria durante TODA la
    ejecución: known_ids (para el dedup al reabrir celdas), cuántas fotos
    hay ya en total (para los informes), qué celdas están completadas, y
    las estadísticas por celda (una fila por celda -- acotado por el
    tamaño del grid, no por el número de fotos).

    A propósito, NO devuelve los embeddings ni las filas de metadata
    completas para que el llamador las guarde en memoria durante toda la
    ejecución -- eso es justo lo que causó un
    `numpy._core._exceptions._ArrayMemoryError` real tras varias horas
    seguidas de ejecución: la lista de embeddings en memoria no paraba de
    crecer durante toda la vida del proceso (varios días), hasta que no
    quedó hueco ni para una imagen de 68MB. Los embeddings/metadata en
    disco se releen solo brevemente dentro de cada `_persist_state()`
    para fusionarlos con el lote nuevo, y se sueltan justo después -- ver
    ese docstring.
    """
    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "index_meta.csv"
    completed_path = output_dir / "_completed_cells.txt"
    stats_path = output_dir / "_cell_stats.csv"

    known_ids: set[str] = set()
    total_photos = 0
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        total_photos = len(meta_df)
        if "id" in meta_df.columns:
            known_ids = set(meta_df["id"])
        del meta_df

    # Comprobación de consistencia (ver también _persist_state):
    # embeddings.npy e index_meta.csv deben tener el mismo número de
    # filas. Se usa mmap_mode="r" para consultar solo shape[0] (el número
    # de filas) sin cargar el array completo en memoria -- este chequeo
    # de arranque no debería, por sí mismo, competir por la misma memoria
    # que se acaba de liberar con el cambio de arriba.
    n_embeddings = 0
    if embeddings_path.exists():
        embeddings_mmap = np.load(embeddings_path, mmap_mode="r")
        n_embeddings = embeddings_mmap.shape[0]
        del embeddings_mmap

    if n_embeddings != total_photos:
        print(f"Error: embeddings.npy tiene {n_embeddings} vectores pero index_meta.csv tiene "
              f"{total_photos} filas -- no coinciden.")
        print("Seguir así arriesga desalinear qué vector corresponde a qué foto en el índice final.")
        print(f"Revisa manualmente {output_dir} antes de continuar (lo más seguro, si no hay forma de saber "
              f"cuál de los dos ficheros es el bueno, es borrar la carpeta entera y volver a empezar).")
        sys.exit(1)

    completed = set(completed_path.read_text().splitlines()) if completed_path.exists() else set()
    cell_stats = pd.read_csv(stats_path).to_dict("records") if stats_path.exists() else []

    if completed:
        print(f"Reanudando: {len(completed)} celdas y {total_photos} fotos ya procesadas.")

    return known_ids, total_photos, completed, cell_stats


def _persist_state(output_dir: Path, new_embeddings: list, new_meta_rows: list, completed: set, cell_stats: list) -> None:
    """Fusiona NEW_EMBEDDINGS/NEW_META_ROWS -- SOLO el lote acumulado
    desde el último flush, NO todo el historial de la ejecución -- con lo
    que ya hay en disco, y escribe el resultado combinado.

    Este cambio (de "recibe todo el historial acumulado" a "recibe solo
    el lote nuevo, y fusiona aquí con lo persistido") es lo que permite
    que _run() vacíe sus listas en memoria después de cada flush, en vez
    de mantener en RAM los embeddings de TODAS las fotos aceptadas
    durante una ejecución de varios días. El lote en memoria en un
    momento dado queda acotado por
    `--flush-every-cells * --cap-per-cell` (unas pocas decenas de miles
    como mucho), no por el total final del índice (hasta ~1.22M
    proyectados). El coste es releer lo ya persistido en cada flush --
    aceptable porque los flushes son cada bastantes celdas
    (`--flush-every-cells`), no por cada foto individual; y solo durante
    ese momento puntual, no de forma continua.

    Escritura atómica: cada fichero final se escribe primero a una ruta
    temporal en el mismo directorio, y todos los renombrados
    (`os.replace`, atómico por fichero en el mismo volumen tanto en
    Windows como en POSIX) se hacen SEGUIDOS al terminar, una vez que los
    temporales ya están completos en disco -- no intercalados con el
    trabajo de construirlos. Esto no hace la operación atómica como GRUPO
    (no hay forma portable de renombrar varios ficheros a la vez en una
    única operación), pero reduce al mínimo posible la ventana en la que
    una interrupción podría dejar unos ficheros con la versión nueva y
    otros con la vieja. Motivo original del cambio a atómico: una
    escritura directa con `to_csv()` sobre el fichero final, interrumpida
    a mitad, dejó `index_meta.csv` completamente vacío en un caso real --
    ver la comprobación de consistencia en `_load_existing_state`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pid_suffix = f".tmp{os.getpid()}"
    pending_renames: list[tuple[Path, Path]] = []

    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "index_meta.csv"

    if new_embeddings or embeddings_path.exists():
        if new_embeddings:
            new_matrix = np.vstack(new_embeddings).astype("float32")
            if embeddings_path.exists():
                with open(embeddings_path, "rb") as f:
                    existing_matrix = np.load(f)
                combined_matrix = np.concatenate([existing_matrix, new_matrix], axis=0)
                del existing_matrix
            else:
                combined_matrix = new_matrix
            del new_matrix
        else:
            with open(embeddings_path, "rb") as f:
                combined_matrix = np.load(f)

        emb_tmp = output_dir / f"embeddings.npy{pid_suffix}"
        with open(emb_tmp, "wb") as f:
            np.save(f, combined_matrix)
        pending_renames.append((emb_tmp, embeddings_path))

        dimension = combined_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(combined_matrix)
        index_tmp = output_dir / f"index.faiss{pid_suffix}"
        faiss.write_index(index, str(index_tmp))
        pending_renames.append((index_tmp, output_dir / "index.faiss"))
        del combined_matrix, index

    if new_meta_rows or meta_path.exists():
        new_df = pd.DataFrame(new_meta_rows)
        if meta_path.exists():
            existing_df = pd.read_csv(meta_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True) if len(new_df) else existing_df
            del existing_df
        else:
            combined_df = new_df

        meta_tmp = output_dir / f"index_meta.csv{pid_suffix}"
        combined_df.to_csv(meta_tmp, index=False)
        pending_renames.append((meta_tmp, meta_path))
        del combined_df, new_df

    completed_tmp = output_dir / f"_completed_cells.txt{pid_suffix}"
    completed_tmp.write_text("\n".join(sorted(completed)))
    pending_renames.append((completed_tmp, output_dir / "_completed_cells.txt"))

    stats_tmp = output_dir / f"_cell_stats.csv{pid_suffix}"
    pd.DataFrame(cell_stats).to_csv(stats_tmp, index=False)
    pending_renames.append((stats_tmp, output_dir / "_cell_stats.csv"))

    # Todos los temporales están completos en disco -- ahora sí, los
    # renombrados seguidos, sin nada más de por medio.
    for tmp_path, final_path in pending_renames:
        os.replace(tmp_path, final_path)


def _acquire_lock(output_dir: Path) -> Path:
    """Evita que dos instancias del script corran a la vez sobre el mismo
    --output. Sin esto, dos procesos escribiendo index_meta.csv/
    embeddings.npy al mismo tiempo pueden pisarse -- visto en un caso
    real: una segunda ejecución lanzada por accidente sobre la misma
    carpeta leyó index_meta.csv justo cuando la primera lo tenía truncado
    a medio escribir, y crasheó con `pandas.errors.EmptyDataError`.

    No detecta automáticamente si el proceso dueño del lock sigue vivo
    (complicaría el script para un caso de uso de TFG en un único
    equipo) -- si el lock queda huérfano tras un corte de luz o un kill
    -9, hay que borrar el fichero .lock a mano. El mensaje de error dice
    esto explícitamente para que no haga falta adivinarlo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".lock"
    if lock_path.exists():
        owner = lock_path.read_text().strip()
        print(f"Error: ya existe {lock_path} (creado por el proceso PID {owner}).")
        print(f"Esto significa que ya hay OTRA ejecución de este script usando --output {output_dir}.")
        print("Lanzar dos instancias a la vez sobre la misma carpeta corrompe index_meta.csv/embeddings.npy")
        print("(dos escrituras simultáneas se pisan entre sí).")
        print(f"Si estás seguro de que NO hay ninguna otra instancia corriendo (p.ej. el lock quedó huérfano")
        print(f"tras un cierre inesperado), borra {lock_path} a mano y vuelve a intentarlo.")
        sys.exit(1)
    lock_path.write_text(str(os.getpid()))
    return lock_path


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../data/commons_spain")
    parser.add_argument("--cell-km", type=float, default=10.0)
    parser.add_argument("--cap-per-cell", type=int, default=400)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--max-pages-per-cell", type=int, default=3)
    parser.add_argument("--flush-every-cells", type=int, default=25)
    parser.add_argument("--max-cells", type=int, default=None,
                         help="Limita el numero de celdas a procesar en esta ejecucion (para probar antes de lanzar todo)")
    parser.add_argument("--near", type=str, default=None,
                         help='"lat,lon" -- ordena las celdas por cercania a este punto en vez del orden de generacion del grid '
                              '(las primeras celdas del grid caen en el Atlantico/frontera portuguesa). Combinar con --max-cells '
                              'para un test rapido en una ciudad conocida, p.ej. --near "40.4168,-3.7038" --max-cells 1 para Madrid.')
    parser.add_argument("--sample-cells", type=int, default=None,
                         help="En vez de procesar las celdas en orden (o cerca de --near), toma una muestra ALEATORIA de N celdas "
                              "repartidas por toda España. Pensado para medir el rendimiento real (tiempo/celda, fotos/celda) antes "
                              "de lanzar la ejecucion completa sobre las ~11700 celdas -- una muestra cerca de una sola ciudad "
                              "(--near) no es representativa del pais entero (zonas rurales rinden mucho menos). Incompatible con --near.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para --sample-cells, para que la muestra sea reproducible")
    args = parser.parse_args()

    if args.near and args.sample_cells:
        print("Error: --near y --sample-cells son incompatibles (uno prueba UN sitio concreto, el otro mide el pais entero).")
        sys.exit(1)

    output_dir = Path(args.output)
    lock_path = _acquire_lock(output_dir)
    try:
        _run(args, output_dir)
    finally:
        _release_lock(lock_path)


def _run(args: argparse.Namespace, output_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device} (modelo {MODEL_NAME})")

    processor, model = load_model(device)
    detector = cv2.FaceDetectorYN.create(
        str(ensure_yunet_model()), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )

    known_ids, total_photos, completed, cell_stats = _load_existing_state(output_dir)
    photos_at_start = total_photos

    all_cells = generate_spain_grid(cell_km=args.cell_km)
    pending_cells = [c for c in all_cells if c.id not in completed]

    if args.near:
        near_lat, near_lon = (float(x) for x in args.near.split(","))
        pending_cells = sort_cells_by_proximity(pending_cells, near_lat, near_lon)
    elif args.sample_cells:
        rng = random.Random(args.seed)
        pending_cells = rng.sample(pending_cells, min(args.sample_cells, len(pending_cells)))
    if args.max_cells:
        pending_cells = pending_cells[:args.max_cells]

    print(f"{len(all_cells)} celdas en el grid ({len(pending_cells)} pendientes en esta ejecucion).")

    # Buffers de SOLO lo acumulado desde el último flush -- NO todo el
    # historial de la ejecución (ver _persist_state). Se vacían tras cada
    # flush para que la memoria del proceso no crezca sin límite durante
    # una ejecución de varios días.
    since_flush_embeddings: list = []
    since_flush_meta: list = []

    client = httpx.Client()
    cells_since_flush = 0
    cells_run_this_execution = 0
    run_start = time.monotonic()
    global_reject_totals = {"sin_imageinfo": 0, "mime_no_valido": 0, "demasiado_pequena": 0, "demasiado_grande": 0, "fallo_descarga": 0, "casi_duplicada": 0}
    global_rejected_mime: dict[str, int] = {}
    global_download_errors: dict[str, int] = {}
    n_duplicates_skipped = 0

    def _flush() -> None:
        nonlocal since_flush_embeddings, since_flush_meta
        _persist_state(output_dir, since_flush_embeddings, since_flush_meta, completed, cell_stats)
        since_flush_embeddings = []
        since_flush_meta = []

    try:
        for cell in tqdm(pending_cells, desc="Celdas"):
            cell_embeddings, cell_meta, diag = _process_cell(
                cell, client, detector, processor, model, device,
                cap=args.cap_per_cell, phash_threshold=args.phash_threshold,
                max_pages=args.max_pages_per_cell,
            )
            # Filtra cualquier foto que ya estuviera en el índice (relevante
            # sobre todo al reabrir una celda ya completada -- en una
            # ejecución normal sobre celdas nuevas esto no debería quitar
            # nada, ya que cada celda solo se procesa una vez).
            new_embeddings, new_meta = [], []
            for emb, meta in zip(cell_embeddings, cell_meta):
                if meta["id"] in known_ids:
                    n_duplicates_skipped += 1
                    continue
                known_ids.add(meta["id"])
                new_embeddings.append(emb)
                new_meta.append(meta)
            since_flush_embeddings.extend(new_embeddings)
            since_flush_meta.extend(new_meta)
            total_photos += len(new_meta)
            # Solo se marca la celda como completada si no hubo un fallo
            # sin recuperar (p.ej. "cirrussearch-too-busy-error" agotando
            # los reintentos) -- si no, esa celda se queda pendiente para
            # la siguiente ejecución en vez de darse por vacía para
            # siempre. Antes de este fix, un fallo transitorio de
            # Wikimedia hacía que la celda se marcara completada con 0
            # fotos igual que una celda rural genuinamente vacía --
            # indistinguibles, y la zona quedaba sin representar en el
            # índice sin que nadie se enterase.
            if not diag["had_unrecovered_error"]:
                completed.add(cell.id)
            else:
                print(f"  {cell.id} queda pendiente (fallo sin recuperar) -- se reintentará en la próxima ejecución.")
            cell_stats.append({
                "cell_id": cell.id, "n_vistas": diag["n_seen"], "n_aceptadas": len(cell_meta),
                "n_pool": diag["n_pool"], "sin_imageinfo": diag["sin_imageinfo"],
                "mime_no_valido": diag["mime_no_valido"], "demasiado_pequena": diag["demasiado_pequena"],
                "demasiado_grande": diag["demasiado_grande"],
                "fallo_descarga": diag["fallo_descarga"], "casi_duplicada": diag["casi_duplicada"],
                "fallo_sin_recuperar": diag["had_unrecovered_error"],
            })
            for k in global_reject_totals:
                global_reject_totals[k] += diag[k]
            for mime, n in diag["rejected_mime_counter"].items():
                global_rejected_mime[mime] = global_rejected_mime.get(mime, 0) + n
            for err, n in diag["download_error_counter"].items():
                global_download_errors[err] = global_download_errors.get(err, 0) + n
            cells_since_flush += 1
            cells_run_this_execution += 1

            if cells_since_flush >= args.flush_every_cells:
                _flush()
                cells_since_flush = 0
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Guardando progreso antes de salir...")
    finally:
        client.close()
        _flush()

    elapsed_s = time.monotonic() - run_start
    n_photos_run = total_photos - photos_at_start

    print(f"\n{total_photos} fotos en el índice total, guardadas en {output_dir}")
    if n_duplicates_skipped:
        print(f"({n_duplicates_skipped} fotos descartadas por ya estar en el índice -- normal solo si has reabierto alguna celda ya completada.)")
    if elapsed_s > 0 and cells_run_this_execution > 0:
        s_per_cell = elapsed_s / cells_run_this_execution
        photos_per_hour = n_photos_run / (elapsed_s / 3600)
        remaining_cells = len(all_cells) - len(completed)
        eta_hours = (remaining_cells * s_per_cell) / 3600
        photos_per_cell_avg = n_photos_run / cells_run_this_execution
        print(f"\nEsta ejecución: {cells_run_this_execution} celdas, {n_photos_run} fotos, {elapsed_s/60:.1f} min "
              f"({s_per_cell:.1f}s/celda, ~{photos_per_hour:.0f} fotos/hora).")
        print(f"Extrapolado a las {remaining_cells} celdas que faltan del grid completo (estimación gruesa, "
              f"el rendimiento real varía mucho entre celdas urbanas y rurales -- para eso es --sample-cells): "
              f"~{eta_hours:.1f}h (~{eta_hours/24:.1f} días) y ~{remaining_cells * photos_per_cell_avg:.0f} fotos más.")

    n_seen_total = sum(r["n_vistas"] for r in cell_stats[-cells_run_this_execution:]) if cells_run_this_execution else 0
    if n_seen_total:
        print(f"\nDesglose de por qué se descartan candidatos (de {n_seen_total} vistos en esta ejecución):")
        for reason, n in global_reject_totals.items():
            print(f"  {reason}: {n} ({100*n/n_seen_total:.1f}%)")
        print(f"  aceptadas: {n_photos_run} ({100*n_photos_run/n_seen_total:.1f}%)")
        if global_rejected_mime:
            print("  Tipos MIME más comunes entre los rechazados por 'mime_no_valido':")
            for mime, n in sorted(global_rejected_mime.items(), key=lambda x: -x[1])[:8]:
                print(f"    {mime}: {n}")
        if global_download_errors:
            print("  Motivos de 'fallo_descarga' más comunes:")
            for err, n in sorted(global_download_errors.items(), key=lambda x: -x[1])[:8]:
                print(f"    {err}: {n}")
    if cell_stats:
        stats_df = pd.DataFrame(cell_stats)
        empty_cells = (stats_df["n_aceptadas"] == 0).sum()
        print(f"Celdas sin ninguna foto aceptada: {empty_cells}/{len(stats_df)} "
              f"({100 * empty_cells / len(stats_df):.1f}%) -- ver _cell_stats.csv.")
    print("\nPara fusionar con otras fuentes, usa scripts/merge_faiss_indices.py.")



if __name__ == "__main__":
    main()
