"""
Utilidades compartidas por los distintos scripts de ingestión de fuentes
externas de fotos geolocalizadas (build_flickr_index.py,
build_commons_index.py, y las que se añadan después). Factorizado aparte
para no duplicar esta lógica en cada script -- si algún día cambias el
umbral de pixelado o el detector de caras, se cambia en un solo sitio.
"""
from pathlib import Path

import cv2
import imagehash
import numpy as np

# Centroides aproximados de las 50 provincias + Ceuta y Melilla. Es una
# aproximación deliberada (vecino más cercano por distancia en grados, no
# una consulta de reverse-geocoding real por foto) -- consistente con que
# el resto del pipeline (app/vision/geolocation.py) ya trabaja a nivel de
# provincia, no de calle, y evita una llamada de red extra por cada una de
# hasta 1M de fotos.
PROVINCE_CENTROIDS = {
    "A Coruña": (43.37, -8.40), "Álava": (42.85, -2.67), "Albacete": (38.99, -1.86),
    "Alicante": (38.35, -0.48), "Almería": (36.84, -2.47), "Asturias": (43.36, -5.85),
    "Ávila": (40.66, -4.70), "Badajoz": (38.88, -6.97), "Baleares": (39.57, 2.65),
    "Barcelona": (41.39, 2.17), "Burgos": (42.34, -3.70), "Cáceres": (39.48, -6.37),
    "Cádiz": (36.53, -6.30), "Cantabria": (43.18, -3.99), "Castellón": (39.99, -0.03),
    "Ciudad Real": (38.99, -3.93), "Córdoba": (37.89, -4.78), "Cuenca": (40.07, -2.14),
    "Girona": (41.98, 2.82), "Granada": (37.18, -3.60), "Guadalajara": (40.63, -3.17),
    "Guipúzcoa": (43.32, -1.98), "Huelva": (37.26, -6.95), "Huesca": (42.14, -0.41),
    "Jaén": (37.77, -3.79), "La Rioja": (42.47, -2.45), "Las Palmas": (28.10, -15.41),
    "León": (42.60, -5.57), "Lleida": (41.61, 0.62), "Lugo": (43.01, -7.56),
    "Madrid": (40.42, -3.70), "Málaga": (36.72, -4.42), "Murcia": (37.98, -1.13),
    "Navarra": (42.82, -1.65), "Ourense": (42.34, -7.86), "Palencia": (42.01, -4.53),
    "Pontevedra": (42.43, -8.64), "Salamanca": (40.97, -5.66), "Segovia": (40.95, -4.12),
    "Sevilla": (37.39, -5.99), "Soria": (41.76, -2.47), "Sta. Cruz de Tenerife": (28.47, -16.25),
    "Tarragona": (41.12, 1.25), "Teruel": (40.34, -1.11), "Toledo": (39.86, -4.03),
    "Valencia": (39.47, -0.38), "Valladolid": (41.65, -4.72), "Vizcaya": (43.26, -2.92),
    "Zamora": (41.50, -5.75), "Zaragoza": (41.65, -0.88), "Ceuta": (35.89, -5.32),
    "Melilla": (35.29, -2.94),
}


def nearest_province(lat: float, lon: float) -> str:
    best_name, best_dist = None, float("inf")
    for name, (plat, plon) in PROVINCE_CENTROIDS.items():
        d = (lat - plat) ** 2 + (lon - plon) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


_YUNET_REPO_ID = "opencv/face_detection_yunet"
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def ensure_yunet_model() -> Path:
    """Descarga el modelo YuNet (~230KB) vía huggingface_hub en vez de
    descargar directamente el .onnx del repo de GitHub de opencv_zoo: ese
    fichero está versionado con Git LFS en GitHub, así que una descarga
    directa de raw.githubusercontent.com/... devuelve el PUNTERO de texto
    de Git LFS (~130 bytes), no el binario real -- OpenCV falla al
    parsearlo. huggingface_hub resuelve LFS correctamente porque habla el
    protocolo real de descarga del Hub; además ya es dependencia del
    proyecto (scripts/download_osv5m_spain.py, y el propio DINOv2 en
    build_faiss_index.py), así que no añade nada nuevo."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=_YUNET_REPO_ID, filename=_YUNET_FILENAME))


def blur_faces(image_bgr: np.ndarray, detector) -> np.ndarray:
    """Detecta caras en image_bgr (numpy array BGR de OpenCV) y las pixela
    in-place. La detección se hace sobre una copia reescalada a 640px de
    lado largo (la parte cara computacionalmente) y las cajas se reescalan
    de vuelta a resolución real para aplicar el pixelado ahí -- el
    pixelado en sí (downscale+upscale de una región pequeña) es
    prácticamente gratis frente a la detección.

    Se usa pixelado en vez de blur gaussiano: más barato de calcular y
    anonimiza de forma más agresiva e irreversible (un gaussiano suave a
    veces se puede revertir parcialmente con deconvolución; un pixelado
    fuerte destruye la información espacial sin posibilidad de
    recuperarla).
    """
    h, w = image_bgr.shape[:2]
    scale = 640 / max(h, w) if max(h, w) > 640 else 1.0
    small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(image_bgr, (small_w, small_h)) if scale != 1.0 else image_bgr

    detector.setInputSize((small_w, small_h))
    _, faces = detector.detect(small)
    if faces is None:
        return image_bgr

    for face in faces:
        x, y, fw, fh = face[:4] / scale
        x, y = int(max(0, x)), int(max(0, y))
        x2, y2 = min(w, x + int(fw)), min(h, y + int(fh))
        if x2 <= x or y2 <= y:
            continue
        roi = image_bgr[y:y2, x:x2]
        pixel_w = max(1, (x2 - x) // 12)
        pixel_h = max(1, (y2 - y) // 12)
        tiny = cv2.resize(roi, (pixel_w, pixel_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(tiny, (x2 - x, y2 - y), interpolation=cv2.INTER_NEAREST)
        image_bgr[y:y2, x:x2] = pixelated

    return image_bgr


def is_near_duplicate(new_hash: imagehash.ImageHash, accepted: list[imagehash.ImageHash], threshold: int) -> bool:
    return any((new_hash - h) <= threshold for h in accepted)
