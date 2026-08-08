import time
from PIL import Image
import numpy as np

# Imagen sintética realista (mismo tamaño típico que una foto de Instagram)
np.random.seed(0)
arr = np.random.randint(0, 255, (1080, 1080, 3), dtype=np.uint8)
image = Image.fromarray(arr)

from app.vision.scene_analysis import _lazy_load, _QUERY
import app.vision.scene_analysis as sa

print("Cargando el modelo (primera vez, puede tardar si tiene que descargarlo)...")
t0 = time.time()
sa._lazy_load()
print(f"Modelo cargado en {time.time() - t0:.1f}s")

print("Analizando UNA foto, sin timeout, midiendo tiempo real...")
t0 = time.time()
answer = sa._model.query(image, _QUERY)["answer"]
elapsed = time.time() - t0
print(f"\n=== TERMINO en {elapsed:.1f} segundos ===")
print("Respuesta:", answer)
