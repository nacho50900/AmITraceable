import time
import torch
from PIL import Image
import numpy as np

from app.vision.scene_analysis import _lazy_load, _QUERY
import app.vision.scene_analysis as sa

print("Cargando el modelo...")
t0 = time.time()
sa._lazy_load()
print(f"Modelo cargado en {time.time() - t0:.1f}s, dtype original: {next(sa._model.parameters()).dtype}")

print("\nConvirtiendo a float32, parametro a parametro (sin .to() sobre el modulo)...")
t0 = time.time()
with torch.no_grad():
    for param in sa._model.parameters():
        param.data = param.data.float()
    for buf in sa._model.buffers():
        if buf.dtype in (torch.bfloat16, torch.float16):
            buf.data = buf.data.float()
print(f"Conversion hecha en {time.time() - t0:.1f}s")
print(f"Dtype tras conversion: {next(sa._model.parameters()).dtype}")

np.random.seed(0)
arr = np.random.randint(0, 255, (1080, 1080, 3), dtype=np.uint8)
image = Image.fromarray(arr)

print("\nAnalizando UNA foto en float32, midiendo tiempo real...")
t0 = time.time()
answer = sa._model.query(image, _QUERY)["answer"]
elapsed = time.time() - t0
print(f"\n=== TERMINO en {elapsed:.1f} segundos ===")
print("Respuesta:", answer)
