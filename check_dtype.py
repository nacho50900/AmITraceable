import time

from app.vision.scene_analysis import _lazy_load
import app.vision.scene_analysis as sa

print("Cargando el modelo...")
t0 = time.time()
sa._lazy_load()
print(f"Modelo cargado en {time.time() - t0:.1f}s")

dtype = next(sa._model.parameters()).dtype
print(f"\n=== DTYPE REAL DEL MODELO: {dtype} ===")

if dtype in (__import__("torch").bfloat16, __import__("torch").float16):
    print(
        "\nConfirmado: el modelo está en un dtype de 16 bits sin aceleración "
        "por hardware en tu CPU. Esto es casi con toda seguridad la causa de "
        "la lentitud extrema. NO se ha lanzado ninguna inferencia todavía."
    )
else:
    print(
        "\nEl modelo YA está en float32 -- la hipótesis del dtype no explica "
        "la lentitud. Hay que seguir investigando por otro lado."
    )
