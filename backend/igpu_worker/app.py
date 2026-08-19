"""
Proceso worker aislado para ejecutar DINOv2 en una GPU DirectML (iGPU
Intel/AMD, vía WSL2), completamente separado del backend principal.

Por qué existe este proceso aparte, en vez de un simple
`import torch_directml` dentro de app/vision/geolocation.py del backend:
torch-directml fija una versión CONCRETA de `torch` (2.4.1 en el momento
de escribir esto, ver requirements.txt de este mismo directorio) como
dependencia -- instalarlo en el MISMO entorno que ya tiene el build CUDA
(`cu121`) que usa Moondream2 (ver backend/requirements-vision.txt) obliga a
pip a reinstalar `torch` con esa otra versión/build, rompiendo el CUDA de
Moondream2 en el proceso. Esto pasó de verdad: la desinstalación del torch
anterior falló a medias con un OSError, dejando el entorno del backend
corrupto y obligando a reconstruir la imagen desde cero.

En vez de reconciliar versiones (bajar el torch de Moondream2 a 2.4.1,
arriesgando toda la maquinaria de dtype casting / device_map ya validada
para la GTX 1650, ver `_upcast_bfloat16_tensors()` en
app/vision/scene_analysis.py del backend), este proceso vive en su propia
imagen Docker con su propio venv: NUNCA comparte site-packages con el
backend. El backend le habla por HTTP (ver `_select_igpu_worker_device_index`
y `_embed_via_igpu_worker` en app/vision/geolocation.py del backend) y, si
este proceso no está arrancado o no responde, el backend cae solo al
comportamiento de siempre (DINOv2 en la misma GPU dedicada que Moondream2)
-- ver ENABLE_IGPU_OFFLOAD en backend/app/config.py.

Solo se levanta con `docker compose --profile igpu up` (ver
docker-compose.yml en la raíz del repo) -- NUNCA con un `docker compose up`
a secas.
"""
import io
import logging

import numpy as np
import torch
import torch_directml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

# Mismo modelo que usa el backend para DINOv2 (ver _MODEL_NAME en
# app/vision/geolocation.py) -- TIENE que coincidir, o los embeddings que
# devuelve este worker no serían comparables contra el índice FAISS que
# construye/consulta el backend con su propia copia del modelo.
_MODEL_NAME = "facebook/dinov2-small"

app = FastAPI(title="AmITraceable - DINOv2 iGPU worker")

# Modelo cargado de forma perezosa, una vez por índice de dispositivo
# pedido -- en la práctica el backend siempre pide el mismo índice (el que
# decidió tras comparar los dispositivos de /devices contra el nombre de
# su GPU dedicada), así que este diccionario normalmente tiene una sola
# entrada, pero soporta varias por si acaso.
_models: dict[int, tuple] = {}


def _load_model(device_index: int):
    if device_index not in _models:
        device = torch_directml.device(device_index)
        logger.info(
            "Cargando DINOv2 (%s) en dispositivo DirectML %d...",
            _MODEL_NAME,
            device_index,
        )
        processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
        model = AutoModel.from_pretrained(_MODEL_NAME).to(device).eval()
        _models[device_index] = (model, processor, device)
    return _models[device_index]


@app.get("/health")
def health():
    """Liveness simple -- NO comprueba que haya un dispositivo DirectML
    utilizable de verdad, eso lo hace /devices (que el backend consulta
    antes de decidir si usar este worker en absoluto)."""
    return {"status": "ok"}


@app.get("/devices")
def devices():
    """Lista los dispositivos DirectML visibles desde ESTE proceso. El
    backend (que conoce el nombre de su propia GPU dedicada vía
    torch.cuda.get_device_name) es quien decide, comparando nombres, si
    alguno de estos dispositivos está de verdad libre -- ver
    `_select_igpu_worker_device_index()` en app/vision/geolocation.py del
    backend: si el único dispositivo DirectML visible resulta ser la
    MISMA GPU dedicada expuesta con otro nombre, usarlo no libera nada y
    el backend lo descarta."""
    try:
        count = torch_directml.device_count()
    except Exception:
        logger.exception("No se pudo enumerar dispositivos DirectML")
        return {"devices": []}
    return {
        "devices": [
            {"index": i, "name": torch_directml.device_name(i)}
            for i in range(count)
        ]
    }


@app.post("/embed")
async def embed(device_index: int, file: UploadFile = File(...)):
    """Calcula el embedding DINOv2 (normalizado L2) de una imagen, en el
    dispositivo DirectML indicado. Réplica exacta del cálculo que hacía
    `estimate_location_from_image` en el backend cuando corría en local --
    tiene que dar el MISMO resultado (mismo modelo, mismo preprocesado,
    misma normalización) para que la búsqueda en el índice FAISS del
    backend siga teniendo sentido."""
    try:
        model, processor, device = _load_model(device_index)
    except Exception:
        logger.exception(
            "Fallo al cargar DINOv2 en el dispositivo DirectML %d", device_index
        )
        raise HTTPException(
            status_code=500, detail="No se pudo cargar el modelo en ese dispositivo"
        )

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    except Exception:
        # Motivo típico: un operador de DINOv2 sin soporte en DirectML
        # todavía (torch-directml va añadiendo cobertura poco a poco, no
        # está completa). El backend interpreta este 500 como "worker no
        # fiable" y cae de forma PERMANENTE al comportamiento de siempre
        # para el resto del proceso -- ver `_igpu_worker_failed` en
        # app/vision/geolocation.py.
        logger.exception(
            "Fallo al calcular el embedding (revisa si es un operador de "
            "DINOv2 sin soporte en DirectML)"
        )
        raise HTTPException(status_code=500, detail="Fallo al calcular el embedding")

    return {"embedding": embedding.tolist()}
