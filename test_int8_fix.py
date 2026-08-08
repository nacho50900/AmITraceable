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

print("\nAplicando cuantizacion dinamica int8 a las capas Linear (inplace=True, sin duplicar el modelo)...")
t0 = time.time()
try:
    torch.quantization.quantize_dynamic(
        sa._model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    quantized_model = sa._model
    print(f"Cuantizacion hecha en {time.time() - t0:.1f}s")
except Exception as e:
    print(f"\n=== FALLO la cuantizacion: {type(e).__name__}: {e} ===")
    raise SystemExit(1)

print("\nConvirtiendo a float32 lo que NO se cuantizo (capas no-Linear: conv, layernorm, embeddings)...")
t0 = time.time()
converted = 0
with torch.no_grad():
    for name, param in quantized_model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.float()
            converted += 1
    for name, buf in quantized_model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.float()
            converted += 1
print(f"{converted} tensores convertidos en {time.time() - t0:.1f}s")

print("\nParcheando prepare_crops() para que devuelva float32 en vez de bfloat16 fijo...")
import sys
vision_mod = None
moondream_mod = None
for _name, _mod in list(sys.modules.items()):
    _f = getattr(_mod, "__file__", "") or ""
    if _f.endswith("vision.py") and "moondream2" in _f:
        vision_mod = _mod
    if _f.endswith("moondream.py") and "moondream2" in _f:
        moondream_mod = _mod

if vision_mod is None:
    print("=== No encontre vision.py en sys.modules, no se puede parchear ===")
    raise SystemExit(1)

_original_prepare_crops = vision_mod.prepare_crops

def _prepare_crops_fp32(image, config, device):
    all_crops, tiling = _original_prepare_crops(image, config, device)
    return all_crops.float(), tiling

vision_mod.prepare_crops = _prepare_crops_fp32
if moondream_mod is not None and hasattr(moondream_mod, "prepare_crops"):
    moondream_mod.prepare_crops = _prepare_crops_fp32
print(f"Parcheado en: vision_mod={vision_mod.__name__}" + (f", moondream_mod={moondream_mod.__name__}" if moondream_mod else ""))

print("\nPlan B por si el anterior no basta: forzar float32 directamente en la entrada de patch_emb...")
patch_emb = None
patch_emb_name = None
for _mod_name, _mod in quantized_model.named_modules():
    if _mod_name.endswith("patch_emb"):
        patch_emb = _mod
        patch_emb_name = _mod_name
        break

if patch_emb is None:
    print("=== No encontre ningun submodulo llamado 'patch_emb' ===")
    raise SystemExit(1)

print(f"patch_emb encontrado en: {patch_emb_name}")
_original_patch_emb_forward = patch_emb.forward

def _patched_patch_emb_forward(x):
    if x.dtype != torch.float32:
        x = x.float()
    return _original_patch_emb_forward(x)

patch_emb.forward = _patched_patch_emb_forward
print("patch_emb.forward parcheado directamente.")

print("\nPlan C: el bias de las capas ya cuantizadas se queda en bfloat16 (quantize_dynamic")
print("solo toca el peso, no el bias) -- arreglando el bias en TODAS las capas cuantizadas...")
import torch.ao.nn.quantized.dynamic as nnqd
fixed_bias_count = 0
checked_count = 0
for _mod_name, _mod in quantized_model.named_modules():
    if isinstance(_mod, nnqd.Linear):
        checked_count += 1
        try:
            b = _mod.bias()
        except Exception as e:
            print(f"  aviso: no pude leer el bias de {_mod_name}: {e}")
            continue
        if b is not None and b.dtype == torch.bfloat16:
            w = _mod.weight()
            try:
                _mod.set_weight_bias(w, b.float())
                fixed_bias_count += 1
            except Exception as e:
                print(f"  aviso: no pude corregir el bias de {_mod_name}: {type(e).__name__}: {e}")
print(f"{checked_count} capas Linear cuantizadas revisadas, {fixed_bias_count} bias corregidos a float32")

np.random.seed(0)
arr = np.random.randint(0, 255, (1080, 1080, 3), dtype=np.uint8)
image = Image.fromarray(arr)

print("\nAnalizando UNA foto cuantizada, midiendo tiempo real...")
t0 = time.time()
answer = quantized_model.query(image, _QUERY)["answer"]
elapsed = time.time() - t0
print(f"\n=== TERMINO en {elapsed:.1f} segundos ===")
print("Respuesta:", answer)
