r"""
Convierte los modelos Helsinki-NLP/opus-mt-{es-en,en-es} (checkpoints
originales de HuggingFace) a formato CTranslate2 cuantizado en int8, para
que app/nlp/translation.py pueda traducir descripciones de fotos
localmente sin depender de Mistral (ver ADR-31).

Se ejecuta UNA SOLA VEZ (o cada vez que se quiera reconvertir/actualizar
los modelos) -- no en cada arranque del backend. A diferencia del propio
uso en producción de app/nlp/translation.py (que solo necesita
`ctranslate2`+`sentencepiece`, ligero), ESTE script sí necesita
`transformers`+`torch` instalados (ya los tienes si construiste la imagen
del backend con WITH_GEOLOCATION=true) -- son las dependencias del
CONVERSOR, no del runtime de traducción.

Uso:
    pip install ctranslate2 transformers torch sentencepiece sacremoses huggingface_hub
    python scripts/convert_translation_models.py

Cada dirección descarga el checkpoint original de HuggingFace (solo en
formato `safetensors`, ~300MB) antes de convertirlo -- si tu caché de
HuggingFace (normalmente `C:\Users\<tú>\.cache\huggingface` en Windows,
`~/.cache/huggingface` en Linux/macOS) vive en una unidad con poco
espacio libre, redirígela a otra con más hueco antes de ejecutar esto:
    # PowerShell:
    $env:HF_HOME = "D:\ruta\con\espacio\hf_cache"
    # bash:
    export HF_HOME=/ruta/con/espacio/hf_cache

Salida:
    ../data/translation_models/es-en/  (model.bin, source.spm, target.spm)
    ../data/translation_models/en-es/  (model.bin, source.spm, target.spm)

(el `model.bin` de la salida es el modelo YA CONVERTIDO a formato
CTranslate2 -- no tiene relación con el `pytorch_model.bin` del
checkpoint original de HuggingFace, que este script evita descargar a
propósito, ver `convert()` más abajo.)

Con esto en su sitio, app/nlp/translation.py detecta los modelos
automáticamente en la siguiente petición de traducción -- no hace falta
reiniciar el backend, la carga es perezosa.
"""
import shutil
from pathlib import Path

from ctranslate2.converters import TransformersConverter

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "translation_models"

# (nombre del checkpoint en HuggingFace, carpeta de salida bajo _OUTPUT_DIR)
_MODELS = [
    ("Helsinki-NLP/opus-mt-es-en", "es-en"),
    ("Helsinki-NLP/opus-mt-en-es", "en-es"),
]


def convert(checkpoint: str, direction: str, force: bool) -> None:
    output_dir = _OUTPUT_DIR / direction
    if output_dir.exists():
        if not force:
            print(f"[{direction}] ya existe en {output_dir} -- omitido (usa --force para reconvertir).")
            return
        shutil.rmtree(output_dir)

    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"[{direction}] descargando {checkpoint} (solo safetensors)...")
    # Sin esto, `TransformersConverter` de más abajo (vía
    # `transformers.from_pretrained`) puede acabar descargando el
    # checkpoint DOS VECES, en dos formatos distintos
    # (`pytorch_model.bin` Y `model.safetensors`, ~300MB cada uno) --
    # visto en la práctica, no es solo una posibilidad teórica. Al
    # pre-descargar aquí explícitamente SOLO `*.safetensors` (con los
    # formatos alternativos vetados vía `ignore_patterns`), esos
    # ficheros ya quedan en la caché local de HuggingFace; cuando el
    # conversor llame después a `from_pretrained`, los encuentra ya
    # descargados y nunca llega a pedir `pytorch_model.bin` -- reduce a
    # la mitad el pico de disco necesario durante la conversión, crítico
    # en máquinas con poco espacio libre.
    snapshot_download(
        repo_id=checkpoint,
        allow_patterns=["*.safetensors", "*.json", "*.spm", "*.txt", "*.model"],
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot"],
    )

    print(f"[{direction}] convirtiendo a CTranslate2 (int8)...")
    converter = TransformersConverter(checkpoint)
    converter.convert(str(output_dir), quantization="int8")

    # TransformersConverter no copia los ficheros SentencePiece del
    # checkpoint original (solo convierte los pesos del modelo) -- hacen
    # falta para tokenizar/detokenizar en app/nlp/translation.py. Ya
    # están en la caché local gracias al snapshot_download de arriba
    # (van incluidos en `allow_patterns`), así que esto no vuelve a
    # tocar la red, solo copia desde disco.
    for spm_file, dest_name in (("source.spm", "source.spm"), ("target.spm", "target.spm")):
        local_path = hf_hub_download(repo_id=checkpoint, filename=spm_file)
        shutil.copy(local_path, output_dir / dest_name)

    print(f"[{direction}] listo en {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Reconvertir aunque ya exista una versión en disco."
    )
    args = parser.parse_args()

    for checkpoint, direction in _MODELS:
        convert(checkpoint, direction, force=args.force)
