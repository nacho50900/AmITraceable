"""
Descarga SOLO las imágenes de España de OpenStreetView-5M (OSV-5M), usando
el modo streaming de la librería `datasets` de Hugging Face para no tener
que bajar el dataset completo (~5.1M imágenes mundiales) antes de filtrar.

Uso:
    pip install datasets huggingface_hub pillow tqdm
    python download_osv5m_spain.py --output ../data/osv5m_spain --max-images 20000

Salida:
    ../data/osv5m_spain/images/<id>.jpg   (una imagen por fila filtrada)
    ../data/osv5m_spain/metadata.csv      (id, lat, lon, city, region si están disponibles)

Nota importante: esto se ejecuta EN TU MÁQUINA, no en un entorno de
Anthropic -- requiere acceso a internet a huggingface.co y puede tardar
un buen rato dependiendo de tu ancho de banda (estimación: 3-6 GB de
descarga real para España, ver conversación).

Si el campo de país en el dataset no se llama exactamente "country" o usa
otro código (p.ej. ISO "ES" en vez de "Spain"), ajusta `_is_spain()` más
abajo tras inspeccionar una fila de ejemplo (el script imprime la primera
fila cruda al arrancar para que puedas verificarlo).
"""
import argparse
import csv
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _is_spain(example: dict) -> bool:
    """Ajusta esto si el nombre/formato del campo de país difiere.
    OSV-5M suele traer 'country' como código ISO-2 (ej. 'ES') en sus
    metadatos de columnas; imprime la primera fila para confirmarlo."""
    country = str(example.get("country", "")).strip().upper()
    return country in ("ES", "ESP", "SPAIN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../data/osv5m_spain")
    parser.add_argument("--max-images", type=int, default=20_000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Abriendo OSV-5M en modo streaming (no descarga el dataset completo)...")
    dataset = load_dataset("osv5m/osv5m", split=args.split, streaming=True)

    # Inspección de la primera fila para verificar nombres de campos reales
    first_row = next(iter(dataset))
    print("Ejemplo de fila cruda (verifica los nombres de campo aquí):")
    print({k: v for k, v in first_row.items() if k != "image"})

    metadata_path = output_dir / "metadata.csv"
    saved = 0

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lat", "lon", "city", "region", "country"])

        progress = tqdm(total=args.max_images, desc="Imágenes de España guardadas")
        for i, example in enumerate(dataset):
            if saved >= args.max_images:
                break
            if not _is_spain(example):
                continue

            image = example["image"]  # objeto PIL.Image
            image_id = example.get("id", f"osv5m_{i}")
            lat = example.get("latitude")
            lon = example.get("longitude")
            city = example.get("city", "")
            region = example.get("region", "")

            image_path = images_dir / f"{image_id}.jpg"
            image.convert("RGB").save(image_path, "JPEG", quality=90)

            writer.writerow([image_id, lat, lon, city, region, "ES"])
            saved += 1
            progress.update(1)

        progress.close()

    print(f"\nListo. {saved} imágenes de España guardadas en {images_dir}")
    print(f"Metadatos en {metadata_path}")
    if saved == 0:
        print(
            "\n⚠️  No se guardó ninguna imagen. Revisa la fila de ejemplo impresa "
            "arriba: probablemente el nombre del campo de país o su valor no "
            "coincide con lo esperado en _is_spain(). Ajusta esa función."
        )


if __name__ == "__main__":
    main()
