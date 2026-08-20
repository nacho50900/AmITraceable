"""
Analiza `backend/data/performance/photo_analysis_log.jsonl` (ver
app/performance_log.py) para comprobar, con datos reales, si la
concurrencia auto-calculada (`_default_photo_analysis_concurrency` en
app/config.py) es una buena elección en la práctica, y para tener cifras
citables en la memoria del TFG (sección "Plan de evaluación pendiente" del
README).

Uso:
    cd backend
    python scripts/analyze_performance_log.py
    python scripts/analyze_performance_log.py --plot   # además guarda un PNG

No requiere ninguna credencial ni acceso a red: solo lee el log local.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

_LOG_PATH = Path(__file__).parent.parent / "data" / "performance" / "photo_analysis_log.jsonl"


def load_log(path: Path = _LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path} todavía -- hay que ejecutar al menos un "
            "análisis con fotos (Instagram, con el índice de geolocalización "
            "construido) para generar entradas."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por la configuración usada (núcleos, concurrencia, si
    Moondream2 estaba activo, si el worker de iGPU se llegó a usar de
    verdad -- ver igpu_offload_used en performance_log.py, no es lo mismo
    que ENABLE_IGPU_OFFLOAD=true en la config: un análisis con el flag
    activado pero el worker caído sale con igpu_offload_used=False, y por
    tanto en el mismo grupo que los análisis sin offload) y calcula el
    tiempo medio por foto y el nº de análisis que aportan cada
    combinación -- para distinguir una media fiable (muchas
    observaciones) de una anecdótica (una sola)."""
    # Logs de antes de que existiera este campo (columna ausente) se
    # tratan como "sin offload" -- es la interpretación correcta, ya que
    # el worker de iGPU no existía todavía cuando se generaron.
    if "igpu_offload_used" not in df.columns:
        df["igpu_offload_used"] = False
    df["igpu_offload_used"] = df["igpu_offload_used"].fillna(False)

    grouped = df.groupby(
        [
            "cpu_count",
            "configured_concurrency",
            "threads_per_inference",
            "enable_scene_analysis",
            "igpu_offload_used",
        ],
        as_index=False,
    ).agg(
        analisis=("total_photos", "count"),
        fotos_totales=("total_photos", "sum"),
        media_seg_por_foto=("avg_seconds_per_photo", "mean"),
        mediana_seg_por_foto=("avg_seconds_per_photo", "median"),
    )
    return grouped.sort_values("media_seg_por_foto")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Guardar también un gráfico PNG")
    args = parser.parse_args()

    df = load_log()
    print(f"{len(df)} análisis registrados, {df['total_photos'].sum()} fotos en total.\n")

    summary = summarize(df)
    print("Resumen por configuración (ordenado de más a menos rápido):\n")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(
        f"\nMás rápida hasta ahora: concurrencia={int(best.configured_concurrency)} "
        f"({int(best.threads_per_inference)} hilos/inferencia) en máquinas de "
        f"{int(best.cpu_count)} núcleos, offload iGPU={'sí' if best.igpu_offload_used else 'no'} "
        f"-> {best.media_seg_por_foto:.2f}s/foto de media ({int(best.analisis)} análisis)."
    )
    if (summary["analisis"] < 3).any():
        print(
            "Aviso: alguna combinación tiene menos de 3 análisis -- la media "
            "todavía puede no ser representativa, conviene acumular más antes "
            "de sacar conclusiones firmes para la memoria."
        )

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for cpu, group in summary.groupby("cpu_count"):
            ax.plot(group["configured_concurrency"], group["media_seg_por_foto"], marker="o", label=f"{cpu} núcleos")
        ax.set_xlabel("Concurrencia (photo_analysis_concurrency)")
        ax.set_ylabel("Segundos por foto (media)")
        ax.set_title("Rendimiento del análisis de fotos por configuración")
        ax.legend()
        out_path = _LOG_PATH.parent / "performance_summary.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nGráfico guardado en {out_path}")


if __name__ == "__main__":
    main()
