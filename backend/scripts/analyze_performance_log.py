"""
Analiza `backend/data/performance/photo_analysis_log.jsonl` (ver
app/log/performance_log.py) para comprobar, con datos reales, si la
concurrencia auto-calculada (`_default_photo_analysis_concurrency` en
app/config.py) es una buena elección en la práctica, si el offload de
DINOv2 a una iGPU ayuda de verdad (ADR-28) y si el pipeline entre fotos
(ADR-29) se refleja en el desglose por modelo -- y para tener cifras
citables en la memoria del TFG (sección "Plan de evaluación pendiente" del
README).

Uso:
    cd backend
    python scripts/analyze_performance_log.py
    python scripts/analyze_performance_log.py --plot   # además guarda dos PNG

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
    observaciones) de una anecdótica (una sola).

    Desde ADR-29 (pipeline entre fotos con semáforos independientes por
    modelo) también se desglosa el tiempo medio de CADA modelo por
    separado (avg_dinov2_seconds_per_photo / avg_scene_seconds_per_photo)
    -- imprescindible para diagnosticar SI el offload a iGPU (ADR-28)
    ayuda: `media_seg_por_foto` por sí sola no distingue "DINOv2 tarda
    más en la iGPU pero el pipeline lo compensa" de "el pipeline no está
    compensando nada"."""
    # Logs de antes de que existieran estos campos (columnas ausentes) se
    # tratan como "sin offload"/"sin dato de etapa" -- es la
    # interpretación correcta, ya que ni el worker de iGPU ni el
    # desglose por etapa existían todavía cuando se generaron.
    if "igpu_offload_used" not in df.columns:
        df["igpu_offload_used"] = False
    df["igpu_offload_used"] = df["igpu_offload_used"].map(lambda v: bool(v) if pd.notna(v) else False)
    for col in ("avg_dinov2_seconds_per_photo", "avg_scene_seconds_per_photo"):
        if col not in df.columns:
            df[col] = pd.NA

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
        media_dinov2_seg=("avg_dinov2_seconds_per_photo", "mean"),
        media_moondream_seg=("avg_scene_seconds_per_photo", "mean"),
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
    desglose = ""
    if pd.notna(best.media_dinov2_seg) or pd.notna(best.media_moondream_seg):
        dinov2_txt = f"{best.media_dinov2_seg:.2f}s" if pd.notna(best.media_dinov2_seg) else "sin dato"
        moondream_txt = f"{best.media_moondream_seg:.2f}s" if pd.notna(best.media_moondream_seg) else "sin dato"
        desglose = f" (DINOv2: {dinov2_txt}/foto, Moondream2: {moondream_txt}/foto)"
    print(
        f"\nMás rápida hasta ahora: concurrencia={int(best.configured_concurrency)} "
        f"({int(best.threads_per_inference)} hilos/inferencia) en máquinas de "
        f"{int(best.cpu_count)} núcleos, offload iGPU={'sí' if best.igpu_offload_used else 'no'} "
        f"-> {best.media_seg_por_foto:.2f}s/foto de media ({int(best.analisis)} análisis){desglose}."
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

        # Desglose DINOv2 vs Moondream2 por combinación offload sí/no --
        # solo si hay al menos un dato de alguna de las dos etapas (logs
        # de antes de ADR-29 no lo tienen, ver summarize()).
        stage_summary = summary.dropna(subset=["media_dinov2_seg", "media_moondream_seg"], how="all")
        if not stage_summary.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            labels = [
                f"offload={'sí' if row.igpu_offload_used else 'no'}\nconc={int(row.configured_concurrency)}"
                for row in stage_summary.itertuples()
            ]
            x = range(len(stage_summary))
            width = 0.35
            ax2.bar(
                [i - width / 2 for i in x], stage_summary["media_dinov2_seg"], width, label="DINOv2"
            )
            ax2.bar(
                [i + width / 2 for i in x], stage_summary["media_moondream_seg"], width, label="Moondream2"
            )
            ax2.set_xticks(list(x))
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_ylabel("Segundos por foto (media)")
            ax2.set_title("Tiempo por modelo, según offload de DINOv2 a iGPU")
            ax2.legend()
            stage_out_path = _LOG_PATH.parent / "performance_summary_por_etapa.png"
            fig2.savefig(stage_out_path, dpi=150, bbox_inches="tight")
            print(f"Gráfico por etapa guardado en {stage_out_path}")


if __name__ == "__main__":
    main()
