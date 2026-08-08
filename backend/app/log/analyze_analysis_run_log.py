"""
Analiza `backend/data/performance/analysis_run_log.jsonl` (ver
app/log/analysis_run_log.py): tiempo total y por ETAPA de cada análisis
completo (huella de escritura, detección de atributos, autodeclaraciones
con IA, espera de geolocalización, estrechamiento de población...).

A diferencia de `analyze_performance_log.py` (que solo mira el tramo de
fotos con el detalle de núcleos/concurrencia), este script da la vista de
conjunto: qué etapa del pipeline pesa más en la práctica, y cómo varía
según el volumen de actividad (nº de posts/comentarios/fotos) analizado.
Sirve como fuente empírica para la sección "Plan de evaluación pendiente"
de la memoria del TFG.

Uso:
    cd backend
    python scripts/analyze_analysis_run_log.py
    python scripts/analyze_analysis_run_log.py --plot   # además guarda un PNG

No requiere ninguna credencial ni acceso a red: solo lee el log local.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

_LOG_PATH = Path(__file__).parent.parent.parent / "data" / "performance" / "analysis_run_log.jsonl"


def load_log(path: Path = _LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path} todavía -- hay que completar al menos un "
            "análisis (POST /api/analyze/{platform} o su variante de "
            "streaming) para generar entradas."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def stage_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Expande la columna `stages_seconds` (un dict por análisis) en una
    fila por (análisis, etapa), para poder agregarlas todas juntas -- no
    todas las filas tienen las mismas etapas (p. ej. sin MISTRAL_API_KEY,
    "autodeclaraciones_ia" no aparece con tiempo significativo, y sin
    Instagram, "espera_geolocalizacion_fotos" no aparece)."""
    records = []
    for _, row in df.iterrows():
        for stage, seconds in row["stages_seconds"].items():
            records.append({"timestamp": row["timestamp"], "platform": row["platform"], "etapa": stage, "segundos": seconds})
    return pd.DataFrame(records)


def summarize_by_stage(stages_df: pd.DataFrame) -> pd.DataFrame:
    grouped = stages_df.groupby("etapa", as_index=False).agg(
        analisis=("segundos", "count"),
        media_segundos=("segundos", "mean"),
        mediana_segundos=("segundos", "median"),
        max_segundos=("segundos", "max"),
    )
    return grouped.sort_values("media_segundos", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Guardar también un gráfico PNG")
    args = parser.parse_args()

    df = load_log()
    print(f"{len(df)} análisis completos registrados.\n")

    print("Totales por plataforma:")
    print(df.groupby("platform")["total_seconds"].agg(["count", "mean", "median"]).to_string())

    stages_df = stage_breakdown(df)
    summary = summarize_by_stage(stages_df)
    print("\nDesglose medio por etapa del pipeline (ordenado de más a menos pesada):\n")
    print(summary.to_string(index=False))

    heaviest = summary.iloc[0]
    print(
        f"\nEtapa que más pesa de media: '{heaviest.etapa}' -> "
        f"{heaviest.media_segundos:.2f}s de media ({int(heaviest.analisis)} análisis)."
    )

    # Correlación simple con el volumen: ¿el tiempo total crece con el nº
    # de posts/fotos analizados? Útil para justificar en la memoria si el
    # pipeline escala razonablemente o si alguna etapa concreta se dispara
    # de forma desproporcionada con el volumen.
    for col in ("n_posts", "n_comments", "n_photos"):
        if df[col].nunique() > 1:
            corr = df[col].corr(df["total_seconds"])
            print(f"Correlación entre {col} y tiempo total: {corr:.2f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(summary["etapa"], summary["media_segundos"])
        ax.set_xlabel("Segundos (media)")
        ax.set_title("Tiempo medio por etapa del pipeline de análisis")
        fig.tight_layout()
        out_path = _LOG_PATH.parent / "analysis_run_summary.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nGráfico guardado en {out_path}")


if __name__ == "__main__":
    main()
