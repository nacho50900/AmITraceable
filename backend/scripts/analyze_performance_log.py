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
    python scripts/analyze_performance_log.py --plot   # además guarda PNGs

No requiere ninguna credencial ni acceso a red: solo lee el log local.

BUG REAL corregido aquí (confirmado en producción, no una precaución
teórica -- ver mensaje de Nacho del 21/8): la versión anterior de este
script comparaba configuraciones usando la media de `avg_seconds_per_photo`,
que mide LATENCIA por foto (desde que arranca su intento hasta que
termina), no rendimiento agregado. Con concurrencia baja y una etapa mucho
más lenta que la otra (p. ej. Moondream2 en CPU frente a DINOv2 recién
descargado a iGPU tras ADR-28), la latencia de las fotos que quedan detrás
en la cola del semáforo crece de forma aproximadamente lineal con su
posición -- así que la media de latencias puede salir varias veces mayor
que el tiempo real que cuesta el análisis completo. Este script ahora usa
`throughput_seconds_per_photo` (`total_wall_seconds / total_photos` de
CADA análisis) como métrica principal -- ver el comentario de ese campo en
app/log/performance_log.py para el detalle completo.

SEPARACIÓN EN DOS SECCIONES (antiguos / con soporte de iGPU en el log):
comparar directamente análisis de antes y después de ADR-28/ADR-29 mezcla
dos versiones distintas del pipeline (antes de ADR-29 los dos modelos
compartían un único semáforo, no dos independientes -- ver docstring de
`_process_photo` en app/vision/geolocation.py) como si fueran la misma
variable, lo cual no es una comparación válida aunque las fotos analizadas
sean las mismas. La señal que se usa para separar NO es una fecha
adivinada, sino un hecho verificable en los propios datos: el campo
`igpu_offload_used` no existe en absoluto en ninguna línea grabada con
versiones del código anteriores al commit que lo introdujo (`de628de`,
2026-08-21) -- antes de eso, la lógica para saber si el offload se había
usado de verdad en cada análisis (worker vivo/caído, ver comentario de ese
campo en performance_log.py) ni siquiera existía. Una línea del log SIN
esa clave es, por definición, de antes de que este tipo de medición fuera
posible; con ella, es comparable de forma consistente independientemente
de si ese análisis en concreto usó offload o no.
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

    # `era`: separa las líneas del log en dos grupos NO comparables entre
    # sí -- ver el docstring del módulo para el porqué. `notna()` sobre
    # `igpu_offload_used` distingue "la clave existía en esa línea del
    # JSON" (True/False, nunca null -- log_photo_analysis_run() siempre
    # pasa un bool) de "la clave no existía porque esa línea es de antes
    # de que este campo se introdujera" (pandas la rellena como NaN al
    # construir el DataFrame a partir de dicts con distintas claves).
    if "igpu_offload_used" in df.columns:
        df["era"] = df["igpu_offload_used"].notna().map(
            {True: "con soporte de iGPU en el log", False: "antiguo"}
        )
    else:
        df["era"] = "antiguo"

    # Logs de antes de que existiera este campo no lo tienen -- se
    # recalcula a partir de total_wall_seconds/total_photos (ambos
    # presentes desde el origen de este log), así que los análisis
    # antiguos siguen siendo comparables ENTRE SÍ (dentro de su propia
    # sección) sin tener que volver a ejecutarlos.
    if "throughput_seconds_per_photo" not in df.columns:
        df["throughput_seconds_per_photo"] = df["total_wall_seconds"] / df["total_photos"]
    else:
        missing = df["throughput_seconds_per_photo"].isna()
        if missing.any():
            df.loc[missing, "throughput_seconds_per_photo"] = (
                df.loc[missing, "total_wall_seconds"] / df.loc[missing, "total_photos"]
            )
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por la configuración usada (núcleos, concurrencia, si
    Moondream2 estaba activo, si el worker de iGPU se llegó a usar de
    verdad -- ver igpu_offload_used en performance_log.py, no es lo mismo
    que ENABLE_IGPU_OFFLOAD=true en la config: un análisis con el flag
    activado pero el worker caído sale con igpu_offload_used=False, y por
    tanto en el mismo grupo que los análisis sin offload) y calcula el
    rendimiento real (throughput) y el nº de análisis que aportan cada
    combinación -- para distinguir una media fiable (muchas
    observaciones) de una anecdótica (una sola).

    `throughput_seg_por_foto` (media de `throughput_seconds_per_photo` por
    análisis) es la cifra que manda para comparar configuraciones DENTRO
    de una misma sección/era (ver `load_log` y el docstring del módulo --
    no tiene sentido comparar esta cifra entre la sección "antiguo" y la
    de "con soporte de iGPU en el log", son pipelines distintos). La
    latencia media (`latencia_media_seg_por_foto`) NO sirve para comparar
    configuraciones cuando la concurrencia es baja -- ver docstring del
    módulo.

    Desde ADR-29 (pipeline entre fotos con semáforos independientes por
    modelo) también se desglosa el tiempo medio de CADA modelo por
    separado (avg_dinov2_seconds_per_photo / avg_scene_seconds_per_photo)
    -- imprescindible para diagnosticar SI el offload a iGPU (ADR-28)
    ayuda: con concurrencia baja, `throughput_seg_por_foto` tiende a
    converger con el mayor de los dos (el cuello de botella), y el
    desglose es lo que permite ver CUÁL de los dos es."""
    df = df.copy()
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
        throughput_seg_por_foto=("throughput_seconds_per_photo", "mean"),
        throughput_mediana_seg_por_foto=("throughput_seconds_per_photo", "median"),
        latencia_media_seg_por_foto=("avg_seconds_per_photo", "mean"),
        media_dinov2_seg=("avg_dinov2_seconds_per_photo", "mean"),
        media_moondream_seg=("avg_scene_seconds_per_photo", "mean"),
    )
    return grouped.sort_values("throughput_seg_por_foto")


def _print_section(df_section: pd.DataFrame, titulo: str) -> pd.DataFrame | None:
    """Imprime el resumen de una sección (antiguo / con soporte de iGPU) y
    devuelve la tabla agregada, o None si la sección no tiene análisis --
    para que el llamador sepa si tiene sentido dibujar un gráfico de esa
    sección."""
    print(f"\n{'=' * len(titulo)}\n{titulo}\n{'=' * len(titulo)}\n")
    if df_section.empty:
        print("(sin análisis en esta sección todavía)")
        return None

    print(f"{len(df_section)} análisis, {df_section['total_photos'].sum()} fotos en total.\n")

    summary = summarize(df_section)
    print("Resumen por configuración (ordenado de más a menos rápido, por throughput real):\n")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    desglose = ""
    if pd.notna(best.media_dinov2_seg) or pd.notna(best.media_moondream_seg):
        dinov2_txt = f"{best.media_dinov2_seg:.2f}s" if pd.notna(best.media_dinov2_seg) else "sin dato"
        moondream_txt = f"{best.media_moondream_seg:.2f}s" if pd.notna(best.media_moondream_seg) else "sin dato"
        desglose = f" (DINOv2: {dinov2_txt}/foto, Moondream2: {moondream_txt}/foto)"
    print(
        f"\nMás rápida en esta sección: concurrencia={int(best.configured_concurrency)} "
        f"({int(best.threads_per_inference)} hilos/inferencia) en máquinas de "
        f"{int(best.cpu_count)} núcleos, offload iGPU={'sí' if best.igpu_offload_used else 'no'} "
        f"-> {best.throughput_seg_por_foto:.2f}s/foto de media ({int(best.analisis)} análisis){desglose}."
    )
    if (summary["analisis"] < 3).any():
        print(
            "Aviso: alguna combinación tiene menos de 3 análisis -- la media "
            "todavía puede no ser representativa, conviene acumular más antes "
            "de sacar conclusiones firmes para la memoria."
        )
    return summary


def _plot_section(summary: pd.DataFrame, suffix: str, etiqueta: str) -> None:
    """Guarda los dos PNG de una sección con un sufijo distinto en el
    nombre de fichero para no pisar los de la otra sección -- así se
    pueden comparar visualmente sin mezclarlas en el mismo gráfico (ver
    docstring del módulo sobre por qué no son comparables entre sí)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for cpu, group in summary.groupby("cpu_count"):
        ax.plot(
            group["configured_concurrency"],
            group["throughput_seg_por_foto"],
            marker="o",
            label=f"{cpu} núcleos",
        )
    ax.set_xlabel("Concurrencia (photo_analysis_concurrency)")
    ax.set_ylabel("Segundos por foto (throughput real)")
    ax.set_title(f"Rendimiento del análisis de fotos -- {etiqueta}")
    ax.legend()
    out_path = _LOG_PATH.parent / f"performance_summary_{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado en {out_path}")

    # Desglose DINOv2 vs Moondream2 por combinación offload sí/no -- solo
    # si hay al menos un dato de alguna de las dos etapas (logs de antes
    # de ADR-29 no lo tienen, ver summarize()).
    stage_summary = summary.dropna(subset=["media_dinov2_seg", "media_moondream_seg"], how="all")
    if stage_summary.empty:
        return
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    labels = [
        f"offload={'sí' if row.igpu_offload_used else 'no'}\nconc={int(row.configured_concurrency)}"
        for row in stage_summary.itertuples()
    ]
    x = range(len(stage_summary))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], stage_summary["media_dinov2_seg"], width, label="DINOv2")
    ax2.bar([i + width / 2 for i in x], stage_summary["media_moondream_seg"], width, label="Moondream2")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Segundos por foto (media)")
    ax2.set_title(f"Tiempo por modelo, según offload de DINOv2 a iGPU -- {etiqueta}")
    ax2.legend()
    stage_out_path = _LOG_PATH.parent / f"performance_summary_por_etapa_{suffix}.png"
    fig2.savefig(stage_out_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Gráfico por etapa guardado en {stage_out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Guardar también gráficos PNG")
    args = parser.parse_args()

    df = load_log()
    print(f"{len(df)} análisis registrados en total, {df['total_photos'].sum()} fotos en total.")
    print(
        "Separados en dos secciones NO comparables entre sí (pipeline distinto -- "
        "ver docstring del módulo): 'antiguo' es de antes de que el log supiera "
        "registrar si se usó offload a iGPU; el resto sí es comparable entre "
        "configuraciones con y sin offload."
    )

    old_df = df[df["era"] == "antiguo"]
    modern_df = df[df["era"] == "con soporte de iGPU en el log"]

    old_summary = _print_section(old_df, "Análisis antiguos (pipeline pre-iGPU)")
    modern_summary = _print_section(modern_df, "Análisis desde que se configuró el iGPU")

    if args.plot:
        print()
        if old_summary is not None:
            _plot_section(old_summary, "antiguos", "análisis antiguos")
        if modern_summary is not None:
            _plot_section(modern_summary, "con_igpu", "análisis desde iGPU")


if __name__ == "__main__":
    main()
