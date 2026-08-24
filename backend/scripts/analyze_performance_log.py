"""
Analiza `backend/data/performance/photo_analysis_log.jsonl` (análisis de
fotos: DINOv2 + Moondream2, ver app/log/performance_log.py) Y
`backend/data/performance/translation_log.jsonl` (traducción local, ver
app/log/translation_log.py y ADR-30/ADR-31) EN UN SOLO SITIO -- pensado
para comprobar, con datos reales, si la concurrencia auto-calculada
(`_default_photo_analysis_concurrency` en app/config.py) es una buena
elección en la práctica, si el offload de DINOv2 a una iGPU ayuda de
verdad (ADR-28), y cuánto de todo el tiempo de CPU del pipeline completo
(fotos + traducción) se va en cada cosa -- y para tener cifras citables en
la memoria del TFG (sección "Plan de evaluación pendiente" del README).

Uso:
    cd backend
    python scripts/analyze_performance_log.py
    python scripts/analyze_performance_log.py --plot   # además guarda PNGs

No requiere ninguna credencial ni acceso a red: solo lee los logs locales.
El log de traducción es OPCIONAL -- si `translation_log.jsonl` no existe
todavía, se avisa y se sigue solo con el análisis de fotos, sin fallar.

FUSIONADO EN UN SOLO SCRIPT A PROPÓSITO (antes había un
`analyze_translation_log.py` hermano, aparte): un solo comando para ver
las dos cosas a la vez, y sobre todo para poder sacar el RESUMEN
COMBINADO DE CPU de más abajo (`_print_combined_cpu_summary`) -- el
análisis de fotos y la traducción son operaciones independientes, no
atadas la una a la otra (ver docstring de app/log/translation_log.py: no
hay ningún identificador que una una entrada de un log con una del otro),
así que NO se puede calcular un "% de CPU de ESTE análisis en concreto"
combinando los dos; lo que sí se puede, y es lo que se hace aquí, es sumar
el total de segundos de CPU de cada log por separado y comparar esos dos
totales agregados -- una foto del reparto de trabajo de CPU en conjunto,
no de una sesión concreta.

BUG REAL corregido aquí (confirmado en producción, no una precaución
teórica -- ver conversación del 21/8): la versión anterior de este script
comparaba configuraciones usando la media de `avg_seconds_per_photo`, que
mide LATENCIA por foto (desde que arranca su intento hasta que termina),
no rendimiento agregado. Con concurrencia baja y una etapa mucho más lenta
que la otra, la latencia de las fotos que quedan detrás en la cola del
semáforo crece de forma aproximadamente lineal con su posición -- así que
la media de latencias puede salir varias veces mayor que el tiempo real
que cuesta el análisis completo. Este script usa
`throughput_seconds_per_photo` (`total_wall_seconds / total_photos` de
CADA análisis) como métrica principal -- ver el comentario de ese campo en
app/log/performance_log.py para el detalle completo.

SEPARACIÓN EN DOS SECCIONES DE FOTOS (antiguos / con soporte de iGPU en el
log): comparar directamente análisis de antes y después de ADR-28/ADR-29
mezcla dos versiones distintas del pipeline (antes de ADR-29 los dos
modelos compartían un único semáforo, no dos independientes) como si
fueran la misma variable, lo cual no es una comparación válida aunque las
fotos analizadas sean las mismas. La señal que se usa para separar NO es
una fecha adivinada, sino un hecho verificable en los propios datos: el
campo `igpu_offload_used` no existe en absoluto en ninguna línea grabada
con versiones del código anteriores al commit que lo introdujo (`de628de`,
2026-08-21). Una línea del log SIN esa clave es, por definición, de antes
de que este tipo de medición fuera posible.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

_LOG_DIR = Path(__file__).parent.parent / "data" / "performance"
_LOG_PATH = _LOG_DIR / "photo_analysis_log.jsonl"
_TRANSLATION_LOG_PATH = _LOG_DIR / "translation_log.jsonl"

# Columnas que no existían en logs antiguos -- se rellenan con este valor
# (no NaN puro) al agregar, para que .mean() no las descarte en silencio
# de forma distinta a como se documenta cada una. Ver load_log().
_NUMERIC_COLS_DEFAULT_NA = (
    "cuda_gpu_seconds",
    "igpu_seconds",
    "cpu_seconds",
    "cuda_gpu_usage_pct",
    "igpu_usage_pct",
    "cpu_usage_pct",
    "avg_dinov2_seconds_per_photo",
    "avg_scene_seconds_per_photo",
)


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

    # cuda_gpu_seconds/igpu_seconds/cpu_seconds/*_pct son más nuevos
    # todavía (añadidos junto con el log de traducción, ver ADR-30/ADR-31
    # y performance_log.py) -- ausentes en logs anteriores a ese cambio,
    # aunque ya tuvieran igpu_offload_used. NaN aquí simplemente
    # significa "no se midió en su momento", no "cero" -- no se
    # recalcula porque, a diferencia del throughput, no hay forma
    # fiable de derivarlo de otros campos si el propio log no separaba
    # ya avg_dinov2/avg_scene por entonces, y sobre todo porque requiere
    # saber el dispositivo REAL de cada modelo (`dinov2_local_device`/
    # `moondream_device`), que tampoco existía todavía.
    for col in _NUMERIC_COLS_DEFAULT_NA:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def load_translation_log(path: Path = _TRANSLATION_LOG_PATH) -> pd.DataFrame | None:
    """Igual que `load_log()` pero para el log de traducción -- `None` si
    todavía no existe (log opcional, ver docstring del módulo), en vez de
    lanzar `FileNotFoundError` como `load_log()`: el análisis de fotos es
    el log principal de este script, pero no tiene sentido exigir que
    también haya traducciones registradas para poder ejecutar nada."""
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
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
    rendimiento real (throughput), el uso de GPU/CPU y el nº de análisis
    que aportan cada combinación -- para distinguir una media fiable
    (muchas observaciones) de una anecdótica (una sola).

    `throughput_seg_por_foto` (media de `throughput_seconds_per_photo` por
    análisis) es la cifra que manda para comparar configuraciones DENTRO
    de una misma sección/era (ver `load_log` y el docstring del módulo --
    no tiene sentido comparar esta cifra entre la sección "antiguo" y la
    de "con soporte de iGPU en el log", son pipelines distintos). La
    latencia media (`latencia_media_seg_por_foto`) NO sirve para comparar
    configuraciones cuando la concurrencia es baja -- ver docstring del
    módulo.

    `cuda_gpu_pct_medio`/`igpu_pct_medio`/`cpu_pct_medio` (media de
    `cuda_gpu_usage_pct`/`igpu_usage_pct`/`cpu_usage_pct`, ver
    performance_log.py): segundos de cómputo de modelo en cada
    dispositivo por cada segundo de reloj -- pueden sumar más de 100% si
    el pipeline entre fotos (ADR-29) solapa bien un modelo con otro; NO
    son un porcentaje de utilización medido por el sistema operativo
    (tipo `nvidia-smi`), sino derivado de los tiempos que el propio
    código mide por etapa. TRES dispositivos, no dos -- ver el docstring
    de `log_photo_analysis_run()` en performance_log.py: sin offload a
    iGPU, DINOv2 comparte la MISMA GPU dedicada (CUDA) que Moondream2,
    no corre en CPU.

    Desde ADR-29 (pipeline entre fotos con semáforos independientes por
    modelo) también se desglosa el tiempo medio de CADA modelo por
    separado (avg_dinov2_seconds_per_photo / avg_scene_seconds_per_photo)
    -- imprescindible para diagnosticar SI el offload a iGPU (ADR-28)
    ayuda: con concurrencia baja, `throughput_seg_por_foto` tiende a
    converger con el mayor de los dos (el cuello de botella), y el
    desglose es lo que permite ver CUÁL de los dos es."""
    df = df.copy()
    if "igpu_offload_used" not in df.columns:
        df["igpu_offload_used"] = False
    df["igpu_offload_used"] = df["igpu_offload_used"].map(lambda v: bool(v) if pd.notna(v) else False)
    for col in _NUMERIC_COLS_DEFAULT_NA:
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
        gpu_pct_medio=("cuda_gpu_usage_pct", "mean"),
        igpu_pct_medio=("igpu_usage_pct", "mean"),
        cpu_pct_medio=("cpu_usage_pct", "mean"),
    )
    return grouped.sort_values("throughput_seg_por_foto")


def summarize_translation(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa el log de traducción por dirección (es-en/en-es) y si el
    modelo estaba realmente disponible (`translation_available` -- ver
    app/log/translation_log.py: si es False, `total_seconds` mide solo la
    comprobación de disco y la ruta de degradación, no una traducción
    real, así que agregarlo junto con traducciones reales falsearía la
    media)."""
    grouped = df.groupby(
        ["direction", "device", "translation_available"],
        as_index=False,
    ).agg(
        llamadas=("num_texts", "count"),
        textos_totales=("num_texts", "sum"),
        seg_totales=("total_seconds", "sum"),
        media_seg_por_texto=("avg_seconds_per_text", "mean"),
        mediana_seg_por_texto=("avg_seconds_per_text", "median"),
    )
    return grouped.sort_values("media_seg_por_texto")


def _print_section(df_section: pd.DataFrame, titulo: str) -> pd.DataFrame | None:
    """Imprime el resumen de una sección de fotos (antiguo / con soporte
    de iGPU) y devuelve la tabla agregada, o None si la sección no tiene
    análisis -- para que el llamador sepa si tiene sentido dibujar un
    gráfico de esa sección."""
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
    gpu_cpu_txt = ""
    if pd.notna(best.gpu_pct_medio) or pd.notna(best.igpu_pct_medio) or pd.notna(best.cpu_pct_medio):
        gpu_txt = f"{best.gpu_pct_medio:.0f}%" if pd.notna(best.gpu_pct_medio) else "sin dato"
        igpu_txt = f"{best.igpu_pct_medio:.0f}%" if pd.notna(best.igpu_pct_medio) else "sin dato"
        cpu_txt = f"{best.cpu_pct_medio:.0f}%" if pd.notna(best.cpu_pct_medio) else "sin dato"
        gpu_cpu_txt = f", uso GPU dedicada: {gpu_txt}, uso iGPU: {igpu_txt}, uso CPU: {cpu_txt}"
    print(
        f"\nMás rápida en esta sección: concurrencia={int(best.configured_concurrency)} "
        f"({int(best.threads_per_inference)} hilos/inferencia) en máquinas de "
        f"{int(best.cpu_count)} núcleos, offload iGPU={'sí' if best.igpu_offload_used else 'no'} "
        f"-> {best.throughput_seg_por_foto:.2f}s/foto de media ({int(best.analisis)} análisis)"
        f"{desglose}{gpu_cpu_txt}."
    )
    if (summary["analisis"] < 3).any():
        print(
            "Aviso: alguna combinación tiene menos de 3 análisis -- la media "
            "todavía puede no ser representativa, conviene acumular más antes "
            "de sacar conclusiones firmes para la memoria."
        )
    return summary


def _print_translation_section(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Igual que `_print_section()` pero para el log de traducción --
    devuelve el propio `df` (sin agrupar) para que `_print_combined_cpu_
    summary()` pueda sumar `total_seconds` sobre las llamadas reales, o
    None si no hay log todavía."""
    titulo = "Traducción local (CPU)"
    print(f"\n{'=' * len(titulo)}\n{titulo}\n{'=' * len(titulo)}\n")
    if df is None:
        print(
            "(sin translation_log.jsonl todavía -- se genera al llamar a "
            "POST /analyze/translate-descriptions, ver app/log/translation_log.py)"
        )
        return None

    print(f"{len(df)} llamadas registradas, {df['num_texts'].sum()} textos traducidos en total.\n")

    summary = summarize_translation(df)
    print("Resumen por dirección (ordenado de más a menos rápido por texto):\n")
    print(summary.to_string(index=False))

    real = df[df["translation_available"]]
    if not real.empty:
        print(
            f"\nDe estas, {len(real)} llamadas ({real['num_texts'].sum()} textos) usaron el "
            f"modelo local de verdad -- media real: {real['avg_seconds_per_text'].mean():.3f}s/texto "
            f"(mediana {real['avg_seconds_per_text'].median():.3f}s/texto)."
        )
    not_available = df[~df["translation_available"]]
    if not not_available.empty:
        print(
            f"\nAviso: {len(not_available)} llamadas se degradaron a devolver los textos "
            "sin traducir (modelo no convertido en disco todavía -- ver "
            "scripts/convert_translation_models.py). Sus tiempos NO reflejan "
            "una traducción real, solo la comprobación de disco -- excluidas "
            "del resumen combinado de CPU de más abajo."
        )
    print("\n(Todas las traducciones corren en CPU -- ver app/nlp/translation.py, "
          "no existe todavía un camino de GPU/iGPU para este modelo.)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Guardar también gráficos PNG")
    args = parser.parse_args()

    df = load_log()
    print(f"{len(df)} análisis de fotos registrados en total, {df['total_photos'].sum()} fotos en total.")
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

    translation_df = load_translation_log()
    _print_translation_section(translation_df)

    # Resumen combinado de CPU -- ver docstring del módulo. Se calcula
    # aquí directamente (no en una función aparte) porque necesita el
    # `modern_df` CRUDO (para sumar `cpu_seconds` de verdad, columna que
    # no sobrevive a `summarize()` al ser una tabla de medias) y el
    # `translation_df` crudo -- ambos ya cargados en este scope.
    titulo = "Resumen combinado de tiempo en CPU (fotos + traducción)"
    print(f"\n{'=' * len(titulo)}\n{titulo}\n{'=' * len(titulo)}\n")
    photo_cpu_seconds = modern_df["cpu_seconds"].sum(skipna=True) if not modern_df.empty else 0.0
    translation_cpu_seconds = 0.0
    if translation_df is not None:
        real_translations = translation_df[translation_df["translation_available"]]
        translation_cpu_seconds = real_translations["total_seconds"].sum()

    total_cpu_seconds = photo_cpu_seconds + translation_cpu_seconds
    if total_cpu_seconds <= 0:
        print(
            "Sin segundos de CPU registrados todavía en ninguno de los dos logs "
            "-- normal si toda la máquina tiene CUDA disponible (DINOv2 y "
            "Moondream2 corren en GPU, ver docstring de "
            "log_photo_analysis_run() en performance_log.py) y aún no se ha "
            "llamado a la traducción."
        )
    else:
        photo_pct = photo_cpu_seconds / total_cpu_seconds * 100
        translation_pct = translation_cpu_seconds / total_cpu_seconds * 100
        print(
            f"Fotos (DINOv2/Moondream2 SIN GPU disponible): {photo_cpu_seconds:.1f}s "
            f"({photo_pct:.0f}%)\n"
            f"Traducción local (siempre CPU): {translation_cpu_seconds:.1f}s "
            f"({translation_pct:.0f}%)\n"
            f"Total: {total_cpu_seconds:.1f}s de CPU acumulados entre los dos logs."
        )
        print(
            "\n(Esto es una suma de dos totales INDEPENDIENTES -- ver docstring "
            "del módulo -- no el reparto de CPU de un análisis concreto. En una "
            "máquina con GPU dedicada disponible, `photo_cpu_seconds` debería "
            "salir en 0 o cerca (DINOv2/Moondream2 corren en GPU, ver "
            "'Análisis desde que se configuró el iGPU' arriba); si sale alto, "
            "es señal de que la GPU no se está usando -- comprobar el log de "
            "arranque de Moondream2, ver scene_analysis._lazy_load().)"
        )

    if args.plot:
        print()
        if old_summary is not None:
            _plot_section(old_summary, "antiguos", "análisis antiguos")
        if modern_summary is not None:
            _plot_section(modern_summary, "con_igpu", "análisis desde iGPU")


def _plot_section(summary: pd.DataFrame, suffix: str, etiqueta: str) -> None:
    """Guarda los PNG de una sección con un sufijo distinto en el nombre
    de fichero para no pisar los de la otra sección -- así se pueden
    comparar visualmente sin mezclarlas en el mismo gráfico (ver
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

    stage_summary = summary.dropna(subset=["media_dinov2_seg", "media_moondream_seg"], how="all")
    if not stage_summary.empty:
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

    usage_summary = summary.dropna(subset=["gpu_pct_medio", "igpu_pct_medio", "cpu_pct_medio"], how="all")
    if not usage_summary.empty:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        labels3 = [
            f"offload={'sí' if row.igpu_offload_used else 'no'}\nconc={int(row.configured_concurrency)}"
            for row in usage_summary.itertuples()
        ]
        x3 = range(len(usage_summary))
        width3 = 0.25
        ax3.bar([i - width3 for i in x3], usage_summary["gpu_pct_medio"].fillna(0), width3, label="GPU dedicada (CUDA)")
        ax3.bar(list(x3), usage_summary["igpu_pct_medio"].fillna(0), width3, label="iGPU (DirectML)")
        ax3.bar([i + width3 for i in x3], usage_summary["cpu_pct_medio"].fillna(0), width3, label="CPU")
        ax3.set_xticks(list(x3))
        ax3.set_xticklabels(labels3, fontsize=8)
        ax3.set_ylabel("% de segundo de reloj en cómputo (puede superar 100%)")
        ax3.set_title(f"Uso de GPU dedicada / iGPU / CPU durante el análisis de fotos -- {etiqueta}")
        ax3.legend()
        usage_out_path = _LOG_PATH.parent / f"performance_summary_uso_gpu_cpu_{suffix}.png"
        fig3.savefig(usage_out_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Gráfico de uso GPU/iGPU/CPU guardado en {usage_out_path}")


if __name__ == "__main__":
    main()
