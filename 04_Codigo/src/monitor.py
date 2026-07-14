"""
Monitor del experimento de larga duracion.

Cada N segundos (default 1800 = 30 min) escribe `results/STATUS_REPORT.md`
con un parte del estado: runs completados, en curso, GPU, ETA, alertas.

Uso:
    python -m src.monitor &
    # o como tarea background

Si se invoca con --once, escribe el reporte una vez y sale.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config as C

REPORT_PATH = C.RESULTS_DIR / "STATUS_REPORT.md"


def gpu_stats() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            text=True, timeout=4).strip()
        u, m_used, m_total, temp, power = [s.strip() for s in out.split(",")]
        return {"util": u, "vram_used": m_used, "vram_total": m_total,
                "temp_c": temp, "power_w": power}
    except Exception as e:
        return {"error": str(e)}


def python_processes() -> int:
    try:
        out = subprocess.check_output(
            ["powershell", "-Command",
             "(Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 100MB } | Measure-Object).Count"],
            text=True, timeout=5).strip()
        return int(out)
    except Exception:
        return -1


def read_master_csv(which: str = "final") -> pd.DataFrame | None:
    """Lee el CSV maestro indicado. which ∈ {'final','baseline'}."""
    fname = "final_master.csv" if which == "final" else "e2_master.csv"
    p = C.RESULTS_DIR / fname
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def read_optuna_summaries() -> dict:
    out: dict = {}
    base = C.RESULTS_DIR / "optuna"
    if not base.exists():
        return out
    for d in base.iterdir():
        s = d / "summary.json"
        if s.exists():
            try:
                out[d.name] = json.loads(s.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


def write_report() -> str:
    now = datetime.now()
    lines = [f"# Status Report — TFM Entrega 2",
             f"_Actualizado: {now.strftime('%Y-%m-%d %H:%M:%S')} (Colombia)_\n"]

    # GPU
    g = gpu_stats()
    lines.append("## GPU")
    if "error" in g:
        lines.append(f"- nvidia-smi falló: `{g['error']}`")
    else:
        lines.append(f"- Utilización: **{g['util']}**  ·  VRAM: **{g['vram_used']} / {g['vram_total']}**")
        lines.append(f"- Temperatura: {g['temp_c']}°C  ·  Potencia: {g['power_w']}")
    lines.append(f"- Procesos python (>100MB): **{python_processes()}**\n")

    # Optuna
    summaries = read_optuna_summaries()
    lines.append("## Fase 1 — Optuna")
    if not summaries:
        lines.append("_Sin studies completos aún._\n")
    else:
        lines.append("| Modelo | Trials | Completos | Pruned | Failed | Best F1 |")
        lines.append("|---|---|---|---|---|---|")
        for m, s in summaries.items():
            lines.append(f"| {m} | {s.get('n_trials','-')} | "
                         f"{s.get('n_complete','-')} | {s.get('n_pruned','-')} | "
                         f"{s.get('n_failed','-')} | {s.get('best_value', float('nan')):.4f} |")
        lines.append("")

    # Baseline pre-tuning (e2_master.csv del plan reducido del 10/05 AM)
    df_b = read_master_csv("baseline")
    lines.append("## Baseline pre-tuning (referencia)")
    if df_b is None or df_b.empty:
        lines.append("_Sin baseline._\n")
    else:
        gb = df_b.groupby("model").agg(
            n=("best_macro_f1","count"),
            f1_mean=("best_macro_f1","mean"),
            f1_std=("best_macro_f1","std"),
        ).round(4)
        lines.append(f"- Runs baseline: **{len(df_b)}** (sin tuning Optuna)")
        lines.append("\n| Modelo | n | Macro F1 (mu +/- sigma) |")
        lines.append("|---|---|---|")
        for m, row in gb.iterrows():
            lines.append(f"| {m} | {int(row['n'])} | {row['f1_mean']:.4f} +/- "
                         f"{row['f1_std'] if not pd.isna(row['f1_std']) else 0:.4f} |")
        lines.append("")

    # Final con hiperparametros Optuna
    df = read_master_csv("final")
    lines.append("## Fase 2 — Entrenamiento final (CV completo, post-Optuna)")
    if df is None or df.empty:
        lines.append("_Sin runs final aún._\n")
    else:
        lines.append(f"- Runs completados: **{len(df)}** de 60 esperados ({len(df)*100//60}%).")
        g = df.groupby("model").agg(
            n=("best_macro_f1", "count"),
            f1_mean=("best_macro_f1", "mean"),
            f1_std=("best_macro_f1", "std"),
            acc_mean=("accuracy", "mean"),
        ).round(4)
        lines.append("\n| Modelo | n | Macro F1 (mu +/- sigma) | Accuracy mu |")
        lines.append("|---|---|---|---|")
        for m, row in g.iterrows():
            lines.append(f"| {m} | {int(row['n'])} | {row['f1_mean']:.4f} +/- "
                         f"{row['f1_std'] if not pd.isna(row['f1_std']) else 0:.4f} | "
                         f"{row['acc_mean']:.4f} |")
        lines.append("")

    # Holdout
    hp = C.RESULTS_DIR / "holdout_eval.json"
    lines.append("## Fase 3 — Holdout 15%")
    if hp.exists():
        try:
            data = json.loads(hp.read_text(encoding="utf-8"))
            lines.append(f"- N imagenes holdout: {data.get('n_holdout','?')}")
            lines.append("\n| Modelo | Modo | Macro F1 | Accuracy |")
            lines.append("|---|---|---|---|")
            for m, modes in data.get("models", {}).items():
                for mode, mdata in modes.items():
                    if "error" in mdata:
                        lines.append(f"| {m} | {mode} | ERROR | - |")
                    else:
                        lines.append(f"| {m} | {mode} | {mdata.get('macro_f1', 0):.4f} | "
                                     f"{mdata.get('accuracy', 0):.4f} |")
            lines.append("")
        except Exception as e:
            lines.append(f"_Error leyendo holdout: {e}_\n")
    else:
        lines.append("_No ejecutado aún._\n")

    # Marker file (last update)
    lines.append("---\n_Reporte generado automáticamente por src.monitor_")
    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def loop(interval_s: int = 1800):
    print(f"Monitor activo. Reportes cada {interval_s}s en {REPORT_PATH}")
    while True:
        try:
            write_report()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Reporte actualizado")
        except Exception as e:
            print(f"  ERROR escribiendo reporte: {e}")
        time.sleep(interval_s)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true",
                   help="Escribe el reporte una vez y sale")
    p.add_argument("--interval", type=int, default=1800,
                   help="Segundos entre reportes (default 1800)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.once:
        write_report()
        print(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        loop(args.interval)
