"""
Fase 2 — Entrenamiento final con CV completo (5 folds × 3 seeds × N modelos).

Carga los hiperparametros ganadores de cada study Optuna y ejecuta los runs
finales bajo el protocolo experimental declarado en Cap 3:
    - 5-fold stratified cross-validation.
    - 3 semillas (42, 137, 2026).
    - Macro F1 como metrica principal.
    - phase = 'final' (tag MLflow).

Persiste un master CSV con todos los resultados, listo para que
analyze_results.py los agregue.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .train import train_one_fold

DEFAULT_MODELS = ["mobilenetv2_100", "resnet50", "efficientnet_b0",
                  "vit_base_patch16_224"]


def load_best_params(model_key: str) -> dict:
    """Lee summary.json del study Optuna y devuelve best_params."""
    path = C.RESULTS_DIR / "optuna" / model_key / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe Optuna summary para {model_key} en {path}")
    return json.loads(path.read_text(encoding="utf-8"))["best_params"]


def run_final(models: list[str] = None,
              folds: list[int] | None = None,
              seeds: list[int] | None = None) -> pd.DataFrame:
    models = models or DEFAULT_MODELS
    folds = folds or list(range(C.N_FOLDS))
    seeds = seeds or C.SEEDS

    rows = []
    total = len(models) * len(folds) * len(seeds)
    i = 0
    for model_key in models:
        try:
            best_params = load_best_params(model_key)
        except FileNotFoundError as e:
            print(f"!! {e}. Saltando {model_key}.")
            continue
        print(f"\n=== {model_key}: aplicando best_params {best_params} ===")
        for seed in seeds:
            for fold in folds:
                i += 1
                print(f"\n--- [{i}/{total}] {model_key} fold={fold} seed={seed} ---")
                res = train_one_fold(
                    model_key=model_key, fold=fold, seed=seed,
                    phase="final", override_cfg=best_params,
                )
                row = {
                    "model": model_key, "fold": fold, "seed": seed,
                    "phase": "final",
                    "best_macro_f1": res["best_macro_f1"],
                    "best_epoch": res["best_epoch"],
                    "accuracy": res["final_metrics"]["accuracy"],
                    "balanced_accuracy": res["final_metrics"]["balanced_accuracy"],
                    "macro_precision": res["final_metrics"]["macro_precision"],
                    "macro_recall": res["final_metrics"]["macro_recall"],
                    "weighted_f1": res["final_metrics"]["weighted_f1"],
                    "total_run_time_s": res["total_run_time_s"],
                    "run_id": res["run_id"],
                }
                for cls, v in res["final_metrics"]["per_class_f1"].items():
                    row[f"f1_{cls}"] = v
                rows.append(row)
                # guardado incremental
                pd.DataFrame(rows).to_csv(
                    C.RESULTS_DIR / "final_master.csv", index=False)
    return pd.DataFrame(rows)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--folds", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    df = run_final(args.models, args.folds, args.seeds)
    out = C.RESULTS_DIR / "final_master.csv"
    df.to_csv(out, index=False)
    print(f"\n[OK] Final master en {out}")
    print(df.to_string())


if __name__ == "__main__":
    main()
