"""
Orquestador del experimento de la Entrega 2.

Ejecuta secuencialmente todas las combinaciones (modelo x fold x semilla)
configuradas y agrega los resultados en un CSV maestro.

Uso:
    # Smoke test rapido (1 fold, 1 semilla, 3 epocas, subset)
    python -m src.run_e2 --smoke

    # Plan minimo viable: MobileNetV2 + ViT, 3 semillas, 5 folds (15 + 15 = 30 runs)
    python -m src.run_e2 --plan minimo

    # Plan reducido: MobileNetV2 + ViT, 3 semillas, 3 folds (9 + 9 = 18 runs)
    python -m src.run_e2 --plan reducido

    # Pruebas individuales
    python -m src.run_e2 --models vit_base_patch16_224 --folds 0 --seeds 42
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .train import train_one_fold


PLANS = {
    "smoke":     {"models": ["mobilenetv2_100"],
                  "folds": [0], "seeds": [42], "epochs": 3, "smoke": True},
    "reducido":  {"models": ["mobilenetv2_100", "vit_base_patch16_224"],
                  "folds": [0, 1, 2], "seeds": [42, 137, 2026],
                  "epochs": None, "smoke": False},
    "minimo":    {"models": ["mobilenetv2_100", "vit_base_patch16_224"],
                  "folds": list(range(C.N_FOLDS)),
                  "seeds": [42, 137, 2026],
                  "epochs": None, "smoke": False},
    "ampliado":  {"models": ["mobilenetv2_100", "vit_base_patch16_224", "resnet50", "efficientnet_b0"],
                  "folds": list(range(C.N_FOLDS)),
                  "seeds": [42, 137, 2026],
                  "epochs": None, "smoke": False},
}


def run_plan(plan: dict) -> pd.DataFrame:
    rows = []
    total = len(plan["models"]) * len(plan["folds"]) * len(plan["seeds"])
    i = 0
    for model in plan["models"]:
        for seed in plan["seeds"]:
            for fold in plan["folds"]:
                i += 1
                print(f"\n=== [{i}/{total}] model={model} fold={fold} seed={seed} ===")
                res = train_one_fold(model, fold, seed,
                                     epochs_override=plan["epochs"],
                                     smoke=plan["smoke"])
                row = {
                    "model": model, "fold": fold, "seed": seed,
                    "best_macro_f1": res["best_macro_f1"],
                    "best_epoch": res["best_epoch"],
                    "accuracy": res["final_metrics"]["accuracy"],
                    "balanced_accuracy": res["final_metrics"]["balanced_accuracy"],
                    "macro_precision": res["final_metrics"]["macro_precision"],
                    "macro_recall": res["final_metrics"]["macro_recall"],
                    "weighted_f1": res["final_metrics"]["weighted_f1"],
                    "run_id": res["run_id"],
                }
                # f1 por clase
                for cls, v in res["final_metrics"]["per_class_f1"].items():
                    row[f"f1_{cls}"] = v
                rows.append(row)

                # guardado incremental por si algo se cae
                pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e2_master.csv", index=False)
    return pd.DataFrame(rows)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", choices=list(PLANS), default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--folds", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.plan:
        plan = dict(PLANS[args.plan])
    else:
        plan = {
            "models": args.models or ["mobilenetv2_100"],
            "folds":  args.folds  or [0],
            "seeds":  args.seeds  or [42],
            "epochs": args.epochs,
            "smoke":  args.smoke,
        }
    df = run_plan(plan)
    out = C.RESULTS_DIR / "e2_master.csv"
    df.to_csv(out, index=False)
    print(f"\n[OK] Maestro guardado en {out}")
    print(df.to_string())


if __name__ == "__main__":
    main()
