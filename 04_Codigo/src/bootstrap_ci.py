"""
Bootstrap CI 95% del macro F1-Score (prometido en Cap 3, no entregado aun).

Base elegida: BOOTSTRAP SOBRE LAS IMAGENES DEL HOLDOUT (15%, N=462) para el
ENSEMBLE de cada modelo. Esta es la cifra de rendimiento FINAL reportada en el
documento (ViT 0.9533, ResNet-50 0.9093, EfficientNet-B0 0.8719,
MobileNetV2 0.8620), y las predicciones por-imagen ya existen en
results/holdout_predictions/<modelo>_ensemble.npz (y_true, y_pred, y_prob).

Procedimiento (percentile bootstrap, no parametrico):
  - B=10000 remuestreos con reemplazo de las N imagenes de test.
  - En cada remuestreo se recalcula el macro F1 (sklearn, average='macro',
    zero_division=0) sobre las predicciones ya guardadas (NO se reentrena ni se
    usa GPU).
  - CI 95% = percentiles 2.5 y 97.5 de la distribucion bootstrap.
  - point_estimate = macro F1 sobre el holdout completo (sin remuestrear).

Reproducibilidad: np.random.default_rng(SEED=2026).

Salidas:
  results/bootstrap_ci.json
  results/bootstrap_ci.csv   (modelo, point_estimate, ci_low, ci_high, n, base)

Uso:
  python -m src.bootstrap_ci
  python -m src.bootstrap_ci --base cv     # variante CV out-of-fold (complementaria)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from . import config as C

SEED = 2026
B_DEFAULT = 10000

MODELS = ["vit_base_patch16_224", "resnet50", "efficientnet_b0", "mobilenetv2_100"]
# Etiqueta legible por modelo para el texto del documento
MODEL_LABEL = {
    "vit_base_patch16_224": "ViT-Base/16",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv2_100": "MobileNetV2",
}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = B_DEFAULT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap del macro F1 remuestreando las N observaciones.

    Devuelve point_estimate (sin remuestrear), ci_low, ci_high (percentiles
    alpha/2 y 1-alpha/2), media y std de la distribucion bootstrap.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = macro_f1(y_true, y_pred)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # muestreo con reemplazo
        stats[b] = macro_f1(y_true[idx], y_pred[idx])

    lo = float(np.percentile(stats, 100 * (alpha / 2)))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "boot_mean": float(stats.mean()),
        "boot_std": float(stats.std(ddof=1)),
        "n": int(n),
        "n_boot": int(n_boot),
        "alpha": alpha,
        "seed": seed,
    }


# --------------------------------------------------------------------------- #
# Carga de datos segun la base elegida
# --------------------------------------------------------------------------- #
def load_holdout_ensemble(model_key: str) -> tuple[np.ndarray, np.ndarray]:
    """y_true, y_pred del ensemble de 15 replicas sobre el holdout."""
    npz = C.RESULTS_DIR / "holdout_predictions" / f"{model_key}_ensemble.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)
    d = np.load(npz)
    return d["y_true"], d["y_pred"]


def load_cv_oof(model_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatena las predicciones out-of-fold de las 15 replicas finales de CV.

    Variante complementaria (CV-based) por si se quiere un CI sobre el CV.
    Cada run final guarda predictions.npz con y_true, y_pred sobre su fold de
    validacion. Concatenamos los 5 folds x 3 seeds.
    """
    runs_dir = C.RESULTS_DIR / "runs"
    pred_files = sorted(runs_dir.glob(f"final__{model_key}_fold*_seed*/predictions.npz"))
    if not pred_files:
        raise FileNotFoundError(f"No hay predictions.npz para {model_key} en {runs_dir}")
    yts, yps = [], []
    for pf in pred_files:
        d = np.load(pf)
        yts.append(d["y_true"])
        yps.append(d["y_pred"])
    return np.concatenate(yts), np.concatenate(yps)


def run(base: str = "holdout", n_boot: int = B_DEFAULT) -> dict:
    loader = load_holdout_ensemble if base == "holdout" else load_cv_oof
    base_desc = (
        "holdout-ensemble (462 imagenes test, ensemble 15 replicas)"
        if base == "holdout"
        else "cv-oof (predicciones out-of-fold concatenadas, 5 folds x 3 seeds)"
    )

    results = {"base": base, "base_desc": base_desc, "seed": SEED,
               "n_boot": n_boot, "models": {}}
    for m in MODELS:
        y_true, y_pred = loader(m)
        res = bootstrap_macro_f1(y_true, y_pred, n_boot=n_boot)
        res["label"] = MODEL_LABEL.get(m, m)
        res["base"] = base
        results["models"][m] = res
        print(
            f"[{base}] {MODEL_LABEL.get(m, m):16s} "
            f"point={res['point_estimate']:.4f}  "
            f"CI95=[{res['ci_low']:.4f}, {res['ci_high']:.4f}]  "
            f"n={res['n']}"
        )
    return results


def save(results: dict) -> tuple[Path, Path]:
    json_path = C.RESULTS_DIR / "bootstrap_ci.json"
    csv_path = C.RESULTS_DIR / "bootstrap_ci.csv"

    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["modelo", "label", "point_estimate", "ci_low", "ci_high",
             "boot_mean", "boot_std", "n", "n_boot", "base"]
        )
        for m, r in results["models"].items():
            w.writerow([
                m, r["label"],
                f"{r['point_estimate']:.6f}",
                f"{r['ci_low']:.6f}",
                f"{r['ci_high']:.6f}",
                f"{r['boot_mean']:.6f}",
                f"{r['boot_std']:.6f}",
                r["n"], r["n_boot"], r["base"],
            ])
    return json_path, csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", choices=["holdout", "cv"], default="holdout",
                    help="holdout (default, ensemble) o cv (out-of-fold)")
    ap.add_argument("--n-boot", type=int, default=B_DEFAULT)
    args = ap.parse_args()

    results = run(base=args.base, n_boot=args.n_boot)
    json_path, csv_path = save(results)
    print(f"\n[OK] {json_path}")
    print(f"[OK] {csv_path}")


if __name__ == "__main__":
    main()
