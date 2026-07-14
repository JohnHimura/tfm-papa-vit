"""
OE4 NORMFIX — Ablacion de manejo de desbalance del ViT, CORREGIDA.

Diferencias respecto a src.run_oe4_ablation (que se conserva intacto):
  1. NORMALIZACION CORRECTA del ViT ((0.5,0.5,0.5)) — automatica via train.py
     tras el fix por-modelo en transforms.py.
  2. FOCAL LOSS arreglada (src.losses.FocalLoss con pt verdadero) — la usa la
     estrategia 'focal'.
  3. LR RE-OPTIMIZADO POR ESTRATEGIA: en vez de heredar el LR de la estrategia
     class_weights (lo que hacia la version original, reutilizando un unico set
     de best_params del ViT), cada estrategia (none/class_weights/focal/
     oversampling) corre un mini-Optuna propio sobre (lr_head, lr_backbone) en
     fold0/seed42, y la CV final de esa estrategia usa SU LR ganador.

PRESERVACION (rutas NUEVAS):
    results/oe4_ablation_normfix.csv        <- tabla agregada por estrategia
    results/oe4_ablation_normfix.json       <- detalle (mini-optuna + replicas)
    results/oe4_ablation_normfix_raw.csv    <- filas crudas por (estrategia,fold,seed)
    optuna/oe4_normfix__<strategy>.db       <- studies mini-optuna por estrategia
    NADA pisa results/oe4_ablation*.csv/json existentes.

REUTILIZACION:
    - Estrategias/flags/labels: importados de src.run_oe4_ablation (STRATEGIES,
      STRATEGY_LABEL, build_strategy_cfg).
    - Entrenamiento: src.train.train_one_fold (phase oe4n_<strategy>).
    - Base de HP: results/optuna_normfix/vit_base_patch16_224/summary.json
      (weight_decay, label_smoothing, batch_size) — el mini-optuna solo mueve el
      LR; el resto se hereda de la sintonia normfix del ViT.

Uso (reproduce la tabla corregida):
    python -m src.run_oe4_ablation_normfix --lr-trials 15

Smoke (CPU, sin GPU larga):
    python -m src.run_oe4_ablation_normfix --smoke   # 1-2 trials, 1 epoca, 1 replica
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .train import train_one_fold
from .run_oe4_ablation import (STRATEGIES, STRATEGY_LABEL, build_strategy_cfg,
                               OE4_MODEL)

OPTUNA_DIR = C.CODE_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

LR_OPTUNA_FOLD = 0
LR_OPTUNA_SEED = 42
DEFAULT_LR_TRIALS = 15

# Base de HP normfix del ViT (sin LR; el LR lo busca el mini-optuna por estrategia).
OPTUNA_NORMFIX_VIT = C.RESULTS_DIR / "optuna_normfix" / OE4_MODEL / "summary.json"


def _load_base_params_normfix() -> dict:
    """HP base del ViT normfix SIN lr_head/lr_backbone (los re-optimizamos)."""
    if OPTUNA_NORMFIX_VIT.exists():
        bp = json.loads(OPTUNA_NORMFIX_VIT.read_text(encoding="utf-8"))["best_params"]
    else:
        # Fallback razonable si aun no se corrio el Optuna normfix del ViT.
        print(f"  [OE4n] AVISO: no existe {OPTUNA_NORMFIX_VIT}; uso defaults "
              f"de la cfg del ViT para weight_decay/label_smoothing/batch_size.")
        bp = {}
    base = {k: v for k, v in bp.items() if k not in ("lr_head", "lr_backbone")}
    return base


def _tune_lr_for_strategy(strategy: str, base_params: dict,
                          n_trials: int, epochs: int | None,
                          smoke: bool) -> dict:
    """Mini-Optuna por estrategia: busca (lr_head, lr_backbone) en fold0/seed42.

    Devuelve dict con best_lr_head, best_lr_backbone, best_value y n_trials.
    """
    import optuna
    from optuna.samplers import TPESampler

    storage = f"sqlite:///{(OPTUNA_DIR / f'oe4_normfix__{strategy}.db').as_posix()}"
    study = optuna.create_study(
        study_name=f"oe4n_{strategy}",
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=min(5, n_trials)),
        storage=storage, load_if_exists=True,
    )

    strat_flags = STRATEGIES[strategy]

    def objective(trial):
        override = dict(base_params)
        override.update(strat_flags)
        override["lr_head"] = trial.suggest_float("lr_head", 1e-5, 1e-2, log=True)
        override["lr_backbone"] = trial.suggest_float("lr_backbone", 1e-6, 1e-3, log=True)
        try:
            res = train_one_fold(
                model_key=OE4_MODEL, fold=LR_OPTUNA_FOLD, seed=LR_OPTUNA_SEED,
                phase=f"oe4n_tune_{strategy}",
                override_cfg=override,
                epochs_override=(1 if smoke else epochs),
                smoke=smoke,
            )
            return res["best_macro_f1"]
        except Exception as e:
            print(f"  [OE4n tune {strategy}] TRIAL FAILED: {type(e).__name__}: {e}")
            return float("nan")

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return {
        "best_lr_head": study.best_params["lr_head"],
        "best_lr_backbone": study.best_params["lr_backbone"],
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }


def run_ablation_normfix(strategies: list[str] | None = None,
                         folds: list[int] | None = None,
                         seeds: list[int] | None = None,
                         lr_trials: int = DEFAULT_LR_TRIALS,
                         epochs: int | None = None,
                         smoke: bool = False) -> dict:
    from scipy.stats import wilcoxon

    strategies = strategies or list(STRATEGIES.keys())
    folds = folds if folds is not None else list(range(C.N_FOLDS))
    seeds = seeds if seeds is not None else C.SEEDS

    base_params = _load_base_params_normfix()
    print(f"[OE4n] base_params (sin LR): {base_params}")
    print(f"[OE4n] estrategias={strategies} folds={folds} seeds={seeds} "
          f"lr_trials={lr_trials}")

    replicas = [(f, s) for s in seeds for f in folds]
    raw_rows: list[dict] = []
    macro_by_strategy: dict[str, dict[tuple[int, int], float]] = {
        s: {} for s in strategies}
    tuned_lr: dict[str, dict] = {}

    for strategy in strategies:
        # 1) Mini-Optuna del LR para ESTA estrategia.
        lr_res = _tune_lr_for_strategy(strategy, base_params, lr_trials,
                                       epochs, smoke)
        tuned_lr[strategy] = lr_res
        print(f"\n=== '{strategy}' ({STRATEGY_LABEL[strategy]}) "
              f"LR* head={lr_res['best_lr_head']:.2e} "
              f"backbone={lr_res['best_lr_backbone']:.2e} "
              f"(val={lr_res['best_value']:.4f}) ===")

        # 2) CV final de la estrategia con SU LR ganador + flags + base normfix.
        cfg_over = build_strategy_cfg(strategy, base_params)
        cfg_over["lr_head"] = lr_res["best_lr_head"]
        cfg_over["lr_backbone"] = lr_res["best_lr_backbone"]

        for (fold, seed) in replicas:
            res = train_one_fold(
                model_key=OE4_MODEL, fold=fold, seed=seed,
                phase=f"oe4n_{strategy}",
                override_cfg=cfg_over,
                epochs_override=(1 if smoke else epochs),
                smoke=smoke,
            )
            fm = res["final_metrics"]
            row = {
                "strategy": strategy,
                "strategy_label": STRATEGY_LABEL[strategy],
                "lr_head": lr_res["best_lr_head"],
                "lr_backbone": lr_res["best_lr_backbone"],
                "fold": fold, "seed": seed,
                "macro_f1": res["best_macro_f1"],
                "macro_f1_final": fm["macro_f1"],
                "accuracy": fm["accuracy"],
                "balanced_accuracy": fm["balanced_accuracy"],
                "f1_Nematode": fm["per_class_f1"]["Nematode"],
                "f1_Healthy": fm["per_class_f1"]["Healthy"],
                "run_id": res["run_id"],
            }
            for cls, v in fm["per_class_f1"].items():
                row[f"f1_{cls}"] = v
            raw_rows.append(row)
            macro_by_strategy[strategy][(fold, seed)] = res["best_macro_f1"]
            pd.DataFrame(raw_rows).to_csv(
                C.RESULTS_DIR / "oe4_ablation_normfix_raw.csv", index=False)

    # ----------------------- Agregacion por estrategia -----------------------
    df = pd.DataFrame(raw_rows)
    summary_rows: list[dict] = []
    baseline = "none" if "none" in strategies else strategies[0]
    base_keys = sorted(macro_by_strategy.get(baseline, {}).keys())

    for strategy in strategies:
        sub = df[df["strategy"] == strategy]
        macro_vals = sub["macro_f1"].to_numpy()
        rec = {
            "strategy": strategy,
            "strategy_label": STRATEGY_LABEL[strategy],
            "lr_head": tuned_lr[strategy]["best_lr_head"],
            "lr_backbone": tuned_lr[strategy]["best_lr_backbone"],
            "lr_tune_trials": tuned_lr[strategy]["n_trials"],
            "n_replicas": int(len(sub)),
            "macro_f1_mean": float(np.mean(macro_vals)) if len(macro_vals) else float("nan"),
            "macro_f1_std": float(np.std(macro_vals, ddof=1)) if len(macro_vals) > 1 else 0.0,
            "f1_Nematode_mean": float(sub["f1_Nematode"].mean()) if len(sub) else float("nan"),
            "f1_Healthy_mean": float(sub["f1_Healthy"].mean()) if len(sub) else float("nan"),
        }
        if strategy == baseline:
            rec["wilcoxon_p_vs_none"] = None
        else:
            cur = macro_by_strategy[strategy]
            common = [k for k in base_keys if k in cur]
            base_arr = np.array([macro_by_strategy[baseline][k] for k in common])
            cur_arr = np.array([cur[k] for k in common])
            if len(common) >= 1 and np.any(cur_arr - base_arr != 0):
                try:
                    stat, p = wilcoxon(cur_arr, base_arr)
                    rec["wilcoxon_p_vs_none"] = float(p)
                except ValueError as e:
                    rec["wilcoxon_p_vs_none"] = None
                    rec["wilcoxon_note"] = str(e)
            else:
                rec["wilcoxon_p_vs_none"] = None
            rec["n_pairs_vs_none"] = int(len(common))
        rec["macro_f1_fmt"] = f"{rec['macro_f1_mean']:.4f} +/- {rec['macro_f1_std']:.4f}"
        summary_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows)
    out_csv = C.RESULTS_DIR / "oe4_ablation_normfix.csv"
    summary_df.to_csv(out_csv, index=False)

    result = {
        "model": OE4_MODEL,
        "base_params_normfix": base_params,
        "tuned_lr_per_strategy": tuned_lr,
        "design": {"folds": folds, "seeds": seeds, "lr_trials": lr_trials,
                   "epochs_override": epochs, "smoke": smoke},
        "strategies": STRATEGIES,
        "summary": summary_rows,
        "raw": raw_rows,
    }
    out_json = C.RESULTS_DIR / "oe4_ablation_normfix.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[OK] OE4 normfix -> {out_csv}")
    print(f"[OK] OE4 normfix -> {out_json}")
    if not summary_df.empty:
        print("\n" + summary_df[[
            "strategy_label", "lr_head", "lr_backbone",
            "macro_f1_fmt", "wilcoxon_p_vs_none"]].to_string(index=False))
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="OE4 ablacion NORMFIX (ViT).")
    p.add_argument("--strategies", nargs="+", default=None,
                   choices=list(STRATEGIES.keys()))
    p.add_argument("--folds", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--lr-trials", type=int, default=DEFAULT_LR_TRIALS,
                   help="trials del mini-optuna de LR por estrategia (default 15)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="CPU smoke: 1-2 trials, 1 epoca, subset chico.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    lr_trials = 2 if args.smoke else args.lr_trials
    folds = args.folds if args.folds is not None else ([0] if args.smoke else None)
    seeds = args.seeds if args.seeds is not None else ([42] if args.smoke else None)
    strategies = args.strategies if args.strategies is not None else (
        ["none", "focal"] if args.smoke else None)
    run_ablation_normfix(strategies=strategies, folds=folds, seeds=seeds,
                         lr_trials=lr_trials, epochs=args.epochs, smoke=args.smoke)


if __name__ == "__main__":
    main()
