"""
Fase 1 — Optimización de hiperparámetros con Optuna (TPE + ASHA pruner).

Para cada modelo:
  - sampler: TPE (Tree-structured Parzen Estimator) — Akiba et al. 2019.
  - pruner: SuccessiveHalving (variante ASHA) — Li et al. 2018.
  - storage: SQLite por modelo en optuna/<modelo>.db (resumible).
  - objective: maximizar val_macro_f1 sobre fold=0, seed=42, hasta 15 epocas.
  - early stopping del study: si no hay mejora del best en 10 trials seguidos.

Cada trial entrena con la misma logica que train_one_fold pero:
  - phase = 'optuna'
  - epochs_override = OPTUNA_EPOCHS (por defecto 15)
  - override_cfg = hiperparametros muestreados por Optuna
  - epoch_callback = pruner que reporta a Optuna y permite cortar trials malos

Uso:
    python -m src.optuna_runner --model mobilenetv2_100 --n-trials 30
    python -m src.optuna_runner --plan all   # corre los 4 modelos del Camino B
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from . import config as C
from .train import train_one_fold

OPTUNA_DIR = C.CODE_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_EPOCHS = 15            # epocas por trial (con pruner)
OPTUNA_FOLD = 0
OPTUNA_SEED = 42
OPTUNA_EARLY_STOP_TRIALS = 10  # trials sin mejora antes de cortar el study

DEFAULT_MODELS = ["mobilenetv2_100", "resnet50", "efficientnet_b0",
                  "vit_base_patch16_224"]

# Espacio de busqueda. batch_size se ajusta al techo de VRAM por modelo.
BATCH_SPACE = {
    "mobilenetv2_100": [32, 64, 96],
    "resnet50":        [16, 32, 48],
    "efficientnet_b0": [32, 64, 96],
    "vit_base_patch16_224": [16, 24, 32],
    # El ViT 1k-only reutiliza la cfg base del ViT (mismo techo de VRAM).
    "vit_base_patch16_224.augreg_in1k": [16, 24, 32],
}

# normfix (29/06/2026) — mapa timm-model-name -> clave de TrainCfg conocida en
# C.MODELS_E2/MODELS_OPCIONAL. Permite sintonizar el ViT 1k-only (cuyo nombre
# timm NO es una clave de cfg) reutilizando la cfg del ViT base e inyectando el
# tag de pesos via override_cfg["model_name"], igual que hace run_vit1k.
CFG_KEY_FOR = {
    "vit_base_patch16_224.augreg_in1k": "vit_base_patch16_224",
}


def make_objective(model_name: str):
    """Construye la funcion objective(trial) para el modelo dado.

    `model_name` es el nombre timm a entrenar (puede llevar tag de pesos, p.ej.
    'vit_base_patch16_224.augreg_in1k'). La cfg base se busca via CFG_KEY_FOR
    (con fallback a `model_name` mismo). La normalizacion correcta del modelo la
    aplica train.py automaticamente desde cfg.model_name (normfix).
    """
    cfg_key = CFG_KEY_FOR.get(model_name, model_name)
    batch_space = BATCH_SPACE.get(model_name, BATCH_SPACE.get(cfg_key))

    def objective(trial: optuna.Trial) -> float:
        override = {
            "lr_head":         trial.suggest_float("lr_head", 1e-5, 1e-2, log=True),
            "lr_backbone":     trial.suggest_float("lr_backbone", 1e-6, 1e-3, log=True),
            "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
            "batch_size":      trial.suggest_categorical("batch_size", batch_space),
        }
        # Si entrenamos un tag de pesos distinto del cfg_key, inyectamos el
        # model_name (asi train.py usa la normalizacion correcta de ESE tag).
        if cfg_key != model_name:
            override["model_name"] = model_name

        # Callback para reportar a Optuna y permitir pruning
        def epoch_cb(epoch: int, metrics: dict) -> bool:
            trial.report(metrics["macro_f1"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return False

        try:
            res = train_one_fold(
                model_key=cfg_key,
                fold=OPTUNA_FOLD,
                seed=OPTUNA_SEED,
                epochs_override=OPTUNA_EPOCHS,
                phase="optuna",
                override_cfg=override,
                epoch_callback=epoch_cb,
            )
            return res["best_macro_f1"]
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  TRIAL FAILED: {type(e).__name__}: {e}")
            return float("nan")

    return objective


def early_stop_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Detiene el study si no hay mejora en N trials consecutivos."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < OPTUNA_EARLY_STOP_TRIALS + 1:
        return
    best = study.best_value
    recent = sorted(completed, key=lambda t: t.number)[-OPTUNA_EARLY_STOP_TRIALS:]
    if all(t.value < best - 1e-5 for t in recent):
        print(f"\n>>> Early stop del study {study.study_name}: "
              f"sin mejora en {OPTUNA_EARLY_STOP_TRIALS} trials")
        study.stop()


def run_optuna_for_model(model_key: str, n_trials: int = 30,
                         normfix: bool = False) -> dict:
    """Crea/reanuda el study, corre los trials, persiste resultados.

    Si normfix=True (29/06/2026), los artefactos van a rutas NUEVAS para NO
    pisar los studies/summaries existentes:
        storage  -> optuna/normfix__<model_key>.db
        summary  -> results/optuna_normfix/<model_key>/summary.json
        trials   -> results/optuna_normfix/<model_key>/trials.csv
    Con normfix=False el comportamiento es identico al original (sin cambios).
    En ambos modos la normalizacion por-modelo ya es correcta (via train.py).
    """
    prefix = "normfix__" if normfix else ""
    storage = f"sqlite:///{(OPTUNA_DIR / f'{prefix}{model_key}.db').as_posix()}"
    study = optuna.create_study(
        study_name=f"tfm_{prefix}{model_key}",
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=10),
        pruner=SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
        storage=storage, load_if_exists=True,
    )

    print(f"\n{'='*70}\nOptuna >> {model_key} "
          f"({'NORMFIX ' if normfix else ''}storage: {storage})\n{'='*70}")
    study.optimize(make_objective(model_key),
                   n_trials=n_trials,
                   callbacks=[early_stop_callback],
                   gc_after_trial=True)

    # Persistir resumen (ruta NUEVA si normfix para no pisar lo existente)
    base = "optuna_normfix" if normfix else "optuna"
    out_dir = C.RESULTS_DIR / base / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    df_trials = study.trials_dataframe(attrs=("number", "value", "state",
                                              "datetime_start", "datetime_complete",
                                              "duration", "params"))
    df_trials.to_csv(out_dir / "trials.csv", index=False)
    summary = {
        "model": model_key,
        "n_trials": len(study.trials),
        "n_complete": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        "n_pruned":   sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "n_failed":   sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {model_key}: best macro_f1 = {study.best_value:.4f}")
    print(f"     params: {study.best_params}")
    print(f"     resumen en {out_dir}")
    return summary


def run_all(models: list[str], n_trials: int = 30, normfix: bool = False) -> dict:
    """Ejecuta Optuna para cada modelo en serie. Persiste un summary global."""
    results = {}
    for m in models:
        results[m] = run_optuna_for_model(m, n_trials=n_trials, normfix=normfix)
    base = "optuna_normfix" if normfix else "optuna"
    out = C.RESULTS_DIR / base / "summary_all.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Resumen global -> {out}")
    return results


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="Modelo individual; si se omite usa el plan elegido por --plan")
    p.add_argument("--plan", choices=["all", "cnn", "vit", "vit_normfix"],
                   default=None)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--normfix", action="store_true",
                   help="Escribe a rutas NUEVAS (optuna_normfix/, normfix__*.db) "
                        "sin pisar los studies existentes. Usar para re-sintonizar "
                        "los ViT con normalizacion correcta.")
    return p.parse_args()


def main():
    args = _parse_args()
    normfix = args.normfix
    if args.plan == "all":
        models = DEFAULT_MODELS
    elif args.plan == "cnn":
        models = ["mobilenetv2_100", "resnet50", "efficientnet_b0"]
    elif args.plan == "vit":
        models = ["vit_base_patch16_224"]
    elif args.plan == "vit_normfix":
        # Re-sintoniza AMBOS ViT con normalizacion correcta a rutas nuevas.
        models = ["vit_base_patch16_224", "vit_base_patch16_224.augreg_in1k"]
        normfix = True
    elif args.model:
        models = [args.model]
    else:
        models = DEFAULT_MODELS
    run_all(models, n_trials=args.n_trials, normfix=normfix)


if __name__ == "__main__":
    main()
