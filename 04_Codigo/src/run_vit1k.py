"""
Opcion (b) — Reentrenamiento del ViT con pesos ImageNet-1K-only.

Motivacion (analisis ImageNet-21K):
    El ViT principal del TFM usa `vit_base_patch16_224` cuyo default de timm es
    `augreg2_in21k_ft_in1k` (preentrenado en ImageNet-21k y afinado en 1k),
    mientras las 3 CNN (ResNet-50, EfficientNet-B0, MobileNetV2) parten de pesos
    1k-only. La comparacion de preentrenamiento NO fue simetrica.

    Este script reentrena el ViT bajo EXACTAMENTE el mismo protocolo (5 folds x
    3 seeds, mismos best_params de Optuna, mismas epocas/scheduler/loss) cambiando
    UNICAMENTE el tag de pesos de partida a `vit_base_patch16_224.augreg_in1k`
    (1k-only). El objetivo es medir empiricamente cuanto cae el macro F1 (se
    espera ~5-6 pts) para decidir el impacto sobre la narrativa del documento.

PRESERVACION DE RESULTADOS 21K:
    - Checkpoints nuevos -> models/vit1k__vit_base_patch16_224_fold<k>_seed<s>_best.pt
    - Runs nuevos        -> results/runs/vit1k__vit_base_patch16_224_fold<k>_seed<s>/
    - Predicciones holdout-> results/holdout_predictions/vit1k_ensemble.npz
    - Resumen comparativo -> results/vit1k_vs_vit21k.json
    NADA sobreescribe los artefactos 'final__vit_base_patch16_224_*' del 21k ni
    su results/holdout_predictions/vit_base_patch16_224_ensemble.npz.

REUTILIZACION (sin duplicar logica):
    - El bucle de entrenamiento es train.train_one_fold (un nuevo parametro
      opcional run_name_override permite el prefijo 'vit1k__').
    - Los best_params salen de results/optuna/vit_base_patch16_224/summary.json
      via final_runner.load_best_params (mismos lr_head, lr_backbone,
      weight_decay, label_smoothing, batch_size).
    - El CI bootstrap reusa bootstrap_ci.bootstrap_macro_f1.
    - McNemar reusa mcnemar.mcnemar_test.
    - El promediado del ensemble replica el patron de holdout_eval.evaluate_ensemble
      apuntando a los checkpoints 1k.

Subcomandos (para que el pipeline nocturno los llame por separado):
    python -m src.run_vit1k train          # 5x3 = 15 runs (GPU)
    python -m src.run_vit1k eval           # ensemble holdout + CI + McNemar (CPU/GPU)
    python -m src.run_vit1k all            # train + eval

Flags utiles:
    --folds 0 1 2 3 4        (subset de folds; default los 5)
    --seeds 42 137 2026      (subset de seeds; default las 3)
    --smoke                  (train: subset chico + pocas epocas, test rapido)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as D
from . import metrics as M
from .bootstrap_ci import bootstrap_macro_f1
from .final_runner import load_best_params
from .mcnemar import mcnemar_test
from .models import build_model
from .train import train_one_fold
from .transforms import eval_transforms, get_model_norm

# --------------------------------------------------------------------------- #
# Constantes de la variante 1k-only
# --------------------------------------------------------------------------- #
# Clave del study Optuna (reutilizamos los best_params del ViT 21k tal cual).
VIT_OPTUNA_KEY = "vit_base_patch16_224"
# Nombre timm CON tag de pesos 1k-only. timm acepta el tag en el nombre, por lo
# que NO hace falta tocar models.build_model: basta pasarlo como model_name.
VIT1K_MODEL_NAME = "vit_base_patch16_224.augreg_in1k"
# Prefijo de artefactos NUEVO (no colisiona con 'final__').
RUN_PREFIX = "vit1k"

# Referencias para la comparacion (predicciones ya existentes en disco).
VIT21K_ENSEMBLE_NPZ = "vit_base_patch16_224_ensemble.npz"   # el ViT 21k actual
RESNET_ENSEMBLE_NPZ = "resnet50_ensemble.npz"
VIT21K_REPORTED_MACRO_F1 = 0.9533   # cifra del documento (holdout ensemble)


def _run_name(fold: int, seed: int) -> str:
    return f"{RUN_PREFIX}__{VIT_OPTUNA_KEY}_fold{fold}_seed{seed}"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================== #
# Subcomando: TRAIN
# =========================================================================== #
def run_train(folds: list[int] | None = None,
              seeds: list[int] | None = None,
              smoke: bool = False,
              resume: bool = False) -> pd.DataFrame:
    """Reentrena el ViT 1k-only sobre folds x seeds reutilizando train_one_fold.

    El UNICO cambio respecto al ViT 21k es model_name (tag de pesos). Todos los
    demas hiperparametros provienen de la cfg del ViT (C.MODELS_E2) + best_params
    de Optuna, identicos al run 21k.
    """
    folds = folds if folds is not None else list(range(C.N_FOLDS))
    seeds = seeds if seeds is not None else C.SEEDS

    best_params = load_best_params(VIT_OPTUNA_KEY)
    # Inyectamos el tag de pesos 1k-only SIN tocar nada mas. La cfg base
    # (epochs=20, warmup_epochs=3, batch_size override de Optuna, etc.) se hereda
    # de C.MODELS_E2['vit_base_patch16_224'] dentro de train_one_fold.
    override_cfg = {**best_params, "model_name": VIT1K_MODEL_NAME}
    print(f"=== ViT 1k-only: model_name={VIT1K_MODEL_NAME} ===")
    print(f"    best_params reutilizados (ViT 21k): {best_params}")

    master_path = C.RESULTS_DIR / "vit1k_master.csv"
    rows: list[dict] = []
    done: set[tuple[int, int]] = set()
    if resume and master_path.exists():
        prev = pd.read_csv(master_path)
        rows = prev.to_dict("records")
        done = {(int(r["fold"]), int(r["seed"])) for r in rows}
        print(f"[RESUME] {len(done)} runs ya completos se omiten: {sorted(done)}")
    total = len(folds) * len(seeds)
    i = 0
    for seed in seeds:
        for fold in folds:
            i += 1
            rname = _run_name(fold, seed)
            if resume and (fold, seed) in done:
                print(f"--- [{i}/{total}] {rname} -> OMITIDO (run ya completo)")
                continue
            print(f"\n--- [{i}/{total}] {rname} ---")
            res = train_one_fold(
                model_key=VIT_OPTUNA_KEY,        # carga cfg del ViT existente
                fold=fold, seed=seed,
                phase="final",                   # mismo protocolo que el 21k
                override_cfg=override_cfg,        # <- unico cambio: tag de pesos
                run_name_override=rname,          # prefijo vit1k__ propio
                epochs_override=(2 if smoke else None),
                smoke=smoke,
            )
            row = {
                "model": VIT1K_MODEL_NAME, "fold": fold, "seed": seed,
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
            # Guardado incremental a un master PROPIO (no toca final_master.csv).
            pd.DataFrame(rows).to_csv(master_path, index=False)
    df = pd.DataFrame(rows)
    print(f"\n[OK] vit1k_master.csv ({len(df)} runs)")
    return df


# =========================================================================== #
# Subcomando: EVAL
# =========================================================================== #
def _holdout_loader(batch_size: int = 32) -> tuple[DataLoader, pd.DataFrame]:
    df = D.load_or_build_splits()
    df_test = df[df["split_holdout"] == "test"].reset_index(drop=True)
    # El ViT 1k-only espera (0.5,0.5,0.5); normalizacion correcta via helper.
    mean, std = get_model_norm(VIT1K_MODEL_NAME)
    ds = D.PotatoLeafDataset(df_test,
                             transform=eval_transforms(mean=mean, std=std))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader, df_test


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(prob)
        gts.append(y.numpy())
    return np.concatenate(gts), np.concatenate(probs)


def find_vit1k_checkpoints() -> list[Path]:
    """Lista los checkpoints best.pt del ViT 1k-only (prefijo vit1k__)."""
    return sorted(C.MODELS_DIR.glob(
        f"{RUN_PREFIX}__{VIT_OPTUNA_KEY}_fold*_seed*_best.pt"))


def evaluate_ensemble_vit1k(loader: DataLoader, device: torch.device) -> dict:
    """Promedia softmax de TODOS los checkpoints vit1k__ (5 folds x 3 seeds).

    Replica holdout_eval.evaluate_ensemble pero construyendo el modelo con el
    nombre con tag 1k-only y apuntando a los checkpoints nuevos.
    """
    ckpts = find_vit1k_checkpoints()
    if not ckpts:
        raise RuntimeError(
            "No hay checkpoints vit1k__ en models/. Corre primero "
            "`python -m src.run_vit1k train`.")
    probs_acc = None
    y_true_ref = None
    for ck in ckpts:
        model = build_model(VIT1K_MODEL_NAME, pretrained=False,
                            num_classes=C.NUM_CLASSES).to(device)
        sd = torch.load(ck, weights_only=False)
        model.load_state_dict(sd["model_state"])
        y_true, y_prob = _predict(model, loader, device)
        if probs_acc is None:
            probs_acc = y_prob
            y_true_ref = y_true
        else:
            probs_acc = probs_acc + y_prob
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    y_prob_avg = probs_acc / len(ckpts)
    y_pred = y_prob_avg.argmax(axis=1)
    met = M.compute_metrics(y_true_ref, y_pred)
    return {
        "mode": "ensemble_15", "n_models": len(ckpts),
        **met,
        "y_true": y_true_ref, "y_pred": y_pred, "y_prob": y_prob_avg,
    }


def _load_ref_preds(npz_name: str) -> tuple[np.ndarray, np.ndarray]:
    npz = C.RESULTS_DIR / "holdout_predictions" / npz_name
    if not npz.exists():
        raise FileNotFoundError(npz)
    d = np.load(npz)
    return d["y_true"], d["y_pred"]


def run_eval() -> dict:
    """Evalua el ensemble 1k sobre holdout y construye la comparacion 1k vs 21k.

    Pasos:
      1. Ensemble de 15 checkpoints vit1k -> holdout_predictions/vit1k_ensemble.npz
      2. Macro F1 + bootstrap CI 95% (reusa bootstrap_ci).
      3. McNemar vit1k vs ViT 21k y vit1k vs ResNet-50 (reusa mcnemar).
      4. Resumen results/vit1k_vs_vit21k.json (incluye F1 por clase comparado).
    """
    device = _device()
    loader, df_test = _holdout_loader()
    pred_dir = C.RESULTS_DIR / "holdout_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Ensemble 1k-only ----
    ens = evaluate_ensemble_vit1k(loader, device)
    y_true = ens["y_true"]
    y_pred_1k = ens["y_pred"]
    out_npz = pred_dir / "vit1k_ensemble.npz"
    np.savez(out_npz, y_true=y_true, y_pred=y_pred_1k, y_prob=ens["y_prob"])
    print(f"[OK] {out_npz}  macro_f1(1k)={ens['macro_f1']:.4f}")

    # ---- 2) Bootstrap CI del 1k ----
    ci_1k = bootstrap_macro_f1(y_true, y_pred_1k)

    # ---- Referencias 21k y ResNet-50 (predicciones ya en disco) ----
    y_true_21k, y_pred_21k = _load_ref_preds(VIT21K_ENSEMBLE_NPZ)
    y_true_rn, y_pred_rn = _load_ref_preds(RESNET_ENSEMBLE_NPZ)
    # Sanidad: el holdout debe ser el mismo conjunto/orden.
    same_order_21k = bool(np.array_equal(y_true, y_true_21k))
    same_order_rn = bool(np.array_equal(y_true, y_true_rn))
    if not same_order_21k:
        print("!! AVISO: y_true del 1k difiere del 21k; revisa el holdout.")

    ci_21k = bootstrap_macro_f1(y_true_21k, y_pred_21k)
    f1_21k = ci_21k["point_estimate"]
    f1_1k = ci_1k["point_estimate"]

    # ---- 3) McNemar ----
    mcn_1k_vs_21k = mcnemar_test(y_true, y_pred_1k, y_pred_21k)
    mcn_1k_vs_rn = mcnemar_test(y_true, y_pred_1k, y_pred_rn)

    # ---- 4) F1 por clase comparado (foco Nematode y Healthy) ----
    met_1k = M.compute_metrics(y_true, y_pred_1k)
    met_21k = M.compute_metrics(y_true_21k, y_pred_21k)
    per_class = {}
    for cls in C.CLASSES:
        a = met_21k["per_class_f1"][cls]
        b = met_1k["per_class_f1"][cls]
        per_class[cls] = {
            "f1_21k": a, "f1_1k": b, "delta_pts": round((b - a) * 100, 2),
        }

    delta_pts = round((f1_1k - f1_21k) * 100, 2)
    summary = {
        "descripcion": (
            "Comparacion ViT preentrenamiento 21k (default timm, augreg2_in21k_ft_in1k) "
            "vs 1k-only (augreg_in1k) bajo el MISMO protocolo (5 folds x 3 seeds, "
            "mismos best_params Optuna). Solo cambia el tag de pesos de partida."),
        "holdout_n": int(len(df_test)),
        "vit_21k": {
            "model_name_timm": "vit_base_patch16_224 (default = augreg2_in21k_ft_in1k)",
            "pretrain": "ImageNet-21k -> ft 1k",
            "macro_f1": f1_21k,
            "macro_f1_reportado_doc": VIT21K_REPORTED_MACRO_F1,
            "ci95": [ci_21k["ci_low"], ci_21k["ci_high"]],
            "source_npz": VIT21K_ENSEMBLE_NPZ,
        },
        "vit_1k": {
            "model_name_timm": VIT1K_MODEL_NAME,
            "pretrain": "ImageNet-1k only (augreg)",
            "macro_f1": f1_1k,
            "ci95": [ci_1k["ci_low"], ci_1k["ci_high"]],
            "source_npz": "vit1k_ensemble.npz",
            "n_checkpoints": ens["n_models"],
        },
        "delta_macro_f1_pts": delta_pts,   # 1k - 21k (negativo = cae)
        "mcnemar_1k_vs_21k": {
            "pvalue": mcn_1k_vs_21k["pvalue"],
            "statistic": mcn_1k_vs_21k["statistic"],
            "n_disagree": mcn_1k_vs_21k["n_disagree"],
            "exact": mcn_1k_vs_21k["exact"],
            "significativo_0.05": bool(mcn_1k_vs_21k["pvalue"] < 0.05),
        },
        "mcnemar_1k_vs_resnet50": {
            "pvalue": mcn_1k_vs_rn["pvalue"],
            "statistic": mcn_1k_vs_rn["statistic"],
            "n_disagree": mcn_1k_vs_rn["n_disagree"],
            "exact": mcn_1k_vs_rn["exact"],
            "significativo_0.05": bool(mcn_1k_vs_rn["pvalue"] < 0.05),
        },
        "f1_por_clase": per_class,
        "sanidad": {
            "holdout_mismo_orden_que_21k": same_order_21k,
            "holdout_mismo_orden_que_resnet50": same_order_rn,
        },
    }
    out_json = C.RESULTS_DIR / "vit1k_vs_vit21k.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[OK] {out_json}")
    print(f"    macro F1  21k={f1_21k:.4f}  1k={f1_1k:.4f}  delta={delta_pts:+.2f} pts")
    print(f"    McNemar 1k vs 21k: p={mcn_1k_vs_21k['pvalue']:.4g} "
          f"(sig={summary['mcnemar_1k_vs_21k']['significativo_0.05']})")
    print(f"    Nematode  21k={per_class['Nematode']['f1_21k']:.4f} "
          f"1k={per_class['Nematode']['f1_1k']:.4f} "
          f"({per_class['Nematode']['delta_pts']:+.2f} pts)")
    print(f"    Healthy   21k={per_class['Healthy']['f1_21k']:.4f} "
          f"1k={per_class['Healthy']['f1_1k']:.4f} "
          f"({per_class['Healthy']['delta_pts']:+.2f} pts)")
    return summary


# =========================================================================== #
# CLI
# =========================================================================== #
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["train", "eval", "all"],
                   help="train (15 runs GPU) | eval (ensemble+CI+McNemar) | all")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="subset de folds (default los 5)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="subset de seeds (default 42 137 2026)")
    p.add_argument("--smoke", action="store_true",
                   help="train: subset chico + 2 epocas (test rapido)")
    p.add_argument("--resume", action="store_true",
                   help="omite (fold,seed) ya registrados en vit1k_master.csv")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.cmd in ("train", "all"):
        run_train(folds=args.folds, seeds=args.seeds, smoke=args.smoke,
                  resume=args.resume)
    if args.cmd in ("eval", "all"):
        run_eval()


if __name__ == "__main__":
    main()
