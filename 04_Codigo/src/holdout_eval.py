"""
Fase 3 — Evaluacion sobre el holdout 15% (462 imagenes nunca vistas).

Para cada modelo:
  (a) Reporta el desempeño del MEJOR checkpoint individual sobre el holdout.
  (b) Reporta el desempeño del ENSEMBLE: promedio de probabilidades softmax
      de las 15 replicas (5 folds x 3 seeds) del CV final.

Salida:
    results/holdout_eval.json
    results/holdout_predictions/<modelo>_<modo>.npz   (y_true, y_pred, y_prob)
    figures/cm_holdout_<modelo>.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as D
from . import metrics as M
from .models import build_model
from .transforms import eval_transforms, get_model_norm

DEFAULT_MODELS = ["mobilenetv2_100", "resnet50", "efficientnet_b0",
                  "vit_base_patch16_224"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _holdout_df() -> pd.DataFrame:
    df = D.load_or_build_splits()
    return df[df["split_holdout"] == "test"].reset_index(drop=True)


def _holdout_loader(model_key: str, df_test: pd.DataFrame | None = None,
                    batch_size: int = 32) -> tuple[DataLoader, pd.DataFrame]:
    """Loader del holdout con la normalizacion CORRECTA para `model_key`.

    CRITICO (normfix 29/06/2026): cada modelo recibe su input normalizado con
    las estadisticas que esperan SUS pesos. NO se comparte un tensor
    pre-normalizado entre una CNN (ImageNet) y el ViT ((0.5,0.5,0.5)) — por eso
    el loader se construye por-modelo, no una sola vez para todos.
    """
    if df_test is None:
        df_test = _holdout_df()
    mean, std = get_model_norm(model_key)
    ds = D.PotatoLeafDataset(df_test,
                             transform=eval_transforms(mean=mean, std=std))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader, df_test


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y_true, y_prob) para todo el loader."""
    model.eval()
    probs, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(prob)
        gts.append(y.numpy())
    return np.concatenate(gts), np.concatenate(probs)


def find_final_checkpoints(model_key: str) -> list[Path]:
    """Lista los checkpoints best.pt de la phase 'final' para este modelo."""
    return sorted(C.MODELS_DIR.glob(f"final__{model_key}_fold*_seed*_best.pt"))


def evaluate_individual(model_key: str, df_test: pd.DataFrame,
                        device: torch.device) -> dict:
    """Evalua el mejor checkpoint individual segun macro_f1 en CV.

    Lo encontramos releyendo final_master.csv para el row con mejor
    best_macro_f1 de este modelo. El loader se construye con la normalizacion
    correcta de `model_key` (normfix).
    """
    loader, _ = _holdout_loader(model_key, df_test)
    df = pd.read_csv(C.RESULTS_DIR / "final_master.csv")
    sub = df[df["model"] == model_key].sort_values("best_macro_f1", ascending=False)
    if sub.empty:
        raise RuntimeError(f"No hay runs final para {model_key}")
    best = sub.iloc[0]
    fold, seed = int(best["fold"]), int(best["seed"])
    ckpt_name = f"final__{model_key}_fold{fold}_seed{seed}_best.pt"
    ckpt_path = C.MODELS_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    model = build_model(model_key, pretrained=False, num_classes=C.NUM_CLASSES).to(device)
    sd = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(sd["model_state"])
    y_true, y_prob = _predict(model, loader, device)
    y_pred = y_prob.argmax(axis=1)
    metrics = M.compute_metrics(y_true, y_pred)
    return {
        "mode": "best_individual",
        "fold": fold, "seed": seed, "checkpoint": ckpt_name,
        **metrics,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }


def evaluate_ensemble(model_key: str, df_test: pd.DataFrame,
                      device: torch.device) -> dict:
    """Promedia probabilidades de TODOS los checkpoints final del modelo.

    El loader se construye con la normalizacion correcta de `model_key`
    (normfix): cada modelo del ensemble ve su input bien normalizado.
    """
    loader, _ = _holdout_loader(model_key, df_test)
    ckpts = find_final_checkpoints(model_key)
    if not ckpts:
        raise RuntimeError(f"No hay checkpoints final para {model_key}")
    probs_acc = None
    y_true_ref = None
    for ck in ckpts:
        model = build_model(model_key, pretrained=False, num_classes=C.NUM_CLASSES).to(device)
        sd = torch.load(ck, weights_only=False)
        model.load_state_dict(sd["model_state"])
        y_true, y_prob = _predict(model, loader, device)
        if probs_acc is None:
            probs_acc = y_prob
            y_true_ref = y_true
        else:
            probs_acc = probs_acc + y_prob
        del model
        torch.cuda.empty_cache()
    y_prob_avg = probs_acc / len(ckpts)
    y_pred = y_prob_avg.argmax(axis=1)
    metrics = M.compute_metrics(y_true_ref, y_pred)
    return {
        "mode": "ensemble_15",
        "n_models": len(ckpts),
        **metrics,
        "y_true": y_true_ref, "y_pred": y_pred, "y_prob": y_prob_avg,
    }


def main():
    device = _device()
    df_test = _holdout_df()
    pred_dir = C.RESULTS_DIR / "holdout_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    out = {"n_holdout": len(df_test), "models": {}}
    for m in DEFAULT_MODELS:
        out["models"][m] = {}
        try:
            ind = evaluate_individual(m, df_test, device)
            np.savez(pred_dir / f"{m}_individual.npz",
                     y_true=ind.pop("y_true"), y_pred=ind.pop("y_pred"),
                     y_prob=ind.pop("y_prob"))
            out["models"][m]["individual"] = ind
            print(f"[OK] {m} individual: macro_f1={ind['macro_f1']:.4f}")
        except Exception as e:
            print(f"!! {m} individual error: {e}")
            out["models"][m]["individual"] = {"error": str(e)}
        try:
            ens = evaluate_ensemble(m, df_test, device)
            np.savez(pred_dir / f"{m}_ensemble.npz",
                     y_true=ens.pop("y_true"), y_pred=ens.pop("y_pred"),
                     y_prob=ens.pop("y_prob"))
            out["models"][m]["ensemble"] = ens
            print(f"[OK] {m} ensemble: macro_f1={ens['macro_f1']:.4f}")
        except Exception as e:
            print(f"!! {m} ensemble error: {e}")
            out["models"][m]["ensemble"] = {"error": str(e)}

    out_path = C.RESULTS_DIR / "holdout_eval.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print(f"\n[OK] holdout_eval.json -> {out_path}")


if __name__ == "__main__":
    main()
