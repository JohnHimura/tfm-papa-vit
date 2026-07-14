"""
Loop de entrenamiento con early stopping, AMP, MLflow y captura completa.

Cambios v2 (Fase 0):
- Captura probabilidades softmax (y_prob) ademas de y_pred argmax.
- Snapshot del entorno (versiones + GPU) al inicio de cada run.
- Hardware/env logged a MLflow.
- Class weights logueados (vector + valores por clase).
- Tags MLflow: phase, model_family, tuning_status.
- Total run time + train time + eval time.
- Confusion matrix por run.
- Early stopping callback opcional (para Optuna pruner).

Uso desde linea de comandos:
    python -m src.train --model mobilenetv2_100 --fold 0 --seed 42
    python -m src.train --model vit_base_patch16_224 --fold 0 --seed 42 --phase final
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import mlflow

from . import config as C
from . import dataset as D
from . import metrics as M
from .env_info import save_env_snapshot
from .losses import build_loss
from .models import build_model, split_param_groups
from .seed_utils import seed_everything
from .transforms import train_transforms, eval_transforms, get_model_norm

# --------------------------- Familias de modelo (para tags) -----------------
MODEL_FAMILY = {
    "mobilenetv2_100": "CNN",
    "resnet50": "CNN",
    "efficientnet_b0": "CNN",
    "vit_base_patch16_224": "Transformer",
    "vit_base_patch16_224.augreg_in1k": "Transformer",
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loaders(df_train: pd.DataFrame, df_val: pd.DataFrame,
                  cfg, seed: int) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)
    # Normalizacion CORRECTA segun los pesos del modelo que se entrena
    # (ImageNet para las CNN; (0.5,0.5,0.5) para el ViT). normfix 29/06/2026.
    mean, std = get_model_norm(cfg.model_name)
    train_ds = D.PotatoLeafDataset(df_train,
                                   transform=train_transforms(mean=mean, std=std))
    val_ds = D.PotatoLeafDataset(df_val,
                                 transform=eval_transforms(mean=mean, std=std))

    # Oversampling (brazo OE4): si use_weighted_sampler=True usamos un
    # WeightedRandomSampler (peso por-muestra = 1/freq(clase)). El sampler y
    # shuffle son mutuamente excluyentes en PyTorch, por lo que shuffle=False
    # cuando hay sampler. Por defecto (use_weighted_sampler=False) el camino
    # original queda intacto: sampler=None, shuffle=True.
    sampler = None
    if getattr(cfg, "use_weighted_sampler", False):
        sampler = D.make_weighted_sampler(df_train, generator=g)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0, generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evalua el modelo y devuelve (loss_avg, y_true, y_pred, y_prob).

    y_prob es un array (N, NUM_CLASSES) con probabilidades softmax.
    """
    model.eval()
    losses, probs, gts = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        prob = F.softmax(logits, dim=1)
        losses.append(loss.item() * x.size(0))
        probs.append(prob.cpu().numpy())
        gts.append(y.cpu().numpy())
    n = sum(len(p) for p in probs)
    y_prob = np.concatenate(probs)
    y_pred = y_prob.argmax(axis=1)
    y_true = np.concatenate(gts)
    return sum(losses) / n, y_true, y_pred, y_prob


def train_one_fold(model_key: str, fold: int, seed: int,
                   epochs_override: int | None = None,
                   smoke: bool = False,
                   phase: str = "baseline",
                   override_cfg: dict | None = None,
                   epoch_callback: Callable[[int, dict], bool] | None = None,
                   run_name_override: str | None = None) -> dict:
    """Entrena un modelo en un fold/seed y devuelve metricas finales.

    Args:
        model_key: clave en C.MODELS_E2 o C.MODELS_OPCIONAL.
        fold: indice del pliegue (0 .. N_FOLDS-1).
        seed: semilla.
        epochs_override: si no es None, sobreescribe cfg.epochs.
        smoke: si True, usa subset chico y pocas epocas (test rapido).
        phase: tag de MLflow ('baseline' | 'optuna' | 'final').
        override_cfg: dict de hiperparametros que sobreescribe los defaults
            (usado por Optuna para inyectar el trial actual).
        epoch_callback: callable(epoch, metrics) -> bool. Si devuelve True,
            se interrumpe el entrenamiento (usado por Optuna pruner).
        run_name_override: si no es None, fuerza el run_name (y por tanto el
            nombre del checkpoint `<run_name>_best.pt` y la carpeta
            results/runs/<run_name>). Permite a scripts derivados (p.ej.
            run_vit1k) escribir a rutas con prefijo propio sin colisionar con
            los artefactos 'final__' existentes. Por defecto None preserva el
            naming original.
    """
    # ----------------------- Configuracion del run -----------------------
    if model_key in C.MODELS_E2:
        cfg = C.MODELS_E2[model_key]
    elif model_key in C.MODELS_OPCIONAL:
        cfg = C.MODELS_OPCIONAL[model_key]
    else:
        raise KeyError(f"Modelo desconocido: {model_key}")

    cfg_dict = asdict(cfg)
    if epochs_override is not None:
        cfg_dict["epochs"] = epochs_override
    if override_cfg:
        cfg_dict.update(override_cfg)
    cfg = type(cfg)(**cfg_dict)

    seed_everything(seed, deterministic=C.PIN_DETERMINISM)
    device = _device()

    t_start = time.time()

    # ----------------------------- Splits -------------------------------
    df = D.load_or_build_splits()
    df_cv = df[df["split_holdout"] == "train_cv"].reset_index(drop=True)
    folds = list(D.kfold_indices(df_cv, n_folds=C.N_FOLDS, seed=42))
    tr_idx, val_idx = folds[fold]
    df_train = df_cv.iloc[tr_idx].reset_index(drop=True)
    df_val = df_cv.iloc[val_idx].reset_index(drop=True)

    if smoke:
        df_train = df_train.sample(min(200, len(df_train)), random_state=seed).reset_index(drop=True)
        df_val = df_val.sample(min(100, len(df_val)), random_state=seed).reset_index(drop=True)

    train_loader, val_loader = _make_loaders(df_train, df_val, cfg, seed)

    # ----------------------------- Modelo -------------------------------
    model = build_model(cfg.model_name, pretrained=cfg.pretrained,
                        num_classes=C.NUM_CLASSES).to(device)
    param_groups = split_param_groups(model, cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = len(train_loader) * cfg.warmup_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[cfg.lr_backbone, cfg.lr_head],
        total_steps=total_steps,
        pct_start=max(warmup_steps / max(total_steps, 1), 0.05),
    )

    # Anti doble-conteo: el oversampling (WeightedRandomSampler) ya re-balancea
    # la distribucion de clases via muestreo. Combinarlo con class_weights en la
    # loss aplicaria la correccion de desbalance DOS veces (sobre-pondera la
    # minoritaria). La decision documentada es: oversampling SIN class-weights.
    use_cw = cfg.use_class_weights
    if getattr(cfg, "use_weighted_sampler", False) and use_cw:
        print("  [OE4] use_weighted_sampler=True -> desactivo class_weights en "
              "la loss para evitar doble conteo del desbalance.")
        use_cw = False
    class_weights = D.compute_class_weights(df_train) if use_cw else None
    criterion = build_loss(cfg.use_focal_loss, class_weights, cfg.label_smoothing, device)
    scaler = GradScaler("cuda", enabled=cfg.mixed_precision)

    # --------------------------- MLflow setup ----------------------------
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    suffix = "_smoke" if smoke else ""
    if run_name_override is not None:
        run_name = f"{run_name_override}{suffix}"
    else:
        run_name = f"{cfg.model_name}_fold{fold}_seed{seed}{suffix}"
        if phase != "baseline":
            run_name = f"{phase}__{run_name}"

    history: list[dict] = []
    best_macro_f1, best_epoch, patience = -1.0, -1, 0
    train_time_total = 0.0

    with mlflow.start_run(run_name=run_name) as run:
        # Tags para filtrar runs en MLflow UI
        mlflow.set_tags({
            "phase": phase,
            "model_family": MODEL_FAMILY.get(cfg.model_name, "Other"),
            "model": cfg.model_name,
            "fold": str(fold),
            "seed": str(seed),
            "smoke": str(smoke),
        })

        # Hiperparams completos
        mlflow.log_params({
            "model": cfg.model_name, "fold": fold, "seed": seed,
            "epochs": cfg.epochs, "batch_size": cfg.batch_size,
            "lr_head": cfg.lr_head, "lr_backbone": cfg.lr_backbone,
            "weight_decay": cfg.weight_decay,
            "label_smoothing": cfg.label_smoothing,
            "warmup_epochs": cfg.warmup_epochs,
            "early_stop_patience": cfg.early_stop_patience,
            "use_class_weights": use_cw,
            "use_focal_loss": cfg.use_focal_loss,
            "use_weighted_sampler": getattr(cfg, "use_weighted_sampler", False),
            "mixed_precision": cfg.mixed_precision,
            "pretrained": cfg.pretrained,
            "n_train": len(df_train), "n_val": len(df_val),
            "n_classes": C.NUM_CLASSES,
            "img_size": C.IMG_SIZE,
        })
        # Class weights por clase
        if class_weights is not None:
            for i, w in enumerate(class_weights.cpu().numpy()):
                mlflow.log_param(f"class_weight_{C.IDX_TO_CLASS[i]}", float(w))

        # Snapshot del entorno
        env_path = C.RESULTS_DIR / "runs" / run_name / "env.json"
        env_snap = save_env_snapshot(env_path)
        mlflow.log_artifact(str(env_path))
        if env_snap.get("gpu"):
            mlflow.log_param("gpu_name", env_snap["gpu"]["name"])
            mlflow.log_param("gpu_vram_gb", env_snap["gpu"]["total_vram_gb"])

        # ----------------------------- Loop --------------------------------
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            model.train()
            running = 0.0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=cfg.mixed_precision):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += loss.item() * x.size(0)
            train_loss = running / len(train_loader.dataset)
            train_time_epoch = time.time() - t0
            train_time_total += train_time_epoch

            t_eval = time.time()
            val_loss, y_true, y_pred, y_prob = _evaluate(model, val_loader, criterion, device)
            eval_time = time.time() - t_eval
            metrics = M.compute_metrics(y_true, y_pred)

            metrics_log = {
                "train_loss": train_loss, "val_loss": val_loss,
                "val_macro_f1": metrics["macro_f1"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_acc": metrics["balanced_accuracy"],
                "epoch_time_s": train_time_epoch + eval_time,
                "train_time_s": train_time_epoch,
                "eval_time_s": eval_time,
            }
            mlflow.log_metrics(metrics_log, step=epoch)
            history.append({"epoch": epoch, **metrics_log})
            print(f"[{run_name}] ep {epoch:02d}/{cfg.epochs} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} "
                  f"({metrics_log['epoch_time_s']:.1f}s)")

            # Best checkpoint
            improved = metrics["macro_f1"] > best_macro_f1 + 1e-4
            if improved:
                best_macro_f1 = metrics["macro_f1"]
                best_epoch = epoch
                patience = 0
                ckpt = C.MODELS_DIR / f"{run_name}_best.pt"
                torch.save({"model_state": model.state_dict(),
                            "epoch": epoch, "metrics": metrics,
                            "config": cfg_dict}, ckpt)
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    print(f"  -> early stop @ epoch {epoch}")
                    break

            # Optuna callback (pruner)
            if epoch_callback is not None:
                if epoch_callback(epoch, metrics):
                    print(f"  -> pruned by callback @ epoch {epoch}")
                    break

        # --------------------- Eval final con best ckpt ---------------------
        ckpt = torch.load(C.MODELS_DIR / f"{run_name}_best.pt", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        val_loss, y_true, y_pred, y_prob = _evaluate(model, val_loader, criterion, device)
        final_metrics = M.compute_metrics(y_true, y_pred)
        cm = M.confusion(y_true, y_pred)

        # Total run time
        total_run_time = time.time() - t_start

        mlflow.log_metric("best_val_macro_f1", best_macro_f1)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("total_run_time_s", total_run_time)
        mlflow.log_metric("total_train_time_s", train_time_total)
        for cls, v in final_metrics["per_class_f1"].items():
            mlflow.log_metric(f"final_f1_{cls}", v)

        # ----------------------- Persistencia local ------------------------
        out_dir = C.RESULTS_DIR / "runs" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        # NPZ con y_true, y_pred y y_prob (matriz Nx7)
        np.savez(out_dir / "predictions.npz",
                 y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        np.save(out_dir / "confusion.npy", cm)
        (out_dir / "final_metrics.json").write_text(
            json.dumps(final_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out_dir / "config_used.json").write_text(
            json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        for fname in ("history.csv", "final_metrics.json",
                      "config_used.json", "predictions.npz"):
            mlflow.log_artifact(str(out_dir / fname))

        return {
            "run_name": run_name,
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "final_metrics": final_metrics,
            "out_dir": str(out_dir),
            "run_id": run.info.run_id,
            "total_run_time_s": total_run_time,
            "phase": phase,
        }


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--phase", default="baseline",
                   choices=["baseline", "optuna", "final"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    epochs = 3 if args.smoke else args.epochs
    result = train_one_fold(args.model, args.fold, args.seed,
                            epochs_override=epochs, smoke=args.smoke,
                            phase=args.phase)
    print(json.dumps(result, indent=2, ensure_ascii=False))
