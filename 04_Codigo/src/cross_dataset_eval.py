"""
Validacion externa (cross-dataset) — Gap #4 del estado del arte.

Evalua los modelos YA ENTRENADOS sobre el dataset de Central Java (7 clases,
campo no controlado) contra un dataset DISTINTO e INDEPENDIENTE: PlantVillage
papa (condiciones de laboratorio, fondo uniforme). Mide cuanto generaliza cada
arquitectura fuera de su distribucion de entrenamiento (domain shift campo->lab)
SIN reentrenar nada: inferencia pura con los checkpoints del CV final.

Solo 3 de las 7 clases del modelo tienen equivalente en PlantVillage. El mapeo
es etiologico (por agente causal), coherente con la taxonomia del dataset propio:

    Potato___Early_blight  (Alternaria solani, hongo)      -> Fungi      (idx 1)
    Potato___Late_blight   (Phytophthora infestans, "gota")-> Phytopthora(idx 5)
    Potato___healthy       (sano)                          -> Healthy    (idx 2)

Se reportan DOS modos de decision:
  * unrestricted: argmax sobre las 7 clases. Una prediccion que caiga en una de
    las 4 clases ausentes (Bacteria/Nematode/Pest/Virus) cuenta como error y se
    contabiliza como "fuga fuera de taxonomia". Es la medida honesta y dura de
    generalizacion.
  * restricted: argmax restringido a los 3 logits presentes (Fungi/Healthy/
    Phytopthora). Simula un despliegue donde solo esas 3 clases son posibles.
    Es la medida mas indulgente.

Macro F1 se calcula SOLO sobre las 3 clases presentes (labels=[1,2,5]); incluir
las 4 clases sin soporte hundiria artificialmente la macro.

Salida:
    results/cross_dataset/cross_dataset_plantvillage.json
    results/cross_dataset/cross_dataset_summary.csv
    results/cross_dataset/predictions/<modelo>_ensemble.npz
    results/cross_dataset/figuras/cm_<modelo>_{unrestricted,restricted}.png

Uso:
    .venv/Scripts/python.exe -m src.cross_dataset_eval
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader

from . import config as C
from .dataset import PotatoLeafDataset
from .models import build_model
from .transforms import eval_transforms, get_model_norm

# =============================================================================
# Dataset externo y mapeo de etiquetas
# =============================================================================
PV_DIR = (C.ROOT / "02_Estado_del_Arte" / "datasets_referencia"
          / "PlantVillage" / "PlantVillage")

# carpeta PlantVillage -> indice de clase en la taxonomia del modelo (config.CLASSES)
PV_MAP = {
    "Potato___Early_blight": 1,   # Alternaria solani (hongo)        -> Fungi
    "Potato___Late_blight":  5,   # Phytophthora infestans ("gota")  -> Phytopthora
    "Potato___healthy":      2,   # sano                             -> Healthy
}
PRESENT = sorted(set(PV_MAP.values()))          # [1, 2, 5]
MODELS = ["vit_base_patch16_224", "resnet50", "efficientnet_b0", "mobilenetv2_100"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def build_pv_catalog() -> pd.DataFrame:
    """Catalogo [path, label, class_name, src_folder] de PlantVillage papa.

    Dedup por ruta resuelta en minusculas (Windows es case-insensitive), igual
    que dataset.build_catalog para el dataset propio.
    """
    seen: set[str] = set()
    rows = []
    for folder, label in PV_MAP.items():
        d = PV_DIR / folder
        if not d.exists():
            raise FileNotFoundError(f"No existe carpeta PlantVillage: {d}")
        for p in d.iterdir():
            if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
                continue
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"path": str(p), "label": label,
                         "class_name": C.IDX_TO_CLASS[label], "src_folder": folder})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Sin imagenes en {PV_DIR}")
    return df


# =============================================================================
# Inferencia ensemble (mismo protocolo que holdout_eval)
# =============================================================================
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _predict_probs(model: torch.nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        gts.append(y.numpy())
    return np.concatenate(gts), np.concatenate(probs)


def ensemble_probs(model_key: str, df: pd.DataFrame, device: torch.device,
                   batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, int]:
    """Promedia softmax de los 15 checkpoints final del modelo.

    Loader con la normalizacion CORRECTA de `model_key` (normfix): el ViT ve
    (0.5,0.5,0.5); las CNN ven ImageNet. Identico a holdout_eval._holdout_loader.
    """
    mean, std = get_model_norm(model_key)
    ds = PotatoLeafDataset(df, transform=eval_transforms(mean=mean, std=std))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    ckpts = sorted(C.MODELS_DIR.glob(f"final__{model_key}_fold*_seed*_best.pt"))
    if not ckpts:
        raise RuntimeError(f"Sin checkpoints final para {model_key}")
    probs_acc = None
    y_true_ref = None
    for ck in ckpts:
        model = build_model(model_key, pretrained=False,
                            num_classes=C.NUM_CLASSES).to(device)
        sd = torch.load(ck, weights_only=False)
        model.load_state_dict(sd["model_state"])
        y_true, y_prob = _predict_probs(model, loader, device)
        probs_acc = y_prob if probs_acc is None else probs_acc + y_prob
        y_true_ref = y_true if y_true_ref is None else y_true_ref
        del model
        torch.cuda.empty_cache()
    return y_true_ref, probs_acc / len(ckpts), len(ckpts)


def restricted_pred(y_prob: np.ndarray) -> np.ndarray:
    """argmax restringido a los logits de las 3 clases presentes."""
    sub = y_prob[:, PRESENT]                    # (N, 3)
    return np.asarray(PRESENT)[sub.argmax(axis=1)]


# =============================================================================
# Metricas (solo sobre las 3 clases presentes)
# =============================================================================
def cross_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    f1s = f1_score(y_true, y_pred, labels=PRESENT, average=None, zero_division=0)
    recs = recall_score(y_true, y_pred, labels=PRESENT, average=None, zero_division=0)
    precs = precision_score(y_true, y_pred, labels=PRESENT, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_present": float(f1_score(y_true, y_pred, labels=PRESENT,
                                           average="macro", zero_division=0)),
        "per_class_f1": {C.IDX_TO_CLASS[c]: float(v) for c, v in zip(PRESENT, f1s)},
        "per_class_recall": {C.IDX_TO_CLASS[c]: float(v) for c, v in zip(PRESENT, recs)},
        "per_class_precision": {C.IDX_TO_CLASS[c]: float(v) for c, v in zip(PRESENT, precs)},
        "out_of_taxonomy_rate": float(np.mean(~np.isin(y_pred, PRESENT))),
    }


# =============================================================================
# Figuras
# =============================================================================
def plot_cm_unrestricted(y_true, y_pred, out_png: Path, title: str) -> None:
    """Matriz 3 filas (real) x 7 columnas (predicho) para ver la fuga de errores."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))
    cm_sub = cm[np.ix_(PRESENT, list(range(C.NUM_CLASSES)))].astype(int)   # 3 x 7
    with np.errstate(all="ignore"):
        cm_norm = cm_sub / cm_sub.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    sns.heatmap(cm_norm, annot=cm_sub, fmt="d", cmap="magma", vmin=0, vmax=1,
                xticklabels=C.CLASSES,
                yticklabels=[C.IDX_TO_CLASS[r] for r in PRESENT],
                cbar_kws={"label": "Proporcion por fila (recall)"}, ax=ax)
    ax.set_xlabel("Predicho (las 7 clases del modelo)")
    ax.set_ylabel("Real (PlantVillage)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_cm_restricted(y_true, y_pred, out_png: Path, title: str) -> None:
    """Matriz 3x3 restringida a las clases presentes."""
    cm = confusion_matrix(y_true, y_pred, labels=PRESENT).astype(int)
    with np.errstate(all="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    names = [C.IDX_TO_CLASS[c] for c in PRESENT]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="magma", vmin=0, vmax=1,
                xticklabels=names, yticklabels=names,
                cbar_kws={"label": "Proporcion por fila (recall)"}, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = _device()

    out_dir = C.RESULTS_DIR / "cross_dataset"
    pred_dir = out_dir / "predictions"
    fig_dir = out_dir / "figuras"
    pred_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = build_pv_catalog()
    dist = {k: int(v) for k, v in df["class_name"].value_counts().items()}
    print(f"[PlantVillage papa] {len(df)} imgs | por clase (mapeada): {dist}")
    print(f"[Device] {device} | ensemble de 15 checkpoints por modelo\n")

    out = {
        "dataset_externo": "PlantVillage potato (laboratorio, fondo uniforme)",
        "dataset_entrenamiento": "Central Java 7 clases (campo no controlado)",
        "n_images": int(len(df)),
        "distribucion": dist,
        "mapeo_etiologico": {k: C.IDX_TO_CLASS[v] for k, v in PV_MAP.items()},
        "clases_presentes": [C.IDX_TO_CLASS[c] for c in PRESENT],
        "protocolo": ("Inferencia sin reentrenar. Ensemble = promedio softmax de "
                      "15 checkpoints (5 folds x 3 seeds). Normalizacion por-modelo."),
        "models": {},
    }
    summary_rows = []

    for mk in MODELS:
        print(f">> {mk} ...")
        y_true, y_prob, n = ensemble_probs(mk, df, device)
        y_pred_u = y_prob.argmax(axis=1)
        y_pred_r = restricted_pred(y_prob)
        mu = cross_metrics(y_true, y_pred_u)
        mr = cross_metrics(y_true, y_pred_r)

        np.savez(pred_dir / f"{mk}_ensemble.npz", y_true=y_true, y_prob=y_prob,
                 y_pred_unrestricted=y_pred_u, y_pred_restricted=y_pred_r)
        plot_cm_unrestricted(y_true, y_pred_u, fig_dir / f"cm_{mk}_unrestricted.png",
                             f"{mk} — cross-dataset PlantVillage (7-way argmax)")
        plot_cm_restricted(y_true, y_pred_r, fig_dir / f"cm_{mk}_restricted.png",
                           f"{mk} — cross-dataset PlantVillage (3-way restringido)")

        out["models"][mk] = {"n_ensemble": n, "unrestricted": mu, "restricted": mr}
        summary_rows.append({
            "model": mk,
            "macroF1_unrestricted": round(mu["macro_f1_present"], 4),
            "acc_unrestricted": round(mu["accuracy"], 4),
            "out_of_taxonomy_rate": round(mu["out_of_taxonomy_rate"], 4),
            "macroF1_restricted": round(mr["macro_f1_present"], 4),
            "acc_restricted": round(mr["accuracy"], 4),
            "recall_Fungi_u": round(mu["per_class_recall"]["Fungi"], 4),
            "recall_Healthy_u": round(mu["per_class_recall"]["Healthy"], 4),
            "recall_Phytopthora_u": round(mu["per_class_recall"]["Phytopthora"], 4),
        })
        print(f"   unrestricted: macroF1={mu['macro_f1_present']:.4f} "
              f"acc={mu['accuracy']:.4f} fuga_fuera_taxonomia={mu['out_of_taxonomy_rate']:.4f}")
        print(f"   restricted  : macroF1={mr['macro_f1_present']:.4f} "
              f"acc={mr['accuracy']:.4f}")
        print(f"   recall/clase (unrestr): Fungi={mu['per_class_recall']['Fungi']:.3f} "
              f"Healthy={mu['per_class_recall']['Healthy']:.3f} "
              f"Phytopthora={mu['per_class_recall']['Phytopthora']:.3f}\n")

    (out_dir / "cross_dataset_plantvillage.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "cross_dataset_summary.csv", index=False)

    print("=" * 60)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 60)
    print(f"\n[OK] JSON  -> {out_dir / 'cross_dataset_plantvillage.json'}")
    print(f"[OK] CSV   -> {out_dir / 'cross_dataset_summary.csv'}")
    print(f"[OK] figs  -> {fig_dir}")


if __name__ == "__main__":
    main()
