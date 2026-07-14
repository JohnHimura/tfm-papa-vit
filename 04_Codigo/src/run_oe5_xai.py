"""
OE5 — Orquestador del estudio de explicabilidad cuantitativa.

Reproduce las Tablas 15 y 16 del documento con datos REALES.

Pipeline (3 etapas desacopladas):
  Etapa A  `generate`  -> genera los mapas Grad-CAM (ResNet-50) y Attention
                          Rollout (ViT-Base/16) para la muestra de 140 imagenes
                          (20/clase) del holdout, reproducible (seed 2026).
                          Guarda overlays PNG + heatmap crudo .npy.
                          [requiere GPU/CPU + checkpoints — paso pesado]
  Etapa B  `metrics`   -> una vez existan las mascaras de lesion en
                          results/xai/masks/, calcula IoU y Pointing Game por
                          imagen, agrega global y por-clase, y escribe
                          results/oe5_metrics.csv (Tablas 15 y 16).
                          [barato — solo numpy]

La SELECCION de la muestra es deterministica (seed 2026), por lo que las dos
etapas operan exactamente sobre el mismo conjunto sin necesidad de re-ejecutar
la generacion.

Healthy se incluye en la muestra (20 img) para inspeccion, pero la FIDELIDAD
de las Tablas 15/16 se reporta sobre las 6 clases CON lesion (120 img); Healthy
no tiene lesion que anotar y se trata aparte (se reporta como N/A en IoU/PG).

Uso:
    python -m src.run_oe5_xai select         # escribe el manifiesto de muestra
    python -m src.run_oe5_xai generate --model resnet50 --variant best
    python -m src.run_oe5_xai generate --model vit_base_patch16_224 --variant best
    python -m src.run_oe5_xai metrics --threshold-method percentile --percentile 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import dataset as D
from . import xai
from .models import build_model
from .transforms import eval_transforms, get_model_norm

# -----------------------------------------------------------------------------
# Rutas OE5
# -----------------------------------------------------------------------------
XAI_DIR = C.RESULTS_DIR / "xai"
HEATMAP_DIR = XAI_DIR / "heatmaps"          # .npy crudos
OVERLAY_DIR = XAI_DIR / "overlays"          # .png para inspeccion
MASKS_DIR = XAI_DIR / "masks"               # mascaras de lesion (BLOQUEO HUMANO)
SAMPLE_MANIFEST = XAI_DIR / "oe5_sample_140.csv"
METRICS_CSV = C.RESULTS_DIR / "oe5_metrics.csv"

OE5_SEED = 2026
N_PER_CLASS = 20
# clases CON lesion para la fidelidad (Healthy se excluye del IoU/PG)
LESION_CLASSES = [c for c in C.CLASSES if c != "Healthy"]

RESNET_KEY = "resnet50"
VIT_KEY = "vit_base_patch16_224"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# ETAPA "select" — muestra reproducible de 140 imagenes (20/clase)
# =============================================================================
def select_sample(n_per_class: int = N_PER_CLASS, seed: int = OE5_SEED) -> pd.DataFrame:
    """Selecciona n_per_class imagenes por clase del holdout, reproducible.

    Si una clase tiene menos de n_per_class imagenes en el holdout, toma todas
    las disponibles (no rellena). Escribe el manifiesto en SAMPLE_MANIFEST.
    """
    df = D.load_or_build_splits()
    test = df[df["split_holdout"] == "test"].copy()
    rng = np.random.default_rng(seed)
    picked = []
    for cls in C.CLASSES:
        sub = test[test["class_name"] == cls].sort_values("rel_path")
        n = min(n_per_class, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        chosen = sub.iloc[np.sort(idx)].copy()
        chosen["idx_in_class"] = range(len(chosen))
        picked.append(chosen)
    out = pd.concat(picked, ignore_index=True)
    # naming estable: clase_idx (p.ej. Bacteria_00)
    out["sample_id"] = out.apply(
        lambda r: f"{r['class_name']}_{int(r['idx_in_class']):02d}", axis=1)
    out["has_lesion"] = out["class_name"].isin(LESION_CLASSES)
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "class_name", "label", "rel_path", "path",
            "idx_in_class", "has_lesion"]
    out[cols].to_csv(SAMPLE_MANIFEST, index=False)
    print(f"[OK] muestra de {len(out)} imagenes -> {SAMPLE_MANIFEST}")
    for cls in C.CLASSES:
        print(f"   {cls:12s}: {(out['class_name'] == cls).sum()}")
    return out


def load_sample() -> pd.DataFrame:
    if not SAMPLE_MANIFEST.exists():
        return select_sample()
    df = pd.read_csv(SAMPLE_MANIFEST)
    # re-resolver path por si se movio el dataset
    df["path"] = df["rel_path"].map(lambda r: str(C.DATA_DIR / r))
    return df


# =============================================================================
# ETAPA "generate" — mapas de saliencia (paso pesado, GPU)
# =============================================================================
def _load_ensemble_or_best(model_key: str, variant: str, device: torch.device):
    """Devuelve lista de modelos cargados.

    variant='best'     -> 1 modelo (mejor fold/seed segun final_master.csv).
    variant='ensemble' -> los 15 checkpoints final del modelo (promedio de mapas).
    """
    if variant == "best":
        fm = pd.read_csv(C.RESULTS_DIR / "final_master.csv")
        sub = fm[fm["model"] == model_key].sort_values("best_macro_f1", ascending=False)
        best = sub.iloc[0]
        fold, seed = int(best["fold"]), int(best["seed"])
        ck = C.MODELS_DIR / f"final__{model_key}_fold{fold}_seed{seed}_best.pt"
        paths = [ck]
    else:
        paths = sorted(C.MODELS_DIR.glob(f"final__{model_key}_fold*_seed*_best.pt"))
    models = []
    for p in paths:
        m = build_model(model_key, pretrained=False, num_classes=C.NUM_CLASSES).to(device)
        sd = torch.load(p, weights_only=False)
        m.load_state_dict(sd["model_state"])
        m.eval()
        models.append((p.name, m))
    return models


def _saliency_for(model_key: str, model: torch.nn.Module, x: torch.Tensor,
                  target_class: int) -> np.ndarray:
    if model_key == RESNET_KEY:
        return xai.grad_cam_resnet(model, x, target_class=target_class)
    elif model_key == VIT_KEY:
        return xai.attention_rollout_vit(model, x)
    raise ValueError(f"modelo no soportado para OE5: {model_key}")


def generate_maps(model_key: str, variant: str = "best",
                  limit: int | None = None) -> None:
    """Genera y guarda heatmaps (.npy) + overlays (.png) para la muestra.

    Para ensemble se promedian los mapas normalizados de los 15 checkpoints y
    se re-normaliza el promedio a [0,1].
    """
    from PIL import Image
    device = _device()
    sample = load_sample()
    if limit:
        sample = sample.head(limit)
    models = _load_ensemble_or_best(model_key, variant, device)
    # Normalizacion correcta por modelo (CNN=ImageNet, ViT=(0.5,0.5,0.5)). normfix.
    _mean, _std = get_model_norm(model_key)
    tfm = eval_transforms(mean=_mean, std=_std)

    sub_h = HEATMAP_DIR / model_key
    sub_o = OVERLAY_DIR / model_key
    sub_h.mkdir(parents=True, exist_ok=True)
    sub_o.mkdir(parents=True, exist_ok=True)

    print(f"[generate] {model_key} variant={variant} | {len(models)} ckpt(s) | "
          f"{len(sample)} imgs | device={device.type}")
    for _, row in sample.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        rgb = np.array(img.resize((C.IMG_SIZE, C.IMG_SIZE)))
        x = tfm(img).unsqueeze(0).to(device)
        target = int(row["label"])
        maps = []
        for _name, m in models:
            maps.append(_saliency_for(model_key, m, x, target))
        heat = np.mean(maps, axis=0)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
        sid = row["sample_id"]
        np.save(sub_h / f"{sid}.npy", heat.astype(np.float32))
        overlay = xai.overlay_heatmap(rgb, heat)
        Image.fromarray(overlay).save(sub_o / f"{sid}.png")
    print(f"[OK] heatmaps -> {sub_h}\n[OK] overlays -> {sub_o}")


# =============================================================================
# ETAPA "metrics" — IoU + Pointing Game -> Tablas 15/16 (requiere mascaras)
# =============================================================================
def _load_mask(sample_id: str, size: int = C.IMG_SIZE) -> np.ndarray | None:
    """Carga la mascara de lesion binaria para sample_id, o None si no existe.

    Formato esperado: results/xai/masks/<sample_id>.png (cualquier pixel >0 es
    lesion). Se redimensiona por vecino mas cercano al tamaño del heatmap.
    """
    from PIL import Image
    p = MASKS_DIR / f"{sample_id}.png"
    if not p.exists():
        return None
    m = Image.open(p).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


def compute_metrics(model_key: str, threshold_method: str = "percentile",
                    percentile: float = 80.0) -> pd.DataFrame:
    """Calcula IoU y PG por imagen para un modelo. Devuelve DataFrame largo.

    Solo procesa imagenes con mascara presente Y clase con lesion.
    """
    sample = load_sample()
    sub_h = HEATMAP_DIR / model_key
    rows = []
    n_missing_heat, n_missing_mask = 0, 0
    for _, r in sample.iterrows():
        if not bool(r["has_lesion"]):
            continue
        sid = r["sample_id"]
        hp = sub_h / f"{sid}.npy"
        if not hp.exists():
            n_missing_heat += 1
            continue
        mask = _load_mask(sid)
        if mask is None:
            n_missing_mask += 1
            continue
        heat = np.load(hp)
        sal_bin = xai.binarize_heatmap(heat, method=threshold_method,
                                       percentile=percentile)
        rows.append({
            "model": model_key,
            "sample_id": sid,
            "class_name": r["class_name"],
            "iou": xai.iou(sal_bin, mask),
            "pg": xai.pointing_game(heat, mask),
        })
    df = pd.DataFrame(rows)
    print(f"[metrics] {model_key}: {len(df)} imgs evaluadas | "
          f"sin heatmap={n_missing_heat} | sin mascara={n_missing_mask}")
    return df


def aggregate_and_write(threshold_method: str = "percentile",
                        percentile: float = 80.0) -> pd.DataFrame:
    """Calcula metricas para ambos modelos, agrega y escribe oe5_metrics.csv.

    El CSV es 'largo' con filas por (modelo, scope) donde scope in
    {global, <clase>}. Reproduce la informacion de las Tablas 15 (global) y
    16 (por clase) para ResNet y ViT.
    """
    per_image = []
    for mk in (RESNET_KEY, VIT_KEY):
        per_image.append(compute_metrics(mk, threshold_method, percentile))
    pi = pd.concat(per_image, ignore_index=True) if per_image else pd.DataFrame()
    if pi.empty:
        print("\n[BLOQUEO] No hay (heatmap + mascara) para ninguna imagen.")
        print("          Genera los mapas y anota las mascaras antes de "
              "calcular las Tablas 15/16.")
        return pi

    # guardar el detalle por imagen tambien
    pi.to_csv(XAI_DIR / "oe5_per_image.csv", index=False)

    agg_rows = []
    for mk in (RESNET_KEY, VIT_KEY):
        sub = pi[pi["model"] == mk]
        if sub.empty:
            continue
        agg_rows.append({"model": mk, "scope": "global", "n": len(sub),
                         "mean_iou": sub["iou"].mean(),
                         "pointing_game": sub["pg"].mean()})
        for cls in LESION_CLASSES:
            cs = sub[sub["class_name"] == cls]
            if cs.empty:
                continue
            agg_rows.append({"model": mk, "scope": cls, "n": len(cs),
                             "mean_iou": cs["iou"].mean(),
                             "pointing_game": cs["pg"].mean()})
    agg = pd.DataFrame(agg_rows)
    agg["threshold_method"] = threshold_method
    agg["percentile"] = percentile
    agg.to_csv(METRICS_CSV, index=False)
    print(f"\n[OK] Tablas 15/16 -> {METRICS_CSV}")
    print(agg.to_string(index=False))
    return agg


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="OE5 XAI cuantitativo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("select", help="escribe el manifiesto de muestra (seed 2026)")

    g = sub.add_parser("generate", help="genera heatmaps + overlays (paso GPU)")
    g.add_argument("--model", required=True,
                   choices=[RESNET_KEY, VIT_KEY])
    g.add_argument("--variant", default="best", choices=["best", "ensemble"])
    g.add_argument("--limit", type=int, default=None,
                   help="procesar solo las primeras N (smoke)")

    mt = sub.add_parser("metrics", help="calcula IoU/PG -> oe5_metrics.csv")
    mt.add_argument("--threshold-method", default="percentile",
                    choices=["percentile", "otsu", "fixed"])
    mt.add_argument("--percentile", type=float, default=80.0)

    args = ap.parse_args()
    if args.cmd == "select":
        select_sample()
    elif args.cmd == "generate":
        generate_maps(args.model, args.variant, args.limit)
    elif args.cmd == "metrics":
        aggregate_and_write(args.threshold_method, args.percentile)


if __name__ == "__main__":
    main()
