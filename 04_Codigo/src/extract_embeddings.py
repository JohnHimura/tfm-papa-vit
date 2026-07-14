"""
Fase 1.6 — Extraccion de embeddings del ViT + proyeccion t-SNE / UMAP.

Extrae los features PRE-CABEZA del ViT-Base/16 (representacion del token CLS /
pooled, justo antes del clasificador) sobre el holdout en UNA pasada, y genera
las proyecciones 2D t-SNE y UMAP coloreadas por clase. Conecta con la materia
"Aprendizaje No Supervisado".

Los features se capturan con un forward HOOK sobre `model.fc_norm` (o el modulo
previo a la cabeza), sin tocar models.py. Para `vit_base_patch16_224` de timm,
`forward_features` + `forward_head(pre_logits=True)` da el embedding pooled; lo
obtenemos de forma robusta hookeando la entrada de `model.head`.

Salidas:
    results/xai/vit_embeddings.npz        (X: (N, D), y: (N,), paths)
    results/figuras/embeddings_tsne_vit.png
    results/figuras/embeddings_umap_vit.png  (solo si umap-learn esta instalado)
    + copia en 05_Documento_TFM/figuras/

Uso:
    python -m src.extract_embeddings --variant best
    python -m src.extract_embeddings --variant best --no-umap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as D
from .models import build_model
from .transforms import eval_transforms, get_model_norm

VIT_KEY = "vit_base_patch16_224"
EMB_NPZ = C.RESULTS_DIR / "xai" / "vit_embeddings.npz"
FIG_TSNE = C.RESULTS_DIR / "figuras" / "embeddings_tsne_vit.png"
FIG_UMAP = C.RESULTS_DIR / "figuras" / "embeddings_umap_vit.png"
DOC_FIGS = C.FIGS_DIR


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _best_vit_checkpoint() -> Path:
    fm = pd.read_csv(C.RESULTS_DIR / "final_master.csv")
    sub = fm[fm["model"] == VIT_KEY].sort_values("best_macro_f1", ascending=False)
    best = sub.iloc[0]
    fold, seed = int(best["fold"]), int(best["seed"])
    return C.MODELS_DIR / f"final__{VIT_KEY}_fold{fold}_seed{seed}_best.pt"


class _PreHeadHook:
    """Captura la ENTRADA del clasificador (model.head) = embedding pooled."""

    def __init__(self, head_module: torch.nn.Module):
        self.feat = None
        self._h = head_module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inp):
        self.feat = inp[0].detach()

    def remove(self):
        self._h.remove()


@torch.no_grad()
def extract_embeddings(variant: str = "best", batch_size: int = 32,
                       limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list]:
    """Extrae embeddings pre-cabeza del ViT sobre el holdout en 1 pasada."""
    device = _device()
    df = D.load_or_build_splits()
    test = df[df["split_holdout"] == "test"].reset_index(drop=True)
    if limit:
        test = test.head(limit)
    _mean, _std = get_model_norm(VIT_KEY)   # ViT espera (0.5,0.5,0.5). normfix.
    ds = D.PotatoLeafDataset(test,
                             transform=eval_transforms(mean=_mean, std=_std),
                             return_path=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model = build_model(VIT_KEY, pretrained=False, num_classes=C.NUM_CLASSES).to(device)
    ck = _best_vit_checkpoint()
    sd = torch.load(ck, weights_only=False)
    model.load_state_dict(sd["model_state"])
    model.eval()

    hook = _PreHeadHook(model.head)
    feats, ys, paths = [], [], []
    try:
        for x, y, p in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            feats.append(hook.feat.cpu().numpy())
            ys.append(y.numpy())
            paths.extend(list(p))
    finally:
        hook.remove()
    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(ys, axis=0)
    EMB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMB_NPZ, X=X, y=Y, paths=np.array(paths))
    print(f"[OK] embeddings {X.shape} (ckpt={ck.name}) -> {EMB_NPZ}")
    return X, Y, paths


def _plot_2d(coords: np.ndarray, y: np.ndarray, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(C.CLASSES):
        m = y == i
        ax.scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.7, label=cls)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=1.4)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    # copia a la carpeta de figuras del documento
    try:
        DOC_FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(DOC_FIGS / out.name, dpi=140)
    except Exception:
        pass
    plt.close(fig)
    print(f"[OK] {title} -> {out}")


def run_tsne(X: np.ndarray, y: np.ndarray) -> None:
    from sklearn.manifold import TSNE
    # perplexity debe ser < n_samples; en holdout real (~462) usa 30.
    perp = min(30, max(2, (len(X) - 1) // 3))
    perp = min(perp, len(X) - 1)
    coords = TSNE(n_components=2, perplexity=perp, init="pca",
                  random_state=C.SEEDS[0]).fit_transform(X)
    _plot_2d(coords, y, "t-SNE de embeddings ViT (holdout)", FIG_TSNE)


def run_umap(X: np.ndarray, y: np.ndarray) -> bool:
    """Devuelve True si umap-learn esta disponible y la figura se genero."""
    try:
        import umap  # noqa
    except Exception:
        print("[SKIP] umap-learn no esta instalado. "
              "Instalar con: pip install umap-learn  (luego re-correr).")
        return False
    import umap
    reducer = umap.UMAP(n_components=2, random_state=C.SEEDS[0])
    coords = reducer.fit_transform(X)
    _plot_2d(coords, y, "UMAP de embeddings ViT (holdout)", FIG_UMAP)
    return True


def main():
    ap = argparse.ArgumentParser(description="Embeddings ViT + t-SNE/UMAP")
    ap.add_argument("--variant", default="best", choices=["best"])
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke: N imagenes")
    args = ap.parse_args()
    X, Y, _ = extract_embeddings(variant=args.variant, limit=args.limit)
    run_tsne(X, Y)
    if not args.no_umap:
        run_umap(X, Y)


if __name__ == "__main__":
    main()
