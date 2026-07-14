"""
Dataset y splits estratificados.

Estrategia:
1. Holdout fijo del 15% (estratificado, semilla 42) → usado SOLO al final.
2. Sobre el 85% restante: Stratified K-Fold (k=5 por defecto, 3 si apuras).
3. La distribucion de clases se mantiene proporcional en cada fold.

Esto es importante por el desbalance 11:1 (Fungi 748 vs Nematode 68).
Un split aleatorio podria dejar 0 imagenes Nematode en validacion.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import Dataset

from . import config as C


# =============================================================================
# Catalogo de muestras
# =============================================================================
def build_catalog(data_dir: Path | None = None) -> pd.DataFrame:
    """Recorre el dataset y devuelve un DataFrame con columnas
    [path, class_name, label].

    NOTA: el filesystem de Windows es case-insensitive, por lo que iterar
    extensiones en mayusculas y minusculas produce duplicados. Usamos un
    `set` con la ruta resuelta en minusculas como clave para deduplicar
    de forma portable entre Windows, Linux y macOS.
    """
    data_dir = data_dir or C.DATA_DIR
    seen: set[str] = set()
    rows = []
    valid_exts = {".jpg", ".jpeg", ".png"}
    for cls in C.CLASSES:
        cls_dir = data_dir / cls
        for p in cls_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in valid_exts:
                continue
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "rel_path": f"{cls}/{p.name}",   # relativa a DATA_DIR -> portable
                "class_name": cls,
                "label": C.CLASS_TO_IDX[cls],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No se encontraron imagenes en {data_dir}")
    return df


def make_holdout_split(df: pd.DataFrame, frac: float = C.TEST_HOLDOUT_FRAC,
                       seed: int = 42) -> pd.DataFrame:
    """Anade columna split_holdout in {'train_cv', 'test'} estratificada."""
    train_idx, test_idx = train_test_split(
        df.index, test_size=frac, stratify=df["label"], random_state=seed
    )
    df = df.copy()
    df["split_holdout"] = "train_cv"
    df.loc[test_idx, "split_holdout"] = "test"
    return df


def kfold_indices(df_train_cv: pd.DataFrame, n_folds: int = C.N_FOLDS,
                  seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Genera pares (train_idx, val_idx) sobre el subset train_cv."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = df_train_cv["label"].values
    for tr, val in skf.split(np.zeros(len(labels)), labels):
        yield tr, val


def save_splits(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def _resolve_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la columna absoluta 'path' a partir de 'rel_path' (relativa a
    C.DATA_DIR). Esto hace el catalogo PORTABLE entre maquinas: el mismo split
    funciona en cualquier equipo apuntando a su propio 03_Dataset/original.
    """
    df = df.copy()
    if "rel_path" not in df.columns and "path" in df.columns:
        # Catalogo antiguo con rutas absolutas -> derivar rel_path (clase/archivo).
        df["rel_path"] = df["path"].map(lambda p: "/".join(Path(str(p)).parts[-2:]))
    df["path"] = df["rel_path"].map(lambda r: str(C.DATA_DIR / r))
    return df


def load_or_build_splits(cache: Path | None = None) -> pd.DataFrame:
    cache = cache or (C.RESULTS_DIR / "catalog_with_holdout.csv")
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = build_catalog()
        df = make_holdout_split(df, seed=42)
        save_splits(df, cache)          # guarda rel_path (portable), sin rutas absolutas
    return _resolve_paths(df)


# =============================================================================
# Torch Dataset
# =============================================================================
@dataclass
class SampleRow:
    path: str
    label: int


class PotatoLeafDataset(Dataset):
    """Dataset PyTorch que carga imagenes JPG/PNG desde rutas absolutas."""

    def __init__(self, df: pd.DataFrame, transform=None, return_path: bool = False):
        self.samples = [SampleRow(r.path, int(r.label)) for r in df.itertuples()]
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, s.label, s.path
        return img, s.label


# =============================================================================
# Pesos de clase (estrategia desbalance: 'balanced')
# =============================================================================
def compute_class_weights(df_train: pd.DataFrame) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia, normalizados."""
    counts = df_train["label"].value_counts().sort_index().values
    total = counts.sum()
    n_classes = len(counts)
    weights = total / (n_classes * counts)
    weights = weights / weights.mean()  # normaliza alrededor de 1.0
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# Oversampling (estrategia desbalance: WeightedRandomSampler)
# =============================================================================
def make_weighted_sampler(df_train: pd.DataFrame,
                          generator: torch.Generator | None = None
                          ) -> torch.utils.data.WeightedRandomSampler:
    """Construye un WeightedRandomSampler con peso por-muestra = 1/freq(clase).

    Cada muestra recibe un peso inversamente proporcional a la frecuencia de su
    clase en df_train, de modo que el muestreo (con reemplazo) genere batches
    aproximadamente balanceados entre las 7 clases pese al desbalance 11:1.

    - replacement=True: las clases minoritarias (p.ej. Nematode, 68 imgs) se
      repiten para igualar la representacion de las mayoritarias.
    - num_samples=len(df_train): una "epoca" sigue viendo tantas muestras como
      el dataset original (no infla el numero de pasos por epoca).

    Devuelve el sampler; el orden de las filas de df_train debe coincidir con el
    orden de PotatoLeafDataset(df_train) (ambos usan el indice posicional del df).
    """
    labels = df_train["label"].to_numpy()
    class_counts = np.bincount(labels, minlength=C.NUM_CLASSES).astype(np.float64)
    # Evita division por cero para clases ausentes en este fold/seed.
    class_counts[class_counts == 0] = 1.0
    per_class_weight = 1.0 / class_counts          # 1/freq por clase
    sample_weights = per_class_weight[labels]      # peso por muestra
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def dump_split_summary(df: pd.DataFrame, out: Path) -> dict:
    """Reporte legible de la distribucion por clase y split."""
    summary = {}
    for split_name in ("train_cv", "test"):
        sub = df[df["split_holdout"] == split_name]
        summary[split_name] = {
            "total": int(len(sub)),
            "by_class": {C.IDX_TO_CLASS[int(k)]: int(v)
                         for k, v in sub["label"].value_counts().sort_index().items()},
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
