"""
Manifest de splits con sha256 por imagen.

Garantiza reproducibilidad exacta: cualquier persona puede recrear los mismos
splits a partir del catalogo + holdout + folds. La clave es el sha256 del
contenido del archivo, no la ruta (que cambia entre maquinas).
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import pandas as pd

from . import config as C
from . import dataset as D


def file_sha256(path: Path, buf_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def build_splits_manifest(out_path: Path | None = None,
                          recompute_hashes: bool = False) -> pd.DataFrame:
    """Genera el manifest con columnas:
    [path, class_name, label, split_holdout, sha256, fold_0..fold_4]
    donde fold_i ∈ {'train','val'}.
    """
    out_path = out_path or (C.RESULTS_DIR / "splits_manifest.csv")

    df = D.load_or_build_splits()
    df_cv = df[df["split_holdout"] == "train_cv"].reset_index(drop=False)
    folds = list(D.kfold_indices(df_cv, n_folds=C.N_FOLDS, seed=42))

    # Inicializar columnas fold_i con 'test' por defecto (las del holdout)
    for fi in range(C.N_FOLDS):
        df[f"fold_{fi}"] = "test_holdout"

    # Para cada fold: mapear los indices originales en df
    for fi, (tr_idx, val_idx) in enumerate(folds):
        train_orig = df_cv.iloc[tr_idx]["index"].values
        val_orig = df_cv.iloc[val_idx]["index"].values
        df.loc[train_orig, f"fold_{fi}"] = "train"
        df.loc[val_orig, f"fold_{fi}"] = "val"

    # Hashes
    if recompute_hashes or "sha256" not in df.columns:
        print(f"Calculando sha256 de {len(df)} imagenes...")
        df["sha256"] = df["path"].apply(lambda p: file_sha256(Path(p)))

    df.to_csv(out_path, index=False)
    print(f"[OK] Manifest guardado en {out_path}")
    print(f"     {len(df)} imagenes, {len(df.columns)} columnas")
    return df


if __name__ == "__main__":
    build_splits_manifest(recompute_hashes=True)
