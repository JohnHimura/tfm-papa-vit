"""Utilidades de reproducibilidad."""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Fija semillas en Python, NumPy y PyTorch (CPU + CUDA).

    Si deterministic=True activa cudnn deterministic — penaliza ~10% velocidad
    pero garantiza que dos runs con misma semilla den el mismo resultado.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
