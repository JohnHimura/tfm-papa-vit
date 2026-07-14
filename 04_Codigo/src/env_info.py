"""
Captura de informacion del entorno (versiones, hardware, sistema).

Se invoca al inicio de cada run para registrar como artefacto/params en MLflow,
garantizando reproducibilidad y respondiendo a la rubrica UNIR.
"""
from __future__ import annotations
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch


def get_env_snapshot() -> dict:
    """Snapshot completo del entorno: SO, Python, dependencias, GPU."""
    snap: dict = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # PyTorch + CUDA
    snap["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

    # GPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        snap["gpu"] = {
            "name": props.name,
            "total_vram_gb": round(props.total_memory / 1e9, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
        }
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=3,
            ).strip()
            snap["gpu"]["driver_version"] = out
        except Exception:
            pass
    else:
        snap["gpu"] = None

    # Dependencias clave
    snap["dependencies"] = {}
    for pkg in ["torch", "torchvision", "timm", "mlflow", "sklearn",
                "PIL", "pandas", "numpy", "matplotlib", "optuna"]:
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", "n/a")
            snap["dependencies"][pkg] = str(v)
        except Exception:
            snap["dependencies"][pkg] = "NOT INSTALLED"

    return snap


def save_env_snapshot(out_path: Path) -> dict:
    """Guarda el snapshot a JSON y lo devuelve."""
    snap = get_env_snapshot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return snap


if __name__ == "__main__":
    import sys
    snap = get_env_snapshot()
    print(json.dumps(snap, indent=2, ensure_ascii=False))
