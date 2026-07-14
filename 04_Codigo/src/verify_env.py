"""
Diagnostico rapido del entorno: dataset, GPU, dependencias, splits.

Ejecutar antes de lanzar entrenamientos:
    python -m src.verify_env
"""
from __future__ import annotations
import json
import sys


def main():
    print("=" * 60)
    print("VERIFICACION DE ENTORNO TFM")
    print("=" * 60)

    # Python
    print(f"\nPython: {sys.version.split()[0]}")

    # Dependencias criticas
    print("\nDependencias:")
    pkgs = ["torch", "torchvision", "timm", "mlflow", "sklearn", "PIL", "pandas", "numpy"]
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", "n/a")
            print(f"  [OK] {pkg:14s} {v}")
        except Exception as e:
            print(f"  [FAIL] {pkg:14s} {type(e).__name__}: {e}")

    # GPU
    print("\nGPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA disponible")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            try:
                cap = torch.cuda.get_device_capability(0)
                print(f"  Compute capability: {cap[0]}.{cap[1]}")
            except Exception:
                pass
        else:
            print("  [WARN] CUDA NO disponible — entrenamientos seran muy lentos en CPU")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Dataset
    print("\nDataset:")
    try:
        from . import config as C
        from . import dataset as D
        print(f"  Path: {C.DATA_DIR}")
        df = D.build_catalog()
        print(f"  Total imagenes: {len(df)}")
        for cls, n in df["class_name"].value_counts().items():
            expected = C.CLASS_COUNTS[cls]
            mark = "[OK]" if n == expected else f"[DIFF! esperado {expected}]"
            print(f"    {cls:14s} {n:>5}  {mark}")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return

    # Splits
    print("\nSplits estratificados:")
    try:
        df = D.load_or_build_splits()
        summary = D.dump_split_summary(df, C.RESULTS_DIR / "split_summary.json")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("  [OK] split_summary.json generado")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Si todo dice [OK] estamos listos para 'python -m src.run_e2 --plan smoke'")
    print("=" * 60)


if __name__ == "__main__":
    main()
