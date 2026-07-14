"""
Computa parámetros, FLOPs y tamaño aproximado del checkpoint para los 4 modelos
del TFM. Usado para llenar la Tabla 5 del Capítulo 4.
"""
from __future__ import annotations
import json
from pathlib import Path

import timm
import torch
from thop import profile

from . import config as C


MODELS = ["mobilenetv2_100", "resnet50", "efficientnet_b0", "vit_base_patch16_224"]


def compute_specs() -> list[dict]:
    rows = []
    dummy = torch.randn(1, 3, 224, 224)
    for name in MODELS:
        m = timm.create_model(name, pretrained=False, num_classes=C.NUM_CLASSES)
        m.eval()
        n_params = sum(p.numel() for p in m.parameters())
        n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        with torch.no_grad():
            macs, _ = profile(m, inputs=(dummy,), verbose=False)
        # macs ~ FLOPs/2 segun convencion thop; usamos FLOPs = 2*MACs
        flops = macs * 2
        # Tamano checkpoint aproximado (fp32 state_dict)
        ckpt_mb = (n_params * 4) / 1e6
        rows.append({
            "model": name,
            "params_M": n_params / 1e6,
            "trainable_M": n_trainable / 1e6,
            "flops_G": flops / 1e9,
            "ckpt_MB_fp32": ckpt_mb,
        })
    return rows


def main():
    rows = compute_specs()
    out = C.RESULTS_DIR / "model_specs.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'Modelo':<28}{'Params (M)':<14}{'FLOPs (G)':<13}{'Ckpt fp32 (MB)':<16}")
    print("-" * 71)
    for r in rows:
        print(f"{r['model']:<28}{r['params_M']:<14.2f}{r['flops_G']:<13.2f}{r['ckpt_MB_fp32']:<16.1f}")
    print(f"\n[OK] Especificaciones en {out}")


if __name__ == "__main__":
    main()
