"""
Factoria de modelos via timm.

Usamos timm porque expone CNN y ViT con la misma interfaz, lo que mantiene
el codigo del experimento absolutamente simetrico — un punto explicito en
el Cap 4 (Tecnologias).

Cabezas reseteadas a NUM_CLASSES (7). Backbone con pesos ImageNet,
fine-tune total (no congelamos para Entrega 2).
"""
from __future__ import annotations
import timm
import torch.nn as nn

from . import config as C


def build_model(model_name: str, pretrained: bool = True,
                num_classes: int = C.NUM_CLASSES) -> nn.Module:
    """Crea un modelo timm con la cabeza reemplazada para nuestro problema."""
    model = timm.create_model(model_name, pretrained=pretrained,
                              num_classes=num_classes)
    return model


def split_param_groups(model: nn.Module, lr_head: float, lr_backbone: float,
                       weight_decay: float):
    """Devuelve param groups con LR diferenciado: cabeza alta, backbone bajo.

    Detecta la cabeza por el nombre del classifier. timm usa 'head' (ViT) o
    'classifier' / 'fc' (CNN) — los manejamos todos.
    """
    head_keys = ("head", "classifier", "fc")
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(k) or f".{k}." in name for k in head_keys):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params,     "lr": lr_head,     "weight_decay": weight_decay},
    ]
