"""
Transformaciones de imagen para train / val / test.

Pipeline conservador alineado con la metodologia declarada en Cap 3:
NO se aplica preprocesamiento clasico (Otsu, mediana, gamma) — el modelo
recibe el RGB normalizado al estandar QUE ESPERAN SUS PESOS DE PARTIDA.

NORMALIZACION POR MODELO (normfix, 29/06/2026):
    Antes este modulo aplicaba SIEMPRE la normalizacion ImageNet
    (mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)) a TODOS los modelos.
    Eso es correcto para las 3 CNN (resnet50.a1_in1k, efficientnet_b0.ra_in1k,
    mobilenetv2_100.ra_in1k), cuyos pesos timm esperan estadisticas ImageNet,
    PERO es incorrecto para el ViT: `vit_base_patch16_224` (default timm =
    augreg2_in21k_ft_in1k) y `vit_base_patch16_224.augreg_in1k` esperan
    mean=std=(0.5,0.5,0.5). El ViT venia mal normalizado en train Y eval.

    Solucion: `train_transforms`/`eval_transforms` ahora aceptan mean/std.
    El helper `get_model_norm(model_name)` lee la estadistica correcta del
    pretrained_cfg de timm (con fallback a ImageNet). Cada call site calcula la
    normalizacion desde el model_name que entrena/evalua. Los defaults siguen
    siendo ImageNet por retrocompatibilidad (las 3 CNN no cambian en nada).

Augmentation solo en train. Fija para val/test asegura comparabilidad
de Macro F1 entre semillas y folds.
"""
from __future__ import annotations
from functools import lru_cache

import torchvision.transforms as T

from . import config as C


@lru_cache(maxsize=None)
def get_model_norm(model_name: str) -> tuple[tuple[float, float, float],
                                             tuple[float, float, float]]:
    """Devuelve (mean, std) que esperan los pesos pre-entrenados de `model_name`.

    Lee `timm.create_model(model_name, pretrained=False).pretrained_cfg` para
    obtener las estadisticas de normalizacion correctas SIN descargar pesos.
    Con fallback a ImageNet si timm no las expone o el nombre no existe.

    Ejemplos verificados (timm 1.0.26):
        resnet50.a1_in1k                 -> (0.485,0.456,0.406)/(0.229,0.224,0.225)
        efficientnet_b0.ra_in1k          -> ImageNet (idem)
        mobilenetv2_100.ra_in1k          -> ImageNet (idem)
        vit_base_patch16_224             -> (0.5,0.5,0.5)/(0.5,0.5,0.5)
        vit_base_patch16_224.augreg2_in21k_ft_in1k -> (0.5,0.5,0.5)/(0.5,0.5,0.5)
        vit_base_patch16_224.augreg_in1k -> (0.5,0.5,0.5)/(0.5,0.5,0.5)
    """
    try:
        import timm
        m = timm.create_model(model_name, pretrained=False)
        cfg = getattr(m, "pretrained_cfg", None) or {}
        mean = cfg.get("mean", C.IMAGENET_MEAN)
        std = cfg.get("std", C.IMAGENET_STD)
        # pretrained_cfg los entrega como tuplas; aseguramos tipo homogeneo.
        return tuple(float(v) for v in mean), tuple(float(v) for v in std)
    except Exception as e:  # nombre invalido, timm ausente, etc.
        print(f"  [get_model_norm] fallback ImageNet para '{model_name}': "
              f"{type(e).__name__}: {e}")
        return C.IMAGENET_MEAN, C.IMAGENET_STD


def train_transforms(img_size: int = C.IMG_SIZE,
                     mean: tuple[float, float, float] = C.IMAGENET_MEAN,
                     std: tuple[float, float, float] = C.IMAGENET_STD):
    """Pipeline de augmentation + normalizacion para TRAIN.

    mean/std por defecto = ImageNet (retrocompat: las CNN no cambian). Para el
    ViT pasar get_model_norm(model_name).
    """
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),       # hojas son rotables
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transforms(img_size: int = C.IMG_SIZE,
                    mean: tuple[float, float, float] = C.IMAGENET_MEAN,
                    std: tuple[float, float, float] = C.IMAGENET_STD):
    """Pipeline determinista (sin augmentation) para VAL/TEST.

    mean/std por defecto = ImageNet (retrocompat). Para el ViT pasar
    get_model_norm(model_name).
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
