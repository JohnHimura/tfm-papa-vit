"""
OE5 — Explicabilidad cuantitativa (XAI).

Implementa los dos mapas de saliencia que el TFM contrasta y las metricas
cuantitativas (IoU y Pointing Game) que pueblan las Tablas 15 y 16.

  - Grad-CAM para ResNet-50 (CNN)            -> grad_cam_resnet()
  - Attention Rollout para ViT-Base/16 (ViT)  -> attention_rollout_vit()
  - iou(), pointing_game()                    -> metricas vs mascara de lesion
  - binarize_heatmap()                        -> umbralizado parametrizable

DISEÑO CLAVE
------------
* NO se modifica src/models.py. Los features/gradientes (Grad-CAM) y las
  matrices de atencion (rollout) se extraen en runtime con forward/backward
  HOOKS sobre los modulos de timm.
* Grad-CAM esta implementado de forma autocontenida (solo torch + numpy) para
  no depender de `pytorch-grad-cam`. Si la libreria esta disponible se puede
  usar como alternativa (ver `grad_cam_resnet(..., use_lib=True)`), pero el
  default es la implementacion propia, que es identica en formulacion y corre
  en cualquiera de los dos entornos del proyecto.
* Attention Rollout sigue Abnar & Zomorrodi (2020): por cada bloque se promedian
  las cabezas, se suma la identidad para modelar la conexion residual, se
  re-normaliza por fila, y se multiplican las matrices capa a capa. El mapa
  final es la fila del token CLS hacia los 196 parches -> 14x14 -> upsample.
  En timm la atencion usa SDPA fundida (`fused_attn`), por lo que NO se puede
  capturar la matriz de atencion con un hook de salida: se hookea `attn.qkv`
  y se recomputa softmax(q·k^T / sqrt(d)) manualmente.

Todas las funciones de mapa devuelven un heatmap float32 normalizado a [0, 1]
con la forma (H, W) del tamaño de entrada (224x224 por defecto).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utilidades de normalizacion / upsample
# =============================================================================
def _normalize01(x: np.ndarray) -> np.ndarray:
    """Lleva un array a [0, 1] de forma robusta (maneja el caso constante)."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _upsample_to(cam: np.ndarray, size: int) -> np.ndarray:
    """Upsample bilineal de un mapa 2D (h, w) -> (size, size)."""
    t = torch.from_numpy(np.asarray(cam, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


# =============================================================================
# GRAD-CAM (ResNet-50) — implementacion propia con hooks
# =============================================================================
class _ActGradHook:
    """Captura activaciones (forward) y gradientes (backward) de un modulo."""

    def __init__(self, module: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fh = module.register_forward_hook(self._forward)
        # full backward hook captura el gradiente respecto a la salida del modulo
        self._bh = module.register_full_backward_hook(self._backward)

    def _forward(self, module, inp, out):
        self.activations = out.detach()

    def _backward(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def get_resnet_target_layer(model: nn.Module) -> nn.Module:
    """Capa objetivo de Grad-CAM para ResNet-50: ultimo bloque de layer4."""
    return model.layer4[-1]


def grad_cam_resnet(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    img_size: int = 224,
    target_layer: Optional[nn.Module] = None,
    use_lib: bool = False,
) -> np.ndarray:
    """Grad-CAM sobre ResNet-50.

    Parameters
    ----------
    model : nn.Module
        ResNet-50 de timm (build_model('resnet50', ...)). Debe estar en el
        mismo device que `input_tensor`.
    input_tensor : torch.Tensor
        Tensor (1, 3, H, W) normalizado a estandar ImageNet.
    target_class : int | None
        Clase para la que se calcula el mapa. Si None, usa argmax del logit.
    img_size : int
        Tamaño de salida del heatmap (lado).
    target_layer : nn.Module | None
        Override de capa objetivo; por defecto model.layer4[-1].
    use_lib : bool
        Si True y `pytorch_grad_cam` esta instalado, delega en la libreria.

    Returns
    -------
    np.ndarray  (img_size, img_size) float32 en [0, 1].
    """
    if use_lib:
        try:
            return _grad_cam_resnet_lib(model, input_tensor, target_class,
                                        img_size, target_layer)
        except ImportError:
            pass  # cae a la implementacion propia

    layer = target_layer or get_resnet_target_layer(model)
    hook = _ActGradHook(layer)
    was_training = model.training
    model.eval()
    try:
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = model(input_tensor)               # (1, num_classes)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        score = logits[0, target_class]
        score.backward()

        acts = hook.activations[0]                 # (C, h, w)
        grads = hook.gradients[0]                  # (C, h, w)
        weights = grads.mean(dim=(1, 2))           # GAP de gradientes -> (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (h, w)
        cam = cam.cpu().numpy()
    finally:
        hook.remove()
        if was_training:
            model.train()

    cam = _upsample_to(cam, img_size)
    return _normalize01(cam)


def _grad_cam_resnet_lib(model, input_tensor, target_class, img_size, target_layer):
    """Alternativa usando pytorch-grad-cam (si esta instalada)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer = target_layer or get_resnet_target_layer(model)
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(int(target_class))]
    with GradCAM(model=model, target_layers=[layer]) as cam:
        grayscale = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)
    grayscale = _upsample_to(grayscale, img_size)
    return _normalize01(grayscale)


# =============================================================================
# ATTENTION ROLLOUT (ViT-Base/16) — Abnar & Zomorrodi (2020)
# =============================================================================
class _QKVHook:
    """Captura la entrada y salida de cada `attn.qkv` para recomputar atencion.

    timm usa scaled_dot_product_attention fundida, asi que la matriz de
    atencion no aparece como salida de ningun modulo. La reconstruimos a mano
    a partir de q y k (la salida de la capa lineal qkv).
    """

    def __init__(self, attn_module: nn.Module):
        self.num_heads = attn_module.num_heads
        self.qkv_out: Optional[torch.Tensor] = None
        self._h = attn_module.qkv.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out: (B, N, 3*dim)
        self.qkv_out = out.detach()

    def attention_matrix(self) -> torch.Tensor:
        """Devuelve A promediada sobre cabezas: (B, N, N)."""
        qkv = self.qkv_out                          # (B, N, 3*dim)
        B, N, three_dim = qkv.shape
        dim = three_dim // 3
        head_dim = dim // self.num_heads
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]                        # (B, heads, N, head_dim)
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale    # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        return attn.mean(dim=1)                      # promedio de cabezas -> (B, N, N)

    def remove(self):
        self._h.remove()


def attention_rollout_vit(
    model: nn.Module,
    input_tensor: torch.Tensor,
    img_size: int = 224,
    num_prefix_tokens: Optional[int] = None,
    grid: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Attention Rollout clasico sobre ViT-Base/16 de timm.

    Algoritmo (Abnar & Zomorrodi, 2020):
      1. Para cada bloque: A = promedio de cabezas de softmax(qk^T/sqrt(d)).
      2. A_hat = 0.5*A + 0.5*I  (suma de identidad por la conexion residual),
         re-normalizada por fila.
      3. Rollout = A_hat_L @ ... @ A_hat_1   (producto de matrices por capa).
      4. Mapa = fila del token CLS hacia los parches -> grid HxW -> upsample.

    Returns
    -------
    np.ndarray (img_size, img_size) float32 en [0, 1].
    """
    if num_prefix_tokens is None:
        num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    if grid is None:
        grid = tuple(model.patch_embed.grid_size)   # (14, 14)

    hooks = [_QKVHook(blk.attn) for blk in model.blocks]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _ = model(input_tensor)                 # forward para poblar hooks

        device = input_tensor.device
        # tomar N de la primera matriz
        a0 = hooks[0].attention_matrix()            # (1, N, N)
        N = a0.shape[-1]
        eye = torch.eye(N, device=device)
        rollout = torch.eye(N, device=device)
        for h in hooks:
            A = h.attention_matrix()[0]             # (N, N)
            A_hat = 0.5 * A + 0.5 * eye             # residual
            A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
            rollout = A_hat @ rollout               # acumula capa a capa
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    # fila del CLS (token 0) hacia los parches (descartando tokens prefijo)
    cls_row = rollout[0, num_prefix_tokens:]        # (num_patches,)
    gh, gw = grid
    cam = cls_row.reshape(gh, gw).cpu().numpy()
    cam = _upsample_to(cam, img_size)
    return _normalize01(cam)


# =============================================================================
# UMBRALIZADO / BINARIZADO de heatmaps
# =============================================================================
def binarize_heatmap(
    heatmap: np.ndarray,
    method: str = "percentile",
    percentile: float = 80.0,
    thresh: Optional[float] = None,
) -> np.ndarray:
    """Binariza un heatmap [0,1] para el calculo de IoU.

    method:
      - "percentile": umbral = percentil `percentile` del heatmap (default 80).
      - "otsu"      : umbral de Otsu (requiere OpenCV; si falta -> percentil 80).
      - "fixed"     : umbral fijo `thresh` (en [0,1]).
    Devuelve mascara booleana (uint8 0/1) de la misma forma.
    """
    h = np.asarray(heatmap, dtype=np.float32)
    if method == "fixed":
        if thresh is None:
            raise ValueError("method='fixed' requiere thresh")
        t = float(thresh)
    elif method == "otsu":
        try:
            import cv2
            img8 = (np.clip(h, 0, 1) * 255).astype(np.uint8)
            t8, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = t8 / 255.0
        except Exception:
            t = float(np.percentile(h, percentile))
    else:  # percentile
        t = float(np.percentile(h, percentile))
    return (h >= t).astype(np.uint8)


# =============================================================================
# METRICAS XAI: IoU y Pointing Game
# =============================================================================
def iou(saliency_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    """Intersection-over-Union entre dos mascaras binarias del mismo tamaño.

    Convencion: si la union es vacia (ambas mascaras vacias) devuelve 0.0.
    """
    a = np.asarray(saliency_bin).astype(bool)
    b = np.asarray(mask_bin).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"shapes distintas: {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def pointing_game(saliency: np.ndarray, mask_bin: np.ndarray) -> int:
    """Pointing Game: ¿el pixel de MAXIMA saliencia cae dentro de la mascara?

    Recibe el heatmap CONTINUO (no binarizado) y la mascara de lesion binaria.
    Devuelve 1 (acierto) o 0 (fallo). Si hay empate de maximos, basta con que
    cualquiera de los pixeles de maximo caiga en la mascara.
    """
    s = np.asarray(saliency, dtype=np.float32)
    m = np.asarray(mask_bin).astype(bool)
    if s.shape != m.shape:
        raise ValueError(f"shapes distintas: {s.shape} vs {m.shape}")
    if m.sum() == 0:
        return 0
    max_val = s.max()
    hits = (s >= max_val - 1e-9) & m
    return int(hits.any())


# =============================================================================
# Overlay para inspeccion visual (PNG)
# =============================================================================
def overlay_heatmap(
    rgb_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Superpone un heatmap [0,1] sobre una imagen RGB uint8 (H,W,3).

    Usa el colormap JET de OpenCV si esta disponible; si no, un mapa manual.
    Devuelve uint8 (H, W, 3) RGB.
    """
    h = _normalize01(heatmap)
    H, W = rgb_img.shape[:2]
    if h.shape != (H, W):
        h = _upsample_to(h, max(H, W))[:H, :W]
    try:
        import cv2
        cmap = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    except Exception:
        cmap = np.stack([h, np.zeros_like(h), 1 - h], axis=-1)
        cmap = (cmap * 255).astype(np.uint8)
    out = (alpha * cmap + (1 - alpha) * rgb_img).astype(np.uint8)
    return out
