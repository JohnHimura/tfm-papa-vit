"""
Funciones de perdida usadas en los experimentos.

CrossEntropyLoss con class_weights por defecto (estrategia A0 de E2).
FocalLoss disponible para ablacion A3 (gamma=2.0 alineado con literatura).

FOCAL LOSS — correccion (normfix, 29/06/2026):
    La implementacion previa hacia `pt = exp(-ce)` donde `ce` ya incluia los
    class_weights (alpha) Y label_smoothing. Por eso `pt` NO era la probabilidad
    softmax verdadera de la clase target y el factor (1-pt)^gamma quedaba
    distorsionado (cuando alpha != 1, pt ni siquiera estaba en [0,1] de forma
    interpretable). Ahora pt se calcula como la softmax verdadera de la clase
    target (sin pesos ni suavizado), y alpha + el factor focal se aplican por
    separado, de forma matematicamente consistente con Lin et al. (2017).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss multiclase (Lin et al., 2017).

        L = - alpha_c * (1 - p_t)^gamma * log(p_t)

    donde p_t = softmax(logits)[target] es la probabilidad VERDADERA de la
    clase correcta (sin ponderar ni suavizar), alpha_c es el peso de clase
    (class weights) y gamma el factor de focalizacion.

    Notas de consistencia:
      - Con gamma=0 y label_smoothing=0, L se reduce EXACTAMENTE a la
        cross-entropy ponderada por alpha de PyTorch
        (CrossEntropyLoss(weight=alpha)), que normaliza por sum(alpha_t) — NO
        por N. Replicamos esa misma reduccion para que las dos ramas de
        build_loss (focal vs CE) esten en la MISMA escala.
      - label_smoothing se aplica sobre el termino log-loss (como en
        F.cross_entropy), no sobre el factor focal: el factor focal usa la
        probabilidad del target "dura" para no diluir la focalizacion.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log-prob estable; pt = prob verdadera del target (SIN weight ni smoothing)
        log_probs = F.log_softmax(logits, dim=1)                 # (N, C)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
        # clamp NO in-place: clamp_() mutaria la salida de exp() que autograd
        # necesita intacta (ExpBackward) -> RuntimeError en backward.
        pt = log_pt.exp().clamp(0.0, 1.0)                        # prob verdadera en [0,1]

        # Termino base de log-loss. label_smoothing se delega a F.cross_entropy
        # SIN weight (el alpha se aplica abajo de forma explicita) para no
        # mezclar dos correcciones distintas en el mismo termino.
        ce = F.cross_entropy(logits, target, reduction="none",
                             weight=None,
                             label_smoothing=self.label_smoothing)  # (N,)

        # Factor focal con la probabilidad VERDADERA del target.
        focal_factor = (1.0 - pt) ** self.gamma                  # (N,)

        # alpha_c (class weights) por muestra, segun la clase target.
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]       # (N,)
            # Reduccion ponderada estilo PyTorch: dividir por sum(alpha_t) para
            # que gamma=0 reproduzca EXACTAMENTE CrossEntropyLoss(weight=alpha).
            return (alpha_t * focal_factor * ce).sum() / alpha_t.sum().clamp(min=1e-12)
        # Sin class weights: media simple (== CrossEntropyLoss sin weight).
        return (focal_factor * ce).mean()


def build_loss(use_focal: bool, class_weights: torch.Tensor | None,
               label_smoothing: float, device: torch.device) -> nn.Module:
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if use_focal:
        return FocalLoss(gamma=2.0, alpha=class_weights,
                         label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weights,
                               label_smoothing=label_smoothing)
