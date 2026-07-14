"""
Metricas de clasificacion.

Macro F1 es la metrica principal del TFM (justificada en Cap 3 por el
desbalance 11:1). Accuracy se reporta como complemento descriptivo.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, precision_score, recall_score,
)

from . import config as C


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Devuelve un diccionario con las metricas clave del experimento."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {
            C.IDX_TO_CLASS[i]: float(v)
            for i, v in enumerate(f1_score(y_true, y_pred, average=None,
                                           labels=list(range(C.NUM_CLASSES)),
                                           zero_division=0))
        },
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))


def report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true, y_pred,
        labels=list(range(C.NUM_CLASSES)),
        target_names=C.CLASSES,
        digits=4, zero_division=0,
    )
