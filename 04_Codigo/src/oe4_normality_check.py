"""
Validación del supuesto de normalidad de las diferencias pareadas (OE4) antes
de aplicar el test de Wilcoxon de rangos con signo.

Responde al comentario del director (E3-50): documentar que se verificó la
normalidad de las 15 diferencias pareadas (5 folds x 3 semillas) de macro F1 de
cada estrategia de mitigación frente a la variante sin mitigación (entropía
cruzada plana) antes de optar por la prueba no paramétrica.

Resultado (05/07/2026): las tres distribuciones de diferencias NO se apartan de
la normalidad (Shapiro-Wilk p = 0,990 / 0,399 / 0,165). Dado el reducido tamaño
muestral (n = 15) se reporta el Wilcoxon por conservadurismo; el t-test pareado
llega a la misma conclusión (ningún contraste significativo), confirmando la
robustez de la elección.

Uso:
    .venv/Scripts/python.exe -m src.oe4_normality_check
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from scipy import stats

from . import config as C

RAW = C.RESULTS_DIR / "oe4_ablation_normfix_raw.csv"
STRATEGIES = [
    ("class_weights", "Ponderacion por clase"),
    ("focal", "Focal Loss gamma=2"),
    ("oversampling", "Oversampling"),
]


def main():
    df = pd.read_csv(RAW)
    base = df[df.strategy == "none"].set_index(["fold", "seed"])["macro_f1"]
    print("Normalidad de las 15 diferencias pareadas (estrategia - baseline CE):\n")
    for strat, label in STRATEGIES:
        s = df[df.strategy == strat].set_index(["fold", "seed"])["macro_f1"]
        idx = base.index.intersection(s.index)
        a, b = s.loc[idx].values, base.loc[idx].values
        diff = a - b
        W, p = stats.shapiro(diff)
        _, w_p = stats.wilcoxon(a, b)
        _, t_p = stats.ttest_rel(a, b)
        print(f"{label:24s} n={len(diff)}  Shapiro W={W:.4f} p={p:.4f}  "
              f"{'NORMAL' if p > 0.05 else 'NO-normal'}  | Wilcoxon p={w_p:.4f} | t-pareado p={t_p:.4f}")


if __name__ == "__main__":
    main()
