"""
Test de McNemar para comparar dos clasificadores sobre el mismo conjunto.

Justificacion (Cap 4): la diferencia de Macro F1 entre modelos puede no ser
significativa si los modelos aciertan/fallan en las mismas imagenes. McNemar
analiza la tabla de discordancias 2x2 y detecta diferencias estadisticamente
distinguibles del azar.

Implementacion: con correccion de continuidad (Edwards). Usamos statsmodels
si esta disponible; si no, calculo manual.
"""
from __future__ import annotations
import numpy as np


def mcnemar_table(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> np.ndarray:
    """Tabla 2x2: filas=A correcto/incorrecto, cols=B correcto/incorrecto."""
    a_ok = y_pred_a == y_true
    b_ok = y_pred_b == y_true
    n00 = int(((~a_ok) & (~b_ok)).sum())
    n01 = int(((~a_ok) & b_ok).sum())
    n10 = int((a_ok & (~b_ok)).sum())
    n11 = int((a_ok & b_ok).sum())
    return np.array([[n11, n10], [n01, n00]])


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                 exact: bool | None = None) -> dict:
    """Ejecuta McNemar con correccion de continuidad (o exacto si b+c<25).

    Devuelve: tabla, statistic, pvalue, n_disagree.
    """
    table = mcnemar_table(y_true, y_pred_a, y_pred_b)
    b = int(table[0, 1])  # A ok, B no
    c = int(table[1, 0])  # A no, B ok
    n_dis = b + c
    use_exact = exact if exact is not None else (n_dis < 25)

    if use_exact:
        # Test binomial: bajo H0, b ~ Binomial(n_dis, 0.5)
        from math import comb
        if n_dis == 0:
            stat, p = 0.0, 1.0
        else:
            k = min(b, c)
            tail = sum(comb(n_dis, i) for i in range(k + 1)) / (2 ** n_dis)
            p = min(2 * tail, 1.0)
            stat = float(min(b, c))
    else:
        # Chi-cuadrado con correccion de continuidad
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        from math import erf, sqrt
        # 1-CDF chi2(df=1, x) = erfc(sqrt(x/2))
        p = 1 - erf(sqrt(stat / 2))

    return {
        "table": table.tolist(),
        "n_disagree": n_dis,
        "b_only_A_correct": b,
        "c_only_B_correct": c,
        "statistic": float(stat),
        "pvalue": float(p),
        "exact": bool(use_exact),
    }
