"""
Fase 4 - Agregacion de resultados y figuras finales para Cap 5.

Lee:
- results/final_master.csv  (60 runs phase=final)
- results/runs/final__*/predictions.npz
- results/holdout_eval.json
- results/holdout_predictions/<modelo>_<modo>.npz

Genera:
- tablas/             tablas en JSON listas para inyectar en el doc
- figuras/cm_<modelo>.png            matriz de confusion normalizada
- figuras/lc_<modelo>.png            curvas de aprendizaje
- mcnemar_holdout.json               McNemar pairwise sobre holdout ensemble
- tokens_replace.json                mapping <<<TOKEN>>> -> valor para Cap 5

Uso:
    python -m src.analyze_results
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from . import config as C
from .mcnemar import mcnemar_test


MODELS = ["mobilenetv2_100", "resnet50", "efficientnet_b0", "vit_base_patch16_224"]
MODEL_SHORT = {
    "mobilenetv2_100": "MOBILE",
    "resnet50": "RESNET",
    "efficientnet_b0": "EFFNET",
    "vit_base_patch16_224": "VIT",
}
CLASS_SHORT = {
    "Bacteria": "BACT", "Fungi": "FUNGI", "Healthy": "HEAL",
    "Nematode": "NEMA", "Pest": "PEST", "Phytopthora": "PHYT", "Virus": "VIRUS",
}


def _set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def load_final_master() -> pd.DataFrame:
    return pd.read_csv(C.RESULTS_DIR / "final_master.csv")


def load_holdout_eval() -> dict:
    return json.loads((C.RESULTS_DIR / "holdout_eval.json").read_text(encoding="utf-8"))


# ----------------------- Tablas agregadas -------------------------
def table_macro_f1(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model").agg(
        macro_f1_mean=("best_macro_f1", "mean"),
        macro_f1_std=("best_macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_acc_mean=("balanced_accuracy", "mean"),
        balanced_acc_std=("balanced_accuracy", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        best_epoch_mean=("best_epoch", "mean"),
        runtime_mean=("total_run_time_s", "mean"),
        n=("best_macro_f1", "count"),
    ).reset_index()
    return g


def table_f1_per_class(df: pd.DataFrame) -> pd.DataFrame:
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    rows = []
    for model, sub in df.groupby("model"):
        for col in f1_cols:
            cls = col.replace("f1_", "")
            rows.append({
                "model": model, "class": cls,
                "f1_mean": float(sub[col].mean()),
                "f1_std": float(sub[col].std()),
            })
    return pd.DataFrame(rows)


def aggregate_predictions_cv(model: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatena y_true / y_pred de todos los runs final__<model>_* del CV."""
    runs_dir = C.RESULTS_DIR / "runs"
    yt, yp = [], []
    for d in sorted(runs_dir.glob(f"final__{model}_fold*_seed*")):
        f = d / "predictions.npz"
        if not f.exists():
            continue
        data = np.load(f)
        yt.append(data["y_true"])
        yp.append(data["y_pred"])
    if not yt:
        return None
    return np.concatenate(yt), np.concatenate(yp)


def plot_confusion(model: str, y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)),
                          normalize="true")
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.NUM_CLASSES))
    ax.set_yticks(range(C.NUM_CLASSES))
    ax.set_xticklabels(C.CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(C.CLASSES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusion normalizada - {model}")
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_learning_curves(model: str, out: Path) -> None:
    runs_dir = C.RESULTS_DIR / "runs"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    n = 0
    for d in sorted(runs_dir.glob(f"final__{model}_fold*_seed*")):
        f = d / "history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        ax1.plot(h["epoch"], h["train_loss"], alpha=0.35, color="#1F4E79")
        ax1.plot(h["epoch"], h["val_loss"], alpha=0.35, color="#9B2C2C", linestyle="--")
        ax2.plot(h["epoch"], h["val_macro_f1"], alpha=0.5, color="#1F4E79")
        n += 1
    ax1.set_xlabel("Epoca"); ax1.set_ylabel("Loss")
    ax1.set_title("Train (azul) / Val loss (rojo punteado)")
    ax2.set_xlabel("Epoca"); ax2.set_ylabel("Macro F1 (val)")
    ax2.set_title("Macro F1 por epoca")
    ax2.set_ylim(0, 1)
    fig.suptitle(f"Curvas de aprendizaje - {model} ({n} replicas)", fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# ----------------------- McNemar pairwise sobre holdout ensemble -------
def run_mcnemar_holdout() -> dict:
    """Aplica McNemar entre los ensembles de cada modelo sobre el holdout."""
    pred_dir = C.RESULTS_DIR / "holdout_predictions"
    preds: dict[str, dict] = {}
    for m in MODELS:
        f = pred_dir / f"{m}_ensemble.npz"
        if f.exists():
            d = np.load(f)
            preds[m] = {"y_true": d["y_true"], "y_pred": d["y_pred"]}
    results = {}
    for a, b in combinations(MODELS, 2):
        if a not in preds or b not in preds:
            continue
        yt_a, yp_a = preds[a]["y_true"], preds[a]["y_pred"]
        yt_b, yp_b = preds[b]["y_true"], preds[b]["y_pred"]
        if not np.array_equal(yt_a, yt_b):
            continue
        results[f"{a}__vs__{b}"] = mcnemar_test(yt_a, yp_a, yp_b)
    return results


# ----------------------- Builder de tokens -------------------------
def build_tokens(df: pd.DataFrame, t_class: pd.DataFrame,
                 holdout: dict, mcnemar: dict) -> dict[str, str]:
    """Construye el mapping de tokens <<<TOKEN>>> -> valor formateado."""
    tokens: dict[str, str] = {}

    # ---- por modelo (CV) ----
    for m in MODELS:
        short = MODEL_SHORT[m]
        sub = df[df["model"] == m]
        tokens[f"F1_{short}"]      = f"{sub['best_macro_f1'].mean():.4f}"
        tokens[f"F1_STD_{short}"]  = f"{sub['best_macro_f1'].std():.4f}"
        tokens[f"ACC_{short}"]     = f"{sub['accuracy'].mean():.4f}"
        tokens[f"BACC_{short}"]    = f"{sub['balanced_accuracy'].mean():.4f}"
        tokens[f"WF1_{short}"]     = f"{sub['weighted_f1'].mean():.4f}"
        tokens[f"BEST_EPOCH_{short}"]   = f"{sub['best_epoch'].mean():.1f}"
        tokens[f"EPOCH_TIME_{short}"]   = f"{sub['total_run_time_s'].mean() / sub['best_epoch'].mean():.1f}"

    # ---- ranking ----
    rank = df.groupby("model")["best_macro_f1"].mean().sort_values(ascending=False)
    best_model, best_f1 = rank.index[0], rank.iloc[0]
    second_model, second_f1 = rank.index[1], rank.iloc[1]
    worst_f1 = rank.iloc[-1]
    tokens["BEST_MODEL_GLOBAL"]   = best_model
    tokens["BEST_F1_GLOBAL"]      = f"{best_f1:.4f}"
    tokens["BEST_F1_STD_GLOBAL"]  = f"{df[df['model']==best_model]['best_macro_f1'].std():.4f}"
    tokens["SECOND_MODEL_GLOBAL"] = second_model
    tokens["SECOND_F1_GLOBAL"]    = f"{second_f1:.4f}"
    tokens["DELTA_F1_GLOBAL"]     = f"{best_f1 - worst_f1:.4f}"
    tokens["DELTA_F1_TOP"]        = f"{best_f1 - second_f1:.4f}"

    # ---- dispersion ----
    stds = df.groupby("model")["best_macro_f1"].std()
    tokens["MIN_F1_STD"] = f"{stds.min():.4f}"
    tokens["MAX_F1_STD"] = f"{stds.max():.4f}"

    # ---- por clase ----
    for cls, short_cls in CLASS_SHORT.items():
        for m in MODELS:
            short_m = MODEL_SHORT[m]
            row = t_class[(t_class["model"] == m) & (t_class["class"] == cls)]
            if not row.empty:
                tokens[f"F1_{short_cls}_{short_m}"] = f"{row['f1_mean'].iloc[0]:.4f}"

    # ---- top / bottom classes por modelo ----
    for m in MODELS:
        short = MODEL_SHORT[m]
        sub = t_class[t_class["model"] == m].sort_values("f1_mean", ascending=False)
        top2 = sub.head(2)["class"].tolist()
        bot2 = sub.tail(2)["class"].tolist()
        tokens[f"TOP_CLASSES_{short}"] = " y ".join(top2)
        tokens[f"BOTTOM_CLASSES_{short}"] = " y ".join(bot2)
        # Main confusions: pairs con mayor confusion off-diagonal
        agg = aggregate_predictions_cv(m)
        if agg is not None:
            yt, yp = agg
            cm = confusion_matrix(yt, yp, labels=list(range(C.NUM_CLASSES)),
                                  normalize="true")
            np.fill_diagonal(cm, 0)
            idxs = np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape)
            top_conf = []
            for k in range(3):
                i, j = idxs[0][k], idxs[1][k]
                top_conf.append(f"{C.CLASSES[i]} confundido como {C.CLASSES[j]} ({cm[i,j]*100:.1f}%)")
            tokens[f"MAIN_CONFUSIONS_{short}"] = "; ".join(top_conf)

    # ---- clase con mejor / peor F1 global ----
    g_class = t_class.groupby("class")["f1_mean"].mean().sort_values(ascending=False)
    tokens["BEST_CLASS_GLOBAL"]  = g_class.index[0]
    tokens["WORST_CLASS_GLOBAL"] = g_class.index[-1]

    # ---- Nematode range ----
    nema = t_class[t_class["class"] == "Nematode"]["f1_mean"]
    tokens["NEMA_MIN"] = f"{nema.min():.4f}"
    tokens["NEMA_MAX"] = f"{nema.max():.4f}"
    tokens["NEMA_RANGE"] = f"{nema.max() - nema.min():.4f}"

    # ---- deltas pre/post tuning (solo MobileNet y ViT) ----
    baseline_mobile = 0.6771
    baseline_vit = 0.8780
    f1_mobile = df[df["model"] == "mobilenetv2_100"]["best_macro_f1"].mean()
    f1_vit = df[df["model"] == "vit_base_patch16_224"]["best_macro_f1"].mean()
    tokens["DELTA_F1_MOBILE"] = f"+{f1_mobile - baseline_mobile:.4f}"
    tokens["DELTA_F1_VIT"]    = f"+{f1_vit - baseline_vit:.4f}"

    # ---- Holdout ----
    for m in MODELS:
        short = MODEL_SHORT[m]
        mdat = holdout.get("models", {}).get(m, {})
        ind = mdat.get("individual", {})
        ens = mdat.get("ensemble", {})
        if "macro_f1" in ind:
            tokens[f"HD_F1_{short}_IND"] = f"{ind['macro_f1']:.4f}"
        if "macro_f1" in ens:
            tokens[f"HD_F1_{short}_ENS"] = f"{ens['macro_f1']:.4f}"
        if "macro_f1" in ind and "macro_f1" in ens:
            tokens[f"HD_DELTA_{short}"] = f"+{ens['macro_f1'] - ind['macro_f1']:.4f}"

    # Ranking holdout ensemble
    hd_rank = []
    for m in MODELS:
        ens = holdout.get("models", {}).get(m, {}).get("ensemble", {})
        if "macro_f1" in ens:
            hd_rank.append((m, ens["macro_f1"]))
    hd_rank.sort(key=lambda x: -x[1])
    if hd_rank:
        tokens["HD_BEST_MODEL"] = hd_rank[0][0]
        tokens["HD_BEST_F1"]    = f"{hd_rank[0][1]:.4f}"

    # Cuantos modelos se benefician del ensemble
    n_ens_helps = 0
    for m in MODELS:
        mdat = holdout.get("models", {}).get(m, {})
        ind = mdat.get("individual", {}).get("macro_f1")
        ens = mdat.get("ensemble", {}).get("macro_f1")
        if ind is not None and ens is not None and ens > ind:
            n_ens_helps += 1
    tokens["N_ENSEMBLE_HELPS"] = str(n_ens_helps)

    # ---- McNemar ----
    pair_map = {
        ("vit_base_patch16_224", "mobilenetv2_100"):  "VIT_MOB",
        ("vit_base_patch16_224", "resnet50"):         "VIT_RES",
        ("vit_base_patch16_224", "efficientnet_b0"):  "VIT_EFF",
        ("resnet50", "mobilenetv2_100"):              "RES_MOB",
        ("efficientnet_b0", "mobilenetv2_100"):       "EFF_MOB",
        ("resnet50", "efficientnet_b0"):              "RES_EFF",
    }
    n_significant = 0
    significant_pairs = []
    nonsignificant_pairs = []
    for (a, b), code in pair_map.items():
        key1 = f"{a}__vs__{b}"
        key2 = f"{b}__vs__{a}"
        mc = mcnemar.get(key1) or mcnemar.get(key2)
        if mc is None: continue
        b_val = mc["b_only_A_correct"] if key1 in mcnemar else mc["c_only_B_correct"]
        c_val = mc["c_only_B_correct"] if key1 in mcnemar else mc["b_only_A_correct"]
        tokens[f"MCN_{code}_BC"]   = f"({b_val}, {c_val})"
        tokens[f"MCN_{code}_STAT"] = f"{mc['statistic']:.4f}"
        tokens[f"MCN_{code}_P"]    = f"{mc['pvalue']:.4f}"
        is_sig = mc['pvalue'] < 0.05
        tokens[f"MCN_{code}_DEC"]  = "Rechazar H0" if is_sig else "No rechazar H0"
        pair_label = f"{a} vs {b}"
        if is_sig:
            n_significant += 1
            significant_pairs.append(pair_label)
        else:
            nonsignificant_pairs.append(pair_label)
    tokens["N_SIGNIFICANT"]          = str(n_significant)
    tokens["SIGNIFICANT_PAIRS"]      = "; ".join(significant_pairs) if significant_pairs else "ninguno"
    tokens["NONSIGNIFICANT_PAIRS"]   = "; ".join(nonsignificant_pairs) if nonsignificant_pairs else "ninguno"

    # ViT vs CNN result
    vit_vs_cnn_codes = ["VIT_MOB", "VIT_RES", "VIT_EFF"]
    n_vit_sig = sum(1 for c in vit_vs_cnn_codes if tokens.get(f"MCN_{c}_DEC") == "Rechazar H0")
    if n_vit_sig == 3:
        tokens["VIT_VS_CNN_RESULT"] = "todos significativos al 5 %"
    elif n_vit_sig == 0:
        tokens["VIT_VS_CNN_RESULT"] = "ninguno significativo al 5 %"
    else:
        tokens["VIT_VS_CNN_RESULT"] = f"{n_vit_sig} de 3 significativos al 5 %"

    # Consistencia CV-holdout
    diffs = []
    for m in MODELS:
        cv = df[df["model"] == m]["best_macro_f1"].mean()
        hd = holdout.get("models", {}).get(m, {}).get("ensemble", {}).get("macro_f1")
        if hd is not None:
            diffs.append(abs(hd - cv))
    max_diff = max(diffs) if diffs else 0
    tokens["CV_HOLDOUT_CONSISTENCY"] = (
        f"alta (diferencia maxima {max_diff:.4f})" if max_diff < 0.05 else
        f"moderada (diferencia maxima {max_diff:.4f})"
    )

    return tokens


def main():
    _set_style()
    df = load_final_master()
    holdout = load_holdout_eval()

    # Tablas
    tabla_f1 = table_macro_f1(df)
    tabla_f1.to_csv(C.RESULTS_DIR / "tabla_macroF1_por_modelo.csv", index=False)
    tabla_cls = table_f1_per_class(df)
    tabla_cls.to_csv(C.RESULTS_DIR / "tabla_f1_por_clase.csv", index=False)

    # Figuras
    figs_dir = C.FIGS_DIR
    for m in MODELS:
        agg = aggregate_predictions_cv(m)
        if agg is None: continue
        yt, yp = agg
        plot_confusion(m, yt, yp, figs_dir / f"cm_{m}.png")
        plot_learning_curves(m, figs_dir / f"lc_{m}.png")
        print(f"[OK] figuras de {m}")

    # McNemar
    mc = run_mcnemar_holdout()
    (C.RESULTS_DIR / "mcnemar_holdout.json").write_text(
        json.dumps(mc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] McNemar pairwise: {len(mc)} contrastes")

    # Tokens
    tokens = build_tokens(df, tabla_cls, holdout, mc)
    (C.RESULTS_DIR / "tokens_replace.json").write_text(
        json.dumps(tokens, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {len(tokens)} tokens generados")

    print(f"\n[OK] Tablas en {C.RESULTS_DIR}")
    print(f"[OK] Figuras en {C.FIGS_DIR}")
    print(f"[OK] tokens_replace.json listo para inyectar en docx")


if __name__ == "__main__":
    main()
