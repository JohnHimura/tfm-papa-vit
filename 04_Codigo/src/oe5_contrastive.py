"""
OE5 — Estudio contrastivo ViT vs ResNet (40 pares divergentes) + kappa de Cohen.

Selecciona los pares de mapas mas DIVERGENTES entre el Attention Rollout del ViT
y el Grad-CAM de ResNet-50 (IoU entre mapas binarizados < 0.3), los exporta para
juicio CIEGO inter-evaluador, y calcula el acuerdo (kappa de Cohen) a partir de
las dos columnas de etiquetas que los evaluadores rellenen.

Las etiquetas posibles del juicio ciego son:
    'vit'    -> el mapa A (anonimo) es fitopatologicamente mas plausible
    'resnet' -> el mapa B (anonimo) es mas plausible
    'tie'    -> empate / ambos igual de plausibles

IMPORTANTE: este script NO inventa los juicios. Genera la plantilla vacia
(results/xai/contrastive/juicios_TEMPLATE.csv) con columnas
[pair_id, sample_id, evaluator1, evaluator2] que los dos evaluadores completan.
El mapeo A/B<->modelo se guarda CIFRADO en un archivo aparte (key) para no
sesgar al evaluador; kappa se calcula despues con --kappa.

Uso:
    python -m src.oe5_contrastive select        # elige 40 pares, exporta laminas
    python -m src.oe5_contrastive kappa          # tras rellenar los juicios
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import xai
from . import run_oe5_xai as OE5

CONTRAST_DIR = OE5.XAI_DIR / "contrastive"
PAIRS_CSV = CONTRAST_DIR / "pares_divergentes.csv"
KEY_JSON = CONTRAST_DIR / "ab_key.json"            # mapeo A/B -> modelo (no mostrar)
JUDGE_TEMPLATE = CONTRAST_DIR / "juicios_TEMPLATE.csv"

N_PAIRS = 40
IOU_DIVERGENCE_THRESHOLD = 0.3
OE5_SEED = OE5.OE5_SEED


PER_IMAGE_CSV = OE5.XAI_DIR / "oe5_per_image.csv"


def _lesion_sample_ids() -> set[str]:
    """sample_id de las imagenes CON lesion delimitada (la misma base de la Tabla 15).

    El juicio contrastivo pregunta que mapa localiza mejor la LESION, de modo que
    solo tiene sentido sobre las clases sintomaticas. Se excluye Healthy (sin lesion)
    y cualquier imagen cuya mascara se descarto en el control de calidad.
    """
    if not PER_IMAGE_CSV.exists():
        print(f"[WARN] no existe {PER_IMAGE_CSV}; no se puede restringir a imagenes "
              f"con lesion. Ejecuta antes `run_oe5_xai`.")
        return set()
    df = pd.read_csv(PER_IMAGE_CSV)
    return set(df["sample_id"].unique())


def select_divergent_pairs(threshold: float = IOU_DIVERGENCE_THRESHOLD,
                           n_pairs: int = N_PAIRS,
                           binarize_method: str = "percentile",
                           percentile: float = 80.0) -> pd.DataFrame:
    """Selecciona los pares con IoU(mapa_ViT, mapa_ResNet) < threshold.

    Requiere que los heatmaps de ambos modelos ya esten generados
    (run_oe5_xai generate para resnet50 y vit_base_patch16_224).

    La seleccion se restringe a las imagenes CON lesion delimitada (misma base que
    la Tabla 15, 6 clases sintomaticas): juzgar que mapa localiza mejor la lesion
    carece de sentido en hojas sanas.

    Si hay mas de n_pairs candidatos, toma los n_pairs de MENOR IoU (mas
    divergentes); si hay menos, los toma todos y avisa.
    """
    sample = OE5.load_sample()
    lesion_ids = _lesion_sample_ids()
    if lesion_ids:
        n_before = len(sample)
        sample = sample[sample["sample_id"].isin(lesion_ids)].copy()
        print(f"[INFO] restringido a imagenes con lesion: {len(sample)} de {n_before} "
              f"(se excluyen Healthy y mascaras descartadas)")
    rdir = OE5.HEATMAP_DIR / OE5.RESNET_KEY
    vdir = OE5.HEATMAP_DIR / OE5.VIT_KEY
    rows = []
    n_missing = 0
    for _, r in sample.iterrows():
        sid = r["sample_id"]
        rp, vp = rdir / f"{sid}.npy", vdir / f"{sid}.npy"
        if not (rp.exists() and vp.exists()):
            n_missing += 1
            continue
        rh, vh = np.load(rp), np.load(vp)
        rb = xai.binarize_heatmap(rh, method=binarize_method, percentile=percentile)
        vb = xai.binarize_heatmap(vh, method=binarize_method, percentile=percentile)
        rows.append({"sample_id": sid, "class_name": r["class_name"],
                     "map_iou": xai.iou(rb, vb)})
    if not rows:
        print(f"[BLOQUEO] No hay heatmaps de ambos modelos. Genera primero los "
              f"mapas (faltan {n_missing}).")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("map_iou").reset_index(drop=True)
    divergent = df[df["map_iou"] < threshold]
    chosen = divergent.head(n_pairs).copy()
    if len(chosen) < n_pairs:
        print(f"[WARN] solo {len(chosen)} pares con IoU<{threshold} "
              f"(se pedian {n_pairs}). Se exportan todos los disponibles.")
    chosen["pair_id"] = [f"P{i:02d}" for i in range(len(chosen))]

    # asignar A/B aleatoriamente por par (cegado), guardar key aparte
    rng = random.Random(OE5_SEED)
    key = {}
    a_is_vit = []
    for pid in chosen["pair_id"]:
        v = rng.random() < 0.5
        a_is_vit.append(v)
        key[pid] = {"A": "vit" if v else "resnet",
                    "B": "resnet" if v else "vit"}
    chosen["A_model"] = ["vit" if v else "resnet" for v in a_is_vit]
    chosen["B_model"] = ["resnet" if v else "vit" for v in a_is_vit]

    CONTRAST_DIR.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(PAIRS_CSV, index=False)
    KEY_JSON.write_text(json.dumps(key, indent=2), encoding="utf-8")

    # plantilla de juicios vacia (los evaluadores la rellenan)
    template = chosen[["pair_id", "sample_id", "class_name"]].copy()
    template["evaluator1"] = ""   # rellenar con vit / resnet / tie
    template["evaluator2"] = ""
    template.to_csv(JUDGE_TEMPLATE, index=False)

    print(f"[OK] {len(chosen)} pares divergentes -> {PAIRS_CSV}")
    print(f"[OK] key cegada (A/B) -> {KEY_JSON}  (NO mostrar a evaluadores)")
    print(f"[OK] plantilla de juicios -> {JUDGE_TEMPLATE}")
    print("     Los evaluadores rellenan evaluator1/evaluator2 con: vit | resnet | tie")
    _export_pair_sheets(chosen)
    return chosen


def _export_pair_sheets(chosen: pd.DataFrame) -> None:
    """Genera laminas 'Hoja | A | B' (overlays anonimos) para el juicio ciego.

    Se incluye la imagen ORIGINAL sin mapa: sin ella el evaluador no puede saber
    donde esta realmente la lesion (el heatmap tapa parte de la lamina) y el juicio
    de plausibilidad fitopatologica seria inviable.

    NO se muestra la mascara de lesion a proposito: revelarla convertiria el juicio
    humano en una simple lectura del solapamiento (que es justo lo que ya mide el
    IoU de la Tabla 15) y lo haria redundante y sesgado.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as e:
        print(f"[WARN] no se pudieron generar laminas: {e}")
        return
    sheets = CONTRAST_DIR / "laminas"
    sheets.mkdir(parents=True, exist_ok=True)
    ov_r = OE5.OVERLAY_DIR / OE5.RESNET_KEY
    ov_v = OE5.OVERLAY_DIR / OE5.VIT_KEY

    # ruta de la imagen original por sample_id
    src = OE5.load_sample()
    orig_path = dict(zip(src["sample_id"], src["path"]))

    n_ok = 0
    for _, r in chosen.iterrows():
        sid, pid = r["sample_id"], r["pair_id"]
        a_path = (ov_v if r["A_model"] == "vit" else ov_r) / f"{sid}.png"
        b_path = (ov_v if r["B_model"] == "vit" else ov_r) / f"{sid}.png"
        if not (a_path.exists() and b_path.exists()):
            continue
        op = orig_path.get(sid)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
        if op and Path(op).exists():
            ax[0].imshow(Image.open(op).convert("RGB"))
        ax[0].set_title("Hoja (sin mapa)", fontsize=11)
        ax[0].axis("off")
        ax[1].imshow(Image.open(a_path)); ax[1].set_title("A", fontsize=13, fontweight="bold"); ax[1].axis("off")
        ax[2].imshow(Image.open(b_path)); ax[2].set_title("B", fontsize=13, fontweight="bold"); ax[2].axis("off")
        fig.suptitle(f"{pid}", fontsize=11)
        fig.tight_layout()
        fig.savefig(sheets / f"{pid}.png", dpi=110)
        plt.close(fig)
        n_ok += 1
    print(f"[OK] {n_ok} laminas 'Hoja | A | B' -> {sheets}")


def merge_votes(e1_csv: Path, e2_csv: Path) -> pd.DataFrame:
    """Fusiona los CSV de los dos evaluadores y los DESCIEGA con ab_key.json.

    Cada evaluador exporta desde el HTML un CSV [pair_id, voto] con voto en
    {A, B, tie}. Aqui se traduce A/B -> vit/resnet usando la key que el evaluador
    nunca vio, y se rellena la plantilla [pair_id, sample_id, class_name,
    evaluator1, evaluator2] que consume `cohen_kappa`.
    """
    key = json.loads(KEY_JSON.read_text(encoding="utf-8"))
    pairs = pd.read_csv(PAIRS_CSV)

    def _load(path: Path, who: str) -> dict:
        df = pd.read_csv(path)
        if "voto" not in df.columns or "pair_id" not in df.columns:
            raise ValueError(f"{path} debe tener columnas [pair_id, voto]")
        out = {}
        for _, r in df.iterrows():
            pid, v = str(r["pair_id"]).strip(), str(r["voto"]).strip().lower()
            if v in ("a", "b"):
                out[pid] = key[pid][v.upper()]      # A/B -> vit|resnet
            elif v in ("tie", "empate"):
                out[pid] = "tie"
            elif v:
                raise ValueError(f"{path}: voto no valido {v!r} en {pid} "
                                 f"(use A, B o tie)")
        print(f"[OK] {who}: {len(out)} juicios leidos de {path.name}")
        return out

    v1, v2 = _load(e1_csv, "evaluator1"), _load(e2_csv, "evaluator2")
    t = pairs[["pair_id", "sample_id", "class_name"]].copy()
    t["evaluator1"] = t["pair_id"].map(v1).fillna("")
    t["evaluator2"] = t["pair_id"].map(v2).fillna("")
    t.to_csv(JUDGE_TEMPLATE, index=False)
    n_both = int(((t.evaluator1 != "") & (t.evaluator2 != "")).sum())
    print(f"[OK] plantilla rellenada -> {JUDGE_TEMPLATE} ({n_both} pares con ambos votos)")
    return t


def cohen_kappa(judge_csv: Path | None = None) -> dict:
    """Calcula kappa de Cohen entre evaluator1 y evaluator2.

    Lee el CSV de juicios (por defecto JUDGE_TEMPLATE una vez rellenado).
    Ignora filas con algun voto vacio. Devuelve dict con kappa, acuerdo
    observado y conteo.
    """
    from sklearn.metrics import cohen_kappa_score
    path = judge_csv or JUDGE_TEMPLATE
    if not path.exists():
        raise FileNotFoundError(f"no existe {path} — rellena los juicios primero")
    df = pd.read_csv(path).fillna("")
    valid = df[(df["evaluator1"].astype(str).str.strip() != "") &
               (df["evaluator2"].astype(str).str.strip() != "")].copy()
    if valid.empty:
        print("[BLOQUEO] el CSV de juicios esta vacio — rellena evaluator1/2.")
        return {"n": 0}
    e1 = valid["evaluator1"].str.strip().str.lower()
    e2 = valid["evaluator2"].str.strip().str.lower()
    labels = ["vit", "resnet", "tie"]
    kappa = float(cohen_kappa_score(e1, e2, labels=labels))
    agree = float((e1.values == e2.values).mean())
    res = {"n": int(len(valid)), "cohen_kappa": kappa,
           "observed_agreement": agree,
           "vit_votes": int((e1 == "vit").sum() + (e2 == "vit").sum()),
           "resnet_votes": int((e1 == "resnet").sum() + (e2 == "resnet").sum()),
           "tie_votes": int((e1 == "tie").sum() + (e2 == "tie").sum())}
    out = CONTRAST_DIR / "kappa_result.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] kappa de Cohen = {kappa:.3f} (n={res['n']}, "
          f"acuerdo={agree:.1%}) -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description="OE5 estudio contrastivo + kappa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="elige 40 pares divergentes y exporta laminas")
    s.add_argument("--threshold", type=float, default=IOU_DIVERGENCE_THRESHOLD)
    s.add_argument("--percentile", type=float, default=80.0)
    m = sub.add_parser("merge", help="fusiona y desciega los CSV de los 2 evaluadores")
    m.add_argument("--e1", type=str, required=True, help="CSV del evaluador 1")
    m.add_argument("--e2", type=str, required=True, help="CSV del evaluador 2")
    k = sub.add_parser("kappa", help="calcula kappa tras rellenar los juicios")
    k.add_argument("--judge-csv", type=str, default=None)
    args = ap.parse_args()
    if args.cmd == "select":
        select_divergent_pairs(threshold=args.threshold, percentile=args.percentile)
    elif args.cmd == "merge":
        merge_votes(Path(args.e1), Path(args.e2))
        cohen_kappa()
    elif args.cmd == "kappa":
        cohen_kappa(Path(args.judge_csv) if args.judge_csv else None)


if __name__ == "__main__":
    main()
