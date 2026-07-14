"""
OE5 — PROPUESTA AUTOMATICA de mascaras de lesion (para REVISION HUMANA).

Este script NO fabrica resultados de las Tablas 15/16. PROPONE, por cada imagen
del holdout con lesion (las de results/xai/to_annotate/, clases != Healthy), una
mascara binaria de la zona enferma, para que el usuario la confirme o comente en
un HTML (ver src/oe5_build_review_html.py). El calculo real de IoU / Pointing
Game (src/run_oe5_xai.py) leera las mascaras FINALES de results/xai/masks/.

Dos metodos de segmentacion, con degradacion automatica:

  1) SAM  (Segment Anything, checkpoint vit_b ~375 MB).  Se usa SOLO si el
     checkpoint existe en models/sam_vit_b_01ec64.pth (o MOBILE_SAM). Estrategia
     de prompting para aislar la LESION (no toda la hoja): se calcula un "indice
     de enfermedad" por anomalia de color (desviacion respecto al verde sano en
     Lab/HSV); los puntos de mayor anomalia se pasan a SAM como prompts
     foreground, y un punto del verde sano como background. La mascara SAM
     resultante se interseca con la anomalia de color para descartar fondo/hoja
     sana.

  2) CLASICO (fallback robusto, SIN descargas).  Segmentacion por color de la
     lesion: se mide la anomalia respecto al verde sano en Lab+HSV, se umbraliza
     con Otsu, se limpia con morfologia (apertura/cierre) y se descarta fondo
     oscuro/claro. cv2 + scipy bastan; siempre disponible.

El script registra POR IMAGEN que metodo uso y por que (masks_meta.json), genera
un overlay visual y una OBSERVACION en lenguaje claro NO tecnico (ubicacion,
tamano y color dominante de la zona marcada).

Salidas:
    results/xai/masks/<sample_id>.png            (PNG binario: lesion>0)
    results/xai/review/overlays/<sample_id>.png  (overlay para revision)
    results/xai/masks_meta.json                  (metodo, area %, centroide, obs)

Uso:
    # Run completo (110 imgs con lesion):
    python -m src.oe5_sam_masks
    # Smoke test (N imgs, no toca el run real si --out-suffix):
    python -m src.oe5_sam_masks --limit 3
    # Aplicar revision humana exportada desde el HTML:
    python -m src.oe5_sam_masks --apply-review <decisiones.json|.csv>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from . import config as C
    _XAI_DIR = C.RESULTS_DIR / "xai"
except Exception:  # permite ejecutar el archivo suelto sin el paquete
    _XAI_DIR = Path(__file__).resolve().parents[1] / "results" / "xai"

# -----------------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------------
XAI_DIR = _XAI_DIR
TO_ANNOTATE_DIR = XAI_DIR / "to_annotate"
MASKS_DIR = XAI_DIR / "masks"
REVIEW_DIR = XAI_DIR / "review"
OVERLAYS_DIR = REVIEW_DIR / "overlays"
META_JSON = XAI_DIR / "masks_meta.json"

# Checkpoints SAM posibles (se usa el primero que exista)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SAM_CHECKPOINTS = [
    (MODELS_DIR / "sam_vit_b_01ec64.pth", "vit_b"),
    (MODELS_DIR / "mobile_sam.pt", "vit_t"),
]

# Clases SIN lesion (no se proponen mascaras; el IoU/PG las ignora)
HEALTHY_CLASS = "Healthy"


# =============================================================================
# 1) INDICE DE ENFERMEDAD POR ANOMALIA DE COLOR (semilla comun a ambos metodos)
# =============================================================================
def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Mascara de PIEL (manos del fotografo) en YCrCb — para EXCLUIR de la hoja."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.int32)
    cb = ycrcb[:, :, 2].astype(np.int32)
    skin = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k, iterations=1)
    return skin


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Rellena huecos internos (p. ej. lesion no-verde rodeada de hoja verde)."""
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 1)          # rellena el fondo exterior
    holes = (ff == 0).astype(np.uint8)             # 0 restantes = huecos internos
    return ((m | holes) > 0).astype(np.uint8)


def _leaf_mask_grabcut(rgb: np.ndarray) -> np.ndarray:
    """Fallback: hoja muy enferma (poco verde) -> GrabCut con rectangulo central."""
    h, w = rgb.shape[:2]
    rect = (int(0.12 * w), int(0.12 * h), int(0.76 * w), int(0.76 * h))
    gc = np.zeros((h, w), np.uint8)
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, gc, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        leaf = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    except Exception:
        leaf = np.zeros((h, w), np.uint8)
        leaf[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1
    leaf[_skin_mask(rgb) > 0] = 0                  # nunca incluir manos
    if leaf.sum() < 0.03 * h * w:                  # GrabCut fallo -> rectangulo
        leaf = np.zeros((h, w), np.uint8)
        leaf[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1
        leaf[_skin_mask(rgb) > 0] = 0
    return _fill_holes(leaf)


# Flag: aislar el sujeto-hoja con SAM (prompt CENTRAL). --no-sam lo apaga.
USE_SAM_LEAF = True
_LEAF_METHOD = "?"


def _leaf_mask_sam_center(rgb: np.ndarray):
    """Hoja-sujeto via SAM con prompt en el CENTRO (fg) y esquinas (bg).

    Arreglo 25/06: los prompts sobre la ANOMALIA de color caian en la mano del
    fotografo. En este dataset el sujeto esta centrado, asi que un prompt central
    aisla la hoja correcta AUNQUE el fondo sea tambien vegetacion verde. Devuelve
    None si SAM no esta disponible (el caller cae a segmentacion por color).
    """
    predictor, _ = _try_load_sam()
    if predictor is None:
        return None
    h, w = rgb.shape[:2]
    fg = [[w // 2, h // 2], [w // 2, int(h * 0.40)], [w // 2, int(h * 0.60)],
          [int(w * 0.40), h // 2], [int(w * 0.60), h // 2]]
    bg = [[int(w * 0.04), int(h * 0.04)], [int(w * 0.96), int(h * 0.04)],
          [int(w * 0.04), int(h * 0.96)], [int(w * 0.96), int(h * 0.96)]]
    pts = np.array(fg + bg)
    lbl = np.array([1] * len(fg) + [0] * len(bg))
    try:
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=lbl, multimask_output=True)
    except Exception:
        return None
    cy, cx = h // 2, w // 2
    best, bs = None, -1.0
    for m, sc in zip(masks, scores):
        frac = float(m.mean())
        if frac > 0.92 or frac < 0.03 or not m[cy, cx]:  # casi todo/nada o sin centro
            continue
        if sc > bs:
            bs, best = float(sc), m
    if best is None:
        best = masks[int(np.argmax(scores))]
    leaf = best.astype(np.uint8)
    leaf[_skin_mask(rgb) > 0] = 0          # nunca incluir la mano
    return _fill_holes(leaf)


def _leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Mascara de la HOJA = vegetacion verde, NO el fondo.

    Arreglo (25/06): la version previa solo descartaba negro/blanco extremo, por
    lo que manos (piel), tierra/tallos rojizos y sombras entraban como 'hoja' y
    luego se marcaban como lesion. Ahora:
      1) vegetacion verde amplia (HSV verde-amarillo saturado U Lab a* verde),
      2) se EXCLUYE la piel (manos),
      3) mayor componente conexo = cuerpo de la hoja,
      4) se rellenan huecos (lesiones internas no-verdes quedan DENTRO de la hoja),
      5) dilatacion moderada para incluir el margen enfermo adyacente.
    Si no hay verde suficiente (hoja muy enferma), cae a GrabCut central.
    Antes de todo, si SAM esta activo, se aisla el sujeto-hoja con prompt central.
    """
    global _LEAF_METHOD
    if USE_SAM_LEAF:
        sam_leaf = _leaf_mask_sam_center(rgb)
        if sam_leaf is not None and 0.07 < float(sam_leaf.mean()) < 0.95:
            _LEAF_METHOD = "sam_center"
            return sam_leaf
        # SAM devolvio una region diminuta (p. ej. hoja marchita sobre mano):
        # mejor caer a color/GrabCut que quedarse con un trozo de fondo.
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].astype(np.int32)   # 0..179
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)
    A = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1].astype(np.int32)  # <128 = verde

    green = (((H >= 25) & (H <= 95) & (S >= 40) & (V >= 40)) | (A <= 116)).astype(np.uint8)
    green[_skin_mask(rgb) > 0] = 0      # las manos NO son hoja
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k, iterations=2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    if n <= 1 or stats[1:, cv2.CC_STAT_AREA].max() < 0.03 * h * w:
        _LEAF_METHOD = "grabcut"
        return _leaf_mask_grabcut(rgb)   # poco verde -> hoja muy enferma
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    core = (lab == biggest).astype(np.uint8)
    core = _fill_holes(core)                         # incluye lesiones internas
    core = cv2.dilate(core, k, iterations=3)          # margen enfermo adyacente
    core = _fill_holes(core)
    core[_skin_mask(rgb) > 0] = 0                     # belt+suspenders: sin manos
    _LEAF_METHOD = "green_cc"
    return (core > 0).astype(np.uint8)


def disease_index(rgb: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    """
    Mapa continuo [0,1] de "anomalia respecto al verde sano".

    Mezcla tres senales tipicas de lesion foliar:
      - menor componente 'a' verde / mayor rojez (Lab canal a alto = rojizo).
      - amarilleo / oscurecimiento (Lab canal b, y baja luminosidad L).
      - perdida de verde (HSV hue fuera del rango verde) y saturacion baja.
    Se normaliza usando la mediana del tejido sano (verde) como referencia.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:, :, 0]  # 0..179 en OpenCV

    leaf_bool = leaf.astype(bool)
    if leaf_bool.sum() < 50:
        leaf_bool = np.ones_like(leaf_bool)

    # "Verde sano" = pixeles de hoja con hue verde (35..85 en OpenCV) y a bajo
    green = leaf_bool & (h > 35) & (h < 85)
    ref = green if green.sum() > 0.05 * leaf_bool.sum() else leaf_bool

    a_ref = np.median(a[ref])
    b_ref = np.median(b[ref])
    L_ref = np.median(L[ref])

    # Componentes de anomalia (clip a 0 = no mas verde-sano que la referencia)
    redness = np.clip(a - a_ref, 0, None)          # rojizo/marron
    yellowing = np.clip(b - b_ref, 0, None)        # amarilleo
    darkening = np.clip(L_ref - L, 0, None)        # oscurecimiento
    s = hsv[:, :, 1] / 255.0                       # saturacion 0..1
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)  # cromaticidad (gris~0)

    # SOMBRA vs NECROSIS: una sombra es oscura PERO neutra (poca saturacion y poca
    # cromaticidad); la necrosis es oscura PERO marron (cromatica). El oscurecimiento
    # solo cuenta donde hay color, y las sombras neutras se ANULAN explicitamente.
    colored = ((s > 0.20) | (chroma > 18)).astype(np.float32)
    shadow = (s < 0.18) & (darkening > 25) & (chroma < 14)

    def _norm(x):
        m = x[leaf_bool]
        p = np.percentile(m, 99) if m.size else 1.0
        return np.clip(x / (p + 1e-6), 0, 1)

    idx = (0.55 * _norm(redness) + 0.35 * _norm(yellowing)
           + 0.10 * _norm(darkening) * colored)
    idx[shadow] = 0.0              # sombras neutras NO son lesion
    idx = idx * leaf_bool  # fuera de la hoja, 0
    # Suavizado leve para semillas mas estables
    idx = cv2.GaussianBlur(idx.astype(np.float32), (0, 0), sigmaX=1.5)
    if idx.max() > 0:
        idx = idx / idx.max()
    return idx


def _clean_binary(mask: np.ndarray) -> np.ndarray:
    """Limpieza morfologica + descarta motas pequenas."""
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    keep = np.zeros_like(m)
    min_area = max(20, int(0.002 * m.size))  # >=0.2% de la imagen
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep


# =============================================================================
# 2) METODO CLASICO (fallback, sin descargas)
# =============================================================================
def segment_classic(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    leaf = _leaf_mask(rgb)
    # Margen interno: erosionar evita el borde de la hoja (suele traer sombra/halo).
    leaf_in = cv2.erode(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        iterations=1)
    if leaf_in.sum() < 0.3 * max(1, int(leaf.sum())):
        leaf_in = leaf
    idx = disease_index(rgb, leaf_in)
    lb = leaf_in.astype(bool)
    vals = (idx[lb] * 255).astype(np.uint8)
    info = {"method": f"leaf:{_LEAF_METHOD}", "leaf_pct": round(100 * float(leaf.mean()), 1)}
    if vals.size == 0 or vals.max() == 0:
        return np.zeros(rgb.shape[:2], np.uint8), {**info, "reason": "sin_anomalia"}
    # Otsu sobre el indice de enfermedad dentro de la hoja
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_frac = thr / 255.0
    # piso de seguridad: al menos percentil alto si Otsu cae muy bajo
    thr_frac = max(thr_frac, float(np.percentile(idx[lb], 75)))
    mask = ((idx >= thr_frac) & lb).astype(np.uint8)
    mask = _clean_binary(mask)
    info["otsu_threshold"] = round(float(thr_frac), 3)
    return mask, info


# =============================================================================
# 3) METODO SAM (solo si hay checkpoint)
# =============================================================================
_SAM_PREDICTOR = None
_SAM_LOADED_FROM = None


def _try_load_sam():
    """Carga perezosa de SAM. Devuelve (predictor, etiqueta) o (None, motivo)."""
    global _SAM_PREDICTOR, _SAM_LOADED_FROM
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR, _SAM_LOADED_FROM
    ckpt = next(((p, t) for p, t in SAM_CHECKPOINTS if p.exists()), None)
    if ckpt is None:
        return None, "no_checkpoint"
    path, model_type = ckpt
    try:
        import torch
        if model_type == "vit_t":  # MobileSAM
            from mobile_sam import sam_model_registry, SamPredictor  # type: ignore
        else:
            from segment_anything import sam_model_registry, SamPredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=str(path)).to(device)
        sam.eval()
        _SAM_PREDICTOR = SamPredictor(sam)
        _SAM_LOADED_FROM = f"sam_{model_type}@{device}"
        return _SAM_PREDICTOR, _SAM_LOADED_FROM
    except Exception as e:  # libreria o pesos rotos -> fallback
        return None, f"sam_load_error:{type(e).__name__}"


def segment_sam(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    predictor, label = _try_load_sam()
    if predictor is None:
        return None, {"method": "sam_unavailable", "reason": label}

    leaf = _leaf_mask(rgb)
    idx = disease_index(rgb, leaf)
    if idx.max() == 0:
        return np.zeros(rgb.shape[:2], np.uint8), {"method": label, "reason": "sin_anomalia"}

    # Semillas foreground = picos de anomalia; background = verde sano
    fg_pts = _peak_points(idx, k=3, min_val=0.5)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = leaf.astype(bool) & (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (idx < 0.15)
    bg_pts = _sample_points(green, k=2)

    if len(fg_pts) == 0:
        return segment_classic(rgb)[0], {"method": label, "fallback": "sin_picos->classic"}

    pts = np.array(fg_pts + bg_pts)
    lbls = np.array([1] * len(fg_pts) + [0] * len(bg_pts))

    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=lbls, multimask_output=True
    )
    # Elegir la mascara que mejor solapa la anomalia y NO cubre toda la hoja
    best, best_score = None, -1.0
    anom = (idx > 0.3).astype(np.uint8)
    leaf_area = max(1, int(leaf.sum()))
    for m in masks:
        mb = (m & leaf.astype(bool))
        area_frac = mb.sum() / leaf_area
        if area_frac > 0.9 or area_frac < 0.005:  # casi toda la hoja o nada
            continue
        inter = np.logical_and(mb, anom).sum()
        overlap = inter / (mb.sum() + 1e-6)
        if overlap > best_score:
            best_score, best = overlap, mb
    if best is None:
        return segment_classic(rgb)[0], {"method": label, "fallback": "sam_sin_region->classic"}
    mask = _clean_binary(best.astype(np.uint8))
    return mask, {"method": label, "n_fg": len(fg_pts), "n_bg": len(bg_pts),
                  "anom_overlap": round(float(best_score), 3)}


def _peak_points(idx: np.ndarray, k: int = 3, min_val: float = 0.5):
    """Hasta k maximos locales separados del mapa de anomalia."""
    work = idx.copy()
    pts = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[y, x] < min_val:
            break
        pts.append([int(x), int(y)])
        cv2.circle(work, (x, y), 25, 0, -1)  # suprimir vecindad
    return pts


def _sample_points(mask_bool: np.ndarray, k: int = 2):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return []
    idxs = np.linspace(0, len(xs) - 1, num=min(k, len(xs))).astype(int)
    return [[int(xs[i]), int(ys[i])] for i in idxs]


# =============================================================================
# 4) OBSERVACION EN LENGUAJE CLARO (no tecnico)
# =============================================================================
def _dominant_color_name(rgb: np.ndarray, mask: np.ndarray) -> str:
    if mask.sum() == 0:
        return "sin zona marcada"
    px = rgb[mask.astype(bool)].reshape(-1, 3).astype(np.float32)
    r, g, b = px.mean(axis=0)
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if val < 60:
        return "marron oscuro casi negro"
    if sat < 40:
        return "gris parduzco" if val < 160 else "claro/blanquecino"
    if hue < 12 or hue > 168:
        return "rojizo"
    if hue < 22:
        return "marron"
    if hue < 35:
        return "amarillo-marron"
    if hue < 45:
        return "amarillento"
    return "verde apagado"


def observation_es(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Observacion humana + metricas simples (area %, centroide normalizado)."""
    h, w = mask.shape
    area_frac = float(mask.sum()) / (h * w)
    if mask.sum() == 0:
        return {
            "observacion": "No detecte una zona enferma clara; conviene revisar a mano.",
            "area_pct": 0.0, "centroid_xy": [0.5, 0.5], "size_label": "ninguna",
            "pos_label": "n/d", "color": "sin zona",
        }
    ys, xs = np.where(mask.astype(bool))
    cx, cy = float(xs.mean()) / w, float(ys.mean()) / h

    # Posicion (combina vertical + horizontal; centro si esta al medio)
    vert = "superior" if cy < 0.38 else "inferior" if cy > 0.62 else "centro"
    horiz = "izquierdo" if cx < 0.38 else "derecho" if cx > 0.62 else "centro"
    if vert == "centro" and horiz == "centro":
        pos = "el centro de la hoja"
    elif vert == "centro":
        pos = f"el borde {horiz} de la hoja"
    elif horiz == "centro":
        pos = f"el borde {vert} de la hoja"
    else:
        pos = f"la esquina {vert} {horiz} de la hoja"
    pos_label = f"{vert}-{horiz}"

    if area_frac < 0.04:
        size = "pequena"
    elif area_frac < 0.18:
        size = "mediana"
    else:
        size = "extensa"

    color = _dominant_color_name(rgb, mask)
    obs = (f"Marque una mancha {color} {size} en {pos} como la zona enferma "
           f"(ocupa ~{area_frac * 100:.0f}% de la imagen).")
    return {
        "observacion": obs,
        "area_pct": round(area_frac * 100, 2),
        "centroid_xy": [round(cx, 3), round(cy, 3)],
        "size_label": size, "pos_label": pos_label, "color": color,
    }


# =============================================================================
# 5) OVERLAY VISUAL
# =============================================================================
def make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    over = rgb.copy()
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    mb = mask.astype(bool)
    over[mb] = (0.55 * rgb[mb] + 0.45 * red[mb]).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (255, 255, 0), 2)
    return over


# =============================================================================
# 6) PIPELINE
# =============================================================================
def list_lesion_images(src_dir: Path):
    imgs = sorted(p for p in src_dir.glob("*.png")
                  if not p.name.startswith(HEALTHY_CLASS))
    return imgs


def propose_mask(rgb: np.ndarray, prefer_sam: bool) -> tuple[np.ndarray, dict]:
    # 'prefer_sam' controla si la HOJA-sujeto se aisla con SAM-centro (USE_SAM_LEAF).
    # La lesion SIEMPRE = anomalia de color DENTRO de la hoja (segment_classic).
    # Antes se prompteaba SAM sobre la anomalia y caia en la mano (arreglado 25/06).
    global USE_SAM_LEAF
    USE_SAM_LEAF = prefer_sam
    return segment_classic(rgb)


def run(limit: int | None = None, out_suffix: str = "", prefer_sam: bool = True,
        src_dir: Path | None = None):
    src = Path(src_dir) if src_dir else TO_ANNOTATE_DIR
    masks_dir = Path(str(MASKS_DIR) + out_suffix)
    overlays_dir = Path(str(OVERLAYS_DIR) + out_suffix)
    meta_path = Path(str(META_JSON).replace(".json", f"{out_suffix}.json"))
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_lesion_images(src)
    if limit:
        imgs = imgs[:limit]

    meta = {}
    method_counts = {}
    for i, p in enumerate(imgs, 1):
        rgb = np.array(Image.open(p).convert("RGB"))
        mask, info = propose_mask(rgb, prefer_sam)
        mask = (mask > 0).astype(np.uint8) * 255

        Image.fromarray(mask, mode="L").save(masks_dir / p.name)
        Image.fromarray(make_overlay(rgb, mask // 255)).save(overlays_dir / p.name)

        obs = observation_es(rgb, mask // 255)
        leaf_pct = info.get("leaf_pct", 100)
        low_conf = (leaf_pct < 10) or (obs["area_pct"] < 1.5) or (obs["area_pct"] > 40)
        if low_conf:
            obs["observacion"] = ("ATENCION, revisa con cuidado (no estoy del todo seguro de "
                                  "esta): " + obs["observacion"])
        cls = p.name.rsplit("_", 1)[0]
        meta[p.name] = {"clase": cls, "revisar": bool(low_conf), **info, **obs}
        m = info.get("method", "?")
        method_counts[m] = method_counts.get(m, 0) + 1
        print(f"[{i}/{len(imgs)}] {p.name:24s} -> {m:16s} area={obs['area_pct']:5.1f}% {obs['pos_label']}")

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMetodos usados: {method_counts}")
    print(f"Mascaras  -> {masks_dir}")
    print(f"Overlays  -> {overlays_dir}")
    print(f"Meta      -> {meta_path}")
    return meta, method_counts


# =============================================================================
# 7) LAZO DE CIERRE: aplicar revision humana exportada del HTML
# =============================================================================
def apply_review(decisions_path: Path):
    """
    Ingesta del JSON/CSV exportado por el usuario desde revision_lesiones.html.

    Estructura esperada por item: {sample_id, decision, comentario}
      decision in {"correcta", "ajuste", "descartar"}.

    Acciones:
      - "correcta"  -> la mascara propuesta queda como final (ya esta en masks/).
      - "descartar" -> se mueve la mascara a masks/_descartadas/ (se excluye del IoU/PG).
      - "ajuste"    -> NO se modifica automaticamente: requiere un SEGUNDO PASO
                       manual (re-anotar / re-promptear). Se LISTA, no se inventa.
    """
    import csv
    import shutil

    p = Path(decisions_path)
    rows = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data["decisiones"] if isinstance(data, dict) and "decisiones" in data else data
    else:
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    discard_dir = MASKS_DIR / "_descartadas"
    discard_dir.mkdir(parents=True, exist_ok=True)

    res = {"correcta": [], "descartar": [], "ajuste": [], "sin_decision": []}
    for r in rows:
        sid = (r.get("sample_id") or r.get("id") or "").strip()
        dec = (r.get("decision") or "").strip().lower()
        if not sid:
            continue
        if dec.startswith("correc"):
            res["correcta"].append(sid)
        elif dec.startswith("descart"):
            src = MASKS_DIR / sid
            if src.exists():
                shutil.move(str(src), str(discard_dir / sid))
            res["descartar"].append(sid)
        elif dec.startswith("ajust"):
            res["ajuste"].append(sid)
        else:
            res["sin_decision"].append(sid)

    print("=== Resultado de aplicar la revision ===")
    print(f"  Correctas (listas para IoU/PG): {len(res['correcta'])}")
    print(f"  Descartadas (movidas a {discard_dir.name}/): {len(res['descartar'])}")
    print(f"  Necesitan AJUSTE (SEGUNDO PASO manual, no automatico): {len(res['ajuste'])}")
    if res["ajuste"]:
        print("   -> Re-anotar/re-promptear estas y volver a exportar:")
        for s in res["ajuste"]:
            print(f"      - {s}")
    if res["sin_decision"]:
        print(f"  Sin decision: {len(res['sin_decision'])}")
    out = MASKS_DIR.parent / "review" / "apply_review_result.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Detalle -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="procesar solo las primeras N imagenes (smoke test)")
    ap.add_argument("--out-suffix", default="",
                    help="sufijo para carpetas de salida (no pisa el run real)")
    ap.add_argument("--src-dir", default=None,
                    help="carpeta de imagenes a procesar (default: to_annotate/)")
    ap.add_argument("--no-sam", action="store_true",
                    help="forzar metodo clasico (no intentar SAM)")
    ap.add_argument("--apply-review", default=None,
                    help="JSON/CSV exportado del HTML -> aplica decisiones")
    args = ap.parse_args()

    if args.apply_review:
        apply_review(Path(args.apply_review))
        return
    run(limit=args.limit, out_suffix=args.out_suffix, prefer_sam=not args.no_sam,
        src_dir=args.src_dir)


if __name__ == "__main__":
    main()
