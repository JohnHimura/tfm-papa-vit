#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orquestador nocturno del TFM (pipeline GPU serial, sin contencion de VRAM).

Ejecuta por etapas, en orden, registrando timestamp/exit-code/duracion de cada una
en un progress.json + progress.log que el orquestador (Claude) lee en cada despertar.

Uso:
    python build/run_overnight_pipeline.py --preflight   # solo valida, no ejecuta
    python build/run_overnight_pipeline.py               # ejecuta el pipeline completo

NO usa Date.now/random; los timestamps salen de time.time() en runtime real (no es un
workflow journaled). Disenado para correr como job en background del harness.
"""
import sys, os, json, time, subprocess, importlib.util, py_compile
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent          # .../04_Codigo
SRC = CODE / "src"
RESULTS = CODE / "results"
MODELS = CODE / "models"
LOGDIR = Path(__file__).resolve().parent / "pipeline_logs"
LOGDIR.mkdir(exist_ok=True)
PROGRESS_JSON = LOGDIR / "progress.json"
PROGRESS_LOG = LOGDIR / "progress.log"
PY = sys.executable                                     # debe ser el python del venv

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def logline(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_progress(state):
    state["updated"] = ts()
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- PRE-FLIGHT
def preflight():
    problems = []
    # 1) venv + GPU
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "NO-CUDA"
        if not cuda:
            problems.append("torch.cuda.is_available() == False (no GPU)")
    except Exception as e:
        problems.append(f"no se pudo importar torch: {e}")
        gpu = "ERR"
    # 2) artefactos de entrada
    need_files = [
        RESULTS / "catalog_with_holdout.csv",
        RESULTS / "optuna" / "vit_base_patch16_224" / "summary.json",
        RESULTS / "holdout_predictions" / "vit_base_patch16_224_ensemble.npz",
        MODELS / "sam_vit_b_01ec64.pth",
    ]
    for p in need_files:
        if not p.exists():
            problems.append(f"falta artefacto: {p}")
    # checkpoints final del ViT y ResNet (15 c/u)
    for fam in ("vit_base_patch16_224", "resnet50"):
        n = len(list(MODELS.glob(f"final__{fam}_fold*_seed*_best.pt")))
        if n < 15:
            problems.append(f"checkpoints final__{fam}: {n}/15")
    # imagenes a anotar para SAM
    n_annot = len(list((RESULTS / "xai" / "to_annotate").glob("*.png"))) if (RESULTS / "xai" / "to_annotate").exists() else 0
    if n_annot < 100:
        problems.append(f"results/xai/to_annotate tiene {n_annot} png (esperaba ~110-130)")
    # 3) py_compile de todo src/ y build/
    for d in (SRC, Path(__file__).resolve().parent):
        for pyf in d.glob("*.py"):
            try:
                py_compile.compile(str(pyf), doraise=True)
            except py_compile.PyCompileError as e:
                problems.append(f"py_compile FALLO en {pyf.name}: {e}")
    # 4) import de los modulos de entrada
    for mod in ("src.run_oe4_ablation", "src.run_vit1k", "src.run_oe5_xai",
                "src.oe5_sam_masks", "src.oe5_build_review_html", "src.extract_embeddings"):
        spec_path = CODE / (mod.replace(".", os.sep) + ".py")
        if not spec_path.exists():
            problems.append(f"falta modulo {mod} ({spec_path})")
    logline(f"PRE-FLIGHT GPU={gpu}  to_annotate={n_annot}  problemas={len(problems)}")
    for p in problems:
        logline(f"  [X] {p}")
    return problems, gpu

# ---------------------------------------------------------------- ETAPAS
# (nombre, descripcion, [args tras 'python -m'], timeout_seg)
STAGES = [
    ("sam_masks",   "SAM propone mascaras de lesion (110 img)",
        ["src.oe5_sam_masks"], 3600),
    ("sam_html",    "HTML de revision para John",
        ["src.oe5_build_review_html"], 1200),
    ("oe5_maps_resnet", "Grad-CAM ResNet-50 (mapas OE5)",
        ["src.run_oe5_xai", "generate", "--model", "resnet50", "--variant", "best"], 3600),
    ("oe5_maps_vit", "Attention Rollout ViT (mapas OE5)",
        ["src.run_oe5_xai", "generate", "--model", "vit_base_patch16_224", "--variant", "best"], 3600),
    ("embeddings",  "Embeddings ViT + t-SNE/UMAP (Fase 1.6)",
        ["src.extract_embeddings", "--variant", "best"], 3600),
    ("vit1k",       "Reentrenamiento ViT 1k-only + comparacion (Opcion b)",
        ["src.run_vit1k", "all", "--resume"], 18000),
    ("oe4_ablation", "Ablacion OE4 (4 estrategias x 15 reps ViT)",
        ["src.run_oe4_ablation"], 43200),
]

def sanity(name):
    """Chequeos baratos post-etapa; devuelven dict para el progress."""
    out = {}
    try:
        if name == "oe4_ablation":
            import csv
            f = RESULTS / "oe4_ablation.csv"
            if f.exists():
                rows = list(csv.DictReader(open(f, encoding="utf-8")))
                for r in rows:
                    key = (r.get("strategy") or r.get("strategy_label") or "").lower()
                    if "class_weight" in key or "ponderaci" in key:
                        out["class_weights_macro_f1"] = r.get("macro_f1_mean") or r.get("macro_f1_fmt")
                out["n_rows"] = len(rows)
        elif name == "vit1k":
            f = RESULTS / "vit1k_vs_vit21k.json"
            if f.exists():
                d = json.load(open(f, encoding="utf-8"))
                out["macro_f1_1k"] = d.get("vit_1k", {}).get("macro_f1")
                out["macro_f1_21k"] = d.get("vit_21k", {}).get("macro_f1")
                out["delta_pts"] = d.get("delta_macro_f1_pts")
                out["mcnemar_1k_vs_resnet_p"] = d.get("mcnemar_1k_vs_resnet50", {}).get("pvalue")
        elif name == "sam_masks":
            out["n_masks"] = len(list((RESULTS / "xai" / "masks").glob("*.png"))) if (RESULTS / "xai" / "masks").exists() else 0
        elif name == "sam_html":
            f = RESULTS / "xai" / "review" / "revision_lesiones.html"
            out["html_existe"] = f.exists()
            out["html_mb"] = round(f.stat().st_size / 1e6, 1) if f.exists() else 0
        elif name in ("oe5_maps_resnet", "oe5_maps_vit"):
            out["n_npy"] = len(list((RESULTS / "xai").rglob("*.npy")))
        elif name == "embeddings":
            figs = list((RESULTS / "figuras").glob("*tsne*")) + list((RESULTS / "figuras").glob("*umap*"))
            out["n_figs"] = len(figs)
    except Exception as e:
        out["sanity_error"] = str(e)
    return out

def run_pipeline():
    problems, gpu = preflight()
    # Reanudacion: conservar etapas ya completadas ('ok') de una corrida previa.
    prior = {}
    if PROGRESS_JSON.exists():
        try:
            prior = json.load(open(PROGRESS_JSON, encoding="utf-8")).get("stages", {})
        except Exception:
            prior = {}
    state = {"started": ts(), "gpu": gpu, "preflight_problems": problems, "stages": {}}
    for s in STAGES:
        name = s[0]
        if prior.get(name, {}).get("status") == "ok":
            state["stages"][name] = prior[name]           # ya completada -> conservar
        else:
            state["stages"][name] = {"status": "pendiente", "desc": s[1]}
    write_progress(state)
    if problems:
        logline("ABORTADO: pre-flight con problemas; no se ejecuta nada.")
        state["aborted"] = True
        write_progress(state)
        return 1
    n_skip = sum(1 for n in state["stages"] if state["stages"][n].get("status") == "ok")
    logline(f"PRE-FLIGHT OK (GPU={gpu}). {n_skip} etapas ya OK se saltan; resto en serie.")
    for name, desc, args, tmo in STAGES:
        if state["stages"][name].get("status") == "ok":
            logline(f"=== SALTO {name}: completada en corrida previa")
            continue
        state["stages"][name]["status"] = "corriendo"
        state["stages"][name]["start"] = ts()
        write_progress(state)
        logline(f"=== INICIO {name}: {desc}")
        stage_log = LOGDIR / f"{name}.log"
        t0 = time.time()
        try:
            with open(stage_log, "w", encoding="utf-8") as lf:
                rc = subprocess.call([PY, "-m"] + args, cwd=str(CODE),
                                     stdout=lf, stderr=subprocess.STDOUT, timeout=tmo)
        except subprocess.TimeoutExpired:
            rc = -9
            logline(f"!!! {name} excedio timeout {tmo}s")
        dur = round(time.time() - t0, 1)
        st = state["stages"][name]
        st["end"] = ts(); st["dur_s"] = dur; st["exit_code"] = rc
        st["status"] = "ok" if rc == 0 else "FALLO"
        st["sanity"] = sanity(name)
        st["log"] = str(stage_log)
        write_progress(state)
        logline(f"=== FIN {name}: exit={rc} dur={dur}s sanity={st['sanity']}")
        # continuar aunque falle (etapas independientes); sam_html depende de sam_masks
        if name == "sam_masks" and rc != 0:
            logline("sam_masks fallo -> sam_html podria no tener insumos, pero se intenta igual.")
    ok = sum(1 for s in state["stages"].values() if s["status"] == "ok")
    state["finished"] = ts(); state["resumen"] = f"{ok}/{len(STAGES)} etapas OK"
    write_progress(state)
    logline(f"PIPELINE TERMINADO: {state['resumen']}")
    return 0

if __name__ == "__main__":
    if "--preflight" in sys.argv:
        probs, gpu = preflight()
        print(json.dumps({"gpu": gpu, "problemas": probs}, ensure_ascii=False, indent=2))
        sys.exit(1 if probs else 0)
    sys.exit(run_pipeline())
