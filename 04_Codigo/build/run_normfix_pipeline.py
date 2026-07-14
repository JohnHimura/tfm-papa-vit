#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orquestador de la CORRIDA CORREGIDA (normfix) del TFM.

Re-entrena el ViT (21k principal + 1k) con la normalizacion correcta por modelo
y Optuna PROPIO, re-evalua holdout + McNemar, y re-corre OE4 con la Focal Loss
arreglada y re-optimizacion por estrategia. NO toca los artefactos viejos.

Orden (valor primero): Optuna ViT -> CV ViT -> EVAL ViT (la respuesta clave) -> OE4.

Uso:
    python build/run_normfix_pipeline.py --preflight
    python build/run_normfix_pipeline.py
"""
import sys, os, json, time, subprocess, py_compile
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
SRC = CODE / "src"
RESULTS = CODE / "results"
MODELS = CODE / "models"
LOGDIR = Path(__file__).resolve().parent / "pipeline_logs"
LOGDIR.mkdir(exist_ok=True)
PROGRESS_JSON = LOGDIR / "progress_normfix.json"
PROGRESS_LOG = LOGDIR / "progress_normfix.log"
PY = sys.executable

def ts(): return time.strftime("%Y-%m-%d %H:%M:%S")

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
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "NO-CUDA"
        if not cuda:
            problems.append("torch.cuda.is_available() == False (no GPU)")
    except Exception as e:
        problems.append(f"no se pudo importar torch: {e}"); gpu = "ERR"
    # artefactos necesarios para el McNemar del eval (predicciones CNN ya en disco, norm correcta)
    need = [
        RESULTS / "catalog_with_holdout.csv",
        RESULTS / "final_master.csv",
        RESULTS / "holdout_predictions" / "resnet50_ensemble.npz",
        RESULTS / "holdout_predictions" / "efficientnet_b0_ensemble.npz",
        RESULTS / "holdout_predictions" / "mobilenetv2_100_ensemble.npz",
    ]
    for p in need:
        if not p.exists():
            problems.append(f"falta artefacto: {p}")
    # checkpoints CNN final (no se re-entrenan, pero deben existir para trazabilidad)
    for fam in ("resnet50", "efficientnet_b0", "mobilenetv2_100"):
        n = len(list(MODELS.glob(f"final__{fam}_fold*_seed*_best.pt")))
        if n < 15:
            problems.append(f"checkpoints final__{fam}: {n}/15")
    # compile + import de los modulos clave
    for pyf in list(SRC.glob("*.py")) + [Path(__file__)]:
        try: py_compile.compile(str(pyf), doraise=True)
        except py_compile.PyCompileError as e: problems.append(f"py_compile {pyf.name}: {e}")
    for mod in ("src.optuna_runner", "src.run_vit_normfix", "src.run_oe4_ablation_normfix",
                "src.transforms", "src.losses", "src.holdout_eval"):
        if not (CODE / (mod.replace('.', os.sep) + '.py')).exists():
            problems.append(f"falta modulo {mod}")
    # verificacion EXTRA: normalizacion por modelo correcta (barata, importante)
    try:
        if str(CODE) not in sys.path:
            sys.path.insert(0, str(CODE))
        from src.transforms import get_model_norm
        cnn = get_model_norm("resnet50")[0][0]
        vit = get_model_norm("vit_base_patch16_224")[0][0]
        if abs(cnn - 0.485) > 1e-3: problems.append(f"norm CNN inesperada: {cnn}")
        if abs(vit - 0.5) > 1e-3: problems.append(f"norm ViT inesperada: {vit}")
        logline(f"PRE-FLIGHT norm: CNN={cnn} ViT={vit}")
    except Exception as e:
        problems.append(f"no se pudo verificar get_model_norm: {e}")
    logline(f"PRE-FLIGHT GPU={gpu}  problemas={len(problems)}")
    for p in problems: logline(f"  [X] {p}")
    return problems, gpu

# ---------------------------------------------------------------- ETAPAS
STAGES = [
    ("optuna_vit", "Optuna propio de ambos ViT (norm correcta)",
        ["src.optuna_runner", "--plan", "vit_normfix", "--n-trials", "30"], 21600),
    ("vit_cv", "CV corregida de ambos ViT (5x3 c/u)",
        ["src.run_vit_normfix", "train", "--resume"], 21600),
    ("vit_eval", "Eval holdout corregido + bootstrap + McNemar (RESPUESTA CLAVE)",
        ["src.run_vit_normfix", "eval"], 3600),
    ("oe4_normfix", "OE4 corregida (focal arreglada + Optuna LR por estrategia)",
        ["src.run_oe4_ablation_normfix", "--lr-trials", "15"], 43200),
]

def sanity(name):
    out = {}
    try:
        if name == "optuna_vit":
            for tag in ("vit_base_patch16_224", "vit_base_patch16_224.augreg_in1k"):
                f = RESULTS / "optuna_normfix" / tag / "summary.json"
                out[tag] = "ok" if f.exists() else "FALTA"
        elif name == "vit_cv":
            import csv
            f = RESULTS / "vit_normfix_master.csv"
            out["n_rows"] = len(list(csv.DictReader(open(f, encoding="utf-8")))) if f.exists() else 0
        elif name == "vit_eval":
            f = RESULTS / "vit_normfix_eval.json"
            if f.exists():
                d = json.load(open(f, encoding="utf-8"))
                out = {"resumen": "ver vit_normfix_eval.json", **{k: v for k, v in d.items() if isinstance(v, (int, float, str))}}
        elif name == "oe4_normfix":
            import csv
            f = RESULTS / "oe4_ablation_normfix.csv"
            out["n_rows"] = len(list(csv.DictReader(open(f, encoding="utf-8")))) if f.exists() else 0
    except Exception as e:
        out["sanity_error"] = str(e)
    return out

def run_pipeline():
    problems, gpu = preflight()
    prior = {}
    if PROGRESS_JSON.exists():
        try: prior = json.load(open(PROGRESS_JSON, encoding="utf-8")).get("stages", {})
        except Exception: prior = {}
    state = {"started": ts(), "gpu": gpu, "preflight_problems": problems, "stages": {}}
    for s in STAGES:
        name = s[0]
        state["stages"][name] = prior[name] if prior.get(name, {}).get("status") == "ok" \
            else {"status": "pendiente", "desc": s[1]}
    write_progress(state)
    if problems:
        logline("ABORTADO: pre-flight con problemas."); state["aborted"] = True
        write_progress(state); return 1
    nsk = sum(1 for n in state["stages"] if state["stages"][n].get("status") == "ok")
    logline(f"PRE-FLIGHT OK (GPU={gpu}). {nsk} etapas ya OK se saltan; resto en serie.")
    for name, desc, args, tmo in STAGES:
        if state["stages"][name].get("status") == "ok":
            logline(f"=== SALTO {name}: completada en corrida previa"); continue
        state["stages"][name]["status"] = "corriendo"; state["stages"][name]["start"] = ts()
        write_progress(state)
        logline(f"=== INICIO {name}: {desc}")
        stage_log = LOGDIR / f"normfix_{name}.log"
        t0 = time.time()
        try:
            with open(stage_log, "w", encoding="utf-8") as lf:
                rc = subprocess.call([PY, "-m"] + args, cwd=str(CODE),
                                     stdout=lf, stderr=subprocess.STDOUT, timeout=tmo)
        except subprocess.TimeoutExpired:
            rc = -9; logline(f"!!! {name} excedio timeout {tmo}s")
        dur = round(time.time() - t0, 1)
        st = state["stages"][name]
        st["end"] = ts(); st["dur_s"] = dur; st["exit_code"] = rc
        st["status"] = "ok" if rc == 0 else "FALLO"; st["sanity"] = sanity(name)
        st["log"] = str(stage_log)
        write_progress(state)
        logline(f"=== FIN {name}: exit={rc} dur={dur}s sanity={st['sanity']}")
    ok = sum(1 for s in state["stages"].values() if s["status"] == "ok")
    state["finished"] = ts(); state["resumen"] = f"{ok}/{len(STAGES)} etapas OK"
    write_progress(state)
    logline(f"PIPELINE NORMFIX TERMINADO: {state['resumen']}")
    return 0

if __name__ == "__main__":
    if "--preflight" in sys.argv:
        probs, gpu = preflight()
        print(json.dumps({"gpu": gpu, "problemas": probs}, ensure_ascii=False, indent=2))
        sys.exit(1 if probs else 0)
    sys.exit(run_pipeline())
