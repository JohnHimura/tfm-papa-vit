"""
Orquestador del plan B completo: Fase 1 + Fase 2 + Fase 3.

Ejecuta en cadena:
  1. Optuna sobre 4 modelos (30 trials cada uno con TPE+ASHA y early-stop).
  2. Entrenamiento final 4 modelos x 5 folds x 3 seeds (60 runs) con
     hiperparametros ganadores.
  3. Evaluacion sobre holdout 15% (individual + ensemble).

Persiste un progress.json con el estado actual para que el monitor pueda
informar avance y para resumir si la maquina se cae.
"""
from __future__ import annotations
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import config as C
from .optuna_runner import run_all as optuna_run_all, DEFAULT_MODELS
from .final_runner import run_final
from .holdout_eval import main as holdout_main


PROGRESS = C.RESULTS_DIR / "progress.json"


def _stamp(state: str, **extra) -> None:
    data = {}
    if PROGRESS.exists():
        try:
            data = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("history", []).append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "state": state, **extra,
    })
    data["current"] = state
    PROGRESS.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def main():
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[{datetime.now()}] === FASE B INICIADA ===")
    _stamp("fase_b_start")

    # ------------------ Fase 1: Optuna ---------------------
    try:
        print("\n>>> FASE 1: Optuna 4 modelos x 30 trials\n")
        _stamp("fase_1_optuna_start")
        optuna_summary = optuna_run_all(DEFAULT_MODELS, n_trials=30)
        _stamp("fase_1_optuna_done", summary=optuna_summary)
    except Exception as e:
        msg = f"FASE 1 FALLO: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(msg)
        _stamp("fase_1_optuna_failed", error=msg)
        return

    # ------------------ Fase 2: entrenamiento final ----------------
    try:
        print("\n>>> FASE 2: Entrenamiento final 4 x 5 x 3 = 60 runs\n")
        _stamp("fase_2_final_start")
        df_final = run_final(DEFAULT_MODELS)
        _stamp("fase_2_final_done", n_runs=int(len(df_final)))
    except Exception as e:
        msg = f"FASE 2 FALLO: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(msg)
        _stamp("fase_2_final_failed", error=msg)
        return

    # ------------------ Fase 3: holdout ----------------------------
    try:
        print("\n>>> FASE 3: Evaluacion holdout 15%\n")
        _stamp("fase_3_holdout_start")
        holdout_main()
        _stamp("fase_3_holdout_done")
    except Exception as e:
        msg = f"FASE 3 FALLO: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(msg)
        _stamp("fase_3_holdout_failed", error=msg)
        return

    elapsed = time.time() - t0
    print(f"\n=== FASE B COMPLETA en {elapsed/3600:.2f}h ===")
    _stamp("fase_b_complete", elapsed_h=elapsed/3600)
    # Marca de finalizacion
    (C.RESULTS_DIR / "EXPERIMENT_COMPLETE.md").write_text(
        f"# EXPERIMENT COMPLETE\n\nPhase B finalizada en {elapsed/3600:.2f} horas.\n"
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
