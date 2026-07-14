"""
Configuracion central del pipeline experimental.

Toda constante reproducible vive aqui. Si necesitas tocar algo, hazlo aqui
y NO en los scripts individuales — eso garantiza que cada run quede trazado
y que MLflow capture el mismo conjunto de hiperparametros.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# RUTAS BASE
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]            # .../UNIR/TFM
CODE_DIR = ROOT / "04_Codigo"
DATA_DIR = ROOT / "03_Dataset" / "original"           # 7 carpetas / 1 por clase
RESULTS_DIR = CODE_DIR / "results"
MODELS_DIR = CODE_DIR / "models"
MLRUNS_DIR = CODE_DIR / "mlruns"
FIGS_DIR = ROOT / "05_Documento_TFM" / "figuras"

for d in (RESULTS_DIR, MODELS_DIR, MLRUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATASET
# =============================================================================
CLASSES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Conteos verificados en disco (10/05/2026)
CLASS_COUNTS = {
    "Bacteria": 569, "Fungi": 748, "Healthy": 201, "Nematode": 68,
    "Pest": 611, "Phytopthora": 347, "Virus": 532,
}
N_TOTAL = sum(CLASS_COUNTS.values())  # 3076

# =============================================================================
# REPRODUCIBILIDAD
# =============================================================================
SEEDS = [42, 137, 2026]                # 3 semillas para estabilidad
N_FOLDS = 5                            # Stratified K-Fold; usar 3 si tiempo aprieta
TEST_HOLDOUT_FRAC = 0.15               # holdout fijo fuera del CV
PIN_DETERMINISM = True                 # cudnn deterministic

# =============================================================================
# IMAGEN
# =============================================================================
IMG_SIZE = 224                         # ImageNet default; ViT-Base/16 tambien usa 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# =============================================================================
# ENTRENAMIENTO
# =============================================================================
@dataclass
class TrainCfg:
    model_name: str                            # "mobilenetv2_100" o "vit_base_patch16_224"
    epochs: int = 25
    batch_size: int = 32
    lr_head: float = 3e-4                      # LR para clasificador
    lr_backbone: float = 3e-5                  # LR para backbone (fine-tune)
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    use_class_weights: bool = True             # estrategia desbalance default
    use_focal_loss: bool = False               # ablacion opcional
    use_weighted_sampler: bool = False         # oversampling (WeightedRandomSampler); aditivo
    early_stop_patience: int = 6               # epocas sin mejora -> stop
    num_workers: int = 4
    mixed_precision: bool = True               # AMP para acelerar
    pretrained: bool = True
    extra: dict = field(default_factory=dict)

# Configuraciones por modelo (overrides)
MODELS_E2 = {
    "mobilenetv2_100": TrainCfg(
        model_name="mobilenetv2_100",
        epochs=25,
        batch_size=64,             # MobileNetV2 cabe holgado en 16GB
        lr_head=3e-4,
        lr_backbone=1e-4,
    ),
    "vit_base_patch16_224": TrainCfg(
        model_name="vit_base_patch16_224",
        epochs=20,
        batch_size=32,             # ViT-Base/16 mas pesado
        lr_head=1e-4,
        lr_backbone=1e-5,          # ViT necesita LR muy bajo en backbone
        warmup_epochs=3,
    ),
}

# Modelos opcionales (si avanzamos rapido)
MODELS_OPCIONAL = {
    "resnet50": TrainCfg(
        model_name="resnet50",
        epochs=25,
        batch_size=48,
        lr_head=3e-4,
        lr_backbone=3e-5,
    ),
    "efficientnet_b0": TrainCfg(
        model_name="efficientnet_b0",
        epochs=25,
        batch_size=64,
        lr_head=3e-4,
        lr_backbone=3e-5,
    ),
}

# =============================================================================
# MLflow
# =============================================================================
MLFLOW_EXPERIMENT = "tfm-papa-e2"
MLFLOW_TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"
