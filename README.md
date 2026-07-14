# Clasificación de enfermedades foliares en papa: CNN vs. Vision Transformer

Código y evidencia del Trabajo Fin de Estudios (Máster Universitario en Inteligencia Artificial, UNIR).
Estudio comparativo de tres arquitecturas convolucionales (MobileNetV2, ResNet-50, EfficientNet-B0)
frente a un Vision Transformer (ViT-Base/16) sobre imágenes de campo no controladas, con análisis de
explicabilidad.

Autor del código: John Jairo Vásquez Acosta. Director: Julián Prieto Velasco.

## Fuentes

| Recurso | Origen |
|---|---|
| **Dataset** | *Potato Leaf Disease Dataset in Uncontrolled Environment* (Shabrina et al., 2023). DOI [10.17632/ptz377bwb8.1](https://doi.org/10.17632/ptz377bwb8.1) — descarga: https://data.mendeley.com/datasets/ptz377bwb8/1 · Licencia CC BY 4.0 |
| **Pesos preentrenados** | ImageNet vía [`timm`](https://github.com/huggingface/pytorch-image-models) 1.0.26 (`resnet50.a1_in1k`, `efficientnet_b0.ra_in1k`, `mobilenetv2_100.ra_in1k`, `vit_base_patch16_224.augreg2_in21k_ft_in1k`) |
| **Modelos entrenados** | Zenodo — DOI [10.5281/zenodo.21381057](https://doi.org/10.5281/zenodo.21381057) · Licencia CC BY 4.0 · 60 checkpoints (4 arquitecturas × 5 pliegues × 3 semillas). Verificar con [`CHECKSUMS_modelos.sha256`](04_Codigo/CHECKSUMS_modelos.sha256) |
| **SAM** (máscaras de lesión, OE5) | `sam_vit_b` de [Segment Anything](https://github.com/facebookresearch/segment-anything) (Meta AI) |

## Instalación

```bash
cd 04_Codigo
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m src.verify_env
```

Descargar el dataset y descomprimirlo en `03_Dataset/original/` con estas 7 carpetas exactas:
`Bacteria  Fungi  Healthy  Nematode  Pest  Phytopthora  Virus`.

## Reproducir

```bash
python -m src.run_phase_b       # Optuna -> CV (5 folds x 3 semillas) -> holdout   (~9 h en GPU)
python -m src.analyze_results   # tablas y figuras
```

Verificar el resultado principal **sin reentrenar** (requiere los modelos de Zenodo en `04_Codigo/models/`):

```bash
python -m src.holdout_eval      # ViT ensemble, macro F1 = 0,9533 sobre 462 imágenes
```

Experimentos específicos de la memoria:

```bash
python -m src.run_oe4_ablation_normfix   # OE4: ablación de desbalance (Tabla 14)
python -m src.run_oe5_xai                # OE5: Grad-CAM / Attention Rollout (Tabla 15)
python -m src.run_vit1k                  # Sensibilidad al preentrenamiento (ImageNet-1k)
python -m src.bootstrap_ci               # Intervalos de confianza 95% (10.000 remuestreos)
```

## Evidencia

Los ficheros de `04_Codigo/results/` respaldan cada cifra de la memoria y permiten contrastar las tablas
sin reentrenar:

| Fichero | Respalda |
|---|---|
| `final_master.csv` | Macro F1 en validación cruzada (Tabla 8) |
| `holdout_eval.json`, `tabla_f1_por_clase.csv` | Holdout individual y ensemble; F1 por clase (Tablas 12, 13) |
| `mcnemar_holdout.json` | Test de McNemar pareado (Tabla 11) |
| `bootstrap_ci.json` | Intervalos de confianza 95% |
| `oe4_ablation_normfix.json` | OE4: ablación de desbalance (Tabla 14) |
| `oe5_metrics.csv`, `xai/oe5_per_image.csv` | OE5: IoU y Pointing Game (Tabla 15) |
| `vit1k_vs_vit21k.json` | Sensibilidad ImageNet-1k vs. 21k |
| `model_specs.json` | Parámetros, GFLOPs y tamaño (Tabla 4) |
| `catalog_with_holdout.csv`, `splits_manifest.csv` | Particiones congeladas (no regenerar: los pliegues dependen del orden de las filas) |

## Reproducibilidad

- **Semillas:** 42, 137, 2026 (3 repeticiones por configuración).
- **Particiones:** holdout fijo del 15 % (462 imágenes) + validación cruzada estratificada de 5 pliegues.
- **Métrica principal:** macro F1.

Resultados de referencia (macro F1):

| Modelo | CV (5×3) | Holdout (individual) | Holdout (ensemble de 15) |
|---|---|---|---|
| MobileNetV2 | 0,823 ± 0,017 | 0,832 | 0,862 |
| EfficientNet-B0 | 0,824 ± 0,014 | 0,825 | 0,872 |
| ResNet-50 | 0,877 ± 0,014 | 0,879 | 0,909 |
| **ViT-Base/16** | **0,898 ± 0,012** | **0,901** | **0,953** |

## Licencia

MIT (ver [LICENSE](LICENSE)).
