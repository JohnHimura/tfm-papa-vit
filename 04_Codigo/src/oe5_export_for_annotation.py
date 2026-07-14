"""
OE5 — Exportador de imagenes para ANOTACION MANUAL de mascaras de lesion.

Este script NO fabrica mascaras. Exporta la muestra de 140 imagenes (20/clase,
seed 2026) a una carpeta plana con naming claro y escribe un README con
instrucciones precisas para que un humano (estudiante / fitopatologo) dibuje
las mascaras binarias de lesion.

==> BLOQUEO HUMANO: sin esas mascaras, las Tablas 15 y 16 NO pueden tener
    numeros reales (hoy son estimadas). Esfuerzo estimado: ~16-24 h-persona
    para las 120 imagenes con lesion (Healthy no se anota).

Salidas:
    results/xai/to_annotate/<sample_id>.png      (224x224, RGB, listo para anotar)
    results/xai/to_annotate/README_ANOTACION.md  (instrucciones)
    results/xai/to_annotate/labelme_classes.txt  (etiqueta unica: 'lesion')

Formato de mascara esperado por el calculo de IoU/PG:
    results/xai/masks/<sample_id>.png   (binaria: lesion>0, fondo=0; 224x224
    o cualquier tamaño — se reescala por vecino mas cercano).

Uso:
    python -m src.oe5_export_for_annotation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C
from . import run_oe5_xai as OE5

TO_ANNOTATE_DIR = OE5.XAI_DIR / "to_annotate"

README_TEXT = """# Anotacion de mascaras de lesion (OE5 — Tablas 15 y 16)

## Que hay que hacer
Para cada imagen de esta carpeta, dibujar una **mascara binaria** que marque la
region de **lesion / sintoma** (tejido enfermo) de la hoja:

- **Blanco (pixel > 0) = lesion** (tejido con el sintoma de la enfermedad).
- **Negro (pixel = 0) = fondo / tejido sano / suelo.**

La mascara se compara contra los mapas de saliencia (Grad-CAM de ResNet-50 y
Attention Rollout de ViT) para medir si el modelo "mira" donde realmente esta
la enfermedad (IoU y Pointing Game).

## Cuantas imagenes
- Total exportadas: **140** (20 por clase, seleccion reproducible seed=2026).
- **A ANOTAR: las 120 con lesion** (Bacteria, Fungi, Nematode, Pest,
  Phytopthora, Virus).
- **Healthy (20 img): NO se anota** — no hay lesion. Estan incluidas solo para
  contexto visual; el calculo de IoU/PG las ignora.
- Esfuerzo estimado: **~16-24 horas-persona** (8-12 min por imagen).

## Donde guardar las mascaras
Guardar cada mascara como PNG **con el MISMO nombre** que la imagen, en:

    results/xai/masks/<sample_id>.png

Ejemplo: la imagen `Bacteria_03.png` -> mascara `masks/Bacteria_03.png`.
El tamaño puede ser cualquiera (se reescala por vecino mas cercano a 224x224).

## Como anotar (dos opciones)

### Opcion A — labelme (recomendada, trazable)
1. `pip install labelme`
2. `labelme results/xai/to_annotate --labels lesion --nodata`
3. Dibujar poligono(s) sobre la lesion, etiqueta `lesion`. Guardar -> genera
   un `.json` por imagen en la misma carpeta.
4. Convertir los `.json` a mascaras PNG binarias en `masks/`:
   `python -m src.oe5_export_for_annotation --labelme-to-masks`

### Opcion B — editor de imagen (GIMP / Photoshop / Paint)
1. Abrir `<sample_id>.png`.
2. Pintar de BLANCO puro la(s) region(es) de lesion y de NEGRO todo lo demas
   (o trabajar en una capa nueva y exportar solo la mascara).
3. Exportar como PNG en escala de grises o RGB a `results/xai/masks/<sample_id>.png`.

## Despues de anotar
Ejecutar el calculo de las Tablas 15/16:

    python -m src.run_oe5_xai metrics --threshold-method percentile --percentile 80

Esto lee los heatmaps ya generados + tus mascaras y escribe
`results/oe5_metrics.csv` con IoU y Pointing Game global y por clase.

## Criterios de anotacion (consistencia)
- Marcar TODAS las lesiones visibles de la enfermedad principal de la clase.
- Si la lesion es difusa (clorosis viral, dano por plaga), marcar el area
  afectada aproximada; preferir cubrir de mas a dejar de menos.
- No marcar reflejos, sombras ni bordes de la hoja sanos.
- Ante duda sobre si una zona es lesion, anotar segun el criterio dominante y
  ser consistente entre imagenes de la misma clase.
"""


def export_for_annotation() -> int:
    """Exporta la muestra a to_annotate/ a 224x224 y escribe el README."""
    sample = OE5.load_sample()
    TO_ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
    OE5.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    n_lesion = 0
    for _, row in sample.iterrows():
        img = Image.open(row["path"]).convert("RGB").resize(
            (C.IMG_SIZE, C.IMG_SIZE))
        img.save(TO_ANNOTATE_DIR / f"{row['sample_id']}.png")
        n += 1
        if bool(row["has_lesion"]):
            n_lesion += 1
    (TO_ANNOTATE_DIR / "README_ANOTACION.md").write_text(
        README_TEXT, encoding="utf-8")
    (TO_ANNOTATE_DIR / "labelme_classes.txt").write_text(
        "lesion\n", encoding="utf-8")
    print(f"[OK] {n} imagenes exportadas -> {TO_ANNOTATE_DIR}")
    print(f"     {n_lesion} con lesion (a anotar), {n - n_lesion} Healthy (no anotar)")
    print(f"[OK] README_ANOTACION.md y labelme_classes.txt escritos")
    print(f"     Guardar mascaras en: {OE5.MASKS_DIR}")
    return n


def labelme_to_masks() -> int:
    """Convierte los .json de labelme en to_annotate/ a mascaras binarias PNG.

    Cada poligono etiquetado 'lesion' se rasteriza a blanco. Requiere que
    labelme haya guardado los .json junto a las imagenes.
    """
    import json
    try:
        from PIL import ImageDraw
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PIL ImageDraw no disponible: {e}")

    OE5.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    jsons = sorted(TO_ANNOTATE_DIR.glob("*.json"))
    if not jsons:
        print(f"[WARN] no hay .json de labelme en {TO_ANNOTATE_DIR}")
        return 0
    n = 0
    for jp in jsons:
        data = json.loads(jp.read_text(encoding="utf-8"))
        h = int(data.get("imageHeight", C.IMG_SIZE))
        w = int(data.get("imageWidth", C.IMG_SIZE))
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for shape in data.get("shapes", []):
            if shape.get("label", "").lower() != "lesion":
                continue
            pts = [tuple(p) for p in shape["points"]]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        mask.save(OE5.MASKS_DIR / f"{jp.stem}.png")
        n += 1
    print(f"[OK] {n} mascaras escritas -> {OE5.MASKS_DIR}")
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Exporta imagenes OE5 para anotacion")
    ap.add_argument("--labelme-to-masks", action="store_true",
                    help="convierte .json de labelme a mascaras PNG binarias")
    args = ap.parse_args()
    if args.labelme_to_masks:
        labelme_to_masks()
    else:
        export_for_annotation()


if __name__ == "__main__":
    main()
