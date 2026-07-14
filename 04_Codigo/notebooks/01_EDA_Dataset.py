# -*- coding: utf-8 -*-
"""
01_EDA_Dataset.py - Análisis Exploratorio del Dataset
TFM: Clasificación de Enfermedades Foliares en Papa con Vision Transformer

Ejecutar con: python 01_EDA_Dataset.py
O convertir a notebook con: jupytext --to notebook 01_EDA_Dataset.py

Autores: John Jairo Vasquez Acosta, Angela
Universidad: UNIR - Máster en Inteligencia Artificial
"""

# %% [markdown]
# # Análisis Exploratorio del Dataset
# ## Potato Leaf Disease Dataset in Uncontrolled Environment

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

DATASET_PATH = Path("../../03_Dataset/original")
if not DATASET_PATH.exists():
    DATASET_PATH = Path("../../../TFM/Potato Leaf Disease Dataset in Uncontrolled Environment")
RESULTS_PATH = Path("../results")
RESULTS_PATH.mkdir(exist_ok=True)
FIGURES_PATH = Path("../../05_Documento_TFM/figuras")
FIGURES_PATH.mkdir(exist_ok=True)

print(f"Dataset path: {DATASET_PATH}")
print(f"Results path: {RESULTS_PATH}")

# %% [markdown]
# ## 1. Estructura del Dataset

# %%
# Catalogar todas las imágenes
data = []
for class_dir in sorted(DATASET_PATH.iterdir()):
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        for img_path in images:
            data.append({
                'class': class_dir.name,
                'filename': img_path.name,
                'path': str(img_path),
                'size_bytes': img_path.stat().st_size
            })

df = pd.DataFrame(data)
print(f"Total imágenes: {len(df)}")
print(f"\nDistribución por clase:")
class_counts = df['class'].value_counts()
print(class_counts)
print(f"\nRatio de desbalance: {class_counts.max()}/{class_counts.min()} = {class_counts.max()/class_counts.min():.1f}:1")

# %%
# Gráfico de distribución de clases
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = sns.color_palette("husl", len(class_counts))
bars = axes[0].bar(class_counts.index, class_counts.values, color=colors)
axes[0].set_title('Distribución de Imágenes por Clase', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Clase de Enfermedad')
axes[0].set_ylabel('Número de Imágenes')
axes[0].tick_params(axis='x', rotation=45)
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontweight='bold')

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Proporción por Clase', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'distribucion_clases.png', dpi=150, bbox_inches='tight')
plt.savefig(RESULTS_PATH / 'distribucion_clases.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Figura guardada en: {FIGURES_PATH / 'distribucion_clases.png'}")

# %% [markdown]
# ## 2. Análisis de Dimensiones y Propiedades de las Imágenes

# %%
# Analizar dimensiones, canales y estadísticas de una muestra
print("Analizando propiedades de las imágenes (muestra de 100)...")
sample_indices = np.random.RandomState(42).choice(len(df), min(100, len(df)), replace=False)
img_stats = []

for idx in sample_indices:
    row = df.iloc[idx]
    try:
        img = Image.open(row['path'])
        img_array = np.array(img)
        img_stats.append({
            'class': row['class'],
            'width': img.size[0],
            'height': img.size[1],
            'channels': img_array.shape[2] if len(img_array.shape) == 3 else 1,
            'mean_r': img_array[:,:,0].mean() if len(img_array.shape) == 3 else img_array.mean(),
            'mean_g': img_array[:,:,1].mean() if len(img_array.shape) == 3 else 0,
            'mean_b': img_array[:,:,2].mean() if len(img_array.shape) == 3 else 0,
            'std_r': img_array[:,:,0].std() if len(img_array.shape) == 3 else img_array.std(),
            'std_g': img_array[:,:,1].std() if len(img_array.shape) == 3 else 0,
            'std_b': img_array[:,:,2].std() if len(img_array.shape) == 3 else 0,
        })
    except Exception as e:
        print(f"Error leyendo {row['path']}: {e}")

stats_df = pd.DataFrame(img_stats)
print(f"\nDimensiones:")
print(f"  Width: {stats_df['width'].unique()}")
print(f"  Height: {stats_df['height'].unique()}")
print(f"  Channels: {stats_df['channels'].unique()}")
print(f"\nEstadísticas de color (RGB):")
print(f"  Mean R: {stats_df['mean_r'].mean():.1f} ± {stats_df['mean_r'].std():.1f}")
print(f"  Mean G: {stats_df['mean_g'].mean():.1f} ± {stats_df['mean_g'].std():.1f}")
print(f"  Mean B: {stats_df['mean_b'].mean():.1f} ± {stats_df['mean_b'].std():.1f}")

# %% [markdown]
# ## 3. Mosaico de Ejemplos por Clase

# %%
# Mostrar 3 ejemplos de cada clase
fig, axes = plt.subplots(7, 3, figsize=(15, 30))
np.random.seed(42)

for i, class_name in enumerate(sorted(df['class'].unique())):
    class_imgs = df[df['class'] == class_name].sample(3, random_state=42)
    for j, (_, row) in enumerate(class_imgs.iterrows()):
        img = Image.open(row['path'])
        axes[i, j].imshow(img)
        axes[i, j].set_title(f"{class_name}" + (f" ({j+1})" if j > 0 else ""), fontsize=11)
        axes[i, j].axis('off')

plt.suptitle('Ejemplos de Imágenes por Clase de Enfermedad', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'mosaico_ejemplos_por_clase.png', dpi=150, bbox_inches='tight')
plt.savefig(RESULTS_PATH / 'mosaico_ejemplos_por_clase.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Figura guardada en: {FIGURES_PATH / 'mosaico_ejemplos_por_clase.png'}")

# %% [markdown]
# ## 4. Distribución de Intensidad de Color por Clase

# %%
# Histogramas de color por clase
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
np.random.seed(42)

for i, class_name in enumerate(sorted(df['class'].unique())):
    class_imgs = df[df['class'] == class_name].sample(min(20, len(df[df['class'] == class_name])), random_state=42)
    r_vals, g_vals, b_vals = [], [], []

    for _, row in class_imgs.iterrows():
        img = np.array(Image.open(row['path']))
        if len(img.shape) == 3:
            r_vals.extend(img[:,:,0].flatten()[::100])  # Subsample for speed
            g_vals.extend(img[:,:,1].flatten()[::100])
            b_vals.extend(img[:,:,2].flatten()[::100])

    axes[i].hist(r_vals, bins=50, alpha=0.5, color='red', label='R', density=True)
    axes[i].hist(g_vals, bins=50, alpha=0.5, color='green', label='G', density=True)
    axes[i].hist(b_vals, bins=50, alpha=0.5, color='blue', label='B', density=True)
    axes[i].set_title(class_name, fontsize=12, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].set_xlim(0, 255)

# Hide last subplot if odd number of classes
if len(df['class'].unique()) < len(axes):
    axes[-1].axis('off')

plt.suptitle('Distribución de Intensidad RGB por Clase', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'distribucion_rgb_por_clase.png', dpi=150, bbox_inches='tight')
plt.savefig(RESULTS_PATH / 'distribucion_rgb_por_clase.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Resumen para el TFM

# %%
print("=" * 60)
print("RESUMEN DEL ANÁLISIS EXPLORATORIO")
print("=" * 60)
print(f"\nDataset: Potato Leaf Disease in Uncontrolled Environment")
print(f"Total imágenes: {len(df)}")
print(f"Clases: {len(df['class'].unique())}")
print(f"Resolución: {stats_df['width'].mode()[0]}x{stats_df['height'].mode()[0]} px")
print(f"Canales: {stats_df['channels'].mode()[0]} (RGB)")
print(f"\nDesbalance:")
print(f"  Clase mayoritaria: {class_counts.index[0]} ({class_counts.values[0]} imgs, {class_counts.values[0]/len(df)*100:.1f}%)")
print(f"  Clase minoritaria: {class_counts.index[-1]} ({class_counts.values[-1]} imgs, {class_counts.values[-1]/len(df)*100:.1f}%)")
print(f"  Ratio: {class_counts.values[0]/class_counts.values[-1]:.1f}:1")
print(f"\nCaracterísticas del dataset:")
print(f"  - Capturado en condiciones de campo no controladas")
print(f"  - Fondos heterogéneos (tierra, plástico, vegetación)")
print(f"  - Iluminación variable (sol directo, nublado, sombra)")
print(f"  - Múltiples hojas por imagen en algunos casos")
print(f"  - Sin anotaciones de bounding box ni segmentación")
print(f"\nImplicaciones para el modelado:")
print(f"  1. Necesidad de estrategia de manejo de desbalance (Nematode={class_counts.get('Nematode', 'N/A')} imgs)")
print(f"  2. Data augmentation diferenciada para clases minoritarias")
print(f"  3. Macro F1-Score como métrica principal (no accuracy)")
print(f"  4. Resize a 224x224 para ViT-Base/16")

# Guardar resumen como CSV
summary = pd.DataFrame({
    'Clase': class_counts.index,
    'Imagenes': class_counts.values,
    'Porcentaje': (class_counts.values / len(df) * 100).round(1)
})
summary.to_csv(RESULTS_PATH / 'resumen_dataset.csv', index=False)
print(f"\nResumen guardado en: {RESULTS_PATH / 'resumen_dataset.csv'}")
