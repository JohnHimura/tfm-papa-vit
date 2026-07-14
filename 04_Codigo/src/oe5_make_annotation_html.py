"""
OE5 — Genera el HTML autocontenido para el juicio CIEGO inter-evaluador.

Toma las laminas A|B producidas por `oe5_contrastive select` y las embebe
(comprimidas) en un unico archivo HTML portable. Cada evaluador lo abre, marca
para cada par si el mapa A o el B es fitopatologicamente mas plausible (o empate)
y descarga su CSV de juicios.

El HTML NO revela que modelo es A y cual es B (el mapeo vive en ab_key.json), ni
la clase de la imagen: el juicio es ciego.

Uso:
    python -m src.oe5_make_annotation_html
    -> results/xai/contrastive/juicio_ciego.html
"""
from __future__ import annotations

import base64
import io as _io
from pathlib import Path

import pandas as pd
from PIL import Image

from . import run_oe5_xai as OE5

CONTRAST_DIR = OE5.XAI_DIR / "contrastive"
PAIRS_CSV = CONTRAST_DIR / "pares_divergentes.csv"
LAMINAS = CONTRAST_DIR / "laminas"
OUT_HTML = CONTRAST_DIR / "juicio_ciego.html"

MAX_W = 1000      # ancho maximo de la lamina embebida
JPEG_Q = 78


def _img_b64(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > MAX_W:
        h = int(im.height * MAX_W / im.width)
        im = im.resize((MAX_W, h), Image.LANCZOS)
    buf = _io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_Q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build() -> Path:
    pairs = pd.read_csv(PAIRS_CSV)
    cards = []
    for _, r in pairs.iterrows():
        pid, sid = r["pair_id"], r["sample_id"]
        lam = LAMINAS / f"{pid}.png"
        if not lam.exists():
            continue
        b64 = _img_b64(lam)
        # OJO: no se emite ni la clase ni el sample_id (el sample_id contiene el
        # nombre de la clase, p.ej. "Fungi_05"): el juicio debe ser ciego tambien
        # respecto de la enfermedad. La correspondencia pair_id -> sample_id vive
        # en pares_divergentes.csv y se recupera al calcular el kappa.
        cards.append(f"""
<div class="card" id="card-{pid}" data-pair="{pid}">
  <div class="pid">{pid}<span class="done" id="done-{pid}"></span></div>
  <img src="data:image/jpeg;base64,{b64}" alt="{pid}">
  <div class="btns">
    <button type="button" onclick="vote('{pid}','A',this)">A es mas plausible</button>
    <button type="button" onclick="vote('{pid}','tie',this)">Empate</button>
    <button type="button" onclick="vote('{pid}','B',this)">B es mas plausible</button>
  </div>
</div>""")

    n = len(cards)
    html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>OE5 — Juicio ciego ViT vs ResNet</title>
<style>
 body {{ font-family: system-ui, "Segoe UI", sans-serif; margin:0; background:#f5f6f8; color:#1a1a1a; }}
 header {{ background:#fff; border-bottom:1px solid #ddd; padding:18px 24px; position:sticky; top:0; z-index:10; }}
 h1 {{ margin:0 0 6px; font-size:19px; }}
 .sub {{ color:#555; font-size:14px; line-height:1.5; max-width:900px; }}
 .bar {{ display:flex; gap:14px; align-items:center; margin-top:12px; flex-wrap:wrap; }}
 input[type=text] {{ padding:7px 10px; border:1px solid #bbb; border-radius:6px; font-size:14px; }}
 .prog {{ font-weight:600; }}
 .wrap {{ padding:22px; display:grid; gap:22px; max-width:1100px; margin:0 auto; }}
 .card {{ background:#fff; border:1px solid #e0e0e0; border-radius:10px; padding:14px; }}
 .card img {{ width:100%; border-radius:6px; display:block; }}
 .pid {{ font-weight:700; margin-bottom:8px; font-size:15px; }}
 .done {{ color:#128a3a; margin-left:10px; font-weight:600; font-size:13px; }}
 .btns {{ display:flex; gap:10px; margin-top:12px; }}
 .btns button {{ flex:1; padding:11px; font-size:14px; border:1px solid #bbb; background:#fafafa;
                 border-radius:7px; cursor:pointer; }}
 .btns button:hover {{ background:#eef2ff; }}
 .btns button.sel {{ background:#1f6feb; color:#fff; border-color:#1f6feb; }}
 .save {{ background:#128a3a; color:#fff; border:none; padding:10px 18px; border-radius:7px;
          font-size:14px; cursor:pointer; font-weight:600; }}
 .save:disabled {{ background:#9bb; cursor:not-allowed; }}
</style></head><body>
<header>
 <h1>OE5 — Juicio ciego: ¿qué mapa señala mejor la lesión?</h1>
 <div class="sub">
  Cada ficha muestra tres imágenes de la <b>misma hoja</b>: a la <b>izquierda, la hoja sin ningún mapa</b>
  (para que pueda ver dónde está realmente la lesión); en el <b>centro, el mapa A</b>; a la <b>derecha, el mapa B</b>.
  Los mapas vienen de dos modelos distintos, pero <b>no se le indica cuál es cuál</b>, ni la enfermedad,
  para que el juicio no esté sesgado. Las zonas cálidas (rojo/amarillo) son las que el modelo considera relevantes.
  <br><b>Pregunta:</b> mirando primero la hoja limpia, ¿cuál de los dos mapas —A o B— concentra mejor su relevancia
  sobre el <b>tejido lesionado real</b> (y no sobre el fondo, la mano, el suelo, las sombras o zona sana)?
  Si no hay una diferencia clara, marque <b>Empate</b>.
  <br>No hay respuestas correctas ni tiempo límite; responda con su criterio. Al terminar, escriba su nombre,
  pulse <b>Descargar mis juicios</b> y envíe el CSV que se descarga.
 </div>
 <div class="bar">
  <label>Su nombre: <input type="text" id="evalname" placeholder="Nombre y apellido"></label>
  <span class="prog"><span id="cnt">0</span> / {n} respondidos</span>
  <button class="save" id="savebtn" onclick="save()" disabled>Descargar mis juicios</button>
 </div>
</header>
<div class="wrap">
{''.join(cards)}
</div>
<script>
const TOTAL = {n};
const votes = {{}};
function vote(pid, val, btn) {{
  votes[pid] = val;
  const card = document.getElementById('card-'+pid);
  card.querySelectorAll('.btns button').forEach(b => b.classList.remove('sel'));
  btn.classList.add('sel');
  document.getElementById('done-'+pid).textContent = '✓ respondido';
  document.getElementById('cnt').textContent = Object.keys(votes).length;
  document.getElementById('savebtn').disabled = Object.keys(votes).length === 0;
}}
function save() {{
  const name = (document.getElementById('evalname').value || 'evaluador').trim();
  let csv = 'pair_id,voto\\n';
  document.querySelectorAll('.card').forEach(c => {{
    const pid = c.dataset.pair;
    if (votes[pid]) csv += pid + ',' + votes[pid] + '\\n';
  }});
  const blob = new Blob([csv], {{type:'text/csv;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'juicios_' + name.replace(/[^A-Za-z0-9]+/g,'_') + '.csv';
  a.click();
}}
</script>
</body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    mb = OUT_HTML.stat().st_size / 1e6
    print(f"[OK] {n} pares -> {OUT_HTML}  ({mb:.1f} MB)")
    print("     Enviar este unico archivo a cada evaluador (se abre con doble clic).")
    return OUT_HTML


if __name__ == "__main__":
    build()
