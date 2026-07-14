"""
OE5 — Constructor del HTML de REVISION de mascaras de lesion propuestas.

Genera UN solo archivo HTML autocontenido (abre con doble clic, sin servidor ni
internet) en results/xai/review/revision_lesiones.html. Por cada imagen muestra:
  - la imagen ORIGINAL y el OVERLAY con la mascara propuesta, lado a lado,
  - la OBSERVACION en lenguaje claro (de masks_meta.json),
  - la clase,
  - controles para decidir: "Correcta" / "Necesita ajuste" / "Descartar imagen",
  - un campo de comentario libre.
Boton "Exportar mis decisiones" -> descarga (Blob, sin internet) JSON y CSV con
las decisiones de TODAS las imagenes, para ingerir con:
    python -m src.oe5_sam_masks --apply-review <archivo>

Las imagenes se embeben en base64 (portable). Si el total estimado supera un
umbral, se cambia automaticamente a rutas relativas (overlays/ y to_annotate/).

Uso:
    python -m src.oe5_build_review_html
    python -m src.oe5_build_review_html --limit 3 --out revision_mini.html \
        --masks-meta masks_meta_smoke.json --overlays-suffix _smoke
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

try:
    from . import config as C
    _XAI_DIR = C.RESULTS_DIR / "xai"
except Exception:
    _XAI_DIR = Path(__file__).resolve().parents[1] / "results" / "xai"

XAI_DIR = _XAI_DIR
TO_ANNOTATE_DIR = XAI_DIR / "to_annotate"
REVIEW_DIR = XAI_DIR / "review"

# Umbral para decidir base64 vs rutas relativas (MB de PNGs embebidos)
EMBED_LIMIT_MB = 60.0


def _b64(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


HEADER_HTML = """
<div class="intro">
  <h1>Revision de zonas enfermas (lesiones)</h1>
  <p>Hola John. El sistema ya <b>marco solo</b> la zona que cree enferma en cada
  hoja. <b>Vos no tenes que dibujar nada.</b> Solo mira cada par de imagenes y
  deci si la marca esta bien.</p>
  <ol>
    <li><b>Izquierda</b> = foto original. <b>Derecha</b> = la misma foto con la
        zona enferma resaltada en rojo.</li>
    <li>Lee mi nota debajo (te digo donde y de que color marque).</li>
    <li>Elegi una opcion:
        <span class="pill ok">Correcta</span> si la marca cubre bien la mancha,
        <span class="pill warn">Necesita ajuste</span> si marco de mas o de menos,
        <span class="pill bad">Descartar imagen</span> si la foto no sirve.</li>
    <li>Si queres, escribi un comentario (opcional).</li>
    <li>Al terminar, toca <b>"Exportar mis decisiones"</b> abajo del todo: se
        descarga un archivo que yo uso para dejar las mascaras finales.</li>
  </ol>
  <p class="muted">Todo funciona sin internet. Tus decisiones se guardan en este
  navegador mientras revisas; el boton de exportar baja el resultado a tu PC.</p>
</div>
"""

PAGE_TMPL = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revision de lesiones — OE5 TFM</title>
<style>
  :root {{ --ok:#1a7f37; --warn:#b08800; --bad:#cf222e; --bg:#f6f8fa; --card:#fff; --line:#d0d7de; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:system-ui,Segoe UI,Arial,sans-serif; margin:0; background:var(--bg); color:#1f2328; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:18px; }}
  .intro {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 22px; margin-bottom:18px; }}
  .intro h1 {{ margin:.2em 0 .4em; }}
  .intro ol {{ line-height:1.7; }}
  .muted {{ color:#656d76; font-size:.92em; }}
  .pill {{ padding:2px 9px; border-radius:999px; color:#fff; font-size:.85em; font-weight:600; }}
  .pill.ok{{background:var(--ok);}} .pill.warn{{background:var(--warn);}} .pill.bad{{background:var(--bad);}}
  .bar {{ position:sticky; top:0; z-index:5; background:var(--card); border:1px solid var(--line);
          border-radius:12px; padding:10px 16px; margin-bottom:16px; display:flex; gap:14px;
          align-items:center; flex-wrap:wrap; }}
  .bar .count {{ font-weight:600; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin-bottom:16px; }}
  .card h3 {{ margin:0 0 8px; font-size:1.05em; }}
  .tag {{ display:inline-block; background:#eaeef2; border-radius:6px; padding:1px 8px; font-size:.85em; margin-left:6px; }}
  .imgs {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .imgs figure {{ margin:0; flex:1 1 300px; text-align:center; }}
  .imgs img {{ width:100%; max-width:360px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
  .imgs figcaption {{ font-size:.85em; color:#656d76; margin-top:4px; }}
  .obs {{ background:#fff8e6; border-left:4px solid var(--warn); padding:8px 12px; border-radius:6px; margin:10px 0; }}
  .opts {{ display:flex; gap:10px; flex-wrap:wrap; margin:8px 0; }}
  .opts label {{ cursor:pointer; border:1px solid var(--line); border-radius:8px; padding:7px 12px; user-select:none; }}
  .opts input {{ margin-right:6px; }}
  .opts label.sel-ok{{border-color:var(--ok); background:#e8f5ec;}}
  .opts label.sel-warn{{border-color:var(--warn); background:#fdf6e3;}}
  .opts label.sel-bad{{border-color:var(--bad); background:#fbe9ea;}}
  textarea {{ width:100%; min-height:48px; border:1px solid var(--line); border-radius:8px; padding:8px; font-family:inherit; }}
  .footer {{ position:sticky; bottom:0; background:var(--card); border:1px solid var(--line);
             border-radius:12px; padding:14px 16px; margin-top:8px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  button {{ font-size:1em; padding:10px 18px; border:0; border-radius:8px; cursor:pointer; }}
  button.primary {{ background:#0969da; color:#fff; }}
  button.ghost {{ background:#eaeef2; }}
  .done {{ color:var(--ok); font-weight:600; }}
</style></head>
<body><div class="wrap">
{header}
<div class="bar">
  <span class="count">Revisadas: <span id="n_done">0</span> / {total}</span>
  <button class="ghost" onclick="window.scrollTo(0,document.body.scrollHeight)">Ir al boton exportar ↓</button>
</div>
{cards}
<div class="footer">
  <button class="primary" onclick="exportJSON()">Exportar mis decisiones (JSON)</button>
  <button class="ghost" onclick="exportCSV()">Tambien como CSV</button>
  <span id="status" class="muted"></span>
</div>
</div>
<script>
const ITEMS = {items_json};
const store = {{}};
function recount() {{
  let n = 0;
  for (const k in store) if (store[k] && store[k].decision) n++;
  document.getElementById('n_done').textContent = n;
}}
function pick(sid, dec, el) {{
  store[sid] = store[sid] || {{}};
  store[sid].decision = dec;
  const group = el.closest('.opts');
  group.querySelectorAll('label').forEach(l => l.className='');
  el.parentElement.className = dec==='correcta' ? 'sel-ok' : dec==='ajuste' ? 'sel-warn' : 'sel-bad';
  recount();
}}
function note(sid, val) {{ store[sid] = store[sid] || {{}}; store[sid].comentario = val; }}
function collect() {{
  return ITEMS.map(it => ({{
    sample_id: it.sample_id,
    clase: it.clase,
    decision: (store[it.sample_id]||{{}}).decision || '',
    comentario: (store[it.sample_id]||{{}}).comentario || '',
    observacion: it.observacion,
    metodo: it.metodo,
    area_pct: it.area_pct
  }}));
}}
function download(name, text, type) {{
  const blob = new Blob([text], {{type}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}}
function exportJSON() {{
  const data = {{ generado: new Date().toISOString(), total: ITEMS.length, decisiones: collect() }};
  download('decisiones_lesiones.json', JSON.stringify(data, null, 2), 'application/json');
  document.getElementById('status').innerHTML = '<span class="done">Descargado decisiones_lesiones.json</span>';
}}
function exportCSV() {{
  const rows = collect();
  const cols = ['sample_id','clase','decision','comentario','area_pct','metodo'];
  const esc = v => '"'+String(v==null?'':v).replace(/"/g,'""')+'"';
  const csv = [cols.join(',')].concat(rows.map(r => cols.map(c => esc(r[c])).join(','))).join('\\n');
  download('decisiones_lesiones.csv', csv, 'text/csv');
  document.getElementById('status').innerHTML = '<span class="done">Descargado decisiones_lesiones.csv</span>';
}}
</script>
</body></html>
"""

CARD_TMPL = """<div class="card" id="card_{sid_safe}">
  <h3>{idx}. {sample_id} <span class="tag">clase: {clase}</span> <span class="tag">metodo: {metodo}</span></h3>
  <div class="imgs">
    <figure><img src="{orig_src}" alt="original"><figcaption>Foto original</figcaption></figure>
    <figure><img src="{over_src}" alt="overlay"><figcaption>Zona enferma resaltada</figcaption></figure>
  </div>
  <div class="obs">{observacion}</div>
  <div class="opts">
    <label><input type="radio" name="d_{sid_safe}" onclick="pick('{sample_id}','correcta',this)">Correcta</label>
    <label><input type="radio" name="d_{sid_safe}" onclick="pick('{sample_id}','ajuste',this)">Necesita ajuste</label>
    <label><input type="radio" name="d_{sid_safe}" onclick="pick('{sample_id}','descartar',this)">Descartar imagen</label>
  </div>
  <textarea placeholder="Comentario (opcional): ej. 'falto la mancha de abajo'..." oninput="note('{sample_id}',this.value)"></textarea>
</div>
"""


def build(limit=None, out_name="revision_lesiones.html",
          meta_name="masks_meta.json", overlays_suffix=""):
    meta_path = XAI_DIR / meta_name
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    overlays_dir = Path(str(REVIEW_DIR / "overlays") + overlays_suffix)

    names = list(meta.keys())
    if limit:
        names = names[:limit]

    # Estimar peso para decidir base64 vs rutas relativas
    total_bytes = 0
    for n in names:
        o = TO_ANNOTATE_DIR / n
        v = overlays_dir / n
        total_bytes += (o.stat().st_size if o.exists() else 0)
        total_bytes += (v.stat().st_size if v.exists() else 0)
    embed = (total_bytes / 1e6) <= EMBED_LIMIT_MB
    mode = "base64 (autocontenido)" if embed else "rutas relativas"

    cards, items = [], []
    for i, n in enumerate(names, 1):
        m = meta[n]
        orig = TO_ANNOTATE_DIR / n
        over = overlays_dir / n
        if embed:
            orig_src = _b64(orig)
            over_src = _b64(over) if over.exists() else orig_src
        else:
            # rutas relativas a la carpeta review/ (donde vive el HTML)
            orig_src = f"../to_annotate/{n}"
            over_src = f"overlays{overlays_suffix}/{n}"
        sid_safe = n.replace(".", "_").replace("-", "_")
        cards.append(CARD_TMPL.format(
            sid_safe=sid_safe, idx=i, sample_id=n, clase=m.get("clase", "?"),
            metodo=m.get("method", "?"), orig_src=orig_src, over_src=over_src,
            observacion=m.get("observacion", ""),
        ))
        items.append({
            "sample_id": n, "clase": m.get("clase", "?"),
            "observacion": m.get("observacion", ""), "metodo": m.get("method", "?"),
            "area_pct": m.get("area_pct", 0),
        })

    html = PAGE_TMPL.format(
        header=HEADER_HTML, total=len(names), cards="\n".join(cards),
        items_json=json.dumps(items, ensure_ascii=False),
    )
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REVIEW_DIR / out_name
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML  -> {out_path}")
    print(f"Imagenes: {len(names)} | modo: {mode} | peso PNGs ~{total_bytes/1e6:.1f} MB")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="revision_lesiones.html")
    ap.add_argument("--masks-meta", default="masks_meta.json")
    ap.add_argument("--overlays-suffix", default="")
    args = ap.parse_args()
    build(limit=args.limit, out_name=args.out,
          meta_name=args.masks_meta, overlays_suffix=args.overlays_suffix)


if __name__ == "__main__":
    main()
