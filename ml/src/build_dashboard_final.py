import json

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

with open(f'{OUT_DIR}/buildings_final_enriched.json') as f:
    buildings = json.load(f)
with open(f'{OUT_DIR}/roads_final.json') as f:
    roads = json.load(f)
with open(f'{OUT_DIR}/water_final.json') as f:
    water = json.load(f)
with open(f'{OUT_DIR}/rgb_b64.txt') as f:
    img_b64 = f.read()

total_area = sum(b['area_m2'] for b in buildings)
total_solar = sum(b['estimated_solar_kwp'] for b in buildings)
total_road_len = sum(r['length_m'] for r in roads)
total_water_area = sum(w['area_m2'] for w in water)
total_tax = sum(b['estimated_annual_tax_inr'] for b in buildings)
n_buildings = len(buildings)
n_high = sum(1 for b in buildings if b['review_priority'].startswith('HIGH'))
n_med = sum(1 for b in buildings if b['review_priority'].startswith('MEDIUM'))
n_low = n_buildings - n_high - n_med

html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SVAMITVA AI Feature Extraction — Village Dashboard v3</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; background:#0f1419; color:#e5e7eb; }}
  header {{ padding: 14px 24px; background:#111827; border-bottom:1px solid #1f2937; display:flex; align-items:center; justify-content:space-between; }}
  header h1 {{ font-size:17px; margin:0; font-weight:600; }}
  header .badge {{ font-size:11px; background:#1d4ed8; padding:4px 10px; border-radius:12px; }}
  .layout {{ display:flex; height:calc(100vh - 54px); }}
  .map-panel {{ flex:1; position:relative; overflow:hidden; background:#000; }}
  #canvasWrap {{ position:relative; width:100%; height:100%; display:flex; align-items:center; justify-content:center; }}
  canvas {{ cursor:crosshair; max-width:100%; max-height:100%; }}
  .sidebar {{ width:340px; background:#111827; border-left:1px solid #1f2937; padding:16px; overflow-y:auto; }}
  .tabs {{ display:flex; gap:4px; margin-bottom:14px; background:#1f2937; border-radius:8px; padding:3px; }}
  .tab {{ flex:1; text-align:center; padding:7px 4px; font-size:12px; border-radius:6px; cursor:pointer; color:#9ca3af; }}
  .tab.active {{ background:#2563eb; color:white; font-weight:600; }}
  .tab-panel {{ display:none; }}
  .tab-panel.active {{ display:block; }}
  .stat-card {{ background:#1f2937; border-radius:10px; padding:12px 14px; margin-bottom:10px; }}
  .stat-card .label {{ font-size:10px; color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em; }}
  .stat-card .value {{ font-size:20px; font-weight:700; margin-top:3px; }}
  .stat-row {{ display:flex; gap:10px; }}
  .stat-row .stat-card {{ flex:1; }}
  .layer-toggle {{ display:flex; align-items:center; gap:8px; padding:9px 10px; background:#1f2937; border-radius:8px; margin-bottom:6px; cursor:pointer; user-select:none; }}
  .layer-toggle input {{ accent-color:#2563eb; width:15px; height:15px; }}
  .layer-toggle .swatch {{ width:12px; height:12px; border-radius:3px; }}
  .layer-toggle .name {{ font-size:12.5px; flex:1; }}
  .layer-toggle .count {{ font-size:11px; color:#9ca3af; }}
  .tooltip {{ position:absolute; background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 12px; font-size:12px; pointer-events:none; display:none; z-index:10; box-shadow:0 8px 20px rgba(0,0,0,0.4); min-width:180px; }}
  .tooltip .t-title {{ font-weight:700; margin-bottom:4px; font-size:13px; }}
  .tooltip .t-row {{ display:flex; justify-content:space-between; gap:14px; color:#d1d5db; }}
  .section-title {{ font-size:12px; font-weight:700; margin: 16px 0 8px; color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em; }}
  .footer-note {{ font-size:10.5px; color:#6b7280; margin-top:16px; line-height:1.5; }}
  .queue-item {{ display:flex; justify-content:space-between; padding:8px 10px; background:#1f2937; border-radius:6px; margin-bottom:5px; font-size:12px; }}
  .queue-badge {{ padding:2px 7px; border-radius:10px; font-size:10px; font-weight:700; }}
  .badge-high {{ background:#dc2626; }}
  .badge-med {{ background:#f59e0b; color:#111; }}
</style>
</head>
<body>

<header>
  <h1>🛰️ SVAMITVA AI Feature Extraction — Village Dashboard</h1>
  <span class="badge">FINAL · TTA + Tuned Threshold + Full Pipeline</span>
</header>

<div class="layout">
  <div class="map-panel">
    <div id="canvasWrap">
      <canvas id="mapCanvas" width="1000" height="1000"></canvas>
    </div>
    <div class="tooltip" id="tooltip"></div>
  </div>

  <div class="sidebar">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('map')">Layers</div>
      <div class="tab" onclick="switchTab('tax')">Tax</div>
      <div class="tab" onclick="switchTab('queue')">Review Queue</div>
    </div>

    <div id="panel-map" class="tab-panel active">
      <div class="stat-row">
        <div class="stat-card"><div class="label">Buildings</div><div class="value">{n_buildings}</div></div>
        <div class="stat-card"><div class="label">Built-up Area</div><div class="value">{total_area/10000:.1f} ha</div></div>
      </div>
      <div class="stat-row">
        <div class="stat-card"><div class="label">Solar Potential</div><div class="value">{total_solar/1000:.1f} MWp</div></div>
        <div class="stat-card"><div class="label">Road Length</div><div class="value">{total_road_len/1000:.2f} km</div></div>
      </div>

      <div class="section-title">Layers</div>
      <div class="layer-toggle" onclick="toggleLayer('buildings')">
        <input type="checkbox" id="chk-buildings" checked>
        <span class="swatch" style="background:#ff5050"></span>
        <span class="name">Buildings (roof-classified)</span>
        <span class="count">{n_buildings}</span>
      </div>
      <div class="layer-toggle" onclick="toggleLayer('roads')">
        <input type="checkbox" id="chk-roads" checked>
        <span class="swatch" style="background:#f5d90a"></span>
        <span class="name">Roads</span>
        <span class="count">{len(roads)}</span>
      </div>
      <div class="layer-toggle" onclick="toggleLayer('water')">
        <input type="checkbox" id="chk-water" checked>
        <span class="swatch" style="background:#00dcdc"></span>
        <span class="name">Waterbodies</span>
        <span class="count">{len(water)}</span>
      </div>

      <div class="section-title">Roof Material Legend</div>
      <div class="layer-toggle" style="cursor:default;"><span class="swatch" style="background:#ff5050"></span><span class="name">RCC (Concrete)</span></div>
      <div class="layer-toggle" style="cursor:default;"><span class="swatch" style="background:#ffb43c"></span><span class="name">Tiled</span></div>
      <div class="layer-toggle" style="cursor:default;"><span class="swatch" style="background:#78c8ff"></span><span class="name">Tin / Metal Sheet</span></div>
      <div class="layer-toggle" style="cursor:default;"><span class="swatch" style="background:#9aa0a6"></span><span class="name">Other</span></div>
    </div>

    <div id="panel-tax" class="tab-panel">
      <div class="stat-card">
        <div class="label">Estimated Total Annual Property Tax</div>
        <div class="value">Rs. {total_tax/100000:.1f} Lakh</div>
      </div>
      <div class="section-title">Rate Basis (illustrative)</div>
      <div style="font-size:12px; color:#d1d5db; line-height:2;">
        RCC (Concrete): Rs 45/m²/yr<br>
        Tiled: Rs 28/m²/yr<br>
        Tin/Metal Sheet: Rs 18/m²/yr<br>
        Other: Rs 15/m²/yr
      </div>
      <div class="footer-note">
        Placeholder slab rates modeled on typical construction-type-based municipal tax structures. Production version pulls real per-district rates from the relevant state Panchayat/ULB tax schedule API. Click any building on the map to see its individual tax estimate.
      </div>
    </div>

    <div id="panel-queue" class="tab-panel">
      <div class="stat-row">
        <div class="stat-card"><div class="label">Auto-Accepted</div><div class="value" style="color:#16a34a">{n_low}</div></div>
        <div class="stat-card"><div class="label">Needs Review</div><div class="value" style="color:#f59e0b">{n_med + n_high}</div></div>
      </div>
      <div class="section-title">Flagged Buildings ({n_high} high, {n_med} medium priority)</div>
      <div id="queue-list"></div>
      <div class="footer-note">
        Review priority is computed from the model's raw pixel-probability map — buildings whose mean internal probability sits close to the 0.5 decision boundary are flagged for a human to check, instead of trusting every detection equally. This is the actual triage mechanism a human-in-the-loop / active-learning pipeline uses.
      </div>
    </div>

    <div class="footer-note" style="margin-top:20px; border-top:1px solid #1f2937; padding-top:12px;">
      Buildings: Attention-ResUNet+ASPP (honest held-out IoU 0.33). Roads: morphological linear detector + PCA vectorization. Water: HSV+texture threshold. Roof material: KMeans clustering. Change detection (see separate slide) validated on a synthetic Time-B tile — no real bi-temporal SVAMITVA pair was available in this environment.
    </div>
  </div>
</div>

<script>
const buildings = {json.dumps(buildings)};
const roads = {json.dumps(roads)};
const water = {json.dumps(water)};
const layerState = {{ buildings: true, roads: true, water: true }};

const img = new Image();
img.src = "data:image/png;base64,{img_b64}";
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

img.onload = () => draw();

function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
}}

// populate review queue list
const queueList = document.getElementById('queue-list');
const flagged = buildings.filter(b => !b.review_priority.startsWith('LOW')).sort((a,b) => a.mean_pixel_probability > b.mean_pixel_probability ? 1 : -1);
flagged.slice(0, 25).forEach(b => {{
  const isHigh = b.review_priority.startsWith('HIGH');
  const div = document.createElement('div');
  div.className = 'queue-item';
  div.innerHTML = `<span>Building #${{b.id}} (${{b.area_m2}} m²)</span><span class="queue-badge ${{isHigh?'badge-high':'badge-med'}}">${{(b.mean_pixel_probability*100).toFixed(0)}}%</span>`;
  queueList.appendChild(div);
}});

function roofColor(label) {{
  const map = {{'RCC (Concrete)':'#ff5050','Tiled':'#ffb43c','Tin/Metal Sheet':'#78c8ff','Other':'#9aa0a6'}};
  return map[label] || '#999';
}}

function toggleLayer(name) {{
  layerState[name] = !layerState[name];
  document.getElementById('chk-' + name).checked = layerState[name];
  draw();
}}

function draw(hover) {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (layerState.water) {{
    water.forEach((w, i) => {{
      const pts = w.polygon_px;
      if (!pts || pts.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let j=1;j<pts.length;j++) ctx.lineTo(pts[j][0], pts[j][1]);
      ctx.closePath();
      const isHover = hover && hover.type==='water' && hover.idx===i;
      ctx.strokeStyle = '#00dcdc'; ctx.lineWidth = isHover?3:1.5;
      ctx.fillStyle = isHover ? '#00dcdc66' : '#00dcdc33';
      ctx.fill(); ctx.stroke();
    }});
  }}

  if (layerState.roads) {{
    roads.forEach((r, i) => {{
      const pts = r.polyline_px;
      if (!pts || pts.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let j=1;j<pts.length;j++) ctx.lineTo(pts[j][0], pts[j][1]);
      const isHover = hover && hover.type==='road' && hover.idx===i;
      ctx.strokeStyle = '#f5d90a'; ctx.lineWidth = isHover?5:3;
      ctx.stroke();
    }});
  }}

  if (layerState.buildings) {{
    buildings.forEach((b, i) => {{
      const pts = b.polygon_px;
      if (!pts || pts.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let j=1;j<pts.length;j++) ctx.lineTo(pts[j][0], pts[j][1]);
      ctx.closePath();
      const col = roofColor(b.roof_material);
      const isHover = hover && hover.type==='building' && hover.idx===i;
      ctx.strokeStyle = col; ctx.lineWidth = isHover?3:1.5;
      ctx.fillStyle = isHover ? col+'55' : col+'22';
      ctx.fill(); ctx.stroke();
      if (!b.review_priority.startsWith('LOW')) {{
        const cx = pts.reduce((s,p)=>s+p[0],0)/pts.length;
        const cy = pts.reduce((s,p)=>s+p[1],0)/pts.length;
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2*Math.PI);
        ctx.fillStyle = b.review_priority.startsWith('HIGH') ? '#dc2626' : '#f59e0b';
        ctx.fill();
      }}
    }});
  }}
}}

function pointInPoly(x, y, pts) {{
  let inside = false;
  for (let i=0, j=pts.length-1; i<pts.length; j=i++) {{
    const xi=pts[i][0], yi=pts[i][1], xj=pts[j][0], yj=pts[j][1];
    const intersect = ((yi>y) !== (yj>y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi);
    if (intersect) inside = !inside;
  }}
  return inside;
}}

function distToSegment(x, y, pts) {{
  let minD = Infinity;
  for (let i=0;i<pts.length-1;i++) {{
    const [x1,y1] = pts[i], [x2,y2] = pts[i+1];
    const A=x-x1, B=y-y1, C=x2-x1, D=y2-y1;
    const dot=A*C+B*D, lenSq=C*C+D*D;
    let t = lenSq!==0 ? dot/lenSq : -1;
    t = Math.max(0, Math.min(1, t));
    const px=x1+t*C, py=y1+t*D;
    const d = Math.hypot(x-px, y-py);
    if (d<minD) minD=d;
  }}
  return minD;
}}

canvas.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  let found = null;
  if (layerState.buildings) {{
    for (let i=0;i<buildings.length;i++) {{
      if (pointInPoly(x,y,buildings[i].polygon_px)) {{ found={{type:'building', idx:i}}; break; }}
    }}
  }}
  if (!found && layerState.roads) {{
    for (let i=0;i<roads.length;i++) {{
      if (distToSegment(x,y,roads[i].polyline_px) < 6) {{ found={{type:'road', idx:i}}; break; }}
    }}
  }}
  if (!found && layerState.water) {{
    for (let i=0;i<water.length;i++) {{
      if (pointInPoly(x,y,water[i].polygon_px)) {{ found={{type:'water', idx:i}}; break; }}
    }}
  }}

  if (found) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY + 14) + 'px';
    if (found.type === 'building') {{
      const b = buildings[found.idx];
      tooltip.innerHTML = `<div class="t-title">Building #${{b.id}}</div>
        <div class="t-row"><span>Area</span><span>${{b.area_m2}} m²</span></div>
        <div class="t-row"><span>Roof</span><span>${{b.roof_material}}</span></div>
        <div class="t-row"><span>Solar Est.</span><span>${{b.estimated_solar_kwp}} kWp</span></div>
        <div class="t-row"><span>Annual Tax</span><span>Rs ${{b.estimated_annual_tax_inr.toLocaleString()}}</span></div>
        <div class="t-row"><span>Review</span><span>${{b.review_priority}}</span></div>`;
    }} else if (found.type === 'road') {{
      const r = roads[found.idx];
      tooltip.innerHTML = `<div class="t-title">Road Segment #${{r.id}}</div>
        <div class="t-row"><span>Length</span><span>${{r.length_m}} m</span></div>`;
    }} else {{
      const w = water[found.idx];
      tooltip.innerHTML = `<div class="t-title">Waterbody #${{w.id}}</div>
        <div class="t-row"><span>Area</span><span>${{w.area_m2}} m²</span></div>`;
    }}
    draw(found);
  }} else {{
    tooltip.style.display = 'none';
    draw();
  }}
}});

canvas.addEventListener('mouseleave', () => {{ tooltip.style.display='none'; draw(); }});
</script>

</body>
</html>
'''

with open(f'{OUT_DIR}/dashboard_final.html', 'w') as f:
    f.write(html)

print('Dashboard v3 written')
