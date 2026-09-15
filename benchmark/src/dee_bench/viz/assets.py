"""The dashboard's stylesheet and behaviour, kept out of the page builder.

Both are plain strings with `__TOKEN__` placeholders rather than f-strings, so
CSS and JavaScript braces stay as written instead of being doubled everywhere —
which is the difference between a stylesheet that can be read and one that can
only be regenerated.
"""

from __future__ import annotations

CSS = """
:root {
  --surface: #fcfcfb; --surface-2: #f4f4f2; --surface-3: #ebebe7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #78776f;
  --grid: #e6e6e2; --border: #dcdcd6; --accent: #2a78d6;
  --good: #008300; --warning: #eda100; --critical: #e34948;
  --shadow: 0 1px 2px rgba(11,11,11,.04), 0 6px 18px rgba(11,11,11,.04);
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #1a1a19; --surface-2: #232322; --surface-3: #2c2c2a;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e85;
    --grid: #333331; --border: #3a3a38; --accent: #3987e5;
    --good: #3aa03a; --warning: #c98500; --critical: #e66767;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.24);
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --surface: #1a1a19; --surface-2: #232322; --surface-3: #2c2c2a;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e85;
  --grid: #333331; --border: #3a3a38; --accent: #3987e5;
  --good: #3aa03a; --warning: #c98500; --critical: #e66767;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.24);
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--text-primary);
  font-family: __FONT__; font-size: 14px; line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
code, .mono { font-family: __MONO__; font-size: .86em; }
code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
a { color: var(--accent); }

.app { display: grid; grid-template-columns: 252px minmax(0, 1fr); min-height: 100vh; }

/* ---------------------------------------------------------------- nav */
.side {
  border-right: 1px solid var(--border); background: var(--surface-2);
  padding: 22px 0 24px; position: sticky; top: 0; height: 100vh;
  overflow-y: auto; display: flex; flex-direction: column; gap: 4px;
}
.brand {
  padding: 0 20px 2px; font-size: 12px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--text-muted);
}
.run-name {
  padding: 0 20px 16px; font-size: 15px; font-weight: 650; letter-spacing: -.01em;
  word-break: break-word; border-bottom: 1px solid var(--border); margin-bottom: 10px;
}
.nav-group {
  padding: 14px 20px 5px; font-size: 10px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text-muted);
}
.nav-item {
  display: flex; align-items: center; gap: 9px; width: calc(100% - 16px);
  margin: 1px 8px; padding: 7px 11px; border: 0; border-radius: 7px;
  background: none; font: inherit; font-size: 13px; color: var(--text-secondary);
  cursor: pointer; text-align: left; line-height: 1.3;
}
.nav-item:hover { background: var(--surface-3); color: var(--text-primary); }
.nav-item.active { background: var(--accent); color: #fff; font-weight: 600; }
.nav-item.active .pill { background: rgba(255,255,255,.22); color: #fff; }
.pill {
  flex: 0 0 auto; min-width: 18px; height: 18px; padding: 0 5px; border-radius: 999px;
  background: var(--surface-3); color: var(--text-muted); font-size: 10px;
  font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
}
.nav-foot { margin-top: auto; padding: 14px 20px 0; border-top: 1px solid var(--border); }
.ghost {
  background: none; border: 1px solid var(--border); border-radius: 7px;
  padding: 5px 10px; font: inherit; font-size: 12px; color: var(--text-secondary);
  cursor: pointer;
}
.ghost:hover { color: var(--text-primary); border-color: var(--text-muted); }
.nav-foot p { font-size: 11px; color: var(--text-muted); margin: 10px 0 0; }

/* ---------------------------------------------------------------- main */
main { min-width: 0; padding: 30px 34px 72px; max-width: 1280px; }
.page-head { border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 24px; }
.eyebrow {
  text-transform: uppercase; letter-spacing: .1em; font-size: 10.5px;
  font-weight: 700; color: var(--text-muted);
}
h1 { margin: 5px 0 6px; font-size: 27px; letter-spacing: -.022em; line-height: 1.2; }
h2 { font-size: 13px; margin: 0; font-weight: 600; letter-spacing: -.005em; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.chip {
  font-size: 11px; color: var(--text-secondary); background: var(--surface-2);
  border: 1px solid var(--border); border-radius: 999px; padding: 2.5px 9px;
}
.question { color: var(--text-secondary); font-size: 14.5px; margin: 0 0 22px; max-width: 76ch; }
.panel { display: none; }
.panel.active { display: block; animation: rise .18s ease-out; }
@keyframes rise { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }

/* ---------------------------------------------------------------- kpis */
.kpis {
  display: grid; gap: 12px; margin: 0 0 26px;
  grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
}
.kpi {
  border: 1px solid var(--border); border-radius: 11px; padding: 13px 15px 14px;
  background: var(--surface-2);
}
.kpi-label {
  font-size: 10.5px; color: var(--text-muted); text-transform: uppercase;
  letter-spacing: .07em; font-weight: 650;
}
.kpi-value {
  font-size: 25px; font-weight: 650; letter-spacing: -.025em; margin-top: 3px;
  font-variant-numeric: tabular-nums; line-height: 1.15;
}
.kpi-value.long { font-size: 17px; letter-spacing: -.01em; }
.kpi-hint { font-size: 11.5px; color: var(--text-muted); margin-top: 5px; line-height: 1.4; }
.kpi.good .kpi-value { color: var(--good); }
.kpi.warning .kpi-value { color: var(--warning); }
.kpi.critical .kpi-value { color: var(--critical); }

/* -------------------------------------------------------------- charts */
.chart {
  margin: 0 0 22px; border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 18px 12px; background: var(--surface); box-shadow: var(--shadow);
}
figure.chart { }
figcaption {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 18px; margin-bottom: 10px;
}
.cap-title { font-size: 15px; font-weight: 650; letter-spacing: -.012em; margin: 0; }
.sub { color: var(--text-secondary); font-size: 12.5px; margin: 3px 0 0; max-width: 74ch; }
.dls { display: flex; gap: 6px; flex-shrink: 0; align-items: center; }
.dls-label {
  font-size: 10px; color: var(--text-muted); text-transform: uppercase;
  letter-spacing: .07em; margin-right: 1px;
}
.dl {
  font-size: 10.5px; font-weight: 700; letter-spacing: .05em; text-decoration: none;
  color: var(--text-secondary); border: 1px solid var(--border); border-radius: 6px;
  padding: 3.5px 8px; background: var(--surface-2); white-space: nowrap;
}
.dl:hover { color: var(--text-primary); border-color: var(--accent); }
.plot { width: 100%; }
.note { font-size: 12.5px; color: var(--text-muted); margin: 8px 0 4px; max-width: 86ch; }

/* -------------------------------------------------------------- tables */
.table-block { margin: 0 0 16px; border: 1px solid var(--border); border-radius: 12px;
               background: var(--surface); overflow: hidden; }
.table-toggle {
  display: flex; align-items: center; gap: 9px; width: 100%; background: none;
  border: 0; padding: 12px 16px; font: inherit; font-size: 13px; font-weight: 600;
  color: var(--text-primary); cursor: pointer; text-align: left;
}
.table-toggle:hover { background: var(--surface-2); }
.caret { color: var(--text-muted); font-size: 10px; transition: transform .15s; }
.table-block.open .caret { transform: rotate(90deg); }
.table-body { display: none; border-top: 1px solid var(--border); }
.table-block.open .table-body { display: block; }
.table-note { font-size: 12px; color: var(--text-muted); padding: 10px 16px 0; margin: 0; }
.table-wrap { overflow: auto; max-height: 460px; margin: 10px 0 0; }
table { border-collapse: collapse; width: 100%; font-size: 12px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 14px; border-bottom: 1px solid var(--grid);
         white-space: nowrap; }
td.num, th.num { text-align: right; }
th { background: var(--surface-2); font-weight: 650; color: var(--text-secondary);
     position: sticky; top: 0; z-index: 1; }
tbody tr:hover td { background: var(--surface-2); }
tbody tr:last-child td { border-bottom: 0; }

/* ------------------------------------------------------------- notices */
.empty, .notice {
  border: 1px dashed var(--border); border-radius: 11px; padding: 20px;
  color: var(--text-secondary); background: var(--surface-2);
}
.empty strong { color: var(--text-primary); }
.empty p { margin: 6px 0 0; font-size: 13px; }
.notice { margin-bottom: 20px; border-style: solid; font-size: 13px; }
footer { margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--border);
         color: var(--text-muted); font-size: 12px; }

@media (max-width: 900px) {
  .app { grid-template-columns: 1fr; }
  .side {
    position: static; height: auto; flex-direction: row; overflow-x: auto;
    align-items: center; padding: 10px 12px; gap: 6px;
    border-right: 0; border-bottom: 1px solid var(--border);
  }
  .brand, .run-name, .nav-group, .nav-foot p { display: none; }
  .side nav { display: flex; gap: 6px; }
  .nav-item { width: auto; margin: 0; white-space: nowrap; }
  .nav-foot { margin: 0 0 0 auto; padding: 0; border: 0; }
  main { padding: 20px 16px 56px; }
  figcaption { flex-direction: column; }
  h1 { font-size: 22px; }
}
"""


JS = """
const FIGURES = __FIGURES__;
const LIGHT = __LIGHT__;
const DARK = __DARK__;
const DRAWN = new Set();

function theme() {
  const stored = localStorage.getItem('dee-bench-theme') || 'system';
  document.documentElement.dataset.theme = stored === 'system' ? '' : stored;
  return stored;
}
function isDark() {
  const stamped = document.documentElement.dataset.theme;
  if (stamped) return stamped === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function themed(entry, dark) {
  const t = dark ? DARK : LIGHT;
  const fig = JSON.parse(JSON.stringify({data: entry.data, layout: entry.layout}));
  fig.data.forEach((tr, i) => {
    const over = dark ? (entry.dark || [])[i] : null;
    if (over) {
      if (over.line && tr.line) tr.line.color = over.line;
      if (over.marker && tr.marker) tr.marker.color = over.marker;
      if (over.fill && tr.fillcolor) tr.fillcolor = over.fill;
      if (over.colorscale) tr.colorscale = over.colorscale;
      if (over.text && tr.textfont) tr.textfont.color = over.text;
      if (over.error && tr.error_y && tr.error_y.color) tr.error_y.color = over.error;
    }
    if (tr.marker && tr.marker.line) tr.marker.line.color = t.surface;
    if (tr.type === 'heatmap') { tr.xgap = 2; tr.ygap = 2; }
  });
  Object.assign(fig.layout, {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: t.text_secondary, size: 11, family: __FONT_JSON__ },
    hoverlabel: { bgcolor: t.surface_2, bordercolor: t.border,
                  font: { color: t.text_primary, size: 11 } },
  });
  ['xaxis', 'yaxis'].forEach(a => {
    fig.layout[a] = Object.assign({}, fig.layout[a], {
      gridcolor: t.grid, linecolor: t.border, zerolinecolor: t.grid,
      tickfont: { color: t.text_secondary },
      title: Object.assign({}, (fig.layout[a] || {}).title,
                           { font: { color: t.text_muted, size: 11 } }),
    });
  });
  (fig.layout.shapes || []).forEach(s => {
    if (s.name !== 'event' && s.line && !s.line.color) s.line.color = t.text_muted;
  });
  (fig.layout.annotations || []).forEach(a => {
    if (a.name !== 'event') a.font = Object.assign({ size: 10 }, a.font,
                                                   { color: a.font && a.font.color ? a.font.color : t.text_muted });
  });
  if (fig.layout.legend) {
    fig.layout.legend.font = { color: t.text_secondary, size: 11 };
    fig.layout.legend.title = Object.assign({}, fig.layout.legend.title,
                                            { font: { color: t.text_muted, size: 10 } });
  }
  return fig;
}

const CONFIG = {
  responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
};

function draw(id, force) {
  const el = document.getElementById('plot-' + id);
  if (!el || !FIGURES[id]) return;
  if (DRAWN.has(id) && !force) return;
  const fig = themed(FIGURES[id], isDark());
  Plotly.react(el, fig.data, fig.layout, CONFIG);
  DRAWN.add(id);
}

// Charts are drawn when their page is first opened, not all at once: a sweep
// with a page per optimization can carry a hundred of them, and drawing the
// ninety nobody has looked at yet is what makes a dashboard feel slow.
function drawPanel(key, force) {
  const panel = document.getElementById('panel-' + key);
  if (!panel) return;
  panel.querySelectorAll('.plot').forEach(el => draw(el.dataset.chart, force));
  window.dispatchEvent(new Event('resize'));
}

function show(key) {
  document.querySelectorAll('.nav-item').forEach(b => {
    const on = b.dataset.panel === key;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.toggle('active', p.id === 'panel-' + key);
  });
  drawPanel(key, false);
  if (history.replaceState) history.replaceState(null, '', '#' + key);
}

document.querySelectorAll('.nav-item').forEach(b => {
  b.addEventListener('click', () => show(b.dataset.panel));
});
document.querySelectorAll('.table-toggle').forEach(b => {
  b.addEventListener('click', () => {
    const block = b.closest('.table-block');
    const open = block.classList.toggle('open');
    b.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
});

const toggle = document.getElementById('theme-toggle');
const LABELS = { system: 'Theme: system', light: 'Theme: light', dark: 'Theme: dark' };
function paintToggle(mode) { toggle.textContent = LABELS[mode]; }
toggle.addEventListener('click', () => {
  const order = ['system', 'light', 'dark'];
  const next = order[(order.indexOf(theme()) + 1) % order.length];
  localStorage.setItem('dee-bench-theme', next);
  theme();
  paintToggle(next);
  DRAWN.forEach(id => draw(id, true));
});

paintToggle(theme());
show((location.hash || '').slice(1) || __FIRST__);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  DRAWN.forEach(id => draw(id, true));
});
"""
