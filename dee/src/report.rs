pub(crate) const REPORT_CSS: &str = r#"
    :root {
      --bg: #f3f6fb;
      --panel: rgba(255,255,255,0.88);
      --panel-strong: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --grid: #dbe4f0;
      --table: #2563eb;
      --table-soft: rgba(37,99,235,0.14);
      --view: #0f766e;
      --view-soft: rgba(15,118,110,0.14);
      --source: #d97706;
      --source-soft: rgba(245,158,11,0.14);
      --temp-table: #7c3aed;
      --temp-table-soft: rgba(124,58,237,0.14);
      --accent: #0f172a;
      --cpu: #ef4444;
      --mem: #8b5cf6;
      --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 28px;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(16,185,129,0.12), transparent 22%),
        linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.88)),
        var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1, h2, h3 { margin: 0; }
    .shell { max-width: 1560px; margin: 0 auto; }
    .hero {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(148,163,184,0.16);
      border-radius: 28px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.68);
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hero h1 {
      margin-top: 14px;
      font-size: 38px;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 900px;
      line-height: 1.6;
      font-size: 15px;
    }
    .tabs {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 20px 0 24px;
    }
    .tab {
      border: 1px solid rgba(148,163,184,0.22);
      border-radius: 999px;
      background: rgba(255,255,255,0.78);
      color: var(--ink);
      padding: 11px 17px;
      cursor: pointer;
      font: inherit;
      font-weight: 600;
      transition: 160ms ease;
    }
    .tab.active {
      background: var(--ink);
      color: white;
      transform: translateY(-1px);
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
    }
    .tab:hover { border-color: rgba(37,99,235,0.35); }
    .page { display: none; gap: 18px; }
    .page.active { display: grid; grid-template-columns: 1fr; }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }
    .card {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(148,163,184,0.14);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 28px rgba(15,23,42,0.05);
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    .value {
      margin-top: 10px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .panel {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(148,163,184,0.14);
      border-radius: 24px;
      padding: 20px;
      box-shadow: var(--shadow);
    }
    .panel h2 {
      font-size: 21px;
      margin-bottom: 8px;
      letter-spacing: -0.03em;
    }
    .subtle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
      margin-bottom: 14px;
    }
    svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .legend {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 10px;
    }
    .swatch {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
      vertical-align: middle;
    }
    .dag-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.65fr) 340px;
      gap: 18px;
      align-items: start;
    }
    .dag-canvas {
      aspect-ratio: 16 / 10;
      width: 100%;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,0.95));
      border: 1px solid rgba(148,163,184,0.16);
      padding: 16px;
      overflow: auto;
      cursor: grab;
    }
    .dag-canvas:active { cursor: grabbing; }
    .dag-canvas .graph,
    .dag-canvas svg text {
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .dag-link {
      fill: none;
      stroke: #c2cfdf;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: 0.95;
    }
    .dag-sidepanel {
      min-height: 480px;
      border-radius: 20px;
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(148,163,184,0.16);
      padding: 18px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }
    .detail-empty {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
      padding-top: 12px;
    }
    .detail-name {
      font-size: 21px;
      font-weight: 700;
      letter-spacing: -0.04em;
      word-break: break-word;
      line-height: 1.2;
    }
    .detail-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 12px 0 16px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 11px;
      border-radius: 999px;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,0.18);
      color: var(--ink);
      font-size: 12px;
      font-weight: 600;
    }
    .pill.table { background: var(--table-soft); color: var(--table); border-color: rgba(37,99,235,0.16); }
    .pill.view { background: var(--view-soft); color: var(--view); border-color: rgba(15,118,110,0.16); }
    .pill.source { background: var(--source-soft); color: #b45309; border-color: rgba(217,119,6,0.18); }
    .pill.temp_table { background: var(--temp-table-soft); color: var(--temp-table); border-color: rgba(124,58,237,0.16); }
    .detail-grid { display: grid; gap: 12px; }
    .detail-box {
      border-radius: 16px;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,0.16);
      padding: 14px;
    }
    .detail-box h3 {
      font-size: 13px;
      margin-bottom: 8px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .detail-box code, .detail-box pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      color: var(--ink);
      line-height: 1.6;
    }
    .query-list { display: grid; gap: 12px; }
    .query-item {
      padding: 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(148,163,184,0.14);
    }
    .query-item code {
      white-space: pre-wrap;
      font-size: 12px;
      color: var(--ink);
    }
    .node-tag {
      display: inline-block;
      font-size: 11px;
      margin-left: 8px;
      padding: 2px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }
    .node-tag.table { background: var(--table-soft); color: var(--table); }
    .node-tag.view { background: var(--view-soft); color: var(--view); }
    .node-tag.temp_table { background: var(--temp-table-soft); color: var(--temp-table); }
    .section-stack { display: grid; gap: 18px; }
    .chart-stack { display: grid; gap: 14px; }
    .svg-wrap {
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(148,163,184,0.14);
      padding: 10px;
      overflow: auto;
    }
    .dag-node, .dag-canvas .node {
      cursor: pointer;
      transition: opacity 140ms ease;
    }
    .dag-node:hover, .dag-canvas .node:hover { opacity: 0.94; }
    .dag-node.selected rect.primary {
      stroke: var(--accent);
      stroke-width: 3;
      filter: drop-shadow(0 10px 22px rgba(15,23,42,0.15));
    }
    .dag-node text {
      user-select: none;
      pointer-events: none;
    }
    .plan-tree {
      margin-top: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }
    .plan-node {
      border-left: 1px solid var(--grid);
      padding-left: 12px;
      margin-bottom: 4px;
    }
    .plan-header {
      display: flex;
      align-items: center;
      gap: 12px;
      cursor: pointer;
      padding: 6px 10px;
      border-radius: 8px;
      transition: background 0.2s;
    }
    .plan-header:hover { background: rgba(0,0,0,0.04); }
    .plan-toggle { 
      width: 16px; 
      height: 16px; 
      display: flex; 
      align-items: center; 
      justify-content: center; 
      transition: transform 0.2s;
      color: var(--muted);
      font-size: 10px;
    }
    .plan-node.folded > .plan-children { display: none; }
    .plan-node.folded > .plan-header .plan-toggle { transform: rotate(-90deg); }
    .plan-type { font-weight: 700; min-width: 140px; }
    .plan-impact-wrap { width: 100px; height: 8px; background: var(--grid); border-radius: 4px; overflow: hidden; flex-shrink: 0; }
    .plan-impact-bar { height: 100%; background: var(--cpu); }
    .plan-rows { color: var(--muted); min-width: 80px; text-align: right; }
    .plan-bytes { color: var(--muted); min-width: 80px; text-align: right; font-size: 11px; }
    .plan-timing { color: var(--muted); min-width: 60px; text-align: right; font-size: 10px; }
    .plan-extra { 
      margin: 4px 0 8px 20px; 
      padding: 10px; 
      background: #f8fafc; 
      border: 1px solid var(--grid); 
      border-radius: 8px; 
      display: none; 
      font-size: 11px;
    }
    .plan-extra.active { display: block; }
    .plan-extra-row { display: flex; gap: 8px; margin-bottom: 2px; }
    .plan-extra-key { font-weight: 700; color: var(--muted); min-width: 120px; }
    .view-plan-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 16px;
      border-radius: 12px;
      background: var(--ink);
      color: white;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 16px;
      transition: opacity 0.2s;
      cursor: pointer;
    }
    .view-plan-btn:hover { opacity: 0.85; }
    .view-plan-btn.disabled {
      background: var(--grid);
      color: var(--muted);
      cursor: not-allowed;
      opacity: 0.7;
    }
    .panel summary {
      cursor: pointer;
      list-style: none;
      outline: none;
    }
    .panel summary::-webkit-details-marker { display: none; }
    .panel summary h2 {
      display: inline-flex;
      align-items: center;
      gap: 12px;
    }
    .panel summary h2::before {
      content: "▼";
      font-size: 14px;
      color: var(--muted);
      transition: transform 0.2s;
    }
    .panel[open] summary h2::before { transform: rotate(0); }
    .panel:not([open]) summary h2::before { transform: rotate(-90deg); }
    
    .back-to-top {
      position: fixed;
      bottom: 30px;
      right: 30px;
      width: 48px;
      height: 48px;
      border-radius: 24px;
      background: var(--ink);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
      opacity: 0;
      visibility: hidden;
      transition: all 0.3s;
      z-index: 1000;
      border: none;
    }
    .back-to-top.visible {
      opacity: 1;
      visibility: visible;
    }
    .back-to-top:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 28px rgba(0,0,0,0.2);
    }

    .compare-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      margin-top: 10px;
    }
    .compare-table th, .compare-table td {
      padding: 14px;
      text-align: left;
      border-bottom: 1px solid var(--grid);
    }
    .compare-table th {
      background: rgba(248, 250, 252, 0.8);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      font-weight: 700;
    }
    .compare-row:hover td {
      background: rgba(0,0,0,0.02);
    }
    .compare-label {
      font-weight: 700;
      color: var(--ink);
      width: 200px;
    }
    .compare-value {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
    }
    .compare-delta {
      font-size: 11px;
      margin-left: 6px;
      font-weight: 600;
    }
    .delta-pos { color: #ef4444; }
    .delta-neg { color: #10b981; }
    .delta-neutral { color: var(--muted); }
    
    .baseline-selector {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--grid);
      background: white;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      transition: all 0.2s;
    }
    .baseline-selector:hover {
      border-color: var(--muted);
    }
    .baseline-selector.active {
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }
    .baseline-tag {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      background: var(--ink);
      color: white;
      font-size: 9px;
      text-transform: uppercase;
      font-weight: 800;
      margin-left: 8px;
      vertical-align: middle;
    }
    .compare-col-highlight {
      background: rgba(15, 23, 42, 0.03) !important;
    }

    @media (max-width: 1100px) {
      body { padding: 18px; }
      .dag-layout { grid-template-columns: 1fr; }
      .dag-canvas, .dag-sidepanel { min-height: 0; }
    }
  "#;

/// Generic HTML report shell shared by the profiling (`run --profile-viz`)
/// and optimizer explain (`opt --explain`) reports: same hero header, tab
/// bar, page container, back-to-top button, and CSS.
///
/// `tabs_html`/`pages_html` are the *initial* contents of the `#tabs`/
/// `#pages` containers -- a caller can leave them empty and populate them
/// itself via `extra_body_script` (as the profiling report does, driving
/// everything off one embedded JSON blob), or pre-render them server-side
/// (as the explain report does). Either way, the generic tab-click/back-to-
/// top wiring below operates on whatever `.tab`/`.page` elements exist once
/// `extra_body_script` has run.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_report_shell(
    title: &str,
    eyebrow: &str,
    heading: &str,
    subtitle_html: &str,
    tabs_html: &str,
    pages_html: &str,
    extra_head: &str,
    extra_body_script: &str,
) -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
{css}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="eyebrow">{eyebrow}</div>
      <h1>{heading}</h1>
      {subtitle_html}
    </div>
    <div class="tabs" id="tabs">{tabs_html}</div>
    <div id="pages">{pages_html}</div>
    <button class="back-to-top" id="backToTop" title="Back to top">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 15l-6-6-6 6"/>
      </svg>
    </button>
  </div>
{extra_head}  <script>
{extra_body_script}
    (function () {{
      const tabEls = [...document.querySelectorAll(".tab")];
      const pageEls = [...document.querySelectorAll(".page")];
      tabEls.forEach(tab => {{
        tab.addEventListener("click", () => {{
          const targetId = tab.dataset.index;
          tabEls.forEach(el => el.classList.toggle("active", el.dataset.index === targetId));
          pageEls.forEach(el => {{
            el.classList.toggle("active", el.dataset.page === targetId);
          }});
        }});
      }});

      const backToTop = document.getElementById("backToTop");
      window.addEventListener("scroll", () => {{
        if (window.pageYOffset > 300) {{
          backToTop.classList.add("visible");
        }} else {{
          backToTop.classList.remove("visible");
        }}
      }});
      backToTop.addEventListener("click", () => {{
        window.scrollTo({{ top: 0, behavior: "smooth" }});
      }});
    }})();
  </script>
</body>
</html>
"##,
        title = title,
        css = REPORT_CSS,
        eyebrow = eyebrow,
        heading = heading,
        subtitle_html = subtitle_html,
        tabs_html = tabs_html,
        pages_html = pages_html,
        extra_head = extra_head,
        extra_body_script = extra_body_script,
    )
}
