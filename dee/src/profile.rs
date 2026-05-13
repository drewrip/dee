use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    dag::Dag,
    executor::ExecStats,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileReport {
    pub generated_at: DateTime<Utc>,
    pub runs: Vec<DagRunProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagRunProfile {
    pub dag_file: String,
    pub db: String,
    pub run_started_at: DateTime<Utc>,
    pub run_finished_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub graph: DagGraphProfile,
    pub node_executions: Vec<NodeExecutionProfile>,
    pub system_samples: Vec<SystemUsageSample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagGraphProfile {
    pub nodes: Vec<DagNodeProfile>,
    pub edges: Vec<DagEdgeProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagNodeProfile {
    pub id: String,
    pub query_text: String,
    pub materialization: String,
    pub depends_on: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagEdgeProfile {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeExecutionProfile {
    pub node_id: String,
    pub start: DateTime<Utc>,
    pub finish: DateTime<Utc>,
    pub duration_ms: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemUsageSample {
    pub timestamp: DateTime<Utc>,
    pub elapsed_ms: i64,
    pub cpu_percent: Option<f64>,
    pub memory_bytes: Option<u64>,
}

pub fn build_dag_run_profile(dag_file: &str, dag: &Dag, exec_stats: &ExecStats) -> DagRunProfile {
    let mut nodes: Vec<_> = dag
        .nodes
        .nodes()
        .map(|node| DagNodeProfile {
            id: node.id.clone(),
            query_text: node.query_text.clone(),
            materialization: node.materialize.as_str().to_string(),
            depends_on: {
                let mut deps: Vec<_> = node.depends_on.iter().cloned().collect();
                deps.sort();
                deps
            },
        })
        .collect();
    nodes.sort_by(|a, b| a.id.cmp(&b.id));

    let mut edges = Vec::new();
    for node in dag.nodes.nodes() {
        for parent in &node.depends_on {
            edges.push(DagEdgeProfile {
                from: parent.clone(),
                to: node.id.clone(),
            });
        }
    }
    edges.sort_by(|a, b| a.from.cmp(&b.from).then(a.to.cmp(&b.to)));

    let mut node_executions: Vec<_> = exec_stats
        .node_stats
        .iter()
        .map(|(node_id, stats)| NodeExecutionProfile {
            node_id: node_id.clone(),
            start: stats.start,
            finish: stats.finish,
            duration_ms: stats.duration.num_milliseconds(),
        })
        .collect();
    node_executions.sort_by(|a, b| a.start.cmp(&b.start).then(a.node_id.cmp(&b.node_id)));

    DagRunProfile {
        dag_file: dag_file.to_string(),
        db: dag.db.clone(),
        run_started_at: exec_stats.start,
        run_finished_at: exec_stats.finish,
        duration_ms: exec_stats.duration.num_milliseconds(),
        graph: DagGraphProfile { nodes, edges },
        node_executions,
        system_samples: exec_stats.system_samples.clone(),
    }
}

pub fn render_profile_summary(report: &ProfileReport) -> String {
    let mut lines = vec!["Profile summary".to_string()];
    for run in &report.runs {
        let peak_memory = run.system_samples.iter().filter_map(|s| s.memory_bytes).max();
        let peak_cpu =
            run.system_samples
                .iter()
                .filter_map(|s| s.cpu_percent)
                .fold(None, |acc: Option<f64>, v| match acc {
                    Some(curr) => Some(curr.max(v)),
                    None => Some(v),
                });
        lines.push(format!(
            "- {}: {} nodes in {} ms{}{}",
            run.dag_file,
            run.graph.nodes.len(),
            run.duration_ms,
            peak_memory
                .map(|v| format!(", peak memory {}", format_bytes(v)))
                .unwrap_or_default(),
            peak_cpu
                .map(|v| format!(", peak cpu {:.1}%", v))
                .unwrap_or_default()
        ));
    }
    lines.join("\n")
}

fn format_bytes(bytes: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < units.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", units[unit])
}

pub fn render_profile_html(report: &ProfileReport) -> Result<String, serde_json::Error> {
    let report_json = serde_json::to_string(report)?.replace("</script>", "<\\/script>");
    Ok(format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>dee profile</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #5b6472;
      --grid: #d7d0c2;
      --table: #b45309;
      --view: #0f766e;
      --accent: #1d4ed8;
      --cpu: #dc2626;
      --mem: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background:
        radial-gradient(circle at top left, rgba(29,78,216,0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(180,83,9,0.08), transparent 24%),
        var(--bg);
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    }}
    h1, h2, h3 {{ margin: 0; }}
    .shell {{ max-width: 1480px; margin: 0 auto; }}
    .hero {{
      background: var(--panel);
      border: 1px solid rgba(31,41,55,0.08);
      border-radius: 22px;
      padding: 24px 28px;
      box-shadow: 0 10px 30px rgba(31,41,55,0.08);
    }}
    .hero p {{ margin: 8px 0 0; color: var(--muted); }}
    .tabs {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 18px 0 22px;
    }}
    .tab {{
      border: 1px solid rgba(31,41,55,0.12);
      border-radius: 999px;
      background: rgba(255,255,255,0.7);
      color: var(--ink);
      padding: 10px 16px;
      cursor: pointer;
      font: inherit;
    }}
    .tab.active {{
      background: var(--ink);
      color: white;
    }}
    .page {{
      display: none;
      gap: 18px;
    }}
    .page.active {{
      display: grid;
      grid-template-columns: 1fr;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(31,41,55,0.08);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: 0 8px 24px rgba(31,41,55,0.05);
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      margin-top: 8px;
      font-size: 26px;
      font-weight: 700;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(31,41,55,0.08);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(31,41,55,0.05);
    }}
    .panel h2 {{
      font-size: 20px;
      margin-bottom: 12px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 10px;
    }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .query-list {{
      display: grid;
      gap: 12px;
    }}
    .query-item {{
      padding: 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(31,41,55,0.08);
    }}
    .query-item code {{
      white-space: pre-wrap;
      font-size: 12px;
      color: var(--ink);
    }}
    .node-tag {{
      display: inline-block;
      font-size: 11px;
      margin-left: 8px;
      padding: 2px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: white;
    }}
    .node-tag.table {{ background: var(--table); }}
    .node-tag.view {{ background: var(--view); }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <h1>dee profiling report</h1>
      <p>{} DAG run(s), generated {}</p>
    </div>
    <div class="tabs" id="tabs"></div>
    <div id="pages"></div>
  </div>
  <script>
    const report = {};

    function formatMs(ms) {{
      return `${{ms.toLocaleString()}} ms`;
    }}

    function formatBytes(bytes) {{
      if (bytes == null) return "n/a";
      const units = ["B", "KiB", "MiB", "GiB", "TiB"];
      let value = bytes;
      let unit = 0;
      while (value >= 1024 && unit < units.length - 1) {{
        value /= 1024;
        unit += 1;
      }}
      return `${{value.toFixed(1)}} ${{units[unit]}}`;
    }}

    function escapeHtml(text) {{
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function buildDagSvg(run) {{
      const width = 1180;
      const margin = {{ top: 36, right: 40, bottom: 36, left: 40 }};
      const nodeWidth = 180;
      const nodeHeight = 64;
      const layerGap = 230;
      const rowGap = 110;

      const nodesById = Object.fromEntries(run.graph.nodes.map(node => [node.id, node]));
      const memo = new Map();
      function depth(id) {{
        if (memo.has(id)) return memo.get(id);
        const node = nodesById[id];
        const value = !node.depends_on.length
          ? 0
          : Math.max(...node.depends_on.map(dep => depth(dep))) + 1;
        memo.set(id, value);
        return value;
      }}

      const layers = new Map();
      run.graph.nodes.forEach(node => {{
        const d = depth(node.id);
        if (!layers.has(d)) layers.set(d, []);
        layers.get(d).push(node);
      }});
      [...layers.values()].forEach(layer => layer.sort((a, b) => a.id.localeCompare(b.id)));

      const positions = new Map();
      let maxRows = 1;
      [...layers.entries()].forEach(([layer, nodes]) => {{
        maxRows = Math.max(maxRows, nodes.length);
        nodes.forEach((node, idx) => {{
          positions.set(node.id, {{
            x: margin.left + layer * layerGap,
            y: margin.top + idx * rowGap,
          }});
        }});
      }});
      const height = margin.top + margin.bottom + maxRows * rowGap;

      const durations = Object.fromEntries(run.node_executions.map(node => [node.node_id, node.duration_ms]));
      const edgeEls = run.graph.edges.map(edge => {{
        const from = positions.get(edge.from);
        const to = positions.get(edge.to);
        return `<line x1="${{from.x + nodeWidth}}" y1="${{from.y + nodeHeight / 2}}" x2="${{to.x}}" y2="${{to.y + nodeHeight / 2}}" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)" />`;
      }}).join("");

      const nodeEls = run.graph.nodes.map(node => {{
        const pos = positions.get(node.id);
        const isTable = node.materialization === "table";
        const fill = isTable ? "rgba(180,83,9,0.12)" : "rgba(15,118,110,0.12)";
        const stroke = isTable ? "var(--table)" : "var(--view)";
        const runtime = durations[node.id];
        return `
          <g transform="translate(${{pos.x}},${{pos.y}})">
            <rect width="${{nodeWidth}}" height="${{nodeHeight}}" rx="18" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{isTable ? 3 : 2}}" />
            <text x="14" y="24" font-size="15" font-weight="700" fill="var(--ink)">${{escapeHtml(node.id)}}</text>
            <text x="14" y="45" font-size="12" fill="var(--muted)">${{node.materialization.toUpperCase()}}${{runtime != null ? ` · ${{runtime}} ms` : ""}}</text>
          </g>
        `;
      }}).join("");

      return `
        <svg viewBox="0 0 ${{width}} ${{height}}" aria-label="dag graph">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"></path>
            </marker>
          </defs>
          ${{edgeEls}}
          ${{nodeEls}}
        </svg>
      `;
    }}

    function buildTimelineSvg(run, accessor, color, label, formatter) {{
      const width = 1180;
      const height = 250;
      const margin = {{ top: 24, right: 24, bottom: 34, left: 64 }};
      const samples = run.system_samples.filter(sample => accessor(sample) != null);
      const maxX = Math.max(run.duration_ms, ...run.system_samples.map(sample => sample.elapsed_ms), 1);

      let maxY = samples.length ? Math.max(...samples.map(sample => accessor(sample))) : 1;
      if (maxY <= 0) maxY = 1;

      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const x = value => margin.left + (value / maxX) * plotWidth;
      const y = value => margin.top + plotHeight - (value / maxY) * plotHeight;

      const grid = [0, 0.25, 0.5, 0.75, 1].map(tick => {{
        const yy = margin.top + plotHeight - tick * plotHeight;
        const value = maxY * tick;
        return `
          <line x1="${{margin.left}}" y1="${{yy}}" x2="${{width - margin.right}}" y2="${{yy}}" stroke="var(--grid)" stroke-dasharray="4 6" />
          <text x="${{margin.left - 10}}" y="${{yy + 4}}" text-anchor="end" font-size="11" fill="var(--muted)">${{formatter(value)}}</text>
        `;
      }}).join("");

      const xTicks = [0, 0.25, 0.5, 0.75, 1].map(tick => {{
        const xx = margin.left + tick * plotWidth;
        const value = Math.round(maxX * tick);
        return `
          <line x1="${{xx}}" y1="${{margin.top}}" x2="${{xx}}" y2="${{height - margin.bottom}}" stroke="var(--grid)" stroke-dasharray="4 6" />
          <text x="${{xx}}" y="${{height - 10}}" text-anchor="middle" font-size="11" fill="var(--muted)">${{value}} ms</text>
        `;
      }}).join("");

      const path = samples.length
        ? samples.map((sample, idx) => `${{idx === 0 ? "M" : "L"}} ${{x(sample.elapsed_ms)}} ${{y(accessor(sample))}}`).join(" ")
        : "";

      return `
        <svg viewBox="0 0 ${{width}} ${{height}}" aria-label="${{label}} chart">
          <text x="${{margin.left}}" y="16" font-size="14" font-weight="700" fill="var(--ink)">${{label}}</text>
          ${{grid}}
          ${{xTicks}}
          <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="var(--ink)" />
          <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="var(--ink)" />
          ${{samples.length ? `<path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />` : `<text x="${{margin.left}}" y="${{height / 2}}" fill="var(--muted)">No samples available.</text>`}}
        </svg>
      `;
    }}

    function buildGanttSvg(run) {{
      const width = 1180;
      const rowHeight = 34;
      const margin = {{ top: 24, right: 24, bottom: 34, left: 220 }};
      const rows = [...run.node_executions].sort((a, b) => a.start.localeCompare(b.start));
      const height = margin.top + margin.bottom + Math.max(rows.length, 1) * rowHeight;
      const maxX = Math.max(run.duration_ms, 1);
      const plotWidth = width - margin.left - margin.right;
      const x = value => margin.left + (value / maxX) * plotWidth;

      const nodesById = Object.fromEntries(run.graph.nodes.map(node => [node.id, node]));
      const grid = [0, 0.25, 0.5, 0.75, 1].map(tick => {{
        const xx = margin.left + tick * plotWidth;
        const value = Math.round(maxX * tick);
        return `
          <line x1="${{xx}}" y1="${{margin.top}}" x2="${{xx}}" y2="${{height - margin.bottom}}" stroke="var(--grid)" stroke-dasharray="4 6" />
          <text x="${{xx}}" y="${{height - 10}}" text-anchor="middle" font-size="11" fill="var(--muted)">${{value}} ms</text>
        `;
      }}).join("");

      const bars = rows.map((row, idx) => {{
        const top = margin.top + idx * rowHeight + 6;
        const start = new Date(row.start).getTime() - new Date(run.run_started_at).getTime();
        const duration = Math.max(row.duration_ms, 4);
        const node = nodesById[row.node_id];
        const fill = node.materialization === "table" ? "rgba(180,83,9,0.78)" : "rgba(15,118,110,0.78)";
        return `
          <text x="${{margin.left - 12}}" y="${{top + 14}}" text-anchor="end" font-size="12" fill="var(--ink)">${{escapeHtml(row.node_id)}}</text>
          <rect x="${{x(start)}}" y="${{top}}" width="${{Math.max((duration / maxX) * plotWidth, 4)}}" height="20" rx="8" fill="${{fill}}" />
          <text x="${{x(start) + 8}}" y="${{top + 14}}" font-size="11" fill="white">${{row.duration_ms}} ms</text>
        `;
      }}).join("");

      return `
        <svg viewBox="0 0 ${{width}} ${{height}}" aria-label="gantt chart">
          ${{grid}}
          <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="var(--ink)" />
          <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="var(--ink)" />
          ${{bars}}
        </svg>
      `;
    }}

    function renderRun(run, index) {{
      const peakMemory = run.system_samples.reduce((acc, sample) => sample.memory_bytes == null ? acc : Math.max(acc, sample.memory_bytes), 0);
      const peakCpu = run.system_samples.reduce((acc, sample) => sample.cpu_percent == null ? acc : Math.max(acc, sample.cpu_percent), 0);
      const nodeIndex = Object.fromEntries(run.node_executions.map(node => [node.node_id, node]));

      return `
        <section class="page${{index === 0 ? " active" : ""}}" data-page="${{index}}">
          <div class="summary">
            <div class="card"><div class="label">Dag file</div><div class="value" style="font-size:20px">${{escapeHtml(run.dag_file)}}</div></div>
            <div class="card"><div class="label">Database</div><div class="value">${{escapeHtml(run.db)}}</div></div>
            <div class="card"><div class="label">Nodes</div><div class="value">${{run.graph.nodes.length}}</div></div>
            <div class="card"><div class="label">Runtime</div><div class="value">${{formatMs(run.duration_ms)}}</div></div>
            <div class="card"><div class="label">Peak memory</div><div class="value">${{peakMemory ? formatBytes(peakMemory) : "n/a"}}</div></div>
            <div class="card"><div class="label">Peak CPU</div><div class="value">${{peakCpu ? `${{peakCpu.toFixed(1)}}%` : "n/a"}}</div></div>
          </div>

          <div class="panel">
            <h2>DAG</h2>
            <div class="legend">
              <span><span class="swatch" style="background: rgba(180,83,9,0.78)"></span>Table nodes are emphasized and annotated with runtime</span>
              <span><span class="swatch" style="background: rgba(15,118,110,0.78)"></span>View nodes</span>
            </div>
            ${{buildDagSvg(run)}}
          </div>

          <div class="panel">
            <h2>Execution Gantt</h2>
            ${{buildGanttSvg(run)}}
          </div>

          <div class="panel">
            <h2>System samples</h2>
            <div class="legend">
              <span><span class="swatch" style="background: var(--cpu)"></span>CPU usage</span>
              <span><span class="swatch" style="background: var(--mem)"></span>Memory usage</span>
            </div>
            ${{buildTimelineSvg(run, sample => sample.cpu_percent, "var(--cpu)", "CPU usage", value => `${{value.toFixed(1)}}%`)}}
            ${{buildTimelineSvg(run, sample => sample.memory_bytes, "var(--mem)", "Memory usage", value => formatBytes(value))}}
          </div>

          <div class="panel">
            <h2>Nodes and SQL</h2>
            <div class="query-list">
              ${{run.graph.nodes.map(node => `
                <div class="query-item">
                  <strong>${{escapeHtml(node.id)}}</strong>
                  <span class="node-tag ${{node.materialization}}">${{node.materialization}}</span>
                  <span style="color: var(--muted); margin-left: 8px;">${{nodeIndex[node.id] ? `${{nodeIndex[node.id].duration_ms}} ms` : "not executed"}}</span>
                  <pre><code>${{escapeHtml(node.query_text)}}</code></pre>
                </div>
              `).join("")}}
            </div>
          </div>
        </section>
      `;
    }}

    const tabs = document.getElementById("tabs");
    const pages = document.getElementById("pages");

    tabs.innerHTML = report.runs.map((run, index) => `
      <button class="tab${{index === 0 ? " active" : ""}}" data-index="${{index}}">
        DAG ${{index + 1}}
      </button>
    `).join("");
    pages.innerHTML = report.runs.map(renderRun).join("");

    const tabEls = [...document.querySelectorAll(".tab")];
    const pageEls = [...document.querySelectorAll(".page")];
    tabEls.forEach(tab => {{
      tab.addEventListener("click", () => {{
        const index = Number(tab.dataset.index);
        tabEls.forEach((el, idx) => el.classList.toggle("active", idx === index));
        pageEls.forEach((el, idx) => el.classList.toggle("active", idx === index));
      }});
    }});
  </script>
</body>
</html>
"##,
        report.runs.len(),
        report.generated_at,
        report_json
    ))
}
