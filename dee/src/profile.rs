use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{dag::Dag, executor::ExecStats};

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
    pub sources: Vec<DagSourceProfile>,
    pub edges: Vec<DagEdgeProfile>,
    pub source_edges: Vec<DagEdgeProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagNodeProfile {
    pub id: String,
    pub query_text: String,
    pub materialization: String,
    pub depends_on: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagSourceProfile {
    pub name: String,
    pub columns: Vec<DagSourceColumnProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagSourceColumnProfile {
    pub name: String,
    pub data_type: String,
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

fn normalize_identifier(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !matches!(ch, '"' | '\'' | '`'))
        .collect::<String>()
        .to_ascii_lowercase()
}

fn node_references_source(query_text: &str, source_name: &str) -> bool {
    let query_lower = query_text.to_ascii_lowercase();
    let source_lower = source_name.to_ascii_lowercase();
    if query_lower.contains(&source_lower) {
        return true;
    }

    let normalized_query = normalize_identifier(query_text);
    let normalized_source = normalize_identifier(source_name);
    normalized_query.contains(&normalized_source)
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

    let mut sources: Vec<_> = dag
        .sources
        .iter()
        .map(|source| DagSourceProfile {
            name: source.name.clone(),
            columns: source
                .schema
                .flattened_fields()
                .iter()
                .map(|field| DagSourceColumnProfile {
                    name: field.name().clone(),
                    data_type: field.data_type().to_string(),
                })
                .collect(),
        })
        .collect();
    sources.sort_by(|a, b| a.name.cmp(&b.name));

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

    let mut source_edges = Vec::new();
    for source in &sources {
        for node in &nodes {
            if node_references_source(&node.query_text, &source.name) {
                source_edges.push(DagEdgeProfile {
                    from: source.name.clone(),
                    to: node.id.clone(),
                });
            }
        }
    }
    source_edges.sort_by(|a, b| a.from.cmp(&b.from).then(a.to.cmp(&b.to)));

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
        graph: DagGraphProfile {
            nodes,
            sources,
            edges,
            source_edges,
        },
        node_executions,
        system_samples: exec_stats.system_samples.clone(),
    }
}

pub fn render_profile_summary(report: &ProfileReport) -> String {
    let mut lines = vec!["Profile summary".to_string()];
    for run in &report.runs {
        let peak_memory = run
            .system_samples
            .iter()
            .filter_map(|s| s.memory_bytes)
            .max();
        let peak_cpu = run
            .system_samples
            .iter()
            .filter_map(|s| s.cpu_percent)
            .fold(None, |acc: Option<f64>, v| match acc {
                Some(curr) => Some(curr.max(v)),
                None => Some(v),
            });
        lines.push(format!(
            "- {}: {} nodes, {} sources in {} ms{}{}",
            run.dag_file,
            run.graph.nodes.len(),
            run.graph.sources.len(),
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
      --accent: #0f172a;
      --cpu: #ef4444;
      --mem: #8b5cf6;
      --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(16,185,129,0.12), transparent 22%),
        linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.88)),
        var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1, h2, h3 {{ margin: 0; }}
    .shell {{ max-width: 1560px; margin: 0 auto; }}
    .hero {{
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(148,163,184,0.16);
      border-radius: 28px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
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
    }}
    .hero h1 {{
      margin-top: 14px;
      font-size: 38px;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 900px;
      line-height: 1.6;
      font-size: 15px;
    }}
    .tabs {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 20px 0 24px;
    }}
    .tab {{
      border: 1px solid rgba(148,163,184,0.22);
      border-radius: 999px;
      background: rgba(255,255,255,0.78);
      color: var(--ink);
      padding: 11px 17px;
      cursor: pointer;
      font: inherit;
      font-weight: 600;
      transition: 160ms ease;
    }}
    .tab.active {{
      background: var(--ink);
      color: white;
      transform: translateY(-1px);
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
    }}
    .tab:hover {{ border-color: rgba(37,99,235,0.35); }}
    .page {{ display: none; gap: 18px; }}
    .page.active {{ display: grid; grid-template-columns: 1fr; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(148,163,184,0.14);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 28px rgba(15,23,42,0.05);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }}
    .value {{
      margin-top: 10px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .panel {{
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(148,163,184,0.14);
      border-radius: 24px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      font-size: 21px;
      margin-bottom: 8px;
      letter-spacing: -0.03em;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
      margin-bottom: 14px;
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
    .dag-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.65fr) 340px;
      gap: 18px;
      align-items: start;
    }}
    .dag-canvas {{
      aspect-ratio: 16 / 10;
      width: 100%;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,0.95));
      border: 1px solid rgba(148,163,184,0.16);
      padding: 16px;
      overflow: auto;
      cursor: grab;
    }}
    .dag-canvas:active {{ cursor: grabbing; }}
    .dag-canvas .graph,
    .dag-canvas svg text {{
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }}
    .dag-link {{
      fill: none;
      stroke: #c2cfdf;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: 0.95;
    }}
    .dag-sidepanel {{
      min-height: 480px;
      border-radius: 20px;
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(148,163,184,0.16);
      padding: 18px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }}
    .detail-empty {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
      padding-top: 12px;
    }}
    .detail-name {{
      font-size: 23px;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    .detail-meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 12px 0 16px;
    }}
    .pill {{
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
    }}
    .pill.table {{ background: var(--table-soft); color: var(--table); border-color: rgba(37,99,235,0.16); }}
    .pill.view {{ background: var(--view-soft); color: var(--view); border-color: rgba(15,118,110,0.16); }}
    .pill.source {{ background: var(--source-soft); color: #b45309; border-color: rgba(217,119,6,0.18); }}
    .detail-grid {{ display: grid; gap: 12px; }}
    .detail-box {{
      border-radius: 16px;
      background: #f8fafc;
      border: 1px solid rgba(148,163,184,0.16);
      padding: 14px;
    }}
    .detail-box h3 {{
      font-size: 13px;
      margin-bottom: 8px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .detail-box code, .detail-box pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      color: var(--ink);
      line-height: 1.6;
    }}
    .query-list {{ display: grid; gap: 12px; }}
    .query-item {{
      padding: 14px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(148,163,184,0.14);
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
      font-weight: 700;
    }}
    .node-tag.table {{ background: var(--table-soft); color: var(--table); }}
    .node-tag.view {{ background: var(--view-soft); color: var(--view); }}
    .section-stack {{ display: grid; gap: 18px; }}
    .chart-stack {{ display: grid; gap: 14px; }}
    .svg-wrap {{
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(148,163,184,0.14);
      padding: 10px;
      overflow: auto;
    }}
    .dag-node, .dag-canvas .node {{
      cursor: pointer;
      transition: opacity 140ms ease;
    }}
    .dag-node:hover, .dag-canvas .node:hover {{ opacity: 0.94; }}
    .dag-node.selected rect.primary {{
      stroke: var(--accent);
      stroke-width: 3;
      filter: drop-shadow(0 10px 22px rgba(15,23,42,0.15));
    }}
    .dag-node text {{
      user-select: none;
      pointer-events: none;
    }}
    @media (max-width: 1100px) {{
      body {{ padding: 18px; }}
      .dag-layout {{ grid-template-columns: 1fr; }}
      .dag-canvas, .dag-sidepanel {{ min-height: 0; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="eyebrow">dee profiler</div>
      <h1>profiling report</h1>
      <p>{} DAG run(s), generated {}.</p>
    </div>
    <div class="tabs" id="tabs"></div>
    <div id="pages"></div>
  </div>
  <script src="https://unpkg.com/d3@7/dist/d3.min.js"></script>
  <script src="https://unpkg.com/d3-dag@1.1.0"></script>
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

    function wrapTextLines(text, maxChars, maxLines) {{
      if (!text || maxChars <= 0 || maxLines <= 0) return [];
      const words = text.split(/\s+/).filter(Boolean);
      const lines = [];
      let current = "";
      const appendWord = (word) => {{
        if (!current) {{
          current = word;
          return;
        }}
        if ([...current].length + 1 + [...word].length <= maxChars) {{
          current += ` ${{word}}`;
        }} else {{
          lines.push(current);
          current = word;
        }}
      }};
      const pushBroken = (word) => {{
        let chunk = "";
        for (const ch of word) {{
          chunk += ch;
          if ([...chunk].length >= maxChars) {{
            lines.push(chunk);
            chunk = "";
            if (lines.length >= maxLines) return true;
          }}
        }}
        current = chunk;
        return false;
      }};

      for (const word of words.length ? words : [text]) {{
        if ([...word].length > maxChars) {{
          if (current) {{
            lines.push(current);
            current = "";
            if (lines.length >= maxLines) break;
          }}
          if (pushBroken(word)) break;
          continue;
        }}
        appendWord(word);
        if (lines.length >= maxLines) break;
      }}
      if (current && lines.length < maxLines) lines.push(current);
      if (lines.length > maxLines) lines.length = maxLines;
      if (lines.length === maxLines) {{
        const last = lines[lines.length - 1];
        if ([...last].length >= maxChars) {{
          lines[lines.length - 1] = [...last].slice(0, Math.max(maxChars - 1, 1)).join("") + "…";
        }}
      }}
      return lines;
    }}

    function appendWrappedText(group, lines, x, y, lineHeight, attrs = {{}}) {{
      const text = group.append("text").attr("x", x).attr("y", y);
      Object.entries(attrs).forEach(([key, value]) => text.attr(key, value));
      lines.forEach((line, index) => {{
        text.append("tspan")
          .attr("x", x)
          .attr("dy", index === 0 ? 0 : lineHeight)
          .text(line);
      }});
      return text;
    }}

    function renderDagCanvas(pageEl, run) {{
      const container = pageEl.querySelector("[data-dag-canvas]");
      const detail = pageEl.querySelector("[data-node-detail]");
      if (!container || !detail) return;

      const sourceParents = new Map();
      run.graph.source_edges.forEach((edge) => {{
        if (!sourceParents.has(edge.to)) sourceParents.set(edge.to, []);
        sourceParents.get(edge.to).push(edge.from);
      }});

      const graphData = [
        ...run.graph.sources.map((source) => ({{
          id: source.name,
          kind: "source",
          parentIds: [],
        }})),
        ...run.graph.nodes.map((node) => ({{
          id: node.id,
          kind: "transform",
          parentIds: [...node.depends_on, ...(sourceParents.get(node.id) || [])],
        }})),
      ];

      const stratify = d3.graphStratify()
        .id((node) => node.id)
        .parentIds((node) => node.parentIds);
      const dag = stratify(graphData);

      const layout = d3.sugiyama()
        .gap([70, 90])
        .nodeSize((node) => node.data.kind === "source" ? [110, 250] : [150, 300]);
      const {{ width, height }} = layout(dag);

      const outerWidth = Math.max(height + 300, (width + 200) * 1.6, 1280);
      const outerHeight = outerWidth / 1.6;
      const svg = d3.select(container)
        .append("svg")
        .attr("viewBox", `0 0 ${{outerWidth}} ${{outerHeight}}`)
        .style("min-width", `${{outerWidth}}px`);
      const viewport = svg.append("g");

      const defs = svg.append("defs");
      defs.append("marker")
        .attr("id", "dag-arrow")
        .attr("markerWidth", 10)
        .attr("markerHeight", 10)
        .attr("refX", 8)
        .attr("refY", 5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M 0 0 L 10 5 L 0 10 z")
        .attr("fill", "#c2cfdf");

      const initialTransform = d3.zoomIdentity.translate(150, 100);
      const root = viewport.append("g").attr("transform", "translate(150,100)");
      const linkLayer = root.append("g");
      const nodeLayer = root.append("g");
      const zoom = d3.zoom()
        .scaleExtent([0.35, 2.5])
        .on("zoom", (event) => {{
          viewport.attr("transform", event.transform);
        }});
      svg.call(zoom).call(zoom.transform, initialTransform).on("dblclick.zoom", null);

      const pointXY = (point) => {{
        if (Array.isArray(point)) {{
          return {{ x: point[0] ?? 0, y: point[1] ?? 0 }};
        }}
        if (point && typeof point === "object") {{
          return {{ x: point.x ?? point[0] ?? 0, y: point.y ?? point[1] ?? 0 }};
        }}
        return {{ x: 0, y: 0 }};
      }};
      const nodeRect = (node) => node.data.kind === "source"
        ? {{ width: 220, height: 64 }}
        : {{ width: 248, height: 112 }};
      const centerPoint = (node) => ({{
        x: node.y,
        y: node.x,
      }});
      const boundaryPoint = (center, toward, dims) => {{
        const dx = toward.x - center.x;
        const dy = toward.y - center.y;
        if (dx === 0 && dy === 0) return center;
        const sx = Math.abs(dx) / (dims.width / 2);
        const sy = Math.abs(dy) / (dims.height / 2);
        const scale = 1 / Math.max(sx, sy);
        return {{
          x: center.x + dx * scale,
          y: center.y + dy * scale,
        }};
      }};
      const line = d3.line()
        .x((point) => point.x)
        .y((point) => point.y)
        .curve(d3.curveMonotoneX);

      linkLayer
        .selectAll("path")
        .data(Array.from(dag.links()))
        .join("path")
        .attr("class", "dag-link")
        .attr("marker-end", "url(#dag-arrow)")
        .attr("d", (link) => {{
          const points = (link.points || []).map(pointXY).map((point) => ({{
            x: point.y,
            y: point.x,
          }}));
          if (!points.length) return null;
          const sourceCenter = centerPoint(link.source);
          const targetCenter = centerPoint(link.target);
          const sourceToward = points[1] || targetCenter;
          const targetToward = points[points.length - 2] || sourceCenter;
          points[0] = boundaryPoint(sourceCenter, sourceToward, nodeRect(link.source));
          points[points.length - 1] = boundaryPoint(targetCenter, targetToward, nodeRect(link.target));
          return points.length ? line(points) : null;
        }});

      const outDegree = Object.fromEntries(run.graph.nodes.map((node) => [node.id, 0]));
      run.graph.edges.forEach((edge) => {{
        outDegree[edge.from] = (outDegree[edge.from] || 0) + 1;
      }});

      const nodeGroups = nodeLayer
        .selectAll("g.dag-node")
        .data(Array.from(dag.nodes()))
        .join("g")
        .attr("class", (node) => `dag-node${{node.data.kind === "source" ? " dag-source-node" : ""}}`)
        .attr("data-node-id", (node) => node.data.id)
        .attr("data-node-kind", (node) => node.data.kind)
        .attr("transform", (node) => {{
          const nodeWidth = node.data.kind === "source" ? 220 : 248;
          const nodeHeight = node.data.kind === "source" ? 64 : 112;
          return `translate(${{node.y - nodeWidth / 2}},${{node.x - nodeHeight / 2}})`;
        }});

      nodeGroups.each(function(node) {{
        const group = d3.select(this);
        if (node.data.kind === "source") {{
          group.append("rect")
            .attr("class", "primary")
            .attr("width", 220)
            .attr("height", 64)
            .attr("rx", 18)
            .attr("fill", "rgba(245, 158, 11, 0.14)")
            .attr("stroke", "#d97706")
            .attr("stroke-width", 2);
          group.append("rect")
            .attr("x", 14)
            .attr("y", 12)
            .attr("width", 74)
            .attr("height", 22)
            .attr("rx", 11)
            .attr("fill", "rgba(245, 158, 11, 0.16)");
          group.append("text")
            .attr("x", 25)
            .attr("y", 27)
            .attr("font-size", 11)
            .attr("font-weight", 700)
            .attr("fill", "#b45309")
            .attr("letter-spacing", "0.08em")
            .text("SOURCE");
          appendWrappedText(group, wrapTextLines(node.data.id, 22, 1), 16, 49, 16, {{
            "font-size": 15,
            "font-weight": 700,
            fill: "#0f172a",
          }});
          return;
        }}

        const info = run.graph.nodes.find((item) => item.id === node.data.id);
        const exec = run.node_executions.find((item) => item.node_id === node.data.id);
        const isTable = info.materialization === "table";
        const fill = isTable ? "rgba(37, 99, 235, 0.14)" : "rgba(15, 118, 110, 0.14)";
        const stroke = isTable ? "#2563eb" : "#0f766e";
        const runtime = exec ? `${{exec.duration_ms}} ms` : "runtime unavailable";
        const nameLines = wrapTextLines(info.id, 23, 2);
        const metaLines = wrapTextLines(
          `${{runtime}} · in=${{info.depends_on.length}} · out=${{outDegree[info.id] || 0}}`,
          30,
          2
        );

        group.append("rect")
          .attr("class", "primary")
          .attr("width", 248)
          .attr("height", 112)
          .attr("rx", 22)
          .attr("fill", fill)
          .attr("stroke", stroke)
          .attr("stroke-width", isTable ? 2.5 : 2);
        group.append("rect")
          .attr("x", 14)
          .attr("y", 14)
          .attr("width", 66)
          .attr("height", 22)
          .attr("rx", 11)
          .attr("fill", fill);
        group.append("text")
          .attr("x", 26)
          .attr("y", 29)
          .attr("font-size", 11)
          .attr("font-weight", 700)
          .attr("fill", stroke)
          .attr("letter-spacing", "0.08em")
          .text(info.materialization.toUpperCase());
        appendWrappedText(group, nameLines, 16, 55, 16, {{
          "font-size": 15,
          "font-weight": 700,
          fill: "#0f172a",
        }});
        appendWrappedText(group, metaLines, 16, 84, 14, {{
          "font-size": 12,
          fill: "#64748b",
        }});
      }});

      const setSelected = (nodeId) => {{
        nodeGroups.classed("selected", (node) => node.data.id === nodeId);
        renderNodeDetail(run, nodeId, detail);
      }};

      nodeGroups.on("click", (_, node) => setSelected(node.data.id));
      const defaultNodeId = detail.dataset.defaultNodeId || (graphData[0] ? graphData[0].id : "");
      if (defaultNodeId) setSelected(defaultNodeId);
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
        const fill = node.materialization === "table" ? "rgba(37,99,235,0.78)" : "rgba(15,118,110,0.78)";
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
      const defaultNodeId = run.graph.nodes[0] ? run.graph.nodes[0].id : (run.graph.sources[0] ? run.graph.sources[0].name : "");

      return `
        <section class="page${{index === 0 ? " active" : ""}}" data-page="${{index}}">
          <div class="summary">
            <div class="card"><div class="label">Dag file</div><div class="value" style="font-size:20px">${{escapeHtml(run.dag_file)}}</div></div>
            <div class="card"><div class="label">Database</div><div class="value">${{escapeHtml(run.db)}}</div></div>
            <div class="card"><div class="label">Nodes</div><div class="value">${{run.graph.nodes.length}}</div></div>
            <div class="card"><div class="label">Sources</div><div class="value">${{run.graph.sources.length}}</div></div>
            <div class="card"><div class="label">Runtime</div><div class="value">${{formatMs(run.duration_ms)}}</div></div>
            <div class="card"><div class="label">Peak memory</div><div class="value">${{peakMemory ? formatBytes(peakMemory) : "n/a"}}</div></div>
            <div class="card"><div class="label">Peak CPU</div><div class="value">${{peakCpu ? `${{peakCpu.toFixed(1)}}%` : "n/a"}}</div></div>
          </div>

          <div class="panel">
            <h2>DAG</h2>
            <div class="subtle">Source tables use a third color, each node labels both in-degree and out-degree, long names wrap inside the node card, and the Sugiyama layout uses wider gaps to reduce crossings and make branch structure easier to follow.</div>
            <div class="legend">
              <span><span class="swatch" style="background: var(--table)"></span>Table nodes</span>
              <span><span class="swatch" style="background: var(--view)"></span>View nodes</span>
              <span><span class="swatch" style="background: var(--source)"></span>Source tables</span>
            </div>
            <div class="dag-layout" data-run-index="${{index}}">
              <div class="dag-canvas" data-dag-canvas></div>
              <aside class="dag-sidepanel" data-node-detail data-default-node-id="${{escapeHtml(defaultNodeId)}}">
                <div class="detail-empty">Select a node to inspect its materialization mode, runtime, dependencies, and SQL or schema.</div>
              </aside>
            </div>
          </div>

          <div class="section-stack">
            <div class="panel">
              <h2>Execution Gantt</h2>
              <div class="subtle">Execution windows for each node, ordered by observed start time.</div>
              <div class="svg-wrap">${{buildGanttSvg(run)}}</div>
            </div>

            <div class="panel">
              <h2>System samples</h2>
              <div class="subtle">Aligned CPU and memory time series make it easier to compare resource pressure with node execution phases.</div>
              <div class="legend">
                <span><span class="swatch" style="background: var(--cpu)"></span>CPU usage</span>
                <span><span class="swatch" style="background: var(--mem)"></span>Memory usage</span>
              </div>
              <div class="chart-stack">
                <div class="svg-wrap">${{buildTimelineSvg(run, sample => sample.cpu_percent, "var(--cpu)", "CPU usage", value => `${{value.toFixed(1)}}%`)}}</div>
                <div class="svg-wrap">${{buildTimelineSvg(run, sample => sample.memory_bytes, "var(--mem)", "Memory usage", value => formatBytes(value))}}</div>
              </div>
            </div>
          </div>

          <div class="panel">
            <h2>Nodes and SQL</h2>
            <div class="subtle">Full node inventory for scanning SQL and materialization choices outside the graph view.</div>
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

    function renderNodeDetail(run, nodeId, container) {{
      const source = run.graph.sources.find(item => item.name === nodeId);
      if (source) {{
        const downstream = run.graph.source_edges.filter(edge => edge.from === nodeId).map(edge => edge.to);
        container.innerHTML = `
          <div class="detail-name">${{escapeHtml(source.name)}}</div>
          <div class="detail-meta">
            <span class="pill source">SOURCE</span>
            <span class="pill">${{source.columns.length}} column${{source.columns.length === 1 ? "" : "s"}}</span>
            <span class="pill">${{downstream.length}} downstream</span>
          </div>
          <div class="detail-grid">
            <div class="detail-box">
              <h3>Connected nodes</h3>
              <pre>${{downstream.length ? downstream.join("\n") : "No downstream nodes inferred from SQL references."}}</pre>
            </div>
            <div class="detail-box">
              <h3>Schema</h3>
              <pre>${{source.columns.length ? source.columns.map(col => `${{col.name}}: ${{col.data_type}}`).join("\n") : "No schema columns recorded."}}</pre>
            </div>
          </div>
        `;
        return;
      }}

      const node = run.graph.nodes.find(item => item.id === nodeId);
      if (!node) {{
        container.innerHTML = `<div class="detail-empty">Node details unavailable.</div>`;
        return;
      }}
      const exec = run.node_executions.find(item => item.node_id === nodeId);
      const downstream = run.graph.edges.filter(edge => edge.from === nodeId).map(edge => edge.to);
      const upstream = node.depends_on;
      container.innerHTML = `
        <div class="detail-name">${{escapeHtml(node.id)}}</div>
        <div class="detail-meta">
          <span class="pill ${{node.materialization}}">${{node.materialization.toUpperCase()}}</span>
          <span class="pill">${{exec ? `${{exec.duration_ms}} ms` : "runtime unavailable"}}</span>
          <span class="pill">${{upstream.length}} upstream</span>
          <span class="pill">${{downstream.length}} downstream</span>
        </div>
        <div class="detail-grid">
          <div class="detail-box">
            <h3>Timing</h3>
            <pre>${{exec ? `start: ${{exec.start}}\nfinish: ${{exec.finish}}\nduration: ${{exec.duration_ms}} ms` : "No execution timing recorded."}}</pre>
          </div>
          <div class="detail-box">
            <h3>Dependencies</h3>
            <pre>${{upstream.length ? upstream.join("\n") : "No upstream dependencies."}}</pre>
          </div>
          <div class="detail-box">
            <h3>Dependents</h3>
            <pre>${{downstream.length ? downstream.join("\n") : "No downstream dependents."}}</pre>
          </div>
          <div class="detail-box">
            <h3>SQL</h3>
            <pre><code>${{escapeHtml(node.query_text)}}</code></pre>
          </div>
        </div>
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
    pageEls.forEach((pageEl, index) => renderDagCanvas(pageEl, report.runs[index]));
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
