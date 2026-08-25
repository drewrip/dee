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
    /// `"warmup"` for untimed leading repetitions, `"measure"` otherwise.
    /// Warmups are reported so they are visible, but must be excluded from
    /// any aggregate.
    #[serde(default = "default_phase")]
    pub phase: String,
    /// 0-based repetition index within this DAG's `--repeat` series.
    #[serde(default)]
    pub rep_index: usize,
    pub run_started_at: DateTime<Utc>,
    pub run_finished_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub estimated_cpu_seconds: f64,
    pub time_executing_nodes_ms: i64,
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
    pub plan: Option<String>,
    /// Rows the backend reported writing for this node, when it reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_produced: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemUsageSample {
    pub timestamp: DateTime<Utc>,
    pub elapsed_ms: i64,
    pub cpu_percent: Option<f64>,
    pub memory_bytes: Option<u64>,
    pub disk_bytes: Option<u64>,
    pub read_bytes: Option<u64>,
    pub written_bytes: Option<u64>,
}

fn default_phase() -> String {
    "measure".to_string()
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
            plan: stats.plan.clone(),
            rows_produced: stats.rows_produced,
        })
        .collect();
    node_executions.sort_by(|a, b| a.start.cmp(&b.start).then(a.node_id.cmp(&b.node_id)));

    let time_executing_nodes_ms = node_executions.iter().map(|e| e.duration_ms).sum();

    let mut estimated_cpu_seconds = 0.0;
    if !exec_stats.system_samples.is_empty() {
        for i in 0..exec_stats.system_samples.len() - 1 {
            let s1 = &exec_stats.system_samples[i];
            let s2 = &exec_stats.system_samples[i + 1];
            if let (Some(cpu1), Some(cpu2)) = (s1.cpu_percent, s2.cpu_percent) {
                let dt_s = (s2.elapsed_ms - s1.elapsed_ms) as f64 / 1000.0;
                let avg_cpu = (cpu1 + cpu2) / 2.0 / 100.0;
                estimated_cpu_seconds += avg_cpu * dt_s;
            }
        }
    }

    DagRunProfile {
        dag_file: dag_file.to_string(),
        phase: default_phase(),
        rep_index: 0,
        db: dag.db.clone(),
        run_started_at: exec_stats.start,
        run_finished_at: exec_stats.finish,
        duration_ms: exec_stats.duration.num_milliseconds(),
        estimated_cpu_seconds,
        time_executing_nodes_ms,
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
            "- {}: {} nodes, {} sources in {} ms, est. cpu {:.3}s, node exec {} ms{}{}",
            run.dag_file,
            run.graph.nodes.len(),
            run.graph.sources.len(),
            run.duration_ms,
            run.estimated_cpu_seconds,
            run.time_executing_nodes_ms,
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

    let extra_body_script = format!(
        r##"    const report = {};

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
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function safeId(id) {{
      return id.replace(/[^a-z0-9_-]/gi, '_');
    }}

    function formatCount(n) {{
      if (n == null) return "0";
      if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
      if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
      if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
      return n.toString();
    }}

    function renderPlanNode(node, totalTiming) {{
      const timing = node.operator_timing || 0;
      const impact = totalTiming > 0 ? (timing / totalTiming) * 100 : 0;
      const children = node.children || [];
      const childrenHtml = children.map(child => renderPlanNode(child, totalTiming)).join("");

      const extraInfoHtml = Object.entries(node.extra_info || {{}})
        .map(([k, v]) => `
          <div class="plan-extra-row">
            <span class="plan-extra-key">${{escapeHtml(k)}}</span>
            <span class="plan-extra-val">${{escapeHtml(JSON.stringify(v))}}</span>
          </div>
        `)
        .join("");

      return `
        <div class="plan-node">
          <div class="plan-header" onclick="
            if (event.target.closest('.plan-toggle')) {{
              this.parentElement.classList.toggle('folded');
            }} else {{
              this.nextElementSibling.classList.toggle('active');
            }}
            event.stopPropagation();
          ">
            <div class="plan-toggle">${{children.length ? "▼" : ""}}</div>
            <span class="plan-type">${{escapeHtml(node.operator_type || node.operator_name || "UNKNOWN")}}</span>
            <div class="plan-impact-wrap">
              <div class="plan-impact-bar" style="width: ${{impact}}%"></div>
            </div>
            <span class="plan-rows">${{formatCount(node.operator_cardinality)}} rows</span>
            <span class="plan-bytes">${{formatBytes(node.result_set_size)}}</span>
            <span class="plan-timing">${{(timing * 1000).toFixed(2)}}ms</span>
          </div>
          <div class="plan-extra">${{extraInfoHtml || "No extra info recorded."}}</div>
          <div class="plan-children">${{childrenHtml}}</div>
        </div>
      `;
    }}


    function renderPlan(planJson) {{
      if (!planJson) return "No plan available.";
      try {{
        const plan = JSON.parse(planJson);
        let totalTiming = 0;
        const walk = (n) => {{
          totalTiming += (n.operator_timing || 0);
          (n.children || []).forEach(walk);
        }};
        (plan.children || []).forEach(walk);
        
        return `
          <div class="plan-tree">
            ${{plan.children.map(child => renderPlanNode(child, totalTiming)).join("")}}
          </div>
        `;
      }} catch (e) {{
        return `<div style="color: var(--cpu)">Error parsing plan: ${{escapeHtml(e.message)}}</div>`;
      }}
    }}

    function scrollToPlan(id) {{
      const el = document.getElementById(id);
      if (el) {{
        el.open = true;
        el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        el.style.outline = '2px solid var(--accent)';
        el.style.outlineOffset = '4px';
        setTimeout(() => {{ el.style.outline = 'none'; }}, 2000);
      }}
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

    function renderDagCanvas(pageEl, run, index) {{
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
        const fill = info.materialization === "table" ? "rgba(37, 99, 235, 0.14)"
          : info.materialization === "temp_table" ? "rgba(124, 58, 237, 0.14)"
          : "rgba(15, 118, 110, 0.14)";
        const stroke = info.materialization === "table" ? "#2563eb"
          : info.materialization === "temp_table" ? "#7c3aed"
          : "#0f766e";
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
          .attr("stroke-width", info.materialization === "table" ? 2.5 : 2);
        const badgeWidth = {{ table: 66, view: 52, temp_table: 104 }}[info.materialization] || 66;
        group.append("rect")
          .attr("x", 14)
          .attr("y", 14)
          .attr("width", badgeWidth)
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
        renderNodeDetail(run, nodeId, detail, index);
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
        const fill = node.materialization === "table" ? "rgba(37,99,235,0.78)"
          : node.materialization === "temp_table" ? "rgba(124,58,237,0.78)"
          : "rgba(15,118,110,0.78)";
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
            <div class="card"><div class="label">Node Exec Time</div><div class="value">${{formatMs(run.time_executing_nodes_ms)}}</div></div>
            <div class="card"><div class="label">Est. CPU Seconds</div><div class="value">${{run.estimated_cpu_seconds.toFixed(3)}}s</div></div>
            <div class="card"><div class="label">Peak memory</div><div class="value">${{peakMemory ? formatBytes(peakMemory) : "n/a"}}</div></div>
            <div class="card"><div class="label">Peak CPU</div><div class="value">${{peakCpu ? `${{peakCpu.toFixed(1)}}%` : "n/a"}}</div></div>
          </div>

          <details class="panel" open>
            <summary><h2>DAG</h2></summary>
            <div class="legend">
              <span><span class="swatch" style="background: var(--table)"></span>Table nodes</span>
              <span><span class="swatch" style="background: var(--view)"></span>View nodes</span>
              <span><span class="swatch" style="background: var(--temp-table)"></span>Temp table nodes</span>
              <span><span class="swatch" style="background: var(--source)"></span>Source tables</span>
            </div>
            <div class="dag-layout" data-run-index="${{index}}">
              <div class="dag-canvas" data-dag-canvas></div>
              <aside class="dag-sidepanel" data-node-detail data-default-node-id="${{escapeHtml(defaultNodeId)}}">
                <div class="detail-empty">Select a node to inspect its materialization mode, runtime, dependencies, and SQL or schema.</div>
              </aside>
            </div>
          </details>

          <div class="section-stack">
            <details class="panel" open>
              <summary><h2>Execution Gantt</h2></summary>
              <div class="subtle">Execution windows for each node, ordered by observed start time.</div>
              <div class="svg-wrap">${{buildGanttSvg(run)}}</div>
            </details>

            <details class="panel" open>
              <summary><h2>System samples</h2></summary>
              <div class="subtle">Aligned CPU and memory time series make it easier to compare resource pressure with node execution phases.</div>
              <div class="legend">
                <span><span class="swatch" style="background: var(--cpu)"></span>CPU usage</span>
                <span><span class="swatch" style="background: var(--mem)"></span>Memory usage</span>
              </div>
              <div class="chart-stack">
                <div class="svg-wrap">${{buildTimelineSvg(run, sample => sample.cpu_percent, "var(--cpu)", "CPU usage", value => `${{value.toFixed(1)}}%`)}}</div>
                <div class="svg-wrap">${{buildTimelineSvg(run, sample => sample.memory_bytes, "var(--mem)", "Memory usage", value => formatBytes(value))}}</div>
              </div>
            </details>

            <details class="panel" open>
              <summary><h2>Table Node Plans</h2></summary>
              <div class="subtle">Collapsible query plans for all nodes materialized as tables.</div>
              <div class="query-list">
                ${{run.graph.nodes.filter(n => n.materialization === "table").map(node => {{
                  const exec = nodeIndex[node.id];
                  const hasPlan = exec && exec.plan;
                  return `
                    <details class="query-item" id="run-${{index}}-plan-${{safeId(node.id)}}">
                      <summary style="cursor: pointer; font-weight: 700;">
                        ${{escapeHtml(node.id)}}
                        <span class="node-tag table">table</span>
                        <span style="color: var(--muted); margin-left: 8px;">${{exec ? `${{exec.duration_ms}} ms` : "not executed"}}</span>
                        ${{!hasPlan ? '<span style="color: var(--cpu); margin-left: 8px; font-size: 11px;">(no plan available)</span>' : ''}}
                      </summary>
                      <div style="margin-top: 12px;">
                        ${{hasPlan ? renderPlan(exec.plan) : "No plan was captured for this node."}}
                      </div>
                    </details>
                  `;
                }}).join("")}}
              </div>
            </details>
          </div>

          <details class="panel" open>
            <summary><h2>Nodes and SQL</h2></summary>
            <div class="subtle">Full node inventory for scanning SQL and materialization choices outside the graph view.</div>
            <div class="query-list">
              ${{run.graph.nodes.map(node => `
                <details class="query-item">
                  <summary style="cursor: pointer;">
                    <strong>${{escapeHtml(node.id)}}</strong>
                    <span class="node-tag ${{node.materialization}}">${{node.materialization}}</span>
                    <span style="color: var(--muted); margin-left: 8px;">${{nodeIndex[node.id] ? `${{nodeIndex[node.id].duration_ms}} ms` : "not executed"}}</span>
                  </summary>
                  <pre style="margin-top: 12px;"><code>${{escapeHtml(node.query_text)}}</code></pre>
                </details>
              `).join("")}}
            </div>
          </details>
        </section>
      `;
    }}

    function renderNodeDetail(run, nodeId, container, runIndex) {{
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
      const hasPlan = (node.materialization === "table" || node.materialization === "temp_table") && exec && exec.plan;
      
      container.innerHTML = `
        <div class="detail-name">${{escapeHtml(node.id)}}</div>
        <div class="detail-meta">
          <span class="pill ${{node.materialization}}">${{node.materialization.toUpperCase()}}</span>
          <span class="pill">${{exec ? `${{exec.duration_ms}} ms` : "runtime unavailable"}}</span>
          <span class="pill">${{upstream.length}} upstream</span>
          <span class="pill">${{downstream.length}} downstream</span>
        </div>
        
        <div class="view-plan-btn-wrap">
          <a class="view-plan-btn ${{hasPlan ? "" : "disabled"}}" 
             ${{hasPlan ? `onclick="scrollToPlan('run-${{runIndex}}-plan-${{safeId(node.id)}}')"` : ""}}>
            View Query Plan
          </a>
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

    const runColors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"];

    function buildCompareTimelineSvg(runs, accessor, label, formatter) {{
      const width = 1180;
      const height = 300;
      const margin = {{ top: 40, right: 180, bottom: 40, left: 64 }};
      
      const allSamples = runs.flatMap((run, i) => 
        run.system_samples
          .filter(sample => accessor(sample) != null)
          .map(sample => ({{ ...sample, runIndex: i }}))
      );

      const maxX = Math.max(...runs.map(run => run.duration_ms), ...allSamples.map(s => s.elapsed_ms), 1);
      let maxY = allSamples.length ? Math.max(...allSamples.map(s => accessor(s))) : 1;
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

      const paths = runs.map((run, i) => {{
        const samples = run.system_samples.filter(sample => accessor(sample) != null);
        if (!samples.length) return "";
        const pathData = samples.map((sample, idx) => `${{idx === 0 ? "M" : "L"}} ${{x(sample.elapsed_ms)}} ${{y(accessor(sample))}}`).join(" ");
        const color = runColors[i % runColors.length];
        return `<path d="${{pathData}}" fill="none" stroke="${{color}}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />`;
      }}).join("");

      const legend = runs.map((run, i) => {{
        const color = runColors[i % runColors.length];
        const yy = margin.top + i * 22;
        return `
          <g transform="translate(${{width - margin.right + 15}}, ${{yy}})">
            <rect width="12" height="12" rx="3" fill="${{color}}" />
            <text x="20" y="10" font-size="11" font-weight="600" fill="var(--ink)">DAG ${{i + 1}}: ${{escapeHtml(run.dag_file)}}</text>
          </g>
        `;
      }}).join("");

      return `
        <svg viewBox="0 0 ${{width}} ${{height}}" aria-label="${{label}} comparison chart">
          <text x="${{margin.left}}" y="20" font-size="16" font-weight="700" fill="var(--ink)">${{label}} Comparison</text>
          ${{grid}}
          ${{xTicks}}
          <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="var(--ink)" />
          <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="var(--ink)" />
          ${{paths}}
          ${{legend}}
        </svg>
      `;
    }}

    function renderComparePage(runs) {{
      const metrics = [
        {{ label: "Runtime", key: "duration_ms", formatter: formatMs, type: "time" }},
        {{ label: "Node Exec Time", key: "time_executing_nodes_ms", formatter: formatMs, type: "time" }},
        {{ label: "Est. CPU Seconds", key: "estimated_cpu_seconds", formatter: (v) => `${{v.toFixed(3)}}s`, type: "cpu" }},
        {{ label: "Peak Memory", key: "peak_memory", formatter: formatBytes, type: "memory" }},
        {{ label: "Peak CPU", key: "peak_cpu", formatter: (v) => `${{v.toFixed(1)}}%`, type: "percent" }},
      ];

      const headerCells = runs.map((run, i) => `
        <th data-run-index="${{i}}">
          <div style="color: ${{runColors[i % runColors.length]}}; margin-bottom: 8px;">DAG ${{i + 1}}</div>
          <div style="font-size: 14px; color: var(--ink); margin-bottom: 12px; font-weight: 700;">${{escapeHtml(run.dag_file)}}</div>
          <button class="baseline-selector" data-run-index="${{i}}" onclick="updateBaseline(${{i}})">
            Set as Baseline
          </button>
        </th>
      `).join("");

      const rows = metrics.map(metric => `
        <tr class="compare-row" data-metric="${{metric.key}}">
          <td class="compare-label">${{metric.label}}</td>
          ${{runs.map((run, i) => {{
            let value;
            if (metric.key === "peak_memory") {{
              value = run.system_samples.reduce((acc, sample) => sample.memory_bytes == null ? acc : Math.max(acc, sample.memory_bytes), 0);
            }} else if (metric.key === "peak_cpu") {{
              value = run.system_samples.reduce((acc, sample) => sample.cpu_percent == null ? acc : Math.max(acc, sample.cpu_percent), 0);
            }} else {{
              value = run[metric.key];
            }}
            return `
              <td data-run-index="${{i}}" data-raw-value="${{value}}">
                <div class="compare-value">${{metric.formatter(value)}}</div>
                <div class="compare-delta" data-delta-for="${{i}}"></div>
              </td>
            `;
          }}).join("")}}
        </tr>
      `).join("");

      return `
        <section class="page" data-page="compare">
          <details class="panel" open>
            <summary><h2>Core Metrics Comparison</h2></summary>
            <div class="subtle">Compare performance metrics side-by-side. Select a baseline DAG to see relative improvements or regressions.</div>
            <div class="svg-wrap" style="padding: 0; overflow-x: auto;">
              <table class="compare-table">
                <thead>
                  <tr>
                    <th style="width: 200px;">Metric</th>
                    ${{headerCells}}
                  </tr>
                </thead>
                <tbody>
                  ${{rows}}
                </tbody>
              </table>
            </div>
          </details>

          <div class="section-stack">
            <details class="panel" open>
              <summary><h2>Resource Comparison</h2></summary>
              <div class="subtle">Overlaid CPU and memory metrics for all DAG runs. Aligned by start time (T=0).</div>
              <div class="chart-stack">
                <div class="svg-wrap">${{buildCompareTimelineSvg(runs, sample => sample.cpu_percent, "CPU usage", value => `${{value.toFixed(1)}}%`)}}</div>
                <div class="svg-wrap">${{buildCompareTimelineSvg(runs, sample => sample.memory_bytes, "Memory usage", value => formatBytes(value))}}</div>
              </div>
            </details>
          </div>
        </section>
      `;
    }}

    const tabs = document.getElementById("tabs");
    const pages = document.getElementById("pages");

    let tabHtml = report.runs.map((run, index) => `
      <button class="tab${{index === 0 ? " active" : ""}}" data-index="${{index}}">
        DAG ${{index + 1}}
      </button>
    `).join("");

    let pageHtml = report.runs.map(renderRun).join("");

    if (report.runs.length > 1) {{
      tabHtml += `
        <button class="tab" data-index="compare">
          Compare
        </button>
      `;
      pageHtml += renderComparePage(report.runs);
    }}

    tabs.innerHTML = tabHtml;
    pages.innerHTML = pageHtml;

    function updateBaseline(baselineIndex) {{
      const baselineSelectors = document.querySelectorAll('.baseline-selector');
      baselineSelectors.forEach(s => {{
        const idx = parseInt(s.dataset.runIndex);
        s.classList.toggle('active', idx === baselineIndex);
        s.innerText = idx === baselineIndex ? 'Baseline' : 'Set as Baseline';
        
        // Add/remove baseline tag in header
        const th = s.closest('th');
        let tag = th.querySelector('.baseline-tag');
        if (idx === baselineIndex) {{
          if (!tag) {{
            tag = document.createElement('span');
            tag.className = 'baseline-tag';
            tag.innerText = 'BASELINE';
            th.querySelector('div:nth-child(2)').appendChild(tag);
          }}
        }} else if (tag) {{
          tag.remove();
        }}

        // Highlight column
        const table = th.closest('table');
        const cells = table.querySelectorAll(`[data-run-index="${{idx}}"]`);
        cells.forEach(c => c.classList.toggle('compare-col-highlight', idx === baselineIndex));
      }});

      // Update deltas
      const rows = document.querySelectorAll('.compare-row');
      rows.forEach(row => {{
        const baselineCell = row.querySelector(`td[data-run-index="${{baselineIndex}}"]`);
        const baselineValue = parseFloat(baselineCell.dataset.rawValue);

        const deltaCells = row.querySelectorAll('.compare-delta');
        deltaCells.forEach(deltaCell => {{
          const runIdx = parseInt(deltaCell.dataset.deltaFor);
          if (runIdx === baselineIndex) {{
            deltaCell.innerText = '';
            return;
          }}

          const cell = row.querySelector(`td[data-run-index="${{runIdx}}"]`);
          const value = parseFloat(cell.dataset.rawValue);
          
          if (baselineValue === 0) {{
            deltaCell.innerText = '';
            return;
          }}

          const pct = ((value - baselineValue) / baselineValue) * 100;
          const sign = pct > 0 ? '+' : '';
          deltaCell.innerText = `(${{sign}}${{pct.toFixed(1)}}%)`;
          deltaCell.className = 'compare-delta ' + (pct > 0.1 ? 'delta-pos' : (pct < -0.1 ? 'delta-neg' : 'delta-neutral'));
        }});
      }});
    }}

    if (report.runs.length > 1) {{
      updateBaseline(0);
    }}

    const tabEls = [...document.querySelectorAll(".tab")];
    const pageEls = [...document.querySelectorAll(".page")];

    report.runs.forEach((_, index) => {{
      renderDagCanvas(pageEls[index], report.runs[index], index);
    }});
"##,
        report_json
    );

    let subtitle_html = format!(
        "<p>{} DAG run(s), generated {}.</p>",
        report.runs.len(),
        report.generated_at
    );

    let extra_head = "  <script src=\"https://unpkg.com/d3@7/dist/d3.min.js\"></script>\n  <script src=\"https://unpkg.com/d3-dag@1.1.0\"></script>\n";

    Ok(crate::report::render_report_shell(
        "dee profile",
        "dee profiler",
        "profiling report",
        &subtitle_html,
        "",
        "",
        extra_head,
        &extra_body_script,
    ))
}
