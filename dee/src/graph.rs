use crate::dag::{MaterializeMode, TransformNode};
use std::collections::{BTreeSet, HashMap, HashSet};
use thiserror::Error;

fn svg_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn node_colors(mode: MaterializeMode) -> (&'static str, &'static str) {
    match mode {
        MaterializeMode::View => ("#eff6ff", "#3b82f6"),
        MaterializeMode::Table => ("#f0fdf4", "#22c55e"),
        MaterializeMode::TempTable => ("#fffbeb", "#f59e0b"),
    }
}

/// Whether `query` reads the relation `name`.
///
/// A plain `contains` would let a source named `orders` match `orders_summary`,
/// which inflates a view's leaf set and makes every containment test above it
/// fail. The name has to sit on an identifier boundary.
fn query_references(query: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    // `.` is not part of the boundary: a query that writes `main.orders`
    // still reads the source declared as `orders`.
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let bytes = query.as_bytes();
    let mut from = 0;
    while let Some(offset) = query[from..].find(name) {
        let start = from + offset;
        let end = start + name.len();
        let before_ok = start == 0 || !is_ident(bytes[start - 1] as char);
        let after_ok = end == query.len() || !is_ident(bytes[end] as char);
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

fn build_children_map(g: &GraphType) -> HashMap<String, Vec<String>> {
    let mut children: HashMap<String, Vec<String>> = HashMap::new();
    for (id, node) in g.iter() {
        for parent in &node.depends_on {
            children.entry(parent.clone()).or_default().push(id.clone());
        }
    }
    children
}

fn barycenter_predecessors(node_id: &str, g: &GraphType, layer_pos: &HashMap<String, f64>) -> f64 {
    let node = match g.get(node_id) {
        Some(n) => n,
        None => return f64::MAX,
    };
    let vals: Vec<f64> = node
        .depends_on
        .iter()
        .filter_map(|p| layer_pos.get(p).copied())
        .collect();
    if vals.is_empty() {
        f64::MAX
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

fn barycenter_successors(
    node_id: &str,
    children_map: &HashMap<String, Vec<String>>,
    layer_pos: &HashMap<String, f64>,
) -> f64 {
    let children = match children_map.get(node_id) {
        Some(c) => c,
        None => return f64::MAX,
    };
    let vals: Vec<f64> = children
        .iter()
        .filter_map(|c| layer_pos.get(c).copied())
        .collect();
    if vals.is_empty() {
        f64::MAX
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("node in graph points to nonexistent node - {0} -> {1}")]
    BadGraph(String, String),
    #[error("node in graph doesn't exist - {0}")]
    MissingNode(String),
}

pub type GraphType = HashMap<String, TransformNode>;

#[derive(Debug, Clone)]
pub struct Graph {
    g: GraphType,
}

impl Graph {
    pub fn new(g: GraphType) -> Self {
        Self { g }
    }

    pub fn get(&self, node: String) -> Option<&TransformNode> {
        self.g.get(&node)
    }

    pub fn get_mut(&mut self, node: String) -> Option<&mut TransformNode> {
        self.g.get_mut(&node)
    }

    pub fn check(&self) -> Result<(), GraphError> {
        for (id, node) in self.g.iter() {
            for parent in &node.depends_on {
                if !self.g.contains_key(parent) {
                    return Err(GraphError::BadGraph(id.clone(), parent.clone()));
                }
            }
        }
        Ok(())
    }

    pub fn check_nodes(&self, nodes: Vec<String>) -> Result<(), GraphError> {
        for node in nodes {
            let child = self
                .g
                .get(&node)
                .ok_or(GraphError::MissingNode(node.clone()))?;
            for parent in &child.depends_on {
                if !self.g.contains_key(parent) {
                    return Err(GraphError::BadGraph(node.clone(), parent.clone()));
                }
            }
        }
        Ok(())
    }

    pub fn remove(&mut self, node_to_remove: String) -> Option<usize> {
        self.g.remove(&node_to_remove)?;
        let mut rem_count = 0;
        for node in &mut self.g {
            rem_count += node.1.depends_on.remove(&node_to_remove) as usize;
        }
        Some(rem_count)
    }

    pub fn add_node(&mut self, new_node: TransformNode) -> Result<(), GraphError> {
        let node_name = new_node.id.clone();
        self.g.insert(node_name.clone(), new_node);
        self.check_nodes(vec![node_name])?;
        Ok(())
    }

    pub fn add_node_unchecked(&mut self, new_node: TransformNode) {
        self.g.insert(new_node.id.clone(), new_node);
    }

    pub fn add_edge(&mut self, src_node: &String, dst_node: &String) -> Result<(), GraphError> {
        if !self.g.contains_key(dst_node) {
            return Err(GraphError::MissingNode(dst_node.clone()));
        }
        let src = self
            .g
            .get_mut(src_node)
            .ok_or(GraphError::MissingNode(src_node.clone()))?;
        src.depends_on.insert(dst_node.clone());
        Ok(())
    }

    pub fn add_edge_unchecked(&mut self, src_node: &String, dst_node: &String) {
        let src = self.g.get_mut(src_node);
        if let Some(good_src) = src {
            good_src.depends_on.insert(dst_node.clone());
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &TransformNode> {
        self.g.values()
    }

    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut TransformNode> {
        self.g.values_mut()
    }

    pub fn num_nodes(&self) -> usize {
        self.g.len()
    }

    /// The most nodes that can ever be in flight at once.
    ///
    /// A scheduler can only run nodes that do not depend on one another, so
    /// the ceiling on concurrency is the largest set of mutually unreachable
    /// nodes -- the widest antichain. By Dilworth's theorem that equals the
    /// fewest chains needed to cover the DAG, and a minimum chain cover of a
    /// transitively closed DAG is `n` minus a maximum bipartite matching over
    /// its reachability relation.
    ///
    /// Worth the trouble because the obvious stand-in, the node count, is far
    /// too loose: a 22-node pipeline that fans into six branches can never run
    /// more than six at once, so anything measuring a cap of 8 against it is
    /// measuring the uncapped DAG twice.
    pub fn max_concurrency(&self) -> usize {
        let ids: Vec<String> = self.g.keys().cloned().collect();
        let n = ids.len();
        if n <= 1 {
            return n;
        }
        let index: HashMap<&str, usize> =
            ids.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();

        // Reachability, by depth-first walk from each node over `depends_on`
        // reversed: `reach[i][j]` is "j runs strictly after i".
        let mut succ = vec![Vec::new(); n];
        for (id, node) in self.g.iter() {
            let Some(&child) = index.get(id.as_str()) else { continue };
            for dep in node.depends_on.iter() {
                if let Some(&parent) = index.get(dep.as_str()) {
                    succ[parent].push(child);
                }
            }
        }
        let mut reach = vec![vec![false; n]; n];
        for start in 0..n {
            let mut stack = succ[start].clone();
            while let Some(cur) = stack.pop() {
                if reach[start][cur] {
                    continue;
                }
                reach[start][cur] = true;
                stack.extend_from_slice(&succ[cur]);
            }
        }

        // Maximum matching over the closure, by augmenting paths. `n` here is
        // a DAG's node count -- tens, not thousands -- so the simple algorithm
        // is not worth replacing with Hopcroft-Karp.
        let mut matched_to: Vec<Option<usize>> = vec![None; n];
        let mut matching = 0;
        for left in 0..n {
            let mut seen = vec![false; n];
            if augment(left, &reach, &mut seen, &mut matched_to) {
                matching += 1;
            }
        }
        // Dilworth: widest antichain == fewest chains == n - matching.
        n - matching
    }

    pub fn num_edges(&self) -> usize {
        self.g.iter().map(|n| n.1.depends_on.len()).sum()
    }

    pub fn in_degree(&self, node: &String) -> Option<usize> {
        self.g.get(node).map(|n| n.depends_on.len())
    }

    pub fn out_degree(&self, node: &String) -> usize {
        self.nodes()
            .map(|n| n.depends_on.contains(node) as usize)
            .sum()
    }

    ///
    /// Nodes that have no dependencies
    ///
    pub fn sources(&self) -> impl Iterator<Item = String> {
        self.g
            .iter()
            .filter(|n| n.1.depends_on.len() == 0)
            .map(|n| n.0.clone())
    }

    ///
    /// Nodes that have no other nodes that depend on them
    ///
    pub fn sinks(&self) -> impl Iterator<Item = String> {
        let mut someone_depends_on = HashSet::new();
        for (_, node) in self.g.iter() {
            someone_depends_on.extend(node.depends_on.clone());
        }

        self.g
            .keys()
            .cloned()
            .filter(move |k| !someone_depends_on.contains(k))
    }

    fn assign_layers(&self) -> HashMap<String, usize> {
        let mut layers: HashMap<String, usize> = HashMap::new();
        for node_id in &self.topological_sort() {
            if let Some(node) = self.g.get(node_id) {
                let layer = node
                    .depends_on
                    .iter()
                    .filter_map(|p| layers.get(p).copied())
                    .max()
                    .map(|l| l + 1)
                    .unwrap_or(0);
                layers.insert(node_id.clone(), layer);
            }
        }
        layers
    }

    pub fn draw_svg(&self, sources: &[String]) -> String {
        const NODE_W: f64 = 180.0;
        const NODE_H: f64 = 52.0;
        const H_GAP: f64 = 120.0;
        const V_GAP: f64 = 48.0;
        const MARGIN: f64 = 40.0;
        const MAX_CHARS: usize = 22;

        if self.g.is_empty() && sources.is_empty() {
            return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"80\">\
                <text x=\"100\" y=\"40\" text-anchor=\"middle\" font-family=\"sans-serif\" fill=\"#64748b\">empty graph</text>\
                </svg>\n"
                .to_string();
        }

        // Assign layers to transform nodes, shifting right by 1 when sources are present
        // so sources can occupy layer 0.
        let shift = if sources.is_empty() { 0usize } else { 1 };
        let mut layers: HashMap<String, usize> = self
            .assign_layers()
            .into_iter()
            .map(|(k, v)| (k, v + shift))
            .collect();

        let mut sorted_sources = sources.to_vec();
        sorted_sources.sort();
        for name in &sorted_sources {
            layers.insert(name.clone(), 0);
        }

        let max_layer = layers.values().max().copied().unwrap_or(0);
        let mut layer_nodes: Vec<Vec<String>> = vec![Vec::new(); max_layer + 1];

        for name in &sorted_sources {
            layer_nodes[0].push(name.clone());
        }
        let mut all_transforms: Vec<String> = self.g.keys().cloned().collect();
        all_transforms.sort();
        for node_id in &all_transforms {
            if let Some(&l) = layers.get(node_id) {
                layer_nodes[l].push(node_id.clone());
            }
        }

        // Infer source → transform edges from query_text.
        // barycenter functions only use depends_on, so source↔staging layers
        // keep their initial alphabetical order; crossing reduction still
        // applies to all deeper transform layers normally.
        let source_edges: Vec<(String, String)> = sorted_sources
            .iter()
            .flat_map(|src| {
                self.g
                    .iter()
                    .filter(|(_, node)| node.query_text.contains(src.as_str()))
                    .map(|(id, _)| (src.clone(), id.clone()))
            })
            .collect();

        let children_map = build_children_map(&self.g);

        for _ in 0..4 {
            for i in 1..=max_layer {
                let prev_pos: HashMap<String, f64> = layer_nodes[i - 1]
                    .iter()
                    .enumerate()
                    .map(|(j, n)| (n.clone(), j as f64))
                    .collect();
                layer_nodes[i].sort_by(|a, b| {
                    barycenter_predecessors(a, &self.g, &prev_pos)
                        .partial_cmp(&barycenter_predecessors(b, &self.g, &prev_pos))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            for i in (0..max_layer).rev() {
                let next_pos: HashMap<String, f64> = layer_nodes[i + 1]
                    .iter()
                    .enumerate()
                    .map(|(j, n)| (n.clone(), j as f64))
                    .collect();
                layer_nodes[i].sort_by(|a, b| {
                    barycenter_successors(a, &children_map, &next_pos)
                        .partial_cmp(&barycenter_successors(b, &children_map, &next_pos))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let mut pos: HashMap<String, (f64, f64)> = HashMap::new();
        for (li, nodes) in layer_nodes.iter().enumerate() {
            let x = MARGIN + li as f64 * (NODE_W + H_GAP);
            for (ri, node_id) in nodes.iter().enumerate() {
                let y = MARGIN + ri as f64 * (NODE_H + V_GAP);
                pos.insert(node_id.clone(), (x, y));
            }
        }

        let max_rows = layer_nodes.iter().map(|v| v.len()).max().unwrap_or(1);
        let svg_w = MARGIN * 2.0 + (max_layer + 1) as f64 * NODE_W + max_layer as f64 * H_GAP;
        let svg_h =
            MARGIN * 2.0 + max_rows as f64 * NODE_H + (max_rows.saturating_sub(1)) as f64 * V_GAP;

        let mut out = String::new();

        out.push_str(&format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">\n\
  <defs>\n\
    <marker id=\"arr\" markerWidth=\"8\" markerHeight=\"6\" refX=\"8\" refY=\"3\" orient=\"auto\" markerUnits=\"userSpaceOnUse\">\n\
      <path d=\"M0,0 L8,3 L0,6 Z\" fill=\"#94a3b8\"/>\n\
    </marker>\n\
  </defs>\n\
",
            w = svg_w as i64,
            h = svg_h as i64
        ));

        // Transform → transform edges (from depends_on)
        for (node_id, node) in &self.g {
            if let Some(&(cx, cy)) = pos.get(node_id) {
                let to_x = cx;
                let to_y = cy + NODE_H / 2.0;
                for parent_id in &node.depends_on {
                    if let Some(&(px, py)) = pos.get(parent_id) {
                        let from_x = px + NODE_W;
                        let from_y = py + NODE_H / 2.0;
                        let offset = (to_x - from_x) * 0.4;
                        out.push_str(&format!(
                            "  <path d=\"M{fx:.1},{fy:.1} C{c1:.1},{fy:.1} {c2:.1},{ty:.1} {tx:.1},{ty:.1}\" stroke=\"#94a3b8\" stroke-width=\"1.5\" fill=\"none\" marker-end=\"url(#arr)\"/>\n",
                            fx = from_x, fy = from_y,
                            c1 = from_x + offset, c2 = to_x - offset,
                            tx = to_x, ty = to_y,
                        ));
                    }
                }
            }
        }

        // Source → transform edges (inferred from query_text)
        for (src, node_id) in &source_edges {
            if let (Some(&(sx, sy)), Some(&(tx, ty))) = (pos.get(src), pos.get(node_id)) {
                let from_x = sx + NODE_W;
                let from_y = sy + NODE_H / 2.0;
                let to_x = tx;
                let to_y = ty + NODE_H / 2.0;
                let offset = (to_x - from_x) * 0.4;
                out.push_str(&format!(
                    "  <path d=\"M{fx:.1},{fy:.1} C{c1:.1},{fy:.1} {c2:.1},{ty:.1} {tx:.1},{ty:.1}\" stroke=\"#94a3b8\" stroke-width=\"1.5\" fill=\"none\" marker-end=\"url(#arr)\"/>\n",
                    fx = from_x, fy = from_y,
                    c1 = from_x + offset, c2 = to_x - offset,
                    tx = to_x, ty = to_y,
                ));
            }
        }

        // Source nodes (orange)
        for src in &sorted_sources {
            if let Some(&(x, y)) = pos.get(src) {
                let label = if src.chars().count() > MAX_CHARS {
                    format!("{}…", src.chars().take(MAX_CHARS - 1).collect::<String>())
                } else {
                    src.clone()
                };
                out.push_str(&format!(
                    "  <g>\n\
    <title>{title}</title>\n\
    <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{nw:.0}\" height=\"{nh:.0}\" rx=\"8\" ry=\"8\" fill=\"#fff7ed\" stroke=\"#ea580c\" stroke-width=\"1.5\"/>\n\
    <text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-family=\"ui-monospace,monospace\" font-size=\"12\" fill=\"#1e293b\">{label}</text>\n\
    <text x=\"{cx:.1}\" y=\"{my:.1}\" text-anchor=\"middle\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"9\" fill=\"#ea580c\">source</text>\n\
  </g>\n",
                    title = svg_escape(src),
                    x = x, y = y, nw = NODE_W, nh = NODE_H,
                    cx = x + NODE_W / 2.0,
                    ly = y + NODE_H / 2.0 - 7.0,
                    label = svg_escape(&label),
                    my = y + NODE_H - 9.0,
                ));
            }
        }

        // Transform nodes (materialize-mode colors)
        for (node_id, node) in &self.g {
            if let Some(&(x, y)) = pos.get(node_id) {
                let (fill, stroke) = node_colors(node.materialize);
                let label = if node_id.chars().count() > MAX_CHARS {
                    format!(
                        "{}…",
                        node_id.chars().take(MAX_CHARS - 1).collect::<String>()
                    )
                } else {
                    node_id.clone()
                };
                out.push_str(&format!(
                    "  <g>\n\
    <title>{title}</title>\n\
    <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{nw:.0}\" height=\"{nh:.0}\" rx=\"8\" ry=\"8\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"/>\n\
    <text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-family=\"ui-monospace,monospace\" font-size=\"12\" fill=\"#1e293b\">{label}</text>\n\
    <text x=\"{cx:.1}\" y=\"{my:.1}\" text-anchor=\"middle\" font-family=\"ui-sans-serif,sans-serif\" font-size=\"9\" fill=\"{stroke}\">{mode}</text>\n\
  </g>\n",
                    title = svg_escape(node_id),
                    x = x, y = y, nw = NODE_W, nh = NODE_H,
                    fill = fill, stroke = stroke,
                    cx = x + NODE_W / 2.0,
                    ly = y + NODE_H / 2.0 - 7.0,
                    label = svg_escape(&label),
                    my = y + NODE_H - 9.0,
                    mode = node.materialize.as_str(),
                ));
            }
        }

        out.push_str("</svg>\n");
        out
    }

    pub fn draw(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        for id in self.g.keys() {
            lines.push(format!("\"{}\"", id.clone().replace("\"", "")));
        }

        for (id, node) in self.g.iter() {
            lines.push(format!("// node={}", id));
            for parent in node.depends_on.iter() {
                lines.push(format!(
                    "\"{}\" -> \"{}\"",
                    parent.replace("\"", ""),
                    id.replace("\"", "")
                ));
            }
        }
        let line_section = lines
            .iter()
            .map(|l| format!("\t{}", l))
            .collect::<Vec<String>>()
            .join("\n");
        format!("digraph G {{\n{}\n}}", line_section)
    }

    pub fn topological_sort(&self) -> Vec<String> {
        let mut result = Vec::new();
        let mut work_graph = self.clone();

        while work_graph.num_nodes() > 0 {
            let sources: Vec<String> = work_graph.sources().collect();
            if sources.is_empty() {
                // Cycle detected or something else wrong, but for now just break
                break;
            }
            for source in sources {
                result.push(source.clone());
                work_graph.remove(source);
            }
        }
        result
    }

    /// Returns the set of node IDs reachable (downstream) from `start` that
    /// satisfy `predicate`.  Traversal follows the direction of data flow:
    /// from a node to every node whose `depends_on` set contains it.
    pub fn reachable(
        &self,
        start: &str,
        predicate: impl Fn(&TransformNode) -> bool,
    ) -> HashSet<String> {
        // Build a children map: parent_id → [child_id, ...]
        let mut children: HashMap<String, Vec<String>> = HashMap::new();
        for (id, node) in &self.g {
            for parent in &node.depends_on {
                children.entry(parent.clone()).or_default().push(id.clone());
            }
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: Vec<String> = vec![start.to_string()];

        while let Some(current) = queue.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            if let Some(kids) = children.get(&current) {
                for kid in kids {
                    if !visited.contains(kid) {
                        queue.push(kid.clone());
                    }
                }
            }
        }

        // `visited` includes `start` itself; filter to only nodes matching the
        // predicate (excluding the start node so callers get pure descendants).
        visited
            .into_iter()
            .filter(|id| id != start)
            .filter(|id| self.g.get(id).map(|n| predicate(n)).unwrap_or(false))
            .collect()
    }

    /// Returns the set of `Table` node IDs reachable downstream from `start`.
    pub fn reachable_tables(&self, start: &str) -> HashSet<String> {
        self.reachable(start, |n| matches!(n.materialize, MaterializeMode::Table))
    }

    /// Returns the set of `TempTable` node IDs reachable downstream from `start`.
    pub fn reachable_temps(&self, start: &str) -> HashSet<String> {
        self.reachable(start, |n| {
            matches!(n.materialize, MaterializeMode::TempTable)
        })
    }

    /// Returns the set of `Table` or `TempTable` node IDs reachable downstream
    /// from `start`.
    pub fn reachable_materializes(&self, start: &str) -> HashSet<String> {
        self.reachable(start, |n| {
            matches!(
                n.materialize,
                MaterializeMode::Table | MaterializeMode::TempTable
            )
        })
    }

    /// Returns the *frontier* of nodes reachable downstream from `start` that
    /// satisfy `predicate`.  Unlike [`reachable`], the search **stops** down a
    /// given path as soon as it encounters a matching node — that node is
    /// included in the result but its descendants are not explored further.
    /// This gives the nearest matching nodes along every downstream path rather
    /// than all matching nodes at any depth.
    pub fn frontier(
        &self,
        start: &str,
        predicate: impl Fn(&TransformNode) -> bool,
    ) -> HashSet<String> {
        let mut children: HashMap<String, Vec<String>> = HashMap::new();
        for (id, node) in &self.g {
            for parent in &node.depends_on {
                children.entry(parent.clone()).or_default().push(id.clone());
            }
        }

        let mut result: HashSet<String> = HashSet::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: Vec<String> = vec![start.to_string()];

        while let Some(current) = queue.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            // Skip the start node itself for the predicate check.
            if current != start {
                if let Some(node) = self.g.get(&current) {
                    if predicate(node) {
                        result.insert(current.clone());
                        // Do not explore further down this path.
                        continue;
                    }
                }
            }
            if let Some(kids) = children.get(&current) {
                for kid in kids {
                    if !visited.contains(kid) {
                        queue.push(kid.clone());
                    }
                }
            }
        }

        result
    }

    /// Returns the nearest `Table` nodes downstream from `start`, stopping
    /// at the first `Table` found on each path.
    pub fn frontier_tables(&self, start: &str) -> HashSet<String> {
        self.frontier(start, |n| matches!(n.materialize, MaterializeMode::Table))
    }

    /// Returns the nearest `TempTable` nodes downstream from `start`, stopping
    /// at the first `TempTable` found on each path.
    pub fn frontier_temps(&self, start: &str) -> HashSet<String> {
        self.frontier(start, |n| {
            matches!(n.materialize, MaterializeMode::TempTable)
        })
    }

    /// Returns the nearest `Table` or `TempTable` nodes downstream from
    /// `start`, stopping at the first match found on each path.
    pub fn frontier_materializes(&self, start: &str) -> HashSet<String> {
        self.frontier(start, |n| {
            matches!(
                n.materialize,
                MaterializeMode::Table | MaterializeMode::TempTable
            )
        })
    }

    /// The mirror image of [`frontier`](Self::frontier): the nearest *upstream*
    /// nodes satisfying `predicate`, walking `depends_on` and stopping down a
    /// path as soon as it matches.
    pub fn upstream_frontier(
        &self,
        start: &str,
        predicate: impl Fn(&TransformNode) -> bool,
    ) -> HashSet<String> {
        let mut result: HashSet<String> = HashSet::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: Vec<String> = vec![start.to_string()];

        while let Some(current) = queue.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            if current != start
                && let Some(node) = self.g.get(&current)
                && predicate(node)
            {
                result.insert(current);
                // Do not explore further up this path.
                continue;
            }
            let Some(node) = self.g.get(&current) else {
                continue;
            };
            for parent in &node.depends_on {
                if !visited.contains(parent) {
                    queue.push(parent.clone());
                }
            }
        }

        result
    }

    /// The base relations the query the engine actually runs for `node` will
    /// read, once the views above it are inlined: declared sources plus
    /// persisted models beneath it.
    ///
    /// This is what lets a view be lined up against a subtree of a consumer's
    /// physical plan -- see [`crate::opt::leafset`]. `sources` is the DAG's
    /// declared source list: a source is not a graph node and is referenced
    /// only by name inside a query, so it is found the same way
    /// [`draw_svg`](Self::draw_svg) finds it -- by looking for the name in the
    /// text of every query that gets inlined into this one.
    ///
    /// Names are normalized by [`crate::plan::normalize_relation`] so they
    /// compare equal to the relation names read off a plan, which arrive
    /// catalog- and schema-qualified.
    pub fn leaf_sources(&self, node: &str, sources: &[String]) -> BTreeSet<String> {
        let persisted = |n: &TransformNode| {
            matches!(
                n.materialize,
                MaterializeMode::Table | MaterializeMode::TempTable
            )
        };

        let mut leaves: BTreeSet<String> = BTreeSet::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: Vec<String> = vec![node.to_string()];

        while let Some(current) = queue.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            let Some(n) = self.g.get(&current) else {
                continue;
            };
            // A persisted ancestor is a relation by the time this node runs, so
            // it is a leaf and nothing above it is inlined here.
            if current != node && persisted(n) {
                leaves.insert(crate::plan::normalize_relation(&current));
                continue;
            }
            // This node's body is part of the query, so the sources it names
            // are scanned by it.
            for src in sources {
                if query_references(&n.query_text, src) {
                    leaves.insert(crate::plan::normalize_relation(src));
                }
            }
            for parent in &n.depends_on {
                queue.push(parent.clone());
            }
        }

        leaves
    }

    /// Longest chain of nodes above each node -- how far it is from the DAG's
    /// output.
    ///
    /// [`crate::opt::leafset`] assigns plan regions consumer-most first, and
    /// this is that order: a view can only be placed inside the region of a
    /// view that depends on it.
    pub fn heights(&self) -> HashMap<String, usize> {
        let children = build_children_map(&self.g);
        let mut height: HashMap<String, usize> = HashMap::new();
        // Reverse topological order: every child is resolved before its parent.
        // A node the sort could not place (a cycle) is left at 0 rather than
        // missing, so callers can index the map for any node of the graph.
        for id in self.g.keys() {
            height.insert(id.clone(), 0);
        }
        for id in self.topological_sort().into_iter().rev() {
            let h = children
                .get(&id)
                .map(|kids| {
                    kids.iter()
                        .filter_map(|k| height.get(k).map(|h| h + 1))
                        .max()
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            height.insert(id, h);
        }
        height
    }

    pub fn paths_to_sinks(&self, node: &String) -> usize {
        let mut children: HashMap<String, Vec<String>> = HashMap::new();
        for n in self.g.values() {
            for parent in &n.depends_on {
                children
                    .entry(parent.clone())
                    .or_default()
                    .push(n.id.clone());
            }
        }

        let mut memo = HashMap::new();
        self.count_paths_helper(node, &children, &mut memo)
    }

    fn count_paths_helper(
        &self,
        node_id: &String,
        children_map: &HashMap<String, Vec<String>>,
        memo: &mut HashMap<String, usize>,
    ) -> usize {
        if let Some(&count) = memo.get(node_id) {
            return count;
        }

        let mut count = 0;
        if let Some(node) = self.g.get(node_id) {
            if matches!(
                node.materialize,
                MaterializeMode::Table | MaterializeMode::TempTable
            ) {
                count += 1;
            }
        }

        if let Some(children) = children_map.get(node_id) {
            for child in children {
                count += self.count_paths_helper(child, children_map, memo);
            }
        }

        memo.insert(node_id.clone(), count);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{MaterializeMode, TransformNode};

    fn node(id: &str, mode: MaterializeMode, deps: &[&str]) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: String::new(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            schema: None,
        }
    }

    fn make_graph(nodes: Vec<TransformNode>) -> Graph {
        let mut g = Graph::new(HashMap::new());
        for n in nodes {
            g.add_node_unchecked(n);
        }
        g
    }

    // raw_a, raw_b are declared sources: not graph nodes, named only in the
    // query text of the nodes that read them.
    //
    //   raw_a   raw_b
    //     |       |
    //    v1(V)  base(T)
    //      \    /
    //       v2(V)
    //         |
    //       out(T)
    fn leafset_graph() -> Graph {
        let with_sql = |id: &str, mode: MaterializeMode, deps: &[&str], sql: &str| {
            let mut n = node(id, mode, deps);
            n.query_text = sql.to_string();
            n
        };
        make_graph(vec![
            with_sql("v1", MaterializeMode::View, &[], "SELECT * FROM raw_a"),
            with_sql("base", MaterializeMode::Table, &[], "SELECT * FROM raw_b"),
            with_sql(
                "v2",
                MaterializeMode::View,
                &["v1", "base"],
                "SELECT * FROM v1 JOIN base USING (k)",
            ),
            with_sql("out", MaterializeMode::Table, &["v2"], "SELECT * FROM v2"),
        ])
    }

    const SOURCES: [&str; 2] = ["raw_a", "raw_b"];

    fn sources() -> Vec<String> {
        SOURCES.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_leaf_sources_descends_views_and_stops_at_persisted() {
        let g = leafset_graph();
        // v2 reads through v1 to the source, and stops at `base` rather than
        // descending to raw_b -- `base` is a relation by the time v2 runs.
        assert_eq!(
            g.leaf_sources("v2", &sources()),
            BTreeSet::from(["raw_a".to_string(), "base".to_string()])
        );
        assert_eq!(
            g.leaf_sources("v1", &sources()),
            BTreeSet::from(["raw_a".to_string()])
        );
        assert_eq!(
            g.leaf_sources("base", &sources()),
            BTreeSet::from(["raw_b".to_string()])
        );
    }

    #[test]
    fn test_leaf_sources_normalizes_the_way_a_plan_spells_a_relation() {
        // A plan prints `warehouse.main.orders`; both sides go through the same
        // normalizer or the coverage test compares nothing.
        let mut n = node("v", MaterializeMode::View, &[]);
        n.query_text = "SELECT * FROM warehouse.main.Orders".to_string();
        let g = make_graph(vec![n]);
        assert_eq!(
            g.leaf_sources("v", &["warehouse.main.Orders".to_string()]),
            BTreeSet::from(["orders".to_string()])
        );
    }

    #[test]
    fn test_leaf_sources_does_not_match_a_source_inside_a_longer_name() {
        // `orders` must not be picked up out of `orders_summary`: an inflated
        // leaf set makes every containment test above this view fail.
        let mut n = node("v", MaterializeMode::View, &[]);
        n.query_text = "SELECT * FROM orders_summary".to_string();
        let g = make_graph(vec![n]);
        assert!(
            g.leaf_sources("v", &["orders".to_string()]).is_empty(),
            "a substring of a longer identifier was read as a source"
        );
    }

    #[test]
    fn test_heights_order_a_chain_consumer_most_first() {
        let g = leafset_graph();
        let h = g.heights();
        assert_eq!(h["out"], 0);
        assert_eq!(h["v2"], 1);
        assert_eq!(h["v1"], 2);
        assert_eq!(h["base"], 2);
    }

    #[test]
    fn test_upstream_frontier_stops_at_the_first_match_on_each_path() {
        //   a(T) -> b(T) -> c(V)
        // From c the nearest persisted ancestor is b alone; a is behind it.
        let g = make_graph(vec![
            node("a", MaterializeMode::Table, &[]),
            node("b", MaterializeMode::Table, &["a"]),
            node("c", MaterializeMode::View, &["b"]),
        ]);
        let found = g.upstream_frontier("c", |n| {
            matches!(
                n.materialize,
                MaterializeMode::Table | MaterializeMode::TempTable
            )
        });
        assert_eq!(found, HashSet::from(["b".to_string()]));
    }

    // DAG layout:
    //
    //   source (View)
    //       │
    //   mid (TempTable)
    //       │
    //   final (Table)
    //
    // From `source`:
    //   - `reachable_materializes` returns both `mid` and `final` because it
    //     keeps walking past the TempTable to find all matches at any depth.
    //   - `frontier_materializes` returns only `mid` because the search stops
    //     at the first matching node on each path and does not continue to
    //     `final`.
    #[test]
    fn test_reachable_vs_frontier_stops_at_first_match_on_path() {
        let g = make_graph(vec![
            node("source", MaterializeMode::View, &[]),
            node("mid", MaterializeMode::TempTable, &["source"]),
            node("final", MaterializeMode::Table, &["mid"]),
        ]);

        let reachable = g.reachable_materializes("source");
        assert!(reachable.contains("mid"), "reachable must include mid");
        assert!(reachable.contains("final"), "reachable must include final");

        let frontier = g.frontier_materializes("source");
        assert!(frontier.contains("mid"), "frontier must include mid");
        assert!(
            !frontier.contains("final"),
            "frontier must NOT include final — path was stopped at mid"
        );
    }

    // DAG layout:
    //
    //   source (View)
    //     ├──► branch_a (TempTable) ──► sink_a (Table)
    //     └──► branch_b (Table)
    //
    // From `source`:
    //   - `reachable_materializes` returns branch_a, sink_a, and branch_b.
    //   - `frontier_materializes` returns branch_a and branch_b only.
    //     The path through branch_a stops there (TempTable matched first), so
    //     sink_a is never reached.  branch_b is its own independent path and
    //     matches immediately.
    #[test]
    fn test_frontier_stops_independently_on_each_path() {
        let g = make_graph(vec![
            node("source", MaterializeMode::View, &[]),
            node("branch_a", MaterializeMode::TempTable, &["source"]),
            node("sink_a", MaterializeMode::Table, &["branch_a"]),
            node("branch_b", MaterializeMode::Table, &["source"]),
        ]);

        let reachable = g.reachable_materializes("source");
        assert_eq!(
            reachable,
            ["branch_a", "sink_a", "branch_b"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>(),
            "reachable must find all materializing nodes at any depth"
        );

        let frontier = g.frontier_materializes("source");
        assert_eq!(
            frontier,
            ["branch_a", "branch_b"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>(),
            "frontier must stop at branch_a (not continue to sink_a) and include branch_b"
        );
    }
}

/// One augmenting step of the bipartite matching behind
/// [`Graph::max_concurrency`]: try to give `left` a partner, displacing an
/// existing pairing only when that pairing can be re-homed.
fn augment(
    left: usize,
    reach: &[Vec<bool>],
    seen: &mut [bool],
    matched_to: &mut [Option<usize>],
) -> bool {
    for right in 0..reach.len() {
        if !reach[left][right] || seen[right] {
            continue;
        }
        seen[right] = true;
        let free = match matched_to[right] {
            None => true,
            Some(other) => augment(other, reach, seen, matched_to),
        };
        if free {
            matched_to[right] = Some(left);
            return true;
        }
    }
    false
}
