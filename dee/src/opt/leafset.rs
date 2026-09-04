//! Attributing a region of a consumer's physical plan to the VIEW that was
//! inlined into it.
//!
//! Once a view is inlined there is no "view" in the plan to read a timing off.
//! What there is, is the consumer's physical plan with per-operator timings and
//! cardinalities. The technique here is **leaf-set matching under DAG-order
//! containment**.
//!
//! [`Graph::leaf_sources`](crate::graph::Graph::leaf_sources) gives the
//! relations a view's own query reads once the views above it are inlined. In a
//! consumer's plan every operator covers some set of base-relation scans, and
//! the view's region is the operator whose leaf set is still contained in the
//! view's -- above that point the plan mixes in relations the view does not
//! read, so it belongs to the consumer.
//!
//! Leaf sets alone cannot separate views that read the same relations. A 3-way
//! join with a window and the `GROUP BY` sitting directly on top of it read
//! exactly the same three relations; matched independently both land on the
//! same node, and the row-level view is reported as producing far fewer rows
//! than it really does -- precisely the error that makes a too-large
//! intermediate look safe to persist. The DAG supplies the missing constraint:
//! if `W` depends on `V` then `V`'s region is strictly inside `W`'s, so regions
//! are assigned consumer-most first and each search is confined to the
//! intersection of the interiors of the already-placed views above it.
//!
//! Times here are only ever used as **ratios within one plan**. That keeps the
//! module indifferent to whether the backend reports CPU time (DuckDB) or wall
//! time (Postgres) -- see [`crate::plan::TimeBasis`]. Never compare a
//! `subtree_time` across two plans.

use std::collections::{BTreeSet, HashMap, HashSet};

use polyglot_sql::{
    dialects::DialectType,
    expressions::{Expression, Select},
};

use crate::plan::PlanNode;

/// Fraction of a view's leaf set an operator must cover to be considered its
/// region. Below this the operator is reading too little of what the view reads
/// for the match to mean anything.
pub const LEAF_COVERAGE_MIN: f64 = 0.5;

/// One operator of a plan, flattened and annotated.
#[derive(Debug, Clone)]
pub struct Annotated {
    pub operator: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Every base relation scanned anywhere in this subtree.
    pub leaves: BTreeSet<String>,
    /// This operator's own time plus every descendant's.
    pub subtree_time: f64,
    pub cardinality: Option<u64>,
    /// Whether this operator collapses cardinality.
    pub is_aggregate: bool,
    /// Whether any operator strictly below it does.
    pub aggregate_below: bool,
}

/// A plan flattened into an index arena, so "the interior of a region" is
/// something that can be intersected.
#[derive(Debug, Clone, Default)]
pub struct PlanArena {
    pub nodes: Vec<Annotated>,
}

impl PlanArena {
    /// Flatten and annotate a parsed plan (a backend may report several roots).
    pub fn build(roots: &[PlanNode]) -> Self {
        let mut arena = PlanArena::default();
        for root in roots {
            arena.push(root, None);
        }
        arena
    }

    fn push(&mut self, node: &PlanNode, parent: Option<usize>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Annotated {
            operator: node.operator.clone(),
            parent,
            children: Vec::new(),
            leaves: BTreeSet::new(),
            subtree_time: 0.0,
            cardinality: node.cardinality,
            is_aggregate: node.is_aggregate_boundary(),
            aggregate_below: false,
        });

        let mut leaves: BTreeSet<String> = BTreeSet::new();
        if let Some(relation) = &node.relation {
            leaves.insert(relation.clone());
        }
        let mut subtree_time = node.exclusive_time_s.unwrap_or(0.0);
        let mut aggregate_below = false;
        let mut children = Vec::with_capacity(node.children.len());

        for child in &node.children {
            let child_idx = self.push(child, Some(idx));
            leaves.extend(self.nodes[child_idx].leaves.iter().cloned());
            subtree_time += self.nodes[child_idx].subtree_time;
            aggregate_below |=
                self.nodes[child_idx].is_aggregate || self.nodes[child_idx].aggregate_below;
            children.push(child_idx);
        }

        let slot = &mut self.nodes[idx];
        slot.children = children;
        slot.leaves = leaves;
        slot.subtree_time = subtree_time;
        slot.aggregate_below = aggregate_below;
        idx
    }

    /// Total time over every operator of the plan -- the denominator a share is
    /// taken against.
    pub fn total_time(&self) -> f64 {
        self.nodes
            .iter()
            .filter(|n| n.parent.is_none())
            .map(|n| n.subtree_time)
            .sum()
    }

    /// The strict descendants of `idx`.
    pub fn interior(&self, idx: usize) -> HashSet<usize> {
        let mut out = HashSet::new();
        let mut stack: Vec<usize> = self.nodes[idx].children.clone();
        while let Some(i) = stack.pop() {
            if out.insert(i) {
                stack.extend(self.nodes[i].children.iter().copied());
            }
        }
        out
    }

    /// The operator that is `view_leaves`'s region, if any.
    ///
    /// Among nodes whose leaves are contained in `view_leaves` and that cover
    /// at least [`LEAF_COVERAGE_MIN`] of them, the one with the largest
    /// `subtree_time`. `confine` restricts the search to a set of indices (the
    /// interiors of the already-placed views above this one); `prefers_aggregate`
    /// is the §5.3 disambiguator -- a view whose SQL has a top-level `GROUP BY`
    /// prefers to land on an aggregate, one without prefers a node with no
    /// aggregate beneath it. The preference partitions the candidates rather
    /// than merely breaking ties: two chained views can have identical leaf sets
    /// and very different regions, which is the whole point of it.
    pub fn match_region(
        &self,
        view_leaves: &BTreeSet<String>,
        confine: Option<&HashSet<usize>>,
        prefers_aggregate: bool,
    ) -> Option<usize> {
        if view_leaves.is_empty() {
            return None;
        }
        let want = view_leaves.len() as f64;

        let candidates: Vec<usize> = (0..self.nodes.len())
            .filter(|idx| confine.is_none_or(|c| c.contains(idx)))
            .filter(|idx| {
                let node = &self.nodes[*idx];
                if node.leaves.is_empty() {
                    return false;
                }
                node.leaves.is_subset(view_leaves)
                    && (node.leaves.len() as f64) / want >= LEAF_COVERAGE_MIN
            })
            .collect();

        let preferred: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|idx| {
                let node = &self.nodes[*idx];
                if prefers_aggregate {
                    node.is_aggregate
                } else {
                    !node.is_aggregate && !node.aggregate_below
                }
            })
            .collect();

        let pool = if preferred.is_empty() {
            &candidates
        } else {
            &preferred
        };
        pool.iter().copied().max_by(|a, b| {
            self.nodes[*a]
                .subtree_time
                .partial_cmp(&self.nodes[*b].subtree_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// One view to place inside a consumer's plan.
#[derive(Debug, Clone)]
pub struct ViewRegionRequest {
    pub id: String,
    pub leaves: BTreeSet<String>,
    /// Views that depend on this one, transitively. Only those already placed
    /// constrain it -- see [`attribute_chain`].
    pub consumers: HashSet<String>,
    pub prefers_aggregate: bool,
}

/// What a view's region cost, and how wide it was.
#[derive(Debug, Clone, Copy)]
pub struct Attribution {
    /// Time in the region, in whatever basis the plan reports.
    pub secs: f64,
    /// Rows the region's root operator emitted.
    pub cardinality: Option<u64>,
    /// Which operator was chosen, for the explain report.
    pub node: usize,
}

/// Place every view inside one consumer's plan, consumer-most first.
///
/// `views` must be ordered consumer-most first (ascending graph height). Each
/// search is confined to the **intersection** of the interiors of every
/// already-placed view above it. Intersecting rather than picking one ancestor
/// matters when a view has several placed descendants; any one alone leaves too
/// much room.
pub fn attribute_chain(
    arena: &PlanArena,
    views: &[ViewRegionRequest],
) -> HashMap<String, Attribution> {
    let mut placed: HashMap<String, usize> = HashMap::new();
    let mut out: HashMap<String, Attribution> = HashMap::new();

    for view in views {
        let mut confine: Option<HashSet<usize>> = None;
        for consumer in &view.consumers {
            let Some(idx) = placed.get(consumer) else {
                continue;
            };
            let interior = arena.interior(*idx);
            confine = Some(match confine {
                None => interior,
                Some(existing) => existing.intersection(&interior).copied().collect(),
            });
        }
        // Every placed consumer's interior excludes this view: the plan does not
        // show it separately, so there is nothing to attribute.
        if confine.as_ref().is_some_and(|c| c.is_empty()) {
            continue;
        }

        let Some(idx) = arena.match_region(&view.leaves, confine.as_ref(), view.prefers_aggregate)
        else {
            continue;
        };
        placed.insert(view.id.clone(), idx);
        out.insert(
            view.id.clone(),
            Attribution {
                secs: arena.nodes[idx].subtree_time,
                cardinality: arena.nodes[idx].cardinality,
                node: idx,
            },
        );
    }

    out
}

/// Whether `sql`'s outermost query collapses cardinality -- a top-level
/// `GROUP BY`, or an aggregate with no `GROUP BY` at all.
///
/// Used only as the §5.3 disambiguator, so a query that will not parse answers
/// `false` rather than failing: the matcher is then no worse than it would be
/// without the hint.
pub fn has_top_level_group_by(sql: &str, dialect: DialectType) -> bool {
    let Ok(parsed) = polyglot_sql::parse_one(sql, dialect) else {
        return false;
    };
    match parsed {
        Expression::Select(select) => select_collapses(&select),
        _ => false,
    }
}

fn select_collapses(select: &Select) -> bool {
    if select.group_by.is_some() {
        return true;
    }
    // `SELECT count(*) FROM t` has no GROUP BY and still emits one row.
    select
        .expressions
        .iter()
        .any(|e| matches!(e.variant_name(), "count" | "sum" | "avg" | "min" | "max"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(relation: &str, time: f64) -> PlanNode {
        PlanNode {
            operator: "SEQ_SCAN".into(),
            exclusive_time_s: Some(time),
            cardinality: Some(100),
            estimated_cardinality: Some(100.0),
            relation: Some(relation.into()),
            children: Vec::new(),
        }
    }

    fn op(operator: &str, time: f64, rows: u64, children: Vec<PlanNode>) -> PlanNode {
        PlanNode {
            operator: operator.into(),
            exclusive_time_s: Some(time),
            cardinality: Some(rows),
            estimated_cardinality: Some(rows as f64),
            relation: None,
            children,
        }
    }

    fn leaves(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    // A consumer that joins three relations, windows the result, then groups it:
    //
    //   HASH_GROUP_BY            <- v_agg's region  (1 row)
    //     WINDOW                 <- v_win's region  (5000 rows)
    //       HASH_JOIN c
    //         HASH_JOIN b
    //           SCAN a
    //           SCAN b
    //         SCAN c
    fn windowed_then_grouped() -> PlanArena {
        let join = op(
            "HASH_JOIN",
            1.0,
            5000,
            vec![
                op(
                    "HASH_JOIN",
                    2.0,
                    5000,
                    vec![scan("a", 0.5), scan("b", 0.5)],
                ),
                scan("c", 0.5),
            ],
        );
        let window = op("WINDOW", 4.0, 5000, vec![join]);
        let group = op("HASH_GROUP_BY", 1.5, 1, vec![window]);
        PlanArena::build(&[group])
    }

    #[test]
    fn leaves_and_times_roll_up_the_subtree() {
        let arena = windowed_then_grouped();
        let root = &arena.nodes[0];
        assert_eq!(root.leaves, leaves(&["a", "b", "c"]));
        // 1.5 + 4.0 + 1.0 + 2.0 + 0.5 * 3
        assert!((root.subtree_time - 10.0).abs() < 1e-9);
        assert!((arena.total_time() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_scan_only_covers_part_of_a_three_relation_view() {
        // One scan covers 1/3 of the view's leaves, below the coverage floor,
        // so it can never be mistaken for the whole view's region.
        let arena = windowed_then_grouped();
        let one_leaf = arena
            .nodes
            .iter()
            .position(|n| n.leaves == leaves(&["a"]))
            .unwrap();
        assert!(
            (arena.nodes[one_leaf].leaves.len() as f64) / 3.0 < LEAF_COVERAGE_MIN,
            "the fixture no longer exercises the coverage floor"
        );
        assert_ne!(
            arena.match_region(&leaves(&["a", "b", "c"]), None, false),
            Some(one_leaf)
        );
    }

    #[test]
    fn chained_views_over_the_same_relations_land_on_different_regions() {
        // Matched independently both views land on the same node and the
        // row-level view is reported as producing 1 row when it really produces
        // 5000 -- exactly the error that makes a too-large intermediate look
        // safe to persist.
        let arena = windowed_then_grouped();
        let three = leaves(&["a", "b", "c"]);

        let placed = attribute_chain(
            &arena,
            &[
                ViewRegionRequest {
                    id: "v_agg".into(),
                    leaves: three.clone(),
                    consumers: HashSet::new(),
                    prefers_aggregate: true,
                },
                ViewRegionRequest {
                    id: "v_win".into(),
                    leaves: three,
                    // v_agg depends on v_win, so v_win is confined below it.
                    consumers: HashSet::from(["v_agg".to_string()]),
                    prefers_aggregate: false,
                },
            ],
        );

        let agg = placed.get("v_agg").expect("the aggregate view was placed");
        let win = placed.get("v_win").expect("the windowed view was placed");
        assert_ne!(agg.node, win.node, "both views landed on the same operator");
        assert_eq!(arena.nodes[agg.node].operator, "HASH_GROUP_BY");
        assert_eq!(agg.cardinality, Some(1));
        assert_eq!(arena.nodes[win.node].operator, "WINDOW");
        assert_eq!(
            win.cardinality,
            Some(5000),
            "the row-level view's real width was lost"
        );
    }

    #[test]
    fn a_placed_consumer_confines_the_search_strictly_below_it() {
        let arena = windowed_then_grouped();
        let group = 0;
        let interior = arena.interior(group);
        assert!(!interior.contains(&group), "interior must be strict");
        // Confined to the group's interior, the aggregate itself is unreachable.
        let found = arena
            .match_region(&leaves(&["a", "b", "c"]), Some(&interior), false)
            .unwrap();
        assert_ne!(found, group);
    }

    #[test]
    fn intersecting_two_placed_consumers_beats_either_alone() {
        //        root (a,b)
        //        /       \
        //   left(a,b)   right(a,b)
        // Two independent branches read the same relations. Confined to `left`
        // alone the search could land in `right`; the intersection cannot.
        let branch = |t: f64| op("HASH_JOIN", t, 10, vec![scan("a", 0.1), scan("b", 0.1)]);
        let arena = PlanArena::build(&[op("UNION", 0.1, 20, vec![branch(1.0), branch(2.0)])]);
        let left = arena.nodes[0].children[0];
        let right = arena.nodes[0].children[1];
        let both: HashSet<usize> = arena
            .interior(left)
            .intersection(&arena.interior(right))
            .copied()
            .collect();
        assert!(
            both.is_empty(),
            "two disjoint branches must intersect to nothing"
        );
    }

    #[test]
    fn a_top_level_group_by_is_recognized() {
        assert!(has_top_level_group_by(
            "SELECT k, count(*) FROM t GROUP BY k",
            DialectType::DuckDB
        ));
        assert!(has_top_level_group_by(
            "SELECT count(*) FROM t",
            DialectType::DuckDB
        ));
        assert!(!has_top_level_group_by(
            "SELECT a, row_number() OVER (ORDER BY a) FROM t",
            DialectType::DuckDB
        ));
        // A query that will not parse must answer false rather than panicking.
        assert!(!has_top_level_group_by("NOT SQL AT ALL (((", DialectType::DuckDB));
    }
}
