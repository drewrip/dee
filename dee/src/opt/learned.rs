//! A cost model fitted to what the engine actually did.
//!
//! Every `CREATE TABLE ... AS` dee runs comes back with an EXPLAIN ANALYZE
//! plan, and every operator in that plan carries how long it took and how much
//! data passed through it. Seconds divided by bytes is a **seconds-per-byte**
//! constant for that operator type --- a small number that says what this
//! engine, on this machine, charges to push a byte through a hash join.
//!
//! Those constants accumulate across runs. Costing a plan is then arithmetic on
//! the plan alone: for each operator, its bytes times the constant for its
//! type, summed. Nothing about it needs the plan to have been executed, which
//! is the point --- a VIEW's plan never is.
//!
//! # Which bytes
//!
//! The ones the operator **consumed**, not the ones it emitted
//! ([`PlanNode::input_bytes`]). This was the other way round first, and the
//! difference is not a refinement. An operator's time tracks the work it did,
//! and an aggregate that scans 3.2M rows to emit twenty of them did the work of
//! the 3.2M: dividing its seconds by twenty rows' worth of bytes produced a
//! constant four orders of magnitude above a scan's, which then transferred to
//! every plan that emitted more. Measured on p05 it over-priced the report
//! tables threefold (5.92 s against 2.06 s measured) and under-priced a large
//! materialization roughly tenfold --- and because
//! [`LearnedCostModel::observe_write`] used to fit the write from what wall
//! clock had left over after compute, that over-pricing kept the residual
//! negative and the write constant was never learned at all.
//!
//! A write is the exception, and for a reason that is not symmetry: it is
//! priced per byte **written**. What a write consumes and what it costs are the
//! same rows, and the plan reports its output as a one-row count rather than
//! the payload --- so the bytes come from the run's own row count, not the plan.
//! [`CostBasis`] records which basis a stored model was fitted against, because
//! the two cannot be mixed and nothing else in a serialized row says.
//!
//! # What "operator type" means
//!
//! The operator's name, except for aggregates, which also carry the functions
//! they compute ([`PlanNode::cost_key`]). `count(*)` and `string_agg` share a
//! `HASH_GROUP_BY` and cost nothing like each other per output byte, and one
//! constant fitted to both is a constant fitted to neither.
//!
//! # Aggregating repeat observations
//!
//! An operator type is seen many times, and each sighting gives its own
//! seconds-per-byte. They are combined **weighted by bytes**: total seconds
//! over total bytes, so an observation counts in proportion to the work it
//! actually represents.
//!
//! The unweighted mean of the ratios was tried first and is kept alongside as
//! [`OperatorSamples::unweighted_seconds_per_byte`], because the difference
//! between them is not academic. On p03_ecommerce a single `HASH_GROUP_BY`
//! whose output was small enough that fixed per-call overhead dwarfed anything
//! proportional to bytes measured 1.6e-4 s/byte --- a hundred thousand times a
//! hash join's 6.2e-10. Counted once, it set the constant for its whole
//! operator type, and multiplied by a 6.9M-row view it priced that view at
//! 13,199 seconds on a DAG that runs in three. Weighting by bytes puts that
//! same sighting where it belongs: it moved a few hundred bytes, so it moves
//! the constant by a few hundred bytes' worth.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use log::warn;
use serde::{Deserialize, Serialize};

use crate::plan::{PlanNode, is_write_operator};

/// Everything observed about one operator type.
///
/// Sums rather than a running mean: sums merge associatively, which is what
/// lets a model loaded from the metadata store be combined with one fitted to
/// the run that just finished without keeping either's observations around.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct OperatorSamples {
    /// Sum over observations of `seconds / bytes`.
    pub seconds_per_byte_sum: f64,
    /// Observations behind `seconds_per_byte_sum`.
    pub seconds_per_byte_n: u64,
    /// Sum over observations of one output tuple's width in bytes.
    pub bytes_per_tuple_sum: f64,
    /// Observations behind `bytes_per_tuple_sum`. Larger than
    /// `seconds_per_byte_n`, because a width is observable on an un-timed plan.
    pub bytes_per_tuple_n: u64,
    /// Total measured seconds and total output bytes across observations. Not
    /// read by the mean below; kept so a byte-weighted constant can be fitted
    /// from stored state without re-collecting plans.
    pub seconds_total: f64,
    pub bytes_total: f64,
}

impl OperatorSamples {
    /// Total seconds over total bytes: the byte-weighted constant.
    ///
    /// This is what prices a plan. A sighting counts for as much as the work it
    /// did, so an operator called once on a handful of bytes cannot outvote a
    /// scan of a million rows.
    pub fn seconds_per_byte(&self) -> Option<f64> {
        (self.bytes_total > 0.0).then(|| self.seconds_total / self.bytes_total)
    }

    /// The plain mean of the per-observation ratios, every sighting counting
    /// once. Not used to price anything; kept because it is the obvious
    /// alternative and the two are worth comparing on a real DAG.
    pub fn unweighted_seconds_per_byte(&self) -> Option<f64> {
        (self.seconds_per_byte_n > 0)
            .then(|| self.seconds_per_byte_sum / self.seconds_per_byte_n as f64)
    }

    /// The mean observed width of one output tuple.
    pub fn bytes_per_tuple(&self) -> Option<f64> {
        (self.bytes_per_tuple_n > 0)
            .then(|| self.bytes_per_tuple_sum / self.bytes_per_tuple_n as f64)
    }

    pub fn merge(&mut self, other: &OperatorSamples) {
        self.seconds_per_byte_sum += other.seconds_per_byte_sum;
        self.seconds_per_byte_n += other.seconds_per_byte_n;
        self.bytes_per_tuple_sum += other.bytes_per_tuple_sum;
        self.bytes_per_tuple_n += other.bytes_per_tuple_n;
        self.seconds_total += other.seconds_total;
        self.bytes_total += other.bytes_total;
    }
}

/// Say once, per distinct substitution, that a write was priced on the wrong
/// path's constant.
///
/// Once rather than every time: `write_cost` runs per member per costed
/// combination, which on a single HMP iteration is scores of calls and would
/// bury the log. There are only ever a handful of distinct pairs --- two paths
/// on DuckDB, one on Postgres --- so the set stays tiny.
fn note_write_path_fallback(wanted: &str, used: &str) {
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(String, String)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let Ok(mut guard) = seen.lock() else { return };
    if guard.insert((wanted.to_string(), used.to_string())) {
        warn!(
            "learned cost: no write constant for `{wanted}`; pricing it with `{used}`'s \
             instead. The two are not interchangeable -- DuckDB's sinks differ by about \
             2x per byte -- so this build's cost is approximate until `{wanted}` is \
             measured, which the first trial that materializes through it will do."
        );
    }
}

/// Seconds per byte written, one observation per relation.
///
/// Not [`OperatorSamples`], and not a running sum, because the write constant
/// is a **median** over relations rather than a mean. Byte-weighting an average
/// hands the constant to the largest table ever written: measured over four
/// dag-bench projects, one 7.2 GB fact table was 65.5% of the buffered path's
/// fit and set the price of every write on the backend, for an out-of-sample
/// median error of 3.21x against 1.64x here. A median cannot be captured that
/// way --- the outlier is one vote.
///
/// The median is also why nothing has to be excluded. A mean in any scale has
/// to be defended against writes too short to be measuring bytes, which report
/// ten to forty times the real rate; the median simply out-votes them, and it
/// was measured doing so (1.64x with every write admitted, against 8.01x for a
/// geometric mean on the same samples).
///
/// **One observation per relation, latest wins.** That bounds the state by the
/// warehouse rather than by how many times a DAG has run, and it is the right
/// statistic besides: a pipeline that runs nightly should not stack three
/// hundred copies of one table's rate and out-vote every other table in the
/// median. Re-measuring a relation replaces its entry, so the constant tracks
/// the machine instead of averaging over its history.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WriteSamples {
    /// Seconds per byte written, keyed by the relation it was observed on.
    #[serde(default)]
    per_relation: HashMap<String, f64>,
}

impl WriteSamples {
    fn observe(&mut self, relation: &str, seconds_per_byte: f64) {
        self.per_relation
            .insert(relation.to_string(), seconds_per_byte);
    }

    /// The median rate across relations, or `None` if none was ever observed.
    pub fn seconds_per_byte(&self) -> Option<f64> {
        let mut ks: Vec<f64> = self.per_relation.values().copied().collect();
        if ks.is_empty() {
            return None;
        }
        ks.sort_by(|a, b| a.partial_cmp(b).expect("rates are finite"));
        let mid = ks.len() / 2;
        Some(if ks.len().is_multiple_of(2) {
            (ks[mid - 1] + ks[mid]) / 2.0
        } else {
            ks[mid]
        })
    }

    /// How many relations are behind the median.
    pub fn len(&self) -> usize {
        self.per_relation.len()
    }

    pub fn is_empty(&self) -> bool {
        self.per_relation.is_empty()
    }

    /// Fold another set in. A relation observed in both takes `other`'s value:
    /// that is the one loaded or measured later, and a rate is a statement
    /// about the relation as it is now.
    fn merge(&mut self, other: &WriteSamples) {
        for (relation, k) in &other.per_relation {
            self.per_relation.insert(relation.clone(), *k);
        }
    }
}

/// Which quantity an operator's seconds were divided by to fit its constant.
///
/// Recorded in the serialized model because the two are not interchangeable and
/// nothing else in a stored row says which one produced it. A model fitted
/// against output bytes and then applied as though it were input bytes is not
/// approximately right, it is wrong by whatever the operator's selectivity was
/// --- and an aggregate's selectivity is measured in orders of magnitude.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostBasis {
    /// Bytes the operator emitted. What the model used to fit, and what no
    /// longer fits anything: retained only so a stored row can say it is one of
    /// these and be discarded.
    #[default]
    OutputBytes,
    /// Bytes the operator consumed --- [`PlanNode::input_bytes`].
    InputBytes,
}

/// How long this plan spent writing its result, in seconds.
///
/// Two shapes, because the backends report it two ways. DuckDB has a real write
/// operator and times it like any other, so the answer is on the plan. Postgres
/// has none --- `EXPLAIN ANALYZE` of a `CREATE TABLE AS` shows the `SELECT` and
/// nothing else --- so the write is what the statement took beyond the sum of
/// everything the plan does account for.
///
/// `None` when neither is available, which is not a write cost of zero.
fn write_timing(roots: &[PlanNode]) -> Option<f64> {
    if let Some(t) = roots.iter().find_map(write_operator_time) {
        return Some(t);
    }
    let root = roots.first()?;
    let total = root.total_execution_time_s?;
    Some((total - accounted_time(root)).max(0.0))
}

/// The write operator's own exclusive time, wherever it sits in the tree.
fn write_operator_time(node: &PlanNode) -> Option<f64> {
    if is_write_operator(&node.operator)
        && let Some(t) = node.exclusive_time_s
    {
        return Some(t);
    }
    node.children.iter().find_map(write_operator_time)
}

/// Every operator's exclusive time, summed --- what the plan explains.
fn accounted_time(node: &PlanNode) -> f64 {
    node.exclusive_time_s.unwrap_or(0.0)
        + node.children.iter().map(accounted_time).sum::<f64>()
}

/// Seconds-per-byte constants, one per operator type, fitted to executed plans.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LearnedCostModel {
    ops: HashMap<String, OperatorSamples>,
    /// Seconds per byte *written*, one set of observations per write path.
    ///
    /// Separate from `ops` because a write operator's own output is a one-row
    /// count of what it wrote rather than the payload, so its bytes cannot be
    /// read off the plan the way every other operator's can --- see
    /// [`is_write_operator`]. A [`WriteSamples`] rather than an
    /// [`OperatorSamples`] because the two are summarized differently, and for
    /// the reason given there.
    ///
    /// Keyed by path, because an engine can persist a relation more than one
    /// way at very different cost. DuckDB has two sinks:
    /// `BATCH_CREATE_TABLE_AS` when the pipeline still carries batch order,
    /// `CREATE_TABLE_AS` when it does not. An engine with one way of writing
    /// has one entry, under
    /// [`SINGLE_WRITE_PATH`](crate::plan::SINGLE_WRITE_PATH).
    ///
    /// Optional in the serialized form, and deliberately under a new shape: a
    /// model stored before the split holds one constant pooled across both
    /// paths, which is the wrong number for either, and one stored before the
    /// median holds a byte-weighted average, which is the number being
    /// replaced. Neither loads into this field; those constants refit.
    #[serde(default)]
    writes: HashMap<String, WriteSamples>,
    /// What `ops` was fitted against. See [`CostBasis`].
    #[serde(default)]
    basis: CostBasis,
}

impl LearnedCostModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether this model has learned anything at all.
    ///
    /// Write constants count: a run that materialized something but priced no
    /// operator still has a constant worth persisting, and reporting it empty
    /// would throw it away on the way out of `publish_learned`.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.writes.is_empty()
    }

    pub fn operators(&self) -> impl Iterator<Item = (&String, &OperatorSamples)> {
        self.ops.iter()
    }

    /// What this model's operator constants were fitted against.
    ///
    /// A model that has observed nothing reports the current basis: there is
    /// nothing in it fitted the old way, so nothing to discard.
    pub fn basis(&self) -> CostBasis {
        // Operator constants specifically: `basis` describes what `ops` was
        // fitted against, and a model holding only write constants has nothing
        // fitted the old way to discard.
        if self.ops.is_empty() {
            return CostBasis::InputBytes;
        }
        self.basis
    }

    /// Fold another model's observations into this one.
    pub fn merge(&mut self, other: &LearnedCostModel) {
        for (key, samples) in &other.ops {
            self.ops.entry(key.clone()).or_default().merge(samples);
        }
        for (path, samples) in &other.writes {
            self.writes.entry(path.clone()).or_default().merge(samples);
        }
        if !other.ops.is_empty() {
            self.basis = other.basis();
        }
    }

    /// Record what every operator of an executed plan revealed.
    ///
    /// Safe to call on an un-executed plan: a timing is required before a
    /// seconds-per-byte observation is recorded, so a plain EXPLAIN contributes
    /// only widths --- which are just as real there as anywhere.
    pub fn observe(&mut self, roots: &[PlanNode]) {
        for root in roots {
            self.observe_node(root);
        }
        self.basis = CostBasis::InputBytes;
    }

    fn observe_node(&mut self, node: &PlanNode) {
        // A write operator's output is a one-row count of what it wrote, not
        // what it worked on, so a constant fitted to it describes nothing ---
        // see `is_write_operator`.
        if !is_write_operator(&node.operator)
            && let Some(width) = node.row_width_bytes.filter(|w| *w > 0.0)
        {
            let entry = self.ops.entry(node.cost_key()).or_default();
            entry.bytes_per_tuple_sum += width;
            entry.bytes_per_tuple_n += 1;

            // Both halves of the ratio have to be measured, not estimated: an
            // operator's cost is fitted to what it did, and `cardinality` is
            // `None` on a plan that was never executed.
            //
            // The denominator is what the operator *consumed*. Output bytes
            // were tried first and are the reason this model could not price a
            // materialization: an aggregate scanning 3.2M rows to emit twenty
            // of them got a seconds-per-byte four orders of magnitude above a
            // scan's, which then transferred to anything that emitted more.
            if let (Some(seconds), Some(bytes)) = (node.exclusive_time_s, node.input_bytes())
                && bytes > 0.0
            {
                entry.seconds_per_byte_sum += seconds / bytes;
                entry.seconds_per_byte_n += 1;
                entry.seconds_total += seconds;
                entry.bytes_total += bytes;
            }
        }
        for child in &node.children {
            self.observe_node(child);
        }
    }

    /// Record what one materialized node revealed about the cost of *writing*.
    ///
    /// The write's own measured time, over the bytes it wrote. Independent of
    /// the compute model by design: this used to be fitted from
    /// `measured - cost(plan)`, which made the write constant a function of
    /// compute-model error, and on every sample we have that error was large
    /// enough to sink the observation entirely. Report tables were over-priced
    /// threefold, the residual came out negative, the guard discarded it, and
    /// the constant stayed unlearned for the whole life of the feature.
    ///
    /// The timing is the write operator's where the plan has one (DuckDB times
    /// `CREATE_TABLE_AS` like any other operator) and otherwise the part of the
    /// statement no operator accounted for (Postgres). The bytes are the rows
    /// written times the payload's width --- not the write operator's own
    /// output, which is a one-row count of what it did.
    ///
    /// Nothing is recorded when no rows were written, when the plan carries no
    /// write timing, or when the payload has no discoverable width. A short
    /// write *is* recorded: it reports a wild rate, and the median out-votes it
    /// rather than needing it excluded --- see [`WriteSamples`].
    pub fn observe_write(&mut self, relation: &str, roots: &[PlanNode], rows_written: f64) {
        if rows_written <= 0.0 {
            return;
        }
        let Some(seconds) = write_timing(roots) else {
            return;
        };
        if seconds <= 0.0 {
            return;
        }
        let Some(width) = roots.iter().find_map(|r| self.payload_width(r)) else {
            return;
        };
        let bytes = rows_written * width;
        if bytes <= 0.0 {
            return;
        }
        // Under the path the engine actually used, read off the executed plan.
        // Never a prediction here: the prediction is for Views that have not
        // been written, and fitting a measurement under a guessed path would
        // put one sink's seconds against the other's constant.
        self.writes
            .entry(crate::plan::observed_write_path(roots))
            .or_default()
            .observe(relation, seconds / bytes);
    }

    /// How wide one row of what this plan wrote is.
    ///
    /// The write operator's own width describes its one-row count, so the
    /// payload is the topmost operator below it that is not itself a write.
    fn payload_width(&self, node: &PlanNode) -> Option<f64> {
        if !is_write_operator(&node.operator) {
            return self.width_of(node);
        }
        node.children.iter().find_map(|c| self.payload_width(c))
    }

    /// The rate to price a write on `path` with, and the path it actually came
    /// from.
    ///
    /// Exact where the model has seen that path written. Where it has not, the
    /// path with the most relations behind it stands in, and the substitution
    /// is logged once per distinct pair --- see [`Self::write_cost`].
    ///
    /// A substitute is a real approximation and not a good one: DuckDB's two
    /// sinks differ by about 2x. It is offered because the alternative is no
    /// price at all, and an unpriced build takes the whole candidate out of a
    /// makespan estimate. The caller is told which path was used so the
    /// difference is never invisible.
    ///
    /// The first write on a DAG is exactly this case: on `p05_hr` the baseline
    /// materializes six batch-sink report tables and one buffered one, while
    /// every candidate HMP wants to price is buffered, so without a substitute
    /// the first ranking prices no write at all.
    pub fn write_rate_for<'a>(&'a self, path: &'a str) -> Option<(f64, &'a str)> {
        if let Some(k) = self
            .writes
            .get(path)
            .and_then(WriteSamples::seconds_per_byte)
        {
            return Some((k, path));
        }
        // The best-evidenced substitute: most relations behind it, and the name
        // as a tie-break so the choice does not depend on map iteration order.
        self.writes
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .max_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| b.0.cmp(a.0)))
            .and_then(|(other, v)| Some((v.seconds_per_byte()?, other.as_str())))
    }

    /// The median seconds-per-byte-written on one write path, or `None` if that
    /// path has never been observed.
    ///
    /// Deliberately not a fallback to another path's constant, or to an average
    /// over all of them. The two DuckDB sinks differ by about 2x, so standing
    /// one in for the other is the error the split exists to remove. `None` is
    /// the honest answer, and `makespan::estimate` already refuses a build it
    /// cannot price rather than charging it as free.
    pub fn write_seconds_per_byte(&self, path: &str) -> Option<f64> {
        self.writes.get(path).and_then(WriteSamples::seconds_per_byte)
    }

    /// Every write path this model has fitted a constant for.
    pub fn write_paths(&self) -> impl Iterator<Item = (&str, f64)> {
        self.writes
            .iter()
            .filter_map(|(k, v)| Some((k.as_str(), v.seconds_per_byte()?)))
    }

    /// What writing this plan's result costs on `path`, in seconds.
    ///
    /// `None` when that path has no constant or the payload has no width ---
    /// which is not a cost of zero. A caller that needs a total has to decide
    /// for itself whether to proceed without this term.
    pub fn write_cost(&self, path: &str, roots: &[PlanNode], rows_written: f64) -> Option<f64> {
        if rows_written <= 0.0 {
            return Some(0.0);
        }
        let (spb, from) = self.write_rate_for(path)?;
        if from != path {
            note_write_path_fallback(path, from);
        }
        let width = roots.iter().find_map(|r| self.payload_width(r))?;
        Some(rows_written * width * spb)
    }

    /// What spooling this plan's result into a `MATERIALIZED` CTE costs, in
    /// seconds.
    ///
    /// # Why this is a fraction of the write constant rather than its own
    ///
    /// A materialized CTE is not a table write. There is no catalog entry, no
    /// durability guarantee, and on an engine that keeps the spool in memory,
    /// no IO at all --- but it is not free either: the rows have to be
    /// buffered somewhere before the readers scan them, and on an engine that
    /// spills that buffer to disk the cost approaches a write. So it is a
    /// per-byte sink cost of the same shape as a write, at a different rate.
    ///
    /// Nothing measures that rate yet. `observe_write` can fit the write
    /// constant because a `CREATE TABLE AS` shows up as its own operator with
    /// its own timing; a CTE materialization does not separate out that way in
    /// either backend's plan. Until it does, `factor` is the caller's estimate
    /// of the ratio, and its per-backend default lives in
    /// [`crate::opt::common::default_spool_factor`] --- **a modelled guess,
    /// not a measurement**, which is why it is a tunable and not a constant
    /// buried here.
    ///
    /// `None` when the write path has no constant or the payload has no width.
    /// Not a cost of zero: a caller that needs a total has to decide for itself
    /// whether to proceed without this term, the same contract
    /// [`Self::write_cost`] has.
    pub fn spool_cost(
        &self,
        path: &str,
        roots: &[PlanNode],
        rows_spooled: f64,
        factor: f64,
    ) -> Option<f64> {
        if factor <= 0.0 {
            // An engine whose spool really is free. Zero because the caller
            // said so, which is a different thing from not being able to price
            // it --- and the difference decides whether the candidate is
            // rankable at all.
            return Some(0.0);
        }
        Some(self.write_cost(path, roots, rows_spooled)? * factor)
    }

    /// [`PlanNode::input_bytes`], but with the model's learned widths standing
    /// in where the plan has none.
    ///
    /// The plan's own method needs `row_width_bytes`, which DuckDB reports only
    /// under profiling --- and the plans this prices are VIEWs, which are never
    /// executed. Same three cases in the same order: what the operator scanned,
    /// else what its children emitted, else its own output.
    fn input_bytes_of(&self, node: &PlanNode) -> Option<f64> {
        if let (Some(rows), Some(width)) = (node.rows_scanned, self.width_of(node))
            && rows > 0.0
        {
            return Some(rows * width);
        }
        let from_children: f64 = node
            .children
            .iter()
            .filter_map(|c| Some(c.rows()? * self.width_of(c)?))
            .sum();
        if from_children > 0.0 {
            return Some(from_children);
        }
        Some(node.rows()? * self.width_of(node)?)
    }

    /// The constant for `key`, falling back to the model's overall mean for an
    /// operator type never seen before.
    ///
    /// Falling back rather than returning zero: an unseen operator priced at
    /// zero is not "unknown", it is "free", and a plan full of unseen operators
    /// would come out cheapest of all --- exactly backwards.
    pub fn seconds_per_byte(&self, key: &str) -> Option<f64> {
        self.ops
            .get(key)
            .and_then(OperatorSamples::seconds_per_byte)
            .or_else(|| self.mean_seconds_per_byte())
    }

    /// The byte-weighted seconds-per-byte over every operator type: the whole
    /// model's total seconds over its total bytes.
    ///
    /// Weighted for the same reason the per-operator constant is, and more
    /// urgently --- this is the fallback an operator type nobody has ever seen
    /// gets priced at, so one wild observation anywhere would otherwise set the
    /// price of everything unknown.
    pub fn mean_seconds_per_byte(&self) -> Option<f64> {
        let (seconds, bytes) = self
            .ops
            .values()
            .fold((0.0, 0.0), |(s, b), o| (s + o.seconds_total, b + o.bytes_total));
        (bytes > 0.0).then(|| seconds / bytes)
    }

    /// The mean output tuple width over every observation of every operator
    /// type.
    pub fn mean_bytes_per_tuple(&self) -> Option<f64> {
        let (sum, n) = self.ops.values().fold((0.0, 0u64), |(s, n), o| {
            (s + o.bytes_per_tuple_sum, n + o.bytes_per_tuple_n)
        });
        (n > 0).then(|| sum / n as f64)
    }

    /// How wide one of `node`'s output tuples is: what the plan says, or what
    /// this operator type was measured at, or the model's overall mean.
    ///
    /// The fallbacks exist for DuckDB, whose plain `EXPLAIN (FORMAT JSON)` --- the
    /// only plan a VIEW ever has --- reports no width at all. Postgres prints
    /// `Plan Width` with or without ANALYZE and takes the first branch.
    fn width_of(&self, node: &PlanNode) -> Option<f64> {
        node.row_width_bytes
            .filter(|w| *w > 0.0)
            .or_else(|| {
                self.ops
                    .get(&node.cost_key())
                    .and_then(OperatorSamples::bytes_per_tuple)
            })
            .or_else(|| self.mean_bytes_per_tuple())
    }

    /// What this plan should cost: output bytes times seconds-per-byte, summed
    /// over every operator.
    ///
    /// `None` when the model has learned nothing at all, or when no operator in
    /// the plan carried enough information to be priced --- neither is a cost of
    /// zero, and reporting one would rank a plan nothing is known about above
    /// every plan something is.
    pub fn cost(&self, roots: &[PlanNode]) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let mut total = 0.0;
        let mut priced = 0usize;
        for root in roots {
            self.cost_node(root, &mut total, &mut priced);
        }
        (priced > 0).then_some(total)
    }

    fn cost_node(&self, node: &PlanNode, total: &mut f64, priced: &mut usize) {
        if !is_write_operator(&node.operator)
            && let (Some(bytes), Some(spb)) =
                (self.input_bytes_of(node), self.seconds_per_byte(&node.cost_key()))
        {
            *total += bytes * spb;
            *priced += 1;
        }
        for child in &node.children {
            self.cost_node(child, total, priced);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{parse_duckdb_plan, parse_postgres_plan};

    /// The DuckDB sink the write fixtures in this module are written through.
    const CTAS: &str = "CREATE_TABLE_AS";

    /// The write's own measured time over the bytes it wrote --- with no
    /// compute model involved at all. This is the property the residual did not
    /// have, and the reason the constant was never learned.
    #[test]
    fn a_write_constant_is_fitted_from_the_write_operators_own_timing() {
        let plans = parse_duckdb_plan(&duckdb_write_profile()).expect("the fixture parses");
        let mut model = LearnedCostModel::new();
        assert!(model.write_seconds_per_byte(CTAS).is_none(), "nothing observed yet");

        // The payload is the operator below the write: 100 rows at 10 bytes.
        model.observe_write("t", &plans, 100.0);

        let spb = model
            .write_seconds_per_byte(CTAS)
            .expect("the write operator carries a timing");
        assert!(
            (spb - 0.26 / 1000.0).abs() < 1e-12,
            "the write's seconds over the bytes it wrote: got {spb}"
        );
        assert!((model.write_cost(CTAS, &plans, 100.0).unwrap() - 0.26).abs() < 1e-12);
    }

    /// The regression this whole change exists for. A model that over-prices
    /// the compute used to teach nothing about writing, because the residual it
    /// was fitted from went negative and the observation was dropped. The write
    /// timing is measured, so an unpriceable --- or wildly wrong --- compute model
    /// cannot suppress it.
    #[test]
    fn an_over_priced_compute_no_longer_suppresses_the_write_constant() {
        let plans = parse_duckdb_plan(&duckdb_write_profile()).expect("the fixture parses");
        let mut model = LearnedCostModel::new();
        assert!(
            model.cost(&plans).is_none(),
            "an empty model prices no compute at all --- the worst case for a residual"
        );

        model.observe_write("t", &plans, 100.0);
        assert!(
            model.write_seconds_per_byte(CTAS).is_some(),
            "the write is observable without any compute model to subtract from"
        );
    }

    /// Postgres has no write operator to time, so the write is what the
    /// statement took beyond everything the plan accounts for.
    #[test]
    fn the_postgres_write_is_what_the_plan_does_not_account_for() {
        // Root inclusive 0.100s; `Execution Time` 150ms. The 0.050s gap is the
        // write. Payload: 100 rows at 10 bytes = 1000 bytes.
        let json = r#"[{"Execution Time": 150.0, "Plan": {
            "Node Type": "Aggregate", "Actual Total Time": 100.0,
            "Actual Rows": 100, "Actual Loops": 1, "Plan Rows": 100, "Plan Width": 10,
            "Plans": []}}]"#;
        let plans = parse_postgres_plan(json).expect("the fixture parses");
        let mut model = LearnedCostModel::new();
        model.observe_write("t", &plans, 100.0);

        let spb = model
            .write_seconds_per_byte(crate::plan::SINGLE_WRITE_PATH)
            .expect("a gap must be recorded");
        assert!((spb - 0.050 / 1000.0).abs() < 1e-12, "got {spb}");
    }

    /// A plan that says nothing about writing yields no constant --- which is
    /// not a constant of zero.
    #[test]
    fn a_plan_with_no_write_teaches_nothing_about_writing() {
        let plans = parse_duckdb_plan(&duckdb_profile()).expect("the fixture parses");
        let mut model = LearnedCostModel::new();
        model.observe(&plans);
        model.observe_write("t", &plans, 100.0);
        assert_eq!(model.write_paths().count(), 0);
    }

    /// Zero rows written is a real answer, not a missing one.
    #[test]
    fn writing_nothing_costs_nothing() {
        let plans = parse_duckdb_plan(&duckdb_profile()).expect("the fixture parses");
        let model = LearnedCostModel::new();
        assert_eq!(model.write_cost(CTAS, &plans, 0.0), Some(0.0));
    }

    /// Models stored before the write constant existed must keep loading ---
    /// and must be recognised as fitted the old way, so they are discarded
    /// rather than mixed with constants on a different basis.
    #[test]
    fn a_stored_model_predating_the_basis_field_loads_and_reports_the_old_basis() {
        let plans = parse_duckdb_plan(&duckdb_profile()).expect("the fixture parses");
        let mut fitted = LearnedCostModel::new();
        fitted.observe(&plans);
        let legacy = serde_json::json!({ "ops": fitted.ops }).to_string();

        let loaded: LearnedCostModel =
            serde_json::from_str(&legacy).expect("a model without `basis` must deserialize");
        assert_eq!(loaded.write_paths().count(), 0);
        assert_eq!(
            loaded.basis(),
            CostBasis::OutputBytes,
            "a row that does not say was written before the basis moved"
        );
    }

    /// A model that has observed nothing has nothing fitted the old way, so it
    /// must not be discarded as though it did.
    #[test]
    fn an_empty_model_is_not_stale() {
        assert_eq!(LearnedCostModel::new().basis(), CostBasis::InputBytes);
    }

    /// Merging has to carry the write observations too, or a model folded in
    /// from the metadata store would silently lose them.
    #[test]
    fn merging_carries_the_write_constant() {
        let plans = parse_duckdb_plan(&duckdb_write_profile()).expect("the fixture parses");
        let mut fitted = LearnedCostModel::new();
        fitted.observe(&plans);
        fitted.observe_write("t", &plans, 100.0);

        let mut empty = LearnedCostModel::new();
        empty.merge(&fitted);
        assert_eq!(empty.write_seconds_per_byte(CTAS), fitted.write_seconds_per_byte(CTAS));
        assert!(empty.write_seconds_per_byte(CTAS).is_some());
        assert_eq!(empty.basis(), CostBasis::InputBytes);
    }

    /// The point of the split. Two sinks with different costs stay apart
    /// instead of averaging into a constant that describes neither.
    #[test]
    fn each_write_path_keeps_its_own_constant() {
        let mut model = LearnedCostModel::new();
        // Buffered: 0.26s for 1000 bytes. Batch: 0.04s for the same 1000.
        model.observe_write(
            "t",
            &parse_duckdb_plan(&duckdb_write_profile()).expect("parses"),
            100.0,
        );
        model.observe_write(
            "t",
            &parse_duckdb_plan(&duckdb_write_profile().replace("CREATE_TABLE_AS", "BATCH_CREATE_TABLE_AS").replace("0.26", "0.04"))
                .expect("parses"),
            100.0,
        );

        let buffered = model.write_seconds_per_byte(CTAS).expect("buffered fitted");
        let batch = model
            .write_seconds_per_byte("BATCH_CREATE_TABLE_AS")
            .expect("batch fitted");
        assert!((buffered - 0.26 / 1000.0).abs() < 1e-12, "got {buffered}");
        assert!((batch - 0.04 / 1000.0).abs() < 1e-12, "got {batch}");
        assert_eq!(model.write_paths().count(), 2);
    }

    /// The exact accessor stays exact: it answers for the path asked about and
    /// nothing else, so a caller that needs to know whether a path has really
    /// been measured can still find out.
    #[test]
    fn the_exact_rate_never_answers_for_a_path_it_has_not_seen() {
        let plans = parse_duckdb_plan(&duckdb_write_profile()).expect("parses");
        let mut model = LearnedCostModel::new();
        model.observe_write("t", &plans, 100.0);

        assert!(model.write_seconds_per_byte(CTAS).is_some());
        assert!(
            model.write_seconds_per_byte("BATCH_CREATE_TABLE_AS").is_none(),
            "the exact accessor substituted a constant"
        );
    }

    /// Pricing does substitute, and says which path it used. Without it the
    /// first ranking on a DAG prices no write at all: on `p05_hr` the baseline
    /// materializes batch-sink report tables while every candidate is buffered,
    /// so the path that needs a constant is exactly the one with no samples.
    #[test]
    fn pricing_falls_back_to_another_paths_constant_and_names_it() {
        let plans = parse_duckdb_plan(&duckdb_write_profile()).expect("parses");
        let mut model = LearnedCostModel::new();
        model.observe_write("t", &plans, 100.0);

        let (rate, from) = model
            .write_rate_for("BATCH_CREATE_TABLE_AS")
            .expect("a substitute is offered");
        assert_eq!(from, CTAS, "the substitute must name where it came from");
        assert_eq!(Some(rate), model.write_seconds_per_byte(CTAS));
        assert!(
            model.write_cost("BATCH_CREATE_TABLE_AS", &plans, 100.0).is_some(),
            "the build stayed unpriced"
        );
    }

    /// The substitution is announced. A silently substituted constant is the
    /// failure mode this whole split exists to avoid, so falling back has to be
    /// visible in the log, once per distinct pair.
    #[test]
    fn falling_back_is_logged() {
        use std::sync::Arc;
        struct Capture(Arc<Mutex<Vec<String>>>);
        impl log::Log for Capture {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }
            fn log(&self, record: &log::Record<'_>) {
                if record.level() <= log::Level::Warn
                    && let Ok(mut v) = self.0.lock()
                {
                    v.push(record.args().to_string());
                }
            }
            fn flush(&self) {}
        }
        let sink = Arc::new(Mutex::new(Vec::new()));
        // A logger can only be installed once per process; if another test got
        // there first this still exercises the path, it just cannot read it.
        static LOGGER: OnceLock<Capture> = OnceLock::new();
        let logger = LOGGER.get_or_init(|| Capture(Arc::clone(&sink)));
        let installed = log::set_logger(logger).is_ok();
        log::set_max_level(log::LevelFilter::Warn);

        let plans = parse_duckdb_plan(&duckdb_write_profile()).expect("parses");
        let mut model = LearnedCostModel::new();
        model.observe_write("t", &plans, 100.0);
        // Asks for the path with no samples, which has to borrow the other's.
        assert!(model.write_cost("BATCH_CREATE_TABLE_AS", &plans, 100.0).is_some());

        if !installed {
            return;
        }
        let lines = sink.lock().unwrap().clone();
        let hit = lines
            .iter()
            .find(|l| l.contains("BATCH_CREATE_TABLE_AS") && l.contains("CREATE_TABLE_AS"));
        let hit = hit.unwrap_or_else(|| panic!("the fallback was silent; saw {lines:?}"));
        assert!(hit.contains("no write constant"), "unhelpful message: {hit}");
        println!("logged: {hit}");

        // Said once, not once per call: this runs per member per costed
        // combination, which is scores of calls per HMP iteration.
        let before = sink.lock().unwrap().len();
        for _ in 0..5 {
            let _ = model.write_cost("BATCH_CREATE_TABLE_AS", &plans, 100.0);
        }
        assert_eq!(sink.lock().unwrap().len(), before, "the warning repeated");
    }

    /// Where several paths could stand in, the one with the most relations
    /// behind it wins --- most evidence, and a deterministic choice rather than
    /// whatever the map happens to iterate first.
    #[test]
    fn the_best_evidenced_path_is_the_one_that_stands_in() {
        let mut model = LearnedCostModel::new();
        model.observe_write("solo", &write_plan(0.100, 100, 1_000_000.0), 100.0);
        let batch = duckdb_write_profile().replace("CREATE_TABLE_AS", "BATCH_CREATE_TABLE_AS");
        for i in 0..3 {
            let plans = parse_duckdb_plan(&batch).expect("parses");
            model.observe_write(&format!("b{i}"), &plans, 100.0);
        }
        let (_, from) = model.write_rate_for("SOME_THIRD_SINK").expect("a substitute");
        assert_eq!(from, "BATCH_CREATE_TABLE_AS", "the thinner path stood in");
    }

    /// A model that has seen no write at all still refuses: there is nothing to
    /// fall back *to*, and a write of zero is not the answer.
    #[test]
    fn an_empty_model_has_nothing_to_fall_back_to() {
        let model = LearnedCostModel::new();
        let plans = parse_duckdb_plan(&duckdb_write_profile()).expect("parses");
        assert!(model.write_rate_for(CTAS).is_none());
        assert!(model.write_cost(CTAS, &plans, 100.0).is_none());
    }

    /// A measurement is always filed under the sink the engine actually used,
    /// read off the executed plan --- never under a predicted one, which would
    /// put one path's seconds against the other's constant.
    #[test]
    fn an_observation_is_filed_under_the_sink_the_plan_names() {
        let batch = duckdb_write_profile().replace("CREATE_TABLE_AS", "BATCH_CREATE_TABLE_AS");
        let mut model = LearnedCostModel::new();
        model.observe_write("t", &parse_duckdb_plan(&batch).expect("parses"), 100.0);
        let paths: Vec<_> = model.write_paths().map(|(p, _)| p.to_string()).collect();
        assert_eq!(paths, vec!["BATCH_CREATE_TABLE_AS".to_string()]);
    }

    /// Merging keeps the paths apart, or a model folded in from the metadata
    /// store would pool them on the way back --- exactly what the split undid.
    #[test]
    fn merging_keeps_each_write_path_separate() {
        let mut a = LearnedCostModel::new();
        a.observe_write(
            "t",
            &parse_duckdb_plan(&duckdb_write_profile()).expect("parses"),
            100.0,
        );
        let mut b = LearnedCostModel::new();
        b.observe_write(
            "t",
            &parse_duckdb_plan(&duckdb_write_profile().replace("CREATE_TABLE_AS", "BATCH_CREATE_TABLE_AS").replace("0.26", "0.04"))
                .expect("parses"),
            100.0,
        );
        a.merge(&b);
        assert!((a.write_seconds_per_byte(CTAS).unwrap() - 0.26 / 1000.0).abs() < 1e-12);
        assert!(
            (a.write_seconds_per_byte("BATCH_CREATE_TABLE_AS").unwrap() - 0.04 / 1000.0).abs()
                < 1e-12
        );
    }

    /// A model stored before the split holds one constant pooled across both
    /// sinks. It is the wrong number for either, so it does not load --- the
    /// write constants refit, while the operator constants beside them survive.
    #[test]
    fn a_stored_model_predating_the_split_refits_its_write_constants() {
        let samples = r#"{"seconds_per_byte_sum":0.001,"seconds_per_byte_n":1,
                          "bytes_per_tuple_sum":10.0,"bytes_per_tuple_n":1,
                          "seconds_total":0.26,"bytes_total":1000.0}"#;
        let legacy =
            format!(r#"{{"ops":{{"SEQ_SCAN":{samples}}},"write":{samples},"basis":"input_bytes"}}"#);
        let loaded: LearnedCostModel =
            serde_json::from_str(&legacy).expect("an older row must still deserialize");
        assert_eq!(
            loaded.write_paths().count(),
            0,
            "a constant pooled across both sinks was adopted for one of them"
        );
        assert_eq!(loaded.operators().count(), 1, "the compute constants survive");
        assert_eq!(loaded.basis(), CostBasis::InputBytes);
    }

    /// A run that materialized something but priced no operator still has a
    /// constant worth keeping: reporting it empty would drop it on the way out
    /// of `publish_learned`.
    #[test]
    fn a_model_holding_only_write_constants_is_not_empty() {
        let mut model = LearnedCostModel::new();
        model.observe_write(
            "t",
            &parse_duckdb_plan(&duckdb_write_profile()).expect("parses"),
            100.0,
        );
        assert!(!model.is_empty());
        assert_eq!(model.basis(), CostBasis::InputBytes, "and is not stale");
    }

    /// The regression this estimator exists for. One enormous table used to own
    /// the write constant outright: byte-weighting gave a 7.2 GB fact table
    /// 65.5% of the buffered path's fit across four dag-bench projects, and
    /// every other write on the backend was priced at its rate.
    #[test]
    fn one_huge_table_no_longer_sets_the_price_of_every_write() {
        // Three ordinary relations at 1e-9 s/byte, and one giant at 4e-9.
        let mut model = LearnedCostModel::new();
        for i in 0..3 {
            model.observe_write(&format!("small_{i}"), &write_plan(0.100, 100, 1_000_000.0), 100.0);
        }
        model.observe_write("giant", &write_plan(40.0, 100, 100_000_000.0), 100.0);

        let k = model.write_seconds_per_byte(CTAS).expect("fitted");
        // Median of [1e-9, 1e-9, 1e-9, 4e-9] is 1e-9: the giant is one vote.
        assert!((k - 1e-9).abs() < 1e-12, "one table captured the constant: {k:e}");

        // For contrast, what byte-weighting would have said: total seconds over
        // total bytes, which is the giant's own rate to two digits.
        let byte_weighted = (0.100 * 3.0 + 40.0) / (1e8 * 3.0 + 1e10);
        assert!(
            byte_weighted / k > 3.0,
            "the two should disagree sharply here: {byte_weighted:e} vs {k:e}"
        );
    }

    /// A write too short to be measuring bytes reports a wild rate --- tens of
    /// times the truth, because at that duration the number is fixed overhead
    /// with the payload lost in it. A median needs no guard against them: they
    /// are out-voted rather than averaged in, which is why nothing is excluded.
    #[test]
    fn a_write_too_short_to_measure_is_outvoted_rather_than_excluded() {
        let mut model = LearnedCostModel::new();
        for i in 0..3 {
            model.observe_write(&format!("real_{i}"), &write_plan(0.100, 100, 1_000_000.0), 100.0);
        }
        // 20 microseconds over a handful of bytes: an implied 2e-8 s/byte.
        model.observe_write("tiny", &write_plan(0.000_02, 100, 10.0), 100.0);

        let k = model.write_seconds_per_byte(CTAS).expect("fitted");
        assert!((k - 1e-9).abs() < 1e-12, "the noise sample moved the median: {k:e}");
        // And it is still recorded, so it counts once and can be re-measured.
        assert_eq!(model.writes.get(CTAS).map(WriteSamples::len), Some(4));
    }

    /// One observation per relation, latest wins. A DAG that runs nightly must
    /// not stack copies of one table's rate and out-vote every other table.
    #[test]
    fn re_measuring_a_relation_replaces_its_rate_rather_than_adding_a_vote() {
        let mut model = LearnedCostModel::new();
        model.observe_write("a", &write_plan(0.100, 100, 1_000_000.0), 100.0);
        model.observe_write("b", &write_plan(0.300, 100, 1_000_000.0), 100.0);
        assert_eq!(model.writes.get(CTAS).map(WriteSamples::len), Some(2));

        // `a` measured again, slower this time.
        model.observe_write("a", &write_plan(0.500, 100, 1_000_000.0), 100.0);
        assert_eq!(
            model.writes.get(CTAS).map(WriteSamples::len),
            Some(2),
            "re-measuring a relation added a second vote for it"
        );
        // Median of [3e-9, 5e-9], not of [1e-9, 3e-9, 5e-9].
        let k = model.write_seconds_per_byte(CTAS).unwrap();
        assert!((k - 4e-9).abs() < 1e-12, "got {k:e}");
    }

    /// Merging has to carry the per-relation observations, or a model folded in
    /// from the metadata store would come back with no write constant.
    #[test]
    fn merging_carries_the_per_relation_observations() {
        let mut a = LearnedCostModel::new();
        a.observe_write("a", &write_plan(0.100, 100, 1_000_000.0), 100.0);
        let mut b = LearnedCostModel::new();
        b.observe_write("b", &write_plan(0.300, 100, 1_000_000.0), 100.0);
        a.merge(&b);
        assert_eq!(a.writes.get(CTAS).map(WriteSamples::len), Some(2));
        // Median of [1e-9, 3e-9].
        assert!((a.write_seconds_per_byte(CTAS).unwrap() - 2e-9).abs() < 1e-12);
    }

    /// A row stored before the median holds a byte-weighted average under the
    /// old shape. It does not load: the number it carries is the one being
    /// replaced, so those write constants refit.
    #[test]
    fn a_stored_row_from_before_the_median_refits() {
        let stored = r#"{"ops":{},"basis":"input_bytes","writes":{"CREATE_TABLE_AS":{
            "seconds_per_byte_sum":1e-9,"seconds_per_byte_n":1,
            "bytes_per_tuple_sum":10.0,"bytes_per_tuple_n":1,
            "seconds_total":0.1,"bytes_total":100000000.0}}}"#;
        let loaded: LearnedCostModel =
            serde_json::from_str(stored).expect("an older row must still deserialize");
        assert!(
            loaded.write_seconds_per_byte(CTAS).is_none(),
            "a byte-weighted average was served as a median"
        );
        assert_eq!(loaded.write_paths().count(), 0);
    }

    /// A `CREATE_TABLE_AS` over one operator: `seconds` on the write, and a
    /// payload of `rows` x `width` under it.
    fn write_plan(seconds: f64, rows: i64, width: f64) -> Vec<PlanNode> {
        let json = format!(
            r#"{{"operator_name":"CREATE_TABLE_AS","operator_timing":{seconds},
                 "operator_cardinality":1,"result_set_size":8,"extra_info":{{}},
                 "children":[{{"operator_name":"SEQ_SCAN","operator_timing":0.0,
                   "operator_cardinality":{rows},"result_set_size":{},
                   "extra_info":{{}},"children":[]}}]}}"#,
            rows as f64 * width
        );
        parse_duckdb_plan(&json).expect("the fixture parses")
    }

    /// The same two operators under a write: DuckDB times `CREATE_TABLE_AS`
    /// like anything else, and reports its output as the one-row count.
    fn duckdb_write_profile() -> String {
        r#"{"operator_name":"CREATE_TABLE_AS","operator_timing":0.26,
            "operator_cardinality":1,"result_set_size":8,
            "extra_info":{},
            "children":[
              {"operator_name":"HASH_GROUP_BY","operator_timing":0.001,
               "operator_cardinality":100,"result_set_size":1000,
               "extra_info":{"Aggregates":["sum(#1)"],"Estimated Cardinality":100},
               "children":[
                 {"operator_name":"SEQ_SCAN","operator_timing":0.002,
                  "operator_cardinality":1000,"result_set_size":24000,
                  "extra_info":{"Table":"orders","Estimated Cardinality":1000},
                  "children":[]}]}]}"#
            .to_string()
    }

    /// A profiling plan: two operators, known bytes, known times.
    fn duckdb_profile() -> String {
        // 1000 rows x 24 bytes = 24000 bytes in 0.002s -> 8.33e-8 s/byte.
        // 100 rows x 10 bytes = 1000 bytes in 0.001s   -> 1e-6   s/byte.
        r#"{"operator_name":"HASH_GROUP_BY","operator_timing":0.001,
            "operator_cardinality":100,"result_set_size":1000,
            "extra_info":{"Aggregates":["sum(#1)"],"Estimated Cardinality":100},
            "children":[
              {"operator_name":"SEQ_SCAN","operator_timing":0.002,
               "operator_cardinality":1000,"result_set_size":24000,
               "extra_info":{"Table":"orders","Estimated Cardinality":1000},
               "children":[]}]}"#
            .to_string()
    }

    /// An operator's constant is its seconds over the bytes it *consumed*.
    ///
    /// The aggregate is the case that matters: it emitted 1000 bytes and read
    /// 24000, and pricing it against the 1000 is what used to make an aggregate
    /// look twenty-four times more expensive per byte than the scan feeding it.
    #[test]
    fn an_operators_constant_is_its_seconds_divided_by_its_input_bytes() {
        let plan = parse_duckdb_plan(&duckdb_profile()).unwrap();
        let mut model = LearnedCostModel::new();
        model.observe(&plan);

        // A leaf with nothing to say about what it read falls back to output.
        assert!((model.seconds_per_byte("SEQ_SCAN").unwrap() - 0.002 / 24000.0).abs() < 1e-15);
        assert!(
            (model.seconds_per_byte("HASH_GROUP_BY[sum]").unwrap() - 0.001 / 24000.0).abs()
                < 1e-15,
            "the aggregate is priced against its child's output, not its own"
        );
    }

    #[test]
    fn an_aggregates_type_carries_the_function_it_computes() {
        let plan = parse_duckdb_plan(&duckdb_profile()).unwrap();
        let mut model = LearnedCostModel::new();
        model.observe(&plan);
        let keys: Vec<&String> = model.ops.keys().collect();
        assert!(keys.iter().any(|k| k.as_str() == "HASH_GROUP_BY[sum]"));
        // The bare name is *not* a key: two aggregates that differ only in
        // their function must not share a constant.
        assert!(!keys.iter().any(|k| k.as_str() == "HASH_GROUP_BY"));
    }

    #[test]
    fn repeat_observations_of_an_operator_are_combined() {
        let mut model = LearnedCostModel::new();
        // 100 bytes in 1s, then 100 bytes in 3s: 0.01 and 0.03 s/byte. Equal
        // byte counts, so weighting changes nothing and both means agree.
        for seconds in [1.0, 3.0] {
            let json = format!(
                r#"{{"operator_name":"FILTER","operator_timing":{seconds},
                     "operator_cardinality":10,"result_set_size":100,
                     "extra_info":{{}},"children":[]}}"#
            );
            model.observe(&parse_duckdb_plan(&json).unwrap());
        }
        assert!((model.seconds_per_byte("FILTER").unwrap() - 0.02).abs() < 1e-12);
    }

    #[test]
    fn a_tiny_observation_cannot_outvote_a_large_one() {
        // The p03 failure in miniature. One call on 100 bytes that took 1s --
        // fixed overhead, nothing to do with bytes -- against a million bytes in
        // 1s. Counted once each the constant would be ~5e-3 s/byte, set almost
        // entirely by the operator that moved a thousandth of the bytes.
        let mut model = LearnedCostModel::new();
        for (seconds, rows, bytes) in [(1.0, 10u64, 100u64), (1.0, 100_000u64, 1_000_000u64)] {
            let json = format!(
                r#"{{"operator_name":"HASH_GROUP_BY","operator_timing":{seconds},
                     "operator_cardinality":{rows},"result_set_size":{bytes},
                     "extra_info":{{"Aggregates":["sum(#1)"]}},"children":[]}}"#
            );
            model.observe(&parse_duckdb_plan(&json).unwrap());
        }

        let key = "HASH_GROUP_BY[sum]";
        // Weighted: 2 seconds over 1,000,100 bytes.
        let weighted = model.seconds_per_byte(key).unwrap();
        assert!((weighted - 2.0 / 1_000_100.0).abs() < 1e-15, "was {weighted}");
        // Unweighted: (0.01 + 0.000001) / 2, ~2500x larger, and the reason the
        // p03 ranking put 13,199 seconds on a three-second DAG.
        let unweighted = model
            .ops
            .get(key)
            .unwrap()
            .unweighted_seconds_per_byte()
            .unwrap();
        assert!(unweighted > weighted * 2000.0, "unweighted was {unweighted}");
    }

    #[test]
    fn merging_two_models_is_the_same_as_observing_both_runs() {
        let plan = parse_duckdb_plan(&duckdb_profile()).unwrap();
        let mut both = LearnedCostModel::new();
        both.observe(&plan);
        both.observe(&plan);

        let mut a = LearnedCostModel::new();
        a.observe(&plan);
        let mut b = LearnedCostModel::new();
        b.observe(&plan);
        a.merge(&b);

        assert_eq!(a, both);
    }

    #[test]
    fn costing_a_plan_prices_every_operators_output_bytes() {
        let mut model = LearnedCostModel::new();
        model.observe(&parse_duckdb_plan(&duckdb_profile()).unwrap());

        // A VIEW's plan: same operators, estimates only, no widths and no
        // timings. Widths come from what the same operator types were measured
        // at, so the cost is the profiled cost reproduced from estimates.
        let view = r#"[{"name":"HASH_GROUP_BY","extra_info":
            {"Aggregates":["sum(#1)"],"Estimated Cardinality":100},"children":[
              {"name":"SEQ_SCAN","extra_info":
                {"Table":"orders","Estimated Cardinality":1000},"children":[]}]}]"#;
        let cost = model.cost(&parse_duckdb_plan(view).unwrap()).unwrap();
        assert!((cost - (0.001 + 0.002)).abs() < 1e-12, "cost was {cost}");
    }

    #[test]
    fn an_operator_type_never_seen_costs_the_average_rather_than_nothing() {
        let mut model = LearnedCostModel::new();
        model.observe(&parse_duckdb_plan(&duckdb_profile()).unwrap());
        let unseen = r#"[{"name":"PIVOT","extra_info":{"Estimated Cardinality":1000},
                          "children":[]}]"#;
        let cost = model.cost(&parse_duckdb_plan(unseen).unwrap()).unwrap();
        assert!(cost > 0.0, "an unpriced operator must not be free");
    }

    #[test]
    fn a_write_operator_teaches_the_model_nothing() {
        // DuckDB gives CREATE_TABLE_AS one row of 8 bytes however large the
        // table it wrote, so its seconds-per-byte is enormous and meaningless.
        // Left in, it sets the model's mean and so prices every operator type
        // the model has never seen.
        let json = r#"{"operator_name":"CREATE_TABLE_AS","operator_timing":0.26,
             "operator_cardinality":1,"result_set_size":8,"extra_info":{},
             "children":[
               {"operator_name":"SEQ_SCAN","operator_timing":0.002,
                "operator_cardinality":1000,"result_set_size":24000,
                "extra_info":{"Table":"orders","Estimated Cardinality":1000},
                "children":[]}]}"#;
        let mut model = LearnedCostModel::new();
        model.observe(&parse_duckdb_plan(json).unwrap());

        assert!(model.ops.get("CREATE_TABLE_AS").is_none());
        // The mean is the scan's constant alone, not 0.0325 s/byte.
        assert!((model.mean_seconds_per_byte().unwrap() - 0.002 / 24000.0).abs() < 1e-15);
    }

    #[test]
    fn a_model_that_has_learned_nothing_declines_to_cost() {
        let model = LearnedCostModel::new();
        assert!(model.cost(&parse_duckdb_plan(&duckdb_profile()).unwrap()).is_none());
    }

    #[test]
    fn postgres_widths_come_from_the_plan_itself() {
        // `Plan Width` is reported with and without ANALYZE, so nothing has to
        // be learned before a Postgres VIEW plan can be priced.
        let analyzed = r#"[{"Plan":{"Node Type":"Aggregate","Strategy":"Hashed",
            "Actual Total Time":2.0,"Actual Rows":100,"Actual Loops":1,
            "Plan Rows":100,"Plan Width":40,
            "Output":["id","sum(f)","count(*)"],"Group Key":["t.id"],
            "Plans":[{"Node Type":"Seq Scan","Relation Name":"t",
                      "Actual Total Time":1.0,"Actual Rows":1000,"Actual Loops":1,
                      "Plan Rows":1000,"Plan Width":36,"Output":["id","f"]}]}}]"#;
        let mut model = LearnedCostModel::new();
        model.observe(&parse_postgres_plan(analyzed).unwrap());

        // The aggregate consumed its child's output --- 1000 rows x 36 bytes ---
        // in (2.0 - 1.0) ms, not the 100 x 40 it emitted.
        let agg = model.seconds_per_byte("AGGREGATE[count,sum]").unwrap();
        assert!((agg - 0.001 / 36000.0).abs() < 1e-15);
        // 1000 rows x 36 bytes in 1.0 ms.
        let scan = model.seconds_per_byte("SEQ SCAN").unwrap();
        assert!((scan - 0.001 / 36000.0).abs() < 1e-15);
    }

    #[test]
    fn a_postgres_group_key_expression_is_not_mistaken_for_an_aggregate() {
        let plan = r#"[{"Plan":{"Node Type":"Aggregate","Plan Rows":10,"Plan Width":8,
            "Output":["date_trunc('day', ts)","count(*)"],
            "Group Key":["date_trunc('day', ts)"]}}]"#;
        let parsed = parse_postgres_plan(plan).unwrap();
        assert_eq!(parsed[0].cost_key(), "AGGREGATE[count]");
    }
}
