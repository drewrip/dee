"""Parquet schemas for benchmark results — the specification.

Every table the harness writes is declared here as a :class:`Table`, with a
one-line description per column. These declarations *are* the documented
contract: ``dee-bench schema`` renders them, the dashboard embeds them, and
:mod:`dee_bench.store` validates every batch against them before writing.

Layout on disk, under ``<run_dir>/results/``::

    <table>/cell_id=<cell_id>/part-<n>.parquet

which is a hive-partitioned dataset: each cell appends its own fragment, so a
crash can never corrupt previously-written cells, and a partial run is still
readable. Query it with duckdb::

    SELECT * FROM 'results/runs/**/*.parquet'

Verbosity
---------
Tables carry a minimum :class:`Verbosity`. A run records a table only when its
configured verbosity is at least that level, so cheap summary sweeps and
deep-dive captures share one code path.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import pyarrow as pa

# Timestamps are UTC microseconds throughout.
TS = pa.timestamp("us", tz="UTC")


class Verbosity(enum.IntEnum):
    """How much of the result set to record.

    Each level is a superset of the one before it.
    """

    SUMMARY = 0
    """Headline numbers only: what ran, how long it took, what changed."""
    STANDARD = 1
    """Adds the optimizer's search trace and the DAG structure. The default."""
    DETAILED = 2
    """Adds per-node timings and the system-metric timeseries."""
    FULL = 3
    """Adds raw backend EXPLAIN plans and full node SQL. Large."""

    @classmethod
    def parse(cls, value: str) -> "Verbosity":
        try:
            return cls[str(value).upper()]
        except KeyError:
            valid = ", ".join(v.name.lower() for v in cls)
            raise ValueError(f"unknown verbosity {value!r}; expected one of {valid}") from None

    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Column:
    name: str
    type: pa.DataType
    doc: str


@dataclass(frozen=True)
class Table:
    name: str
    grain: str
    doc: str
    columns: list[Column]
    min_verbosity: Verbosity = Verbosity.SUMMARY
    # Written by `dee-bench analyze` from the raw tables rather than by the
    # runner, so it is recomputable without re-running any benchmark.
    derived: bool = False
    partition_by: tuple[str, ...] = field(default=("cell_id",))

    @property
    def arrow_schema(self) -> pa.Schema:
        return pa.schema([pa.field(c.name, c.type) for c in self.columns])

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def doc_for(self, column: str) -> str:
        for c in self.columns:
            if c.name == column:
                return c.doc
        raise KeyError(column)


# --------------------------------------------------------------------------
# cells — the experiment matrix, and the join key for everything else.
# --------------------------------------------------------------------------

CELLS = Table(
    name="cells",
    grain="one row per matrix cell",
    doc=(
        "The expanded experiment matrix. One row per unique combination of "
        "project, backend, scale factor, optimizer variant and dee options. "
        "`cell_id` is the join key for every other table, and is a hash of "
        "the effective parameter set, so it is stable across runs."
    ),
    columns=[
        Column("cell_id", pa.string(), "Stable hash of the effective parameter set. Primary key."),
        Column("run_name", pa.string(), "Name of the benchmark run this cell belongs to."),
        Column("project", pa.string(), "dag-bench project, e.g. 'p01_iot'."),
        Column("backend", pa.string(), "'duckdb' or 'postgres'."),
        Column("sf", pa.float64(), "dag-bench scale factor. Row counts scale ~linearly with it."),
        Column("variant", pa.string(), "Named optimizer variant, e.g. 'unopt', 'hmp', 'hmp_pushdown'."),
        Column("passes", pa.list_(pa.string()), "Optimizer passes enabled for this variant, in pipeline order."),
        Column("dee_opt", pa.string(), "JSON of the dee optimizer options in effect, after pruning options the enabled passes ignore."),
        Column("backend_config", pa.string(), "JSON of the backend tuning in effect (threads, memory, connection count)."),
        Column("repetitions", pa.int32(), "Measured repetitions requested per DAG variant."),
        Column("warmups", pa.int32(), "Untimed warmup repetitions run before the measured ones."),
        Column("repeat_mode", pa.string(), "How the repetitions were executed: 'group' (one server-side run group sharing one warm engine) or 'queue' (one queued run group each, run strictly in sequence with a fresh engine per repetition). Timings are only comparable across cells that agree on this."),
        Column("optimization_mode", pa.string(), "How the cell's optimization was driven: 'batch' (run to convergence up front, buying its own DAG runs) or 'continuous' (registered on the DAG and stepped around the measured runs, spending none). Optimization cost means different things under the two, so payback is only comparable within one. Always 'batch' for a baseline, which runs no optimization."),
        Column("dee_git_sha", pa.string(), "git SHA of the dee checkout under test."),
        Column("dag_bench_git_sha", pa.string(), "git SHA of the dag-bench checkout supplying the workload."),
        Column("harness_version", pa.string(), "dee-bench version that produced this row."),
        Column("host", pa.string(), "Hostname the benchmark ran on."),
        Column("cpu_count", pa.int32(), "Logical CPUs visible on the host."),
        Column("mem_total_bytes", pa.int64(), "Total host RAM, for interpreting memory figures."),
        Column("created_at", TS, "When this cell was expanded into the matrix."),
    ],
)


# --------------------------------------------------------------------------
# runs — the headline measurement table.
# --------------------------------------------------------------------------

RUNS = Table(
    name="runs",
    grain="one row per DAG execution",
    doc=(
        "Every execution of a DAG, including warmups. This is the primary "
        "table for studies 1, 2, 4 and 7. Always filter on "
        "`phase = 'measure'` before aggregating — warmups are recorded so "
        "they are visible, not so they are averaged in."
    ),
    columns=[
        Column("cell_id", pa.string(), "Cell this run belongs to."),
        Column("run_id", pa.string(), "Unique id for this single execution."),
        Column("run_group_id", pa.string(), "dee run group this execution belonged to. One group holds every repetition under repeat_mode 'group', and exactly one under 'queue' -- which is how the two modes are told apart in the data."),
        Column("dag_variant", pa.string(), "'unopt' for the baseline DAG, 'optimized' for the post-optimizer DAG, 'converging' for a run under a continuous optimization -- which may be the DAG as authored, a candidate under test, or the converged result, told apart by `dag_version`."),
        Column("dag_version", pa.int32(), "dee version this run executed. Constant in batch mode. Under a continuous optimization it is how a converged run is told from one that measured a candidate: the version rises once when the search promotes its result, and every run at that version is a run of the optimized DAG."),
        Column("phase", pa.string(), "'warmup' or 'measure'. Exclude warmups from every aggregate."),
        Column("rep_index", pa.int32(), "0-based repetition index within its phase."),
        Column("started_at", TS, "Wall-clock start of this execution."),
        Column("finished_at", TS, "Wall-clock finish of this execution."),
        Column("engine_wall_ms", pa.int64(), "Time dee spent executing the DAG, measured in-process. The authoritative runtime metric: excludes CLI startup, pool creation and cleanup."),
        Column("phase_wall_ms", pa.int64(), "Client-observed wall time of the whole measure phase, divided across its repetitions by each one's share of engine time. Includes the API round trip; useful only as a sanity check against engine_wall_ms."),
        Column("node_time_ms", pa.int64(), "Sum of every node's execution time. Exceeds engine_wall_ms when nodes run in parallel; their ratio is the achieved parallelism."),
        Column("node_count", pa.int32(), "Nodes in the DAG as executed."),
        Column("rows_produced", pa.int64(), "Total rows written by materialized nodes, when the backend reports them."),
        Column("cpu_seconds", pa.float64(), "CPU seconds consumed during this run, from counter deltas (/proc or cgroup), not sampled percentages. Covers the dee process tree and, on postgres, the server container."),
        Column("peak_rss_bytes", pa.int64(), "Peak resident memory of the dee server process tree during this phase. The server is long-lived and shared across cells, so unlike cpu_seconds and the IO counters -- which are deltas from an attach-time baseline -- this is an absolute figure that carries a previous cell's buffer pool with it. Prefer peak_engine_mem_bytes for memory studies."),
        Column("peak_engine_mem_bytes", pa.int64(), "Peak memory the engine itself reported (DuckDB buffer manager). Null where the backend reports nothing comparable."),
        Column("read_bytes", pa.int64(), "Bytes read from disk during this run."),
        Column("written_bytes", pa.int64(), "Bytes written to disk during this run."),
        Column("db_size_bytes", pa.int64(), "On-disk size of the database after this run, where the backend reports it."),
        Column("plan_time_basis", pa.string(), "'cpu_time' on duckdb, 'wall_time' on postgres: what the backend's per-operator plan timings actually measure. Recorded because the optimizer's cost ranking means different things on each backend."),
        Column("status", pa.string(), "'ok', 'error' or 'timeout'."),
        Column("error", pa.string(), "Failure message when status is not 'ok'."),
    ],
)


# --------------------------------------------------------------------------
# optimizations — the cost side of the payback analysis.
# --------------------------------------------------------------------------

OPTIMIZATIONS = Table(
    name="optimizations",
    grain="one row per cell whose variant runs at least one pass",
    doc=(
        "What it cost to optimize the DAG, and what the optimizer believed it "
        "gained. `opt_wall_ms` and `opt_cpu_seconds` are the numerator of the "
        "payback analysis (study 3); the denominator comes from comparing this "
        "cell's `runs` against the matching unoptimized cell."
    ),
    columns=[
        Column("cell_id", pa.string(), "Cell this optimization belongs to."),
        Column("started_at", TS, "Start of the optimize phase."),
        Column("finished_at", TS, "End of the optimize phase."),
        Column("opt_wall_ms", pa.int64(), "Wall time of the whole optimization, including every candidate DAG the passes executed while searching."),
        Column("opt_cpu_seconds", pa.float64(), "CPU seconds consumed while optimizing, from the external sampler."),
        Column("opt_peak_rss_bytes", pa.int64(), "Peak RSS during optimization."),
        Column("dag_runs_used", pa.int32(), "Full DAG executions the optimizer spent searching. The dominant term in optimization cost."),
        Column("baseline_runtime_ms", pa.int64(), "Runtime of the unoptimized DAG as the optimizer measured it. Not a substitute for the `runs` table: it is a single unrepeated measurement."),
        Column("final_runtime_ms", pa.int64(), "Runtime of the chosen DAG as the optimizer last measured it."),
        Column("total_changes_applied", pa.int32(), "Changes applied across every pass. Sum of pass_stats.changes_applied."),
        Column("nodes_before", pa.int32(), "Nodes in the DAG before optimization."),
        Column("nodes_after", pa.int32(), "Nodes after optimization. Grows when passes insert landing-pad nodes."),
        Column("status", pa.string(), "'ok', 'error' or 'timeout' for a batch optimization; 'converged' or 'converging' for a continuous one."),
        Column("error", pa.string(), "Failure message when status is not 'ok'."),
        Column("optimization_type", pa.string(), "'continuous' or 'once'. Null for a batch optimization, which drives every pass the same way regardless."),
        Column("step_phase", pa.string(), "Which side of each run a continuous optimization stepped on: 'before', 'after' or 'both'. Null in batch mode."),
        Column("result_version", pa.int32(), "DAG version a continuous optimization promoted, or null if it converged on the DAG as authored. Null in batch mode, where the version is the cell's measured one."),
    ],
)


# --------------------------------------------------------------------------
# pass_stats — study 5.
# --------------------------------------------------------------------------

PASS_STATS = Table(
    name="pass_stats",
    grain="one row per cell x optimizer pass",
    doc=(
        "What each optimizer pass did. `changes_applied` is the single "
        "comparable 'how many changes did this pass make' number across "
        "passes (study 5): materializations for HMP and OMP, query rewrites "
        "for Pushdown. `detail` holds the pass-specific fields as JSON."
    ),
    columns=[
        Column("cell_id", pa.string(), "Cell this pass ran in."),
        Column("pass_name", pa.string(), "'HMPPass', 'OMPPass' or 'PushdownPass'."),
        Column("pass_order", pa.int32(), "0-based position in the pass pipeline as actually executed."),
        Column("wall_ms", pa.int64(), "Wall time of this pass alone."),
        Column("dag_runs_used", pa.int32(), "DAG executions this pass spent. Pushdown is a static analysis and spends none."),
        Column("changes_applied", pa.int32(), "Changes this pass made to the DAG. The study-5 metric."),
        Column("candidates_considered", pa.int32(), "Candidates the pass evaluated."),
        Column("working_set_size", pa.int32(), "Candidates in the pass's working set before searching."),
        Column("detail", pa.string(), "JSON of the pass-specific detail (strategy, rankings, chosen plan)."),
    ],
)


# --------------------------------------------------------------------------
# pass_iterations — the optimizer's search trace.
# --------------------------------------------------------------------------

PASS_ITERATIONS = Table(
    name="pass_iterations",
    grain="one row per candidate DAG the optimizer tried",
    doc=(
        "The optimizer's search trace: every candidate DAG it executed while "
        "searching, in order. Iteration 1 is the baseline. Shows how the "
        "search converges and what each additional run budget buys."
    ),
    min_verbosity=Verbosity.STANDARD,
    columns=[
        Column("cell_id", pa.string(), "Cell this search ran in."),
        Column("pass_name", pa.string(), "Pass that ran this iteration."),
        Column("iteration", pa.int32(), "1-based position in the search."),
        Column("runtime_ms", pa.int64(), "Measured runtime of this candidate. A lower bound only when outcome is 'cancelled'."),
        Column("combo", pa.list_(pa.string()), "Nodes materialized in this candidate. Empty for the baseline."),
        Column("outcome", pa.string(), "'baseline', 'ok', 'cancelled' (killed for exceeding the best-so-far budget) or 'skipped'."),
        Column("cpu_seconds", pa.float64(), "CPU seconds during this candidate, when iteration profiling is enabled."),
        Column("peak_rss_bytes", pa.int64(), "Peak memory during this candidate, when iteration profiling is enabled."),
    ],
)


# --------------------------------------------------------------------------
# dag_graph — DAG structure, before and after optimization.
# --------------------------------------------------------------------------

DAG_GRAPH = Table(
    name="dag_graph",
    grain="one row per cell x dag_variant x node",
    doc=(
        "The DAG's structure, captured for both the unoptimized and optimized "
        "variants so the optimizer's structural effect can be diffed. "
        "`query_text` is recorded only at FULL verbosity."
    ),
    min_verbosity=Verbosity.STANDARD,
    columns=[
        Column("cell_id", pa.string(), "Cell this DAG belongs to."),
        Column("dag_variant", pa.string(), "'unopt' or 'optimized'."),
        Column("node_id", pa.string(), "Fully-qualified node identifier."),
        Column("materialization", pa.string(), "'view', 'table' or 'temp_table'."),
        Column("depends_on", pa.list_(pa.string()), "Immediate upstream node ids."),
        Column("out_degree", pa.int32(), "Number of direct downstream consumers. Views with out-degree > 1 are the optimizer's materialization candidates."),
        Column("paths_to_sinks", pa.int32(), "Materialization points (table or temp-table nodes) reachable from this node, as computed by dee's own Graph::paths_to_sinks. This is the number OMP filters and ranks candidates by, so it explains the optimizer's choices. It is not a count of distinct paths to childless nodes, which is what an earlier harness-side reimplementation computed."),
        Column("query_text", pa.string(), "The node's SQL. FULL verbosity only; null otherwise."),
    ],
)


# --------------------------------------------------------------------------
# node_executions — per-node timing.
# --------------------------------------------------------------------------

NODE_EXECUTIONS = Table(
    name="node_executions",
    grain="one row per run x node",
    doc=(
        "Per-node execution timing within a run. Shows which nodes the "
        "optimizer's changes actually moved, and where a DAG's time goes."
    ),
    min_verbosity=Verbosity.DETAILED,
    columns=[
        Column("cell_id", pa.string(), "Cell this execution belongs to."),
        Column("run_id", pa.string(), "Run this node executed in."),
        Column("node_id", pa.string(), "Node identifier."),
        Column("materialization", pa.string(), "'view', 'table' or 'temp_table' as executed."),
        Column("started_at", TS, "Start of this node's execution."),
        Column("duration_ms", pa.int64(), "How long this node took."),
        Column("rows_produced", pa.int64(), "Rows written by this node. 0 for views, which materialize nothing."),
        Column("has_plan", pa.bool_(), "Whether a backend plan was captured for this node (see the `plans` table)."),
    ],
)


# --------------------------------------------------------------------------
# system_samples — the resource timeseries.
# --------------------------------------------------------------------------

SYSTEM_SAMPLES = Table(
    name="system_samples",
    grain="one row per sample per source per run",
    doc=(
        "Resource-usage timeseries (study 6). `source` distinguishes the "
        "harness's own external sampling from what the engine self-reports: "
        "'harness_process' reads /proc for the dee process tree, "
        "'harness_container' reads cgroup counters for the postgres "
        "container, and 'engine_internal' is dee's own connector sampling. "
        "Prefer the harness sources for anything comparable across backends; "
        "engine_internal is supplementary detail with no cross-backend "
        "equivalent."
    ),
    min_verbosity=Verbosity.DETAILED,
    columns=[
        Column("cell_id", pa.string(), "Cell this sample belongs to."),
        Column("run_id", pa.string(), "Run or optimize phase being sampled."),
        Column("phase", pa.string(), "'optimize', 'warmup' or 'measure'."),
        Column("source", pa.string(), "'harness_process', 'harness_container' or 'engine_internal'."),
        Column("elapsed_ms", pa.int64(), "Milliseconds since the sampled phase started."),
        Column("timestamp", TS, "Absolute sample time."),
        Column("cpu_seconds_cum", pa.float64(), "Cumulative CPU seconds since the phase started. Differentiate for a rate; the endpoint is the phase's total."),
        Column("rss_bytes", pa.int64(), "Resident set size at this instant."),
        Column("engine_mem_bytes", pa.int64(), "Engine-reported memory (DuckDB buffer manager). Null for harness sources."),
        Column("read_bytes", pa.int64(), "Cumulative bytes read since the phase started."),
        Column("written_bytes", pa.int64(), "Cumulative bytes written since the phase started."),
    ],
)


# --------------------------------------------------------------------------
# plans — raw backend plans.
# --------------------------------------------------------------------------

PLANS = Table(
    name="plans",
    grain="one row per run x node with a captured plan",
    doc=(
        "Raw backend EXPLAIN / EXPLAIN ANALYZE JSON, exactly as the engine "
        "emitted it. Large, so FULL verbosity only. Kept verbatim rather than "
        "parsed so the capture never loses information the analysis did not "
        "anticipate needing."
    ),
    min_verbosity=Verbosity.FULL,
    columns=[
        Column("cell_id", pa.string(), "Cell this plan belongs to."),
        Column("run_id", pa.string(), "Run this plan was captured in."),
        Column("node_id", pa.string(), "Node this plan describes."),
        Column("plan_format", pa.string(), "'duckdb_json' or 'postgres_json'."),
        Column("plan_json", pa.string(), "The plan, verbatim."),
    ],
)


# --------------------------------------------------------------------------
# payback — derived by `dee-bench analyze` (study 3).
# --------------------------------------------------------------------------

PAYBACK = Table(
    name="payback",
    grain="one row per project x backend x sf x variant",
    doc=(
        "How many DAG runs it takes to repay the cost of optimizing (study "
        "3). Derived from `optimizations` and `runs` by `dee-bench analyze`, "
        "never written by the runner, so it can be recomputed from raw "
        "results without re-running anything. `payback_runs_*` is null when "
        "the variant did not actually improve on the baseline, since the cost "
        "is then never repaid."
    ),
    derived=True,
    partition_by=(),
    columns=[
        Column("run_name", pa.string(), "Benchmark run these figures come from."),
        Column("project", pa.string(), "dag-bench project."),
        Column("backend", pa.string(), "'duckdb' or 'postgres'."),
        Column("sf", pa.float64(), "Scale factor."),
        Column("variant", pa.string(), "Optimizer variant being repaid."),
        Column("cell_id", pa.string(), "Cell the optimized measurements come from."),
        Column("baseline_cell_id", pa.string(), "Cell the unoptimized baseline comes from."),
        Column("opt_cost_wall_s", pa.float64(), "Wall seconds spent optimizing."),
        Column("opt_cost_cpu_s", pa.float64(), "CPU seconds spent optimizing."),
        Column("baseline_wall_s", pa.float64(), "Median measured runtime of the unoptimized DAG."),
        Column("variant_wall_s", pa.float64(), "Median measured runtime of the optimized DAG."),
        Column("baseline_cpu_s", pa.float64(), "Median CPU seconds per run, unoptimized."),
        Column("variant_cpu_s", pa.float64(), "Median CPU seconds per run, optimized."),
        Column("savings_per_run_wall_s", pa.float64(), "Wall seconds saved per run. Negative means the variant is slower."),
        Column("savings_per_run_cpu_s", pa.float64(), "CPU seconds saved per run."),
        Column("speedup", pa.float64(), "baseline_wall_s / variant_wall_s. Above 1 is faster."),
        Column("payback_runs_wall", pa.float64(), "Runs to repay the optimization in wall time. Null when never repaid."),
        Column("payback_runs_cpu", pa.float64(), "Runs to repay the optimization in CPU time. Null when never repaid."),
        Column("payback_runs_wall_lo", pa.float64(), "Lower bound of the bootstrap 95% CI on payback_runs_wall."),
        Column("payback_runs_wall_hi", pa.float64(), "Upper bound of the bootstrap 95% CI on payback_runs_wall."),
        Column("n_baseline", pa.int32(), "Measured repetitions behind the baseline median."),
        Column("n_variant", pa.int32(), "Measured repetitions behind the variant median."),
    ],
)


ALL_TABLES: list[Table] = [
    CELLS,
    RUNS,
    OPTIMIZATIONS,
    PASS_STATS,
    PASS_ITERATIONS,
    DAG_GRAPH,
    NODE_EXECUTIONS,
    SYSTEM_SAMPLES,
    PLANS,
    PAYBACK,
]

BY_NAME: dict[str, Table] = {t.name: t for t in ALL_TABLES}


def tables_for(verbosity: Verbosity) -> list[Table]:
    """The non-derived tables a run at `verbosity` should record."""
    return [t for t in ALL_TABLES if not t.derived and t.min_verbosity <= verbosity]


def render_markdown() -> str:
    """Render every schema as markdown, for `dee-bench schema` and the dashboard."""
    out: list[str] = [
        "# dee-bench result schemas",
        "",
        "Results are written as a hive-partitioned parquet dataset under",
        "`<run_dir>/results/<table>/cell_id=<cell_id>/part-<n>.parquet`.",
        "Query with duckdb: `SELECT * FROM 'results/runs/**/*.parquet'`.",
        "",
        "| Table | Grain | Verbosity | Written by |",
        "|---|---|---|---|",
    ]
    for t in ALL_TABLES:
        out.append(
            f"| [`{t.name}`](#{t.name}) | {t.grain} | "
            f"{'derived' if t.derived else t.min_verbosity.label()} | "
            f"{'`dee-bench analyze`' if t.derived else 'runner'} |"
        )
    for t in ALL_TABLES:
        out += ["", f"## {t.name}", "", t.doc, "", f"**Grain:** {t.grain}  "]
        out.append(
            f"**Recorded at:** {'derived by `dee-bench analyze`' if t.derived else t.min_verbosity.label() + ' verbosity and above'}",
        )
        out += ["", "| Column | Type | Description |", "|---|---|---|"]
        for c in t.columns:
            out.append(f"| `{c.name}` | `{c.type}` | {c.doc} |")
    return "\n".join(out) + "\n"
