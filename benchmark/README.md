# dee-bench

Benchmarking harness for [`dee`](../) against
[dag-bench](https://github.com/drewrip/dag-bench).

It expands a declarative experiment matrix into a queue of runs, manages the
backend infrastructure each run needs, records everything to documented
parquet, and renders an interactive dashboard.

## Quick start

```bash
cd benchmark
uv venv && uv pip install -e .          # dbt is installed here, and used from here
export DAG_BENCH=/path/to/dag-bench
cargo build --release --manifest-path ../Cargo.toml

.venv/bin/dee-bench doctor              # check the environment
.venv/bin/dee-bench run -c configs/smoke.yaml
```

`doctor` reports which container runtime, dbt and dag-bench projects it found,
and cleans up stray containers with `--clean`.

## Commands

| Command | What it does |
|---|---|
| `run -c <config>` | Run in the foreground, then build the dashboard |
| `submit -c <config>` | Run in a detached background worker |
| `status <run_dir>` | Progress, current cell, ETA; `--failed` lists errors |
| `resume <run_dir>` | Continue an interrupted or partially failed run |
| `cancel <run_dir>` | Stop the background worker, tearing infrastructure down |
| `viz <run_dir>` | Rebuild the dashboard and charts from results alone |
| `analyze <run_dir>` | Recompute the derived `payback` table |
| `schema` | Print the result schemas as markdown |
| `doctor` | Check tooling; `--clean` removes leftover containers |

Long sweeps should be submitted rather than run in the foreground:

```bash
dee-bench submit -c configs/full.yaml
dee-bench status results/full-eval
tail -f results/full-eval/worker.log
```

## Configuration

A config describes an *experiment matrix*, not a single run. Any key under
`matrix` or `dee_opt` may be a list, and the harness expands the cross product.

```yaml
name: my-eval
dag_bench: ${DAG_BENCH}
dee_cli: ../../target/release/dee-cli
output_dir: ../results/my-eval
verbosity: detailed              # summary | standard | detailed | full

matrix:
  project: [p01_iot, p03_ecommerce]
  backend: [duckdb, postgres]
  sf:      [0.1, 0.5]
  variant: [unopt, hmp, hmp_pushdown]

variants:                        # the ablation ladder
  unopt:        {passes: []}
  hmp:          {passes: [hmp], hmp_use_pushdown: false}
  hmp_pushdown: {passes: [hmp, pushdown]}

dee_opt:                         # every dee optimizer option; lists sweep
  hmp_max_runs: [1, 4]
  hmp_strategy: [breadth, greedy]

backends:
  duckdb:   {threads: 16, max_memory: 16GB, num_connections: 16}
  postgres:
    provider: container          # or `external` to use a server you run
    image: postgres:18
    port: 55433
    cpus: 16
    memory: 16g
    settings: {shared_buffers: 4GB, work_mem: 256MB}

execution:
  repetitions: 5
  warmups: 1
  sample_interval_ms: 100
```

Every `dee_opt` key mirrors a field of `OptimizerConfig` in
`dee/src/opt.rs`; `src/dee_bench/config.py` holds the mapping, including which
passes read each option.

### Why the matrix does not explode

Options are pruned to the passes that actually read them before cells are
deduplicated. Sweeping `hmp_strategy` does not multiply `unopt` or `omp` cells,
because neither consults it. In practice this is a 3-4x reduction and, more
importantly, stops identical experiments being double-counted in aggregates.

### Scheduling

Cells are ordered by `(backend, sf, project)` so infrastructure and data
preparation are amortized: a scale factor's data is generated and loaded once,
then every variant and repetition sharing it runs back to back. Cells run
strictly one at a time — concurrency would destroy timing fidelity.

## Results

Results are a hive-partitioned parquet dataset:

```
<run_dir>/results/<table>/cell_id=<cell_id>/part-<n>.parquet
```

Query them directly:

```bash
duckdb -c "SELECT variant, median(engine_wall_ms) FROM 'results/runs/**/*.parquet' GROUP BY 1"
```

`dee-bench schema` documents every table and column. Ten tables, gated by
verbosity: `cells`, `runs`, `optimizations`, `pass_stats` (summary);
`pass_iterations`, `dag_graph` (standard); `node_executions`, `system_samples`
(detailed); `plans` (full); plus `payback`, derived by `analyze`.

Each cell writes its own fragment and nothing rewrites an existing one, so a
crash can only lose the cell in flight. A partial run stays fully queryable and
renderable, and `resume` picks up exactly where it stopped.

## The seven studies

| # | Study | Where the data is |
|---|---|---|
| 1 | Runtime vs scale factor | `runs` × `cells.sf` |
| 2 | Runtime vs optimization | `runs` grouped by `cells.variant` |
| 3 | Optimization payback | `payback` |
| 4 | Ablation ladder | the `variant` ladder |
| 5 | Changes per pass | `pass_stats.changes_applied` |
| 6 | System usage | `system_samples` |
| 7 | Runtime / memory / CPU response | `runs.{engine_wall_ms, peak_rss_bytes, cpu_seconds}` |

`dee-bench viz` renders all seven as an interactive dashboard, plus a static
png and pdf per chart. It reads only the parquet, so it can be re-run at any
time, on a partial run, without re-benchmarking:

```bash
dee-bench viz results/my-eval --only payback --format png,pdf
```

## How measurements are taken

**Timing.** `dee-cli run --repeat N --warmups W` executes every repetition
inside one process, so `runs.engine_wall_ms` measures the DAG rather than CLI
startup, connection-pool construction and cleanup. `subprocess_wall_ms` records
the outer bound for comparison.

**CPU and memory.** Sampled externally by the harness from `/proc` (the dee
process tree) and cgroup counters (the Postgres container), as counter deltas
rather than sampled percentages. This matters more than it sounds: on Postgres
the dee process burns ~0.1 CPU-seconds while the server does ~96, so anything
measured only inside dee would describe the orchestrator, not the work. dee's
own connector samples are still recorded, tagged `source: engine_internal`, as
supplementary engine-level detail.

**A cross-backend caveat.** DuckDB's per-operator plan timings are CPU time;
Postgres's are wall time. The optimizer's cost ranking therefore means
something different on each, so every run records `plan_time_basis` and the two
must not be silently averaged together.

## Infrastructure

Postgres runs in a container the harness brings up and tears down, on both
normal exit and interruption — an orphaned container holding the port would
silently break the next run. Data lives in a named volume keyed by scale
factor, so re-running a scale factor skips the reload; `--fresh` drops it.
`--keep-infra` leaves the server up for debugging.

The container runtime is auto-detected (docker if its socket is usable, else
podman) and can be forced with `DEE_BENCH_CONTAINER_RUNTIME`. Results are never
touched by teardown.

Set `provider: external` to point at a Postgres you run yourself.

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

The harness invokes the `dbt` installed in its own environment rather than
whatever is on `PATH`, because dbt-core pins a narrow Python range and a
system-wide dbt on a newer interpreter fails in a way that looks unrelated.
Override with `DEE_BENCH_DBT`.
