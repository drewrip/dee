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
dee_bin: ../../target/release/dee
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
  repeat_mode: group           # or `queue` — see "How measurements are taken"
```

Every `dee_opt` key mirrors a field of `OptimizerConfig` in
`dee/src/opt.rs`; `src/dee_bench/config.py` holds the mapping, including which
passes read each option.

### Batch and continuous optimization

`execution.optimization_mode` chooses how a cell's optimization is driven, and
the two answer different questions:

| | `batch` (default) | `continuous` |
|---|---|---|
| how | `dee optimize` runs the search to convergence up front | the optimization is registered on the DAG and steps around the measured runs |
| cost | the DAG runs its search buys, in `opt_wall_ms` and `dag_runs_used` | none of its own — it spends runs the DAG was performing anyway |
| measured runs | all execute the result | those before it converges include its baseline and candidates; only runs at the promoted version are runs of the optimized DAG |
| asks | how good a plan can be found, and what did finding it cost | how quickly does a DAG converge while doing its normal work |

`batch` is the default and is what every existing result in `results/` was
produced under, so those numbers stay comparable. `configs/continuous.yaml`
sweeps `optimization_mode` to put the two side by side on the same DAGs.

In continuous mode `runs.dag_version` is what separates the two halves of a
cell: it rises once, when the search promotes its result, and `analyze` computes
payback from the runs at that version only. A cell whose search had not
converged by its last run is recorded `converging` rather than `converged`, and
contributes no payback row — a search that did not finish has no result to
price. Give continuous cells generous `repetitions` for that reason.

Only a `continuous` optimization can be driven this way. Pushdown runs `once`,
so a variant naming it alone is refused up front rather than measured as though
it had been applied; `dee-bench doctor` lists which is which.

A cell's variant and `dee_opt` are submitted **with its DAG**, as the DAG's
optimizer settings, so `dee dag optimizer c<cell_id>` answers what any DAG in
the sweep's registry is for. The optimize request itself then carries no
settings at all — an optimization with none of its own runs under the DAG's.
dee echoes back the configuration it resolved to, and a cell fails loudly if
that disagrees with what it asked for, so the indirection is checked rather
than assumed.

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

**Timing.** A cell's `warmups` and `repetitions` execute inside one dee server
against one already-warm connection pool, so `runs.engine_wall_ms` measures the
DAG rather than process startup, pool construction and cleanup. `phase_wall_ms`
records the client-observed outer bound for comparison.

`execution.repeat_mode` decides how the repetitions get there:

`group` *(default)*
:   One trigger carrying the whole series. dee runs it back to back inside a
    single driver against one engine — the tightest measurement of the DAG
    itself, and what every result to date was produced with.

`queue` *(needs dee's run queue)*
:   One entry per repetition on the server's run queue, drained strictly in
    sequence. Each repetition gets a fresh engine and its own run group in
    dee's history, which is closer to what a repetition looks like in
    production, where nothing shares an engine with the run before it. Use it
    to ask whether the shared engine is flattering the numbers.

Both modes produce the same rows: `repetitions` measured runs with `rep_index`
counting `0..n-1` within a phase, and `warmups` run once at the front.
`runs.run_group_id` is what tells them apart — one group for the whole cell
under `group`, one per repetition under `queue` — and `cells.repeat_mode`
records which was asked for. **Timings are only comparable across cells that
agree on it**, so it is part of `cell_id`. That also makes it sweepable like
any other matrix axis, which is how the two modes get compared inside one run
directory rather than by diffing two:

```yaml
matrix:
  variant: [unopt, hmp]
  repeat_mode: [group, queue]    # four cells, one baseline, two ways to measure
```

```sql
select c.variant, c.repeat_mode,
       count(distinct r.run_group_id) as groups,
       median(r.engine_wall_ms) as median_ms
from runs r join cells c using (cell_id)
where r.phase = 'measure'
group by 1, 2;
```

**CPU and memory.** Sampled externally by the harness from `/proc` (the dee
server's process tree) and cgroup counters (the Postgres container), as counter
deltas rather than sampled percentages. This matters more than it sounds: on
Postgres the dee process burns ~0.1 CPU-seconds while the server does ~96, so
anything measured only inside dee would describe the orchestrator, not the work.
dee's own connector samples are still recorded, tagged
`source: engine_internal`, as supplementary engine-level detail.

**What the shared server costs.** dee is now one long-lived process for the
whole sweep rather than a fresh child per phase, which is what makes pool reuse
possible. CPU and IO are unaffected, because the sampler baselines its counters
when it attaches and reports deltas. `peak_rss_bytes` *is* affected: resident
memory is absolute, so a previous cell's DuckDB buffer pool is still resident
when the next one starts. Use `peak_engine_mem_bytes`, which comes from dee's
own connector sampling, for memory studies.

**A cross-backend caveat.** DuckDB's per-operator plan timings are CPU time;
Postgres's are wall time. The optimizer's cost ranking therefore means
something different on each, so every run records `plan_time_basis` and the two
must not be silently averaged together.

## The dee server

The harness starts one `dee serve` per sweep, on an ephemeral port so
concurrent sweeps cannot collide, and stops it on exit. Each prepared project
registers its own connection and submits its DAG — with its optimizer settings
attached — and because every cell builds its own warehouse, connections are
re-registered per preparation, which is what evicts the pool still holding the
previous cell's database file.

The server's metadata database is written to `<run_dir>/metadata.duckdb` and is
an artifact of the sweep in its own right — it holds every run, node timing and
optimizer decision as queryable tables, alongside the parquet:

```sql
SELECT d.name, d.optimizer_config, r.dag_version,
       count(*) AS runs, round(median(r.duration_ms))
FROM runs r JOIN dags d USING (dag_id)
WHERE r.status = 'succeeded' AND r.phase = 'measure'
GROUP BY 1, 2, 3;
```

A sweep needs a dee new enough to have the run queue and per-DAG optimizer
settings. Neither is detectable from a version — dee keeps one schema rather
than a migration chain — so the queue is probed by calling its endpoint
(`dee-bench doctor` reports it, and a sweep attached to a server it did not
start checks before its first cell), and the settings are checked per cell by
comparing the configuration dee says it resolved to against the cell's own.

To point a sweep at a server you manage yourself, set `server.url`; the harness
will use it and will not stop it.

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
