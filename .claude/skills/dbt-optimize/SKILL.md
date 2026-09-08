---
name: dbt-optimize
description: Make a dbt project's DAG run faster without changing what it produces. Use when asked to optimize, speed up, or reduce the cost of a dbt project - choosing materializations, rewriting model SQL, and tuning adapter settings against a fixed database, dataset, and machine. Balances end-to-end runtime against the queries spent finding the answer.
---

# Optimizing a dbt project

You are given a dbt project that runs against postgres or duckdb. Make it
finish faster. You may not change the database, the data, or the machine, and
the optimized project must produce the same relations under the same model
names.

Two signals, and both count:

1. **End-to-end DAG runtime.** Shorter is better. This is the goal.
2. **Queries sent to the database while optimizing.** Every trial run costs
   money and time. An optimization that saves 10s per run but costs 40 runs to
   find has to be run 40+ times before it is worth anything. Spend
   measurement like it comes out of the savings, because it does.

The whole job is buying information about the DAG as cheaply as possible, and
then spending it well.

## The constraints, exactly

These are fixed. Everything not listed here is fair game.

1. The database is what it is (postgres or duckdb) - do not switch engines.
2. The source data is what it is - do not sample, filter, pre-aggregate, or
   otherwise reduce it.
3. The machine is what it is - no scaling up, no adding hardware.
4. **Every model in the original DAG has a same-named model in the optimized
   DAG returning an equivalent relation.** The direction matters: you may
   **add** models freely, and you may not **drop** one or change what it
   returns. Equivalence means *for all data*, not *for this data* - matching
   output on one dataset is a failed falsification, not a proof.
5. **A model the user declared as a table must exist as a table when the DAG
   finishes.** You may change how it gets there; you may not leave it a view.

Constraint 4 gives you more room than it first appears. The original models
are a floor, not a ceiling: every one of them must survive with its relation
intact, and everything else is yours to arrange. New intermediate models are
allowed and are one of the best tools here, because they let you place a
materialization at exactly the right cut point instead of at whichever node
the original author happened to write. What you cannot do is delete a model
someone may be querying - which also rules out making an original model
`ephemeral`, since an ephemeral model has no relation at all.

## Workflow

### 1. Read the DAG before you run anything (free)

```bash
dbt parse                                     # writes target/manifest.json, sends no queries
python3 scripts/analyze.py <project> --top 20
```

`analyze.py` reports, per model, its **expansion count**: how many times its
SQL is expanded into a query the database actually plans. A table is built
once. A view is re-expanded once per materialized consumer, transitively,
counting every path. A view with expansion 6 does its work six times.

Note what you are looking at:

- **Declared materializations.** Which models the user pinned as `table`
  (constraint 5) and which are views you are free to move.
- **Shape.** A wide fan-in of view layers ending in one table means the whole
  DAG collapses into a single enormous query, built by a single thread, with
  every shared subtree recomputed per path. That is the classic case and its
  fix is materialization placement.
- **Obvious SQL waste**, readable without measuring: `order by` in a model
  nothing consumes ordered, `select *` chains dragging unread columns through
  every layer, `distinct` on an already-unique key, a filter applied late that
  could be applied before a join.
- **Repeated SQL that is not a shared node.** The same subquery, join, or
  aggregation written out inline in two different models. `analyze.py` cannot
  see this - textually duplicated SQL is not a shared DAG node - so it is
  found by reading, and it is frequently the largest single win, because
  factoring it into one new materialized model removes work no tool reported.

Write down what you expect to win before you measure. It keeps you honest
about whether the measurement taught you anything.

### 2. Baseline, and learn the noise floor (2-4 runs)

```bash
python3 scripts/bench.py --project P --label baseline --runs 3 --warmups 1
```

This records wall clock, statements sent, and database time. Keep the
`--command` identical for every measurement you intend to compare
(`run --full-refresh` by default; if you compare against a `build`, compare
against `build` throughout).

The standard deviation is the important output. **Any candidate whose
improvement is smaller than 2 sigma has not been shown to do anything.** If
sigma is a large fraction of the runtime, add runs before you add candidates -
otherwise every later trial is unreadable and you will spend the whole budget
chasing noise.

Do the same for values, not just time - see step 5.

### 2b. Decompose the baseline before spending anything (free)

```bash
python3 scripts/decompose.py --project P \
    --measurement .opt/measurements/baseline.json --time-parse --dbt <dbt>
```

This reads the debug log the baseline already wrote and costs nothing. It is
the highest-value step in the workflow and it is allowed to **end the
exercise**. It reports:

- **the node execution window vs wall clock.** This is the ceiling: making
  every model instant saves exactly the window, and not a second more. Use
  the *window*, not the summed database time - statements run concurrently,
  so the sum routinely exceeds the window and can exceed wall clock outright
  (one sweep project measured 4.28s of database work inside a 1.46s window at
  3.3x concurrency). Treating the sum as a ceiling overstates the opportunity
  by exactly the concurrency factor. If the window is 30% of wall clock, the
  DAG is not what is slow - dbt's startup and parse are, and `--time-parse`
  measures that directly with zero queries.
- **total database work**, which is the *work*, not the elapsed cost. Divided
  by achieved concurrency it approximates elapsed time; on its own it says how
  much the engine is doing, not how long you wait.
- **where the database time is.** If three nodes hold 70% of it, read those
  three models. A materialization search over the other twenty cannot find
  anything, because there is nothing there to find.
- **achieved concurrency** against the configured threads.

Act on the verdict before buying anything else:

| what it shows | what to do |
|---|---|
| the window is a small share of wall clock | **Stop.** Report the decomposition. Optimizing this DAG under these conditions cannot pay back. Say what would change it - a larger scale factor, the other adapter. |
| database time concentrated in a few nodes | Read those models. Go straight to targeted rewrites; skip the broad search. |
| database time large and spread out | The probe below is worth its run. Continue. |

A negative result delivered in four runs is a good outcome. The same negative
result delivered in twenty-five is a bad one, and the difference is entirely
this step.

### 3. Buy the cost model in one run

Per-model `execution_time` from a normal run is nearly useless on a
view-heavy DAG: a view costs ~0ms to create and its real work is billed to
whichever downstream table finally materializes it. So the profile you need
is not there.

One probe run buys it. Materialize **every** model as a table (temporarily,
e.g. `+materialized: table` at the project level in `dbt_project.yml`), run
once, and read per-model times out of `run_results.json`:

```bash
python3 scripts/bench.py --project P --label probe_all_tables --runs 1 --warmups 0
```

Now you have an isolated cost `c(v)` for every node. From that plus the
expansion counts you can *estimate* any candidate plan without running it:

```
cost(plan) ~ sum over nodes v of c(v) * expansions(v under plan)
           + write cost of each node materialized under plan
```

parallelized across `threads`, so the floor is roughly
`max(total_work / threads, critical_path)`. Use this to rank candidates,
never as a prediction: engines pipeline and fuse operators, so isolated costs
do not add up exactly, and a materialized node pays a write the estimate only
approximates. It gets the *ordering* right, which is all you need to decide
what to measure. The all-tables plan is itself a legitimate candidate - keep
its measurement.

### 4. Derive your own budget

Nobody knows in advance how much optimization is available here. Work it out,
and say so out loud:

- **Ceiling.** From the probe, the redundant work is
  `sum over views v of c(v) * (expansions(v) - 1)`. Compare that to baseline
  runtime. If it is 3% of the DAG, materialization placement is not the
  lever - stop and look at SQL and adapter settings instead. If it is 60%,
  it is worth several trials. Cap the ceiling at the measured node execution
  window: an estimate larger than that means the engine is already fusing the
  shared work, or running it concurrently, not that the headroom exists. And subtract dbt's
  own per-node overhead, which on a small DAG can be most of the wall clock -
  compare `wall_s` against `db_s` before believing any of this is about SQL.
- **Price of a trial.** One measurement point at n runs costs
  `n * baseline_runtime` seconds and `n * queries` statements. You measured
  both in step 2.
- **Payback.** A change saving `S` per run, found by spending `B` seconds of
  measurement, repays itself after `B / S` runs. **Both terms are measured,
  so always compute this - never leave it as a question for the user.** `B`
  is the ledger total (`bench.py` accumulates it, warmups included); `S` is
  the wall-clock delta. `bench.py --compare` prints the answer on every
  comparison.

  How often the DAG actually runs is *not* an input to that calculation. It
  is what you compare the answer against: "payback in 16 runs" is a fact you
  computed, "and this DAG runs nightly, so that is two weeks" is the
  judgement on top of it. If you do not know the run frequency, still report
  the payback number, then say what it implies at a few plausible
  frequencies. Reporting "I cannot compute payback without knowing the
  schedule" is wrong - the schedule was never the denominator.
- **Stop rule.** Stop when the best remaining candidate's estimated gain is
  below the cost of measuring it, or below 2 sigma, or when the budget you set
  is spent. Say which one stopped you.

**Keep a ledger, in the report, updated as you go:**

```
spent   4 baseline + 1 probe                        =  5 runs
budget 15 runs; remaining 10
next   materialize order_line_facts (est. -0.4s)    =  3 runs
```

Before each candidate, check three things and write the answer down:

1. Is the estimated gain above 2 sigma? If not, measuring it cannot tell you
   anything - skip it.
2. Is the estimated gain above the cap from step 2b? If your estimate exceeds
   the measured node execution window, the estimate is wrong, not the
   opportunity.
3. **At current spend, what payback would this change need?** `runs_spent *
   baseline_runtime / estimated_saving`. If the answer already exceeds the
   DAG's plausible lifetime run count, stop - you cannot get the money back
   even if the candidate works perfectly.

That third check is the one that actually binds, and it tightens as you
spend: every trial makes the *next* trial harder to justify. When two
consecutive candidates fail, the prior on the third is worse, not the same -
stop earlier than feels natural. Overrunning a budget to find a change you
then cannot pay for is worse than reporting the negative result.

State the budget before spending it, and report what you actually spent -
including verification, which is not free: a fingerprint pass is a full scan
per model, so on a 25-model project it costs as much as a DAG run or more.
Budget one baseline fingerprint and one per kept change, not one per
candidate.

### 5. Establish the value noise floor, then fingerprint

Before trusting any equivalence check, find out whether the *unchanged*
project is even reproducible:

```bash
python3 scripts/fingerprint.py --project P --out .opt/fp_a.json
dbt run --full-refresh
python3 scripts/fingerprint.py --project P --out .opt/fp_b.json
python3 scripts/fingerprint.py --diff .opt/fp_a.json .opt/fp_b.json
```

It usually is not. Parallel floating-point aggregation is not associative, so
a rebuilt-but-identical project can move a `sum()` by 1e-5 relative. That is
the tolerance floor you have to work above: pass `--rtol` slightly above the
self-diff's worst relative difference, and record which models are unstable
and why. A model that is *exactly* reproducible must stay exactly
reproducible - do not blanket the whole project in the loosest tolerance.

Fingerprint the baseline once, then diff every candidate against it. See
`reference/equivalence.md` for what this does and does not prove; the short
version is that it falsifies rewrites, and the argument that a rewrite is
correct still has to be made about the SQL.

### 6. Change, measure, keep or revert

Work in priority order (`reference/candidates.md` has the full catalogue with
expected win, risk, and how to verify each):

1. **Free and safe first.** Static SQL fixes that cannot be wrong and need no
   search: drop intermediate `order by`, prune unread columns, hoist filters
   above joins, remove redundant `distinct`. Bundle them into one candidate
   and measure once.
2. **Materialization placement, including new cut points.** Ranked by the cost
   model, greedy: take the best-estimated single change, measure, keep it if it
   beats baseline by >2 sigma, then re-rank. Greedy over a ranked list beats
   enumerating 2^N plans, and the cost model means most of the enumeration
   happens for free. Remember that the candidate set is not limited to the
   existing nodes - a new model holding exactly the shared prefix, materialized,
   is often better than materializing any node the original author wrote. Plan
   changes need no equivalence argument, which makes them the cheapest wins to
   justify as well as to find.
3. **Adapter and engine settings.** Often the largest single win and among
   the cheapest to test - postgres `jit`, `work_mem`, `unlogged` intermediate
   tables, `ANALYZE` after materializing (postgres does not gather statistics
   for a CTAS result, so every downstream join plans blind), indexes on join
   keys; duckdb `threads`, `memory_limit`, `preserve_insertion_order`. And
   dbt's own `threads`, which changes no SQL at all.
4. **Structural rewrites.** Merging one-to-one view chains, replacing a
   self-join with a window function, turning correlated subqueries into
   joins. Highest risk to equivalence - argue each one, and re-fingerprint.

Bundle changes that are independent into one measurement, and bisect only if
the bundle regresses. That is the single biggest saving on signal 2.

After every kept change: `fingerprint.py --diff` against the baseline, and
`dbt build` (tests included) at least once before you call it done - the
project's own tests are free equivalence evidence you did not have to write.

### 7. Report

Say plainly:

- baseline vs final: wall clock (median, spread, n), statements, database time
- every change made, and *why it should be faster* - the mechanism, not just
  the number
- equivalence evidence: what was fingerprinted, at what tolerance, which
  models are unstable at baseline, what argument covers each rewrite
- what you spent: total runs, statements, wall time on measurement
- payback: runs before this is net positive
- what you did not try, and what it would cost to try it

## Hard rules

Three of these follow from the constraints; the rest are about not fooling
yourself. Note what they do *not* forbid - see "what these do not rule out".

- **Do not make it fast by doing less work.** No sampling, no `limit`, no
  approximate aggregates (`approx_count_distinct` is not `count(distinct)`),
  no dropping rows a filter "obviously" never matches. This is constraint 4
  restated: a different relation is not a faster relation.
- **Separate cost claims from correctness claims.** Anything you learn from
  the data about *cost* - this table is small, this join fans out 8x, this
  filter keeps 2% of rows - is exactly what you should be using to choose a
  plan. But a fact about the data may not license a rewrite that is only
  *correct* because of it. "This join key happens to be unique here" is not a
  precondition; a `unique` test asserting it is. If you find such a property
  and want to rely on it, **add the test that enforces it**, so the project
  fails loudly rather than silently returning wrong data when it stops
  holding. See the precondition table in `reference/equivalence.md`.
- **The reasoning must transfer, even though the change need not.** Every
  optimization is specific to the project it is made in - that is fine. What
  must generalize is *why* it works: "materialize this node because it is
  recomputed on four paths and costs 0.5s per path" is a reason; "materialize
  this node because it was the answer last time" is a memorized answer. Never
  bake a constant discovered from the data into SQL.
- **Do not cache across runs by accident.** An `incremental` model makes the
  second run cheap by not rebuilding, which is a different thing from the DAG
  being faster - and it is only equivalent if the incremental predicate
  provably captures every new or changed row. It is a legitimate and often
  large win when the user's data really is append-only; it is the user's call,
  not yours. Propose it with the semantics spelled out, and keep any
  measurement full-refresh to full-refresh so the comparison stays honest.
- **Do not report a win inside the noise.** Report the spread with the median,
  always.
- **Do not drop a model (constraint 4).** Every original model still exists
  and still returns its relation - someone may be querying it directly. That
  also rules out `ephemeral` for an original model, since an ephemeral model
  has no relation at all. `fingerprint.py --diff` flags this as `DROPPED`.
- **Do not quietly break constraint 5.** A model declared `table` ends as a
  table. `fingerprint.py --diff` flags this as `UNTABLED`; do not ignore it.

### What these do not rule out

Be aggressive within them. In particular, all of the following are fine:

- **Adding models.** New intermediate models are allowed and are one of the
  sharpest tools available - see "adding models" in `reference/candidates.md`.
- **Changing any original model's SQL**, as long as the relation it returns is
  equivalent. Rewrite it completely if you have the argument.
- **Changing any materialization** except turning a declared table into a
  non-table.
- **Using the data to estimate cost.** Row counts, cardinalities, selectivity,
  fan-out - measure all of it and plan with it.
- **Adding tests, constraints, indexes, hooks, and adapter settings.** Adding
  a `unique` test to license a rewrite is not overhead, it is the licence.
- **Project-specific answers.** The optimized project is expected to look
  nothing like the original.

## Files

Paths below are relative to this skill's directory; invoke the scripts by
absolute path. They need only Python 3 and a working `dbt` - if `dbt` is not on
`PATH` (it often lives in a project's own virtualenv), pass `--dbt
/path/to/dbt`. Work on a copy of the project, not the checkout: the workflow
rebuilds relations and rewrites model files.


- `scripts/analyze.py` - static DAG analysis, expansion counts. Free.
- `scripts/decompose.py` - where the wall clock goes, from a log you already
  have. Free, and empowered to end the exercise.
- `scripts/bench.py` - measure a variant (wall clock, statements, db time);
  `--compare` two measurements with a noise verdict.
- `scripts/fingerprint.py` - per-model relation digests and tolerance-aware
  diffing, for falsifying rewrites.
- `reference/measurement.md` - measuring honestly, the cost model, budget math.
- `reference/candidates.md` - the optimization catalogue, per adapter.
- `reference/equivalence.md` - what the fingerprint proves, and the rewrite
  rules that need an argument instead.
