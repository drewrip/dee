# Measuring honestly, and paying for it

## What a dbt run actually costs

A `dbt run` sends far more statements than it has models. Per node dbt issues
a transaction, a `create or replace`, a `describe`/catalog lookup, and a
commit; before the run it lists relations in every schema it touches. On a
12-model project expect ~80 statements, most of them trivial. That matters for
signal 2 in one direction only: **cutting statement count is not where the
money is - cutting the expensive ones is.** `bench.py` reports both count and
total database time; when they disagree, believe database time.

`bench.py` gets both from the debug log dbt already writes to `logs/dbt.log`
(`SQL status: OK in N seconds`), so counting costs nothing extra. It deletes
the log before each run so the parse covers exactly one invocation.

## Wall clock is not the sum of node times

`run_results.json` gives `execution_time` per node, and on a view-heavy DAG it
lies about where the work is, for two reasons:

1. A `view` model is a `create or replace view` - microseconds. Its work is
   billed to whichever downstream table materializes it.
2. Nodes run concurrently across `threads`, so the sum exceeds wall clock, and
   the critical path may be much shorter than either.

So: use `execution_time` to see *where cost moved* between two variants (that
is what `bench.py --compare` prints), and use wall clock to decide whether the
DAG got faster.

And do not expect the two to reconcile. One measured change improved wall
clock by 6.0% while *increasing* total node execution time (5.572s -> 5.627s)
and leaving database work identical. It shortened the critical path: the same
work, arranged so the longest chain through the DAG got shorter. Above
concurrency 1, wall clock is governed by the critical path, and total work is
nearly irrelevant to it - which is also why adding a node can make a DAG
faster even though it adds work.

## The one-run cost model

Build every model as a table once (`+materialized: table` at the project root
in `dbt_project.yml`, one probe run). Now every node's `execution_time` is the
isolated cost of its own SQL over its inputs, `c(v)`, and you can estimate any
materialization plan without running it:

```
work(plan) = sum_v c(v) * expansions(v | plan)   +   write_cost(materialized nodes)
runtime    ~ max(work(plan) / threads, critical_path(plan))
```

`expansions(v | plan)` is what `analyze.py` computes: 1 if `v` is materialized,
otherwise the sum over its consumers - every distinct path through views is a
separate recomputation.

Where this model is wrong, and how to use it anyway:

- **Pipelining.** A view fused into its consumer can be cheaper than `c(v)`
  measured standalone, because the engine never materializes intermediate rows.
  So the model overestimates the benefit of materializing.
- **Predicate pushdown.** A filter in the consumer can be pushed into the view
  but not into a table you already wrote. Materializing can therefore make a
  node's own work *larger* than `c(v)`. Again the model overestimates.
- **Statistics.** On postgres, a freshly created table has no statistics until
  `ANALYZE`. Materializing without analyzing can make downstream joins choose
  worse plans, which the model does not see at all.
- **Memory.** Materializing relieves memory pressure and can turn a spilling
  hash join into an in-memory one - a step change the model does not predict.
- **Fixed per-node overhead.** Every materialized node costs dbt a
  transaction, a create, a catalog lookup and a commit regardless of how
  little data it moves. On a small DAG that overhead can exceed the database
  time entirely - one measured 12-model project spent 2.4s wall against 0.7s
  of database time, and the all-tables probe pushed wall to 4.5s while node
  work totalled 1.6s. Adding materializations is not free even when the write
  is trivial.

**Sanity-check the ceiling against reality.** The redundancy estimate
`sum_v c(v) * (expansions(v) - 1)` cannot exceed the elapsed node execution
window you actually measured at baseline. Compare against the window, not the
summed database time: statements run concurrently, so the sum overstates the
elapsed opportunity by the concurrency factor, and on a well-parallelized DAG
it can exceed wall clock entirely. When it does - and it often does by 2x or
more - the engine is already sharing or fusing that work in the single fused
plan, and the headroom is not there. Cap your ceiling at measured baseline
`db_s`, and treat a wildly larger estimate as evidence the baseline plan is
better than the model assumes, not as a promise.

Which is why it is a **ranking** device. Use it to pick the two or three
candidates worth a real measurement, and let the measurement decide.

## Reading a comparison

`bench.py --compare A.json B.json` prints the delta, the spread, and a
verdict:

```
wall  -1.24s (18.3% faster)   REAL (2*sd = 0.31s)
query +1 statements
db    -0.98s
```

`WITHIN NOISE` means exactly that: you learned nothing except that the effect
is smaller than your resolution. The verdict tests the delta against the
**standard error of the difference**, `sqrt(sd_a^2/n_a + sd_b^2/n_b)`, not
against one sample's `sd`. That distinction matters: `sd` describes how much a
single run varies, but what is uncertain is the gap between two estimates, and
that shrinks as `sqrt(n)`. Testing against `2*sd` is about `sqrt(n)` too
strict and buries small real effects. When the verdict is `WITHIN NOISE` it
prints how many runs per side would resolve an effect of the size you just
measured - use that number rather than guessing.

Two projects in one evaluation sat below the threshold at n=2 and came back
`REAL` at n=8, at -5.8% and -6.0%. Both were genuine. So "within noise" is a
statement about your budget, not about the change.

**Interleave, and balance the order.** When the effect is small, measuring all
of A and then all of B lets machine drift masquerade as the effect. Alternate
in blocks - but alternate the *order* too, ABBA rather than ABAB. If the
candidate always runs second within a block, any within-block warming lands
entirely on the candidate and reads as a win. One measurement in this
evaluation had to be redone for exactly that reason; it survived, but the
ABAB design could not have shown that.

Sources of noise to control before adding runs:

- **Cold vs warm cache.** The first build of a session reads from disk; later
  ones hit the page cache. `--warmups 1` absorbs it. Never compare a cold
  baseline against a warm candidate, and never let one variant always occupy
  the warmer position - see the ABBA note above.
- **Other load.** The database is fixed but not necessarily idle. Interleave
  candidates (A, B, A, B) if the machine drifts.
- **Partial parse.** dbt's `target/partial_parse.msgpack` makes later parses
  faster; the first run after editing a model reparses. Warmups absorb this
  too, and it is dbt-side time, not database time - if wall clock moves but
  `db_s` does not, suspect parsing rather than the DAG.

## Budget arithmetic

Let `T` be baseline runtime, `Q` baseline statements, `n` runs per
measurement point.

- One measurement point costs `n*T` seconds and `n*Q` statements.
- The probe (step 3) costs one run and buys the cost model for every candidate.
  It is the highest-value run in the whole process; spend it early.
- A candidate that the cost model ranks below an already-kept change is not
  worth measuring - the ranking is free, the measurement is not.
- A change saving `S` per run costs `B` seconds to find and pays back after
  `B/S` runs of the DAG in production. `bench.py` maintains the ledger that
  makes `B` a real number rather than a recollection: every measurement point
  appends its runs, seconds, and statements to `.opt/ledger.json`, warmups
  counted, and `--compare` divides. Compute it on every comparison, not at the
  end of the exercise - it is the number that decides whether to take the next
  measurement, and it gets worse with every trial you spend.

A defensible default when nobody knows the run frequency: keep total
measurement under ~15 full DAG runs, and require the final result to pay back
within ~20 production runs. State it as an assumption, and say what a larger
budget would have bought - if the cost model still shows unclaimed headroom
when you stop, say how much and what it would cost to chase.

Diminishing returns are real: the first materialization decision usually
captures most of the available win, and the fifth is usually noise. Track the
cumulative improvement per run spent; when the last two trials bought less
than the next trial costs, stop.

## Hygiene

- `dbt show` (used by `fingerprint.py`) **overwrites `target/run_results.json`**.
  Take timings first, or copy the file aside.
- Keep every measurement JSON under `.opt/measurements/` so the final report
  can cite real numbers rather than recollection.
- Snapshot the original `models/` tree before editing (`git stash`, a copy, or
  a branch), so reverting a candidate is free and the final diff is legible.
- Record the exact `--command` used. A `run` baseline compared against a
  `build` candidate is a meaningless number that looks like a regression.
