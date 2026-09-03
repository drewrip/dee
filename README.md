# dee

An experimental SQL transformation orchestrator with an optimizing DAG runner.

dee is a server. You start it, register the warehouses it can reach, submit
DAGs, and put them on a schedule. It runs them, and records everything it
observed — every run, every node's timing and plan, every optimizer decision —
into a DuckDB metadata database you can query directly.

That history is the point. dee's optimizer decides which intermediate views are
worth materializing by *measuring* the DAG, and those measurements are only
worth their cost if they can be amortized across the many times a DAG runs.

## Quick start

```bash
cargo build --release

# Start the server. It keeps its state in ~/.dee/dee.duckdb by default.
./target/release/dee serve &

# Tell it about a warehouse.
dee connection add wh --type duckdb --database warehouse.duckdb
dee connection test wh

# Submit a DAG and run it once.
dee dag submit test.json --name churn --target wh
dee trigger churn --wait

# Put it on a schedule and walk away.
dee schedule set churn --cron "0 3 * * *" --timezone America/New_York
```

Every command talks to the server over HTTP. Point it elsewhere with
`--server` or `$DEE_SERVER`. The two exceptions are `dee draw` and
`dee convert`, which are pure local file transformations.

## Overview

**Connections** are named warehouses. They live in the metadata database, so
they survive restarts and can be edited at runtime. An existing
`connections.json` imports directly:

```bash
dee connection add -f connections.json
```

**DAGs** are content-addressed and versioned. Submitting a definition that
hashes to a version dee already has is a no-op that returns the existing
version, so resubmitting from CI does not grow history. The hash ignores things
that carry no meaning — node order, dependency order — so an optimizer round
trip does not look like a change.

```bash
dee dag submit pipeline.json --name pipeline --target wh
dee dag versions pipeline
dee dag graph pipeline -o pipeline.svg
```

A DAG also carries the optimizer settings it is meant to be worked on under —
which passes to run and their parameters — so a benchmark cell is submitted in
one shot rather than repeated on every command that touches it:

```bash
dee dag submit pipeline.json --name pipeline --target wh \
    --enable omp --omp-top 3 --omp-node-centrality paths --omp-exhaust

dee dag optimizer pipeline                       # show
dee dag optimizer pipeline --enable hmp,omp      # replace
dee dag optimizer pipeline --clear               # back to dee's defaults
```

`--optimizer-config settings.json` reads the whole thing from a file, and
individual flags override it — which is how a sweep generates one cell per
parameter combination. The settings live on the DAG, not on a version, so
resubmitting a definition under new settings does not mint a version, and
resubmitting without any settings leaves the stored ones alone.

**Runs** record what happened. A trigger produces a *run group*: one run
normally, or a whole series when you ask for repetitions.

```bash
dee trigger pipeline --warmups 1 --repeat 5 --wait
dee runs list --dag pipeline
dee runs nodes <run-id>
dee runs report <run-id> --html -o profile.html
```

Repetitions execute back to back inside one server against one already-warm
connection pool, so their timings measure the DAG rather than process startup
and pool construction.

**The queue** runs a DAG N times in succession. Each entry is its own run
group, started only once the one before it has finished.

```bash
dee queue add pipeline -n 20                 # twenty runs, back to back
dee queue add pipeline -n 20 --repeat 3 --warmups 1 --wait
dee queue list --dag pipeline
dee queue drop <run-group-id>
dee queue clear --dag pipeline
```

This is not the same thing as `--repeat`. A run group pins one version and
executes under one driver, so nothing that happens between its repetitions can
change what it runs — with one exception, a continuous optimization registered
on the DAG, which is *meant* to change what it runs and records the version
each repetition actually executed. Queue entries are separate groups, so an entry that did
not name a `--version` resolves to whatever version is current *when its turn
comes*. Submit a new version halfway through a queue of twenty and the
remaining runs execute it, so one `dee runs list` shows the DAG before and
after the change under otherwise identical conditions:

```bash
dee queue add pipeline -n 20
dee dag submit rewritten.json --name pipeline   # lands mid-drain, as v2
dee runs list --dag pipeline --limit 40         # v1 before it, v2 after
```

Pass `--version` to pin an entry instead.

Enqueueing never conflicts — it is `dee trigger` that refuses while a DAG is
busy, and the queue is where that run goes instead. Nothing cuts the line
either: a manual trigger, a scheduled window and `dee optimize` are all refused
while a queue is draining, since all three execute the DAG against the same
warehouse. Entries still waiting when the server stops are marked `orphaned` on
the next start rather than replayed, the same no-catchup rule the scheduler
follows.

**Schedules** are a cron expression and an IANA timezone.

```bash
dee schedule set pipeline --cron "0 * * * *" --timezone UTC
dee schedule list
dee schedule pause pipeline
dee schedule skips pipeline
```

dee does not catch up. A window that elapses while the server is down is
recorded as skipped and never replayed — coming back from an overnight outage
gives you one run, not eight hours of backlog. `dee schedule skips` shows every
window that produced nothing and why, so "the schedule did nothing" is never
indistinguishable from "the server was down".

At most one job runs per DAG at a time. A window that collides with a job still
in flight is recorded as an `overlap` skip naming what blocked it. A
`dee optimize` counts as a job, because it executes the DAG too. A registered
continuous optimization does not: it spends no runs of its own, so there is
nothing for it to collide with.

**Optimization** rewrites a DAG and, with `--save`, stores the result as a new
version attributed to the one it came from. With no flags it runs under the
DAG's own settings; a flag overrides that one setting for that one run and
leaves the rest alone.

```bash
dee optimize pipeline --save --explain explain.html   # the DAG's settings
dee optimize pipeline --omp-top 8                     # ...with one changed
dee dag versions pipeline
```

Every optimization begins by printing the configuration it resolved to, since
with two places settings can come from, "what did this actually run" should not
be something you have to reconstruct. A DAG with no settings of its own falls
back to dee's defaults — HMP and OMP on, pushdown and parallelism off — so
`--enable`/`--disable` is still how you pin the pass set explicitly.

```
optimized pipeline v1 in 1669ms using 2 dag run(s)
runtime 634ms -> 378ms (40.4% faster)
  pays for itself after 7 run(s)
```

**ParallelismTuning** tunes one thing: how many of the DAG's nodes run at once.
dee's default is no limit, which assumes more concurrency is never worse. On a
warehouse that already parallelizes inside a single query that assumption is
backwards — one heavy node can saturate the machine on its own, and three of
them at once make each slower rather than the set faster. On one that cannot
fill the box with a single query, the opposite holds. Neither answer is safe to
hardcode, so it ladders over caps and measures.

```bash
dee optimize pipeline --enable parallelism --parallelism-ladder 1,2,4,8
```

It measures the DAG's current setting a couple of times, then tries each cap
that could actually mean something on this DAG — one at or above the node count
can never bind, and is not paid for. A cap is accepted only when *every*
measurement of it beats *every* measurement of the incumbent, which costs one
extra run per promising rung and is what stops a setting that is merely
sometimes fast from being adopted. What it settles on rides along with the DAG
as `metadata.max_parallelism`, so every later run inherits it.

Run it **before** the materialization passes, which is where
`--enable parallelism,hmp` already puts it: a materialization speedup measured
against an untuned concurrency setting is crediting HMP with a win that belongs
to the ladder.

**Continuous optimization** is the other way to reach the same thing. `dee
optimize` above is a job: it decides now, and it buys the DAG runs its search
needs to decide. But dee is already running this DAG — on a schedule, from a
trigger, out of the queue — and those runs are measurements too. Registering an
optimization attaches it to the DAG so it steps around each of those runs
instead, proposing a candidate before one and learning from it after.

```bash
dee optimization available                    # what dee can register, and how each behaves
dee optimization register pipeline hmp        # attach it
dee optimization list pipeline
dee optimization deregister pipeline hmp      # detach it, dropping its state
```

It costs no runs of its own. The DAG is a little slower while the search is
exploring, and faster once it has converged; when it does, it stores its result
as a new version exactly as `dee optimize --save` would, and everything after
runs that. `dee optimization list` says whether it is still `stepping` or has
`converged`, and which version it promoted.

```
NAME       TYPE        PHASE   STATE     TABLES
hmp        continuous  both    converged opt_hmp_state, opt_hmp_trials
```

Which of the two applies is a property of the optimization, not a preference.
HMP, OMP and ParallelismTuning decide by measuring the DAG run, so they are
`continuous` and step around runs. Pushdown decides from the DAG in front of
it, so it is `once`: there is no measurement to wait for and nothing a later
run could teach it.
`dee optimize` drives either — for a continuous optimization it supplies the
runs itself rather than waiting for the schedule to provide them, which is why
it is still the right command when you want an answer today.

A continuous optimization keeps its state in the metadata database, in tables it
creates itself and that `deregister` drops. That is what lets a search survive a
restart: what to try on this run is decided by reading what previous runs
measured, so nothing is lost when the process goes away. Those tables are the
optimization's own — it can reach nothing else in that database, which also
holds every run dee has recorded and every connection's credentials.

Because it spends no runs of its own, a continuous optimization does not take
the exclusive per-DAG claim that `dee optimize` does, and does not block the
schedule it is improving.
