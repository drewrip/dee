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
in flight is recorded as an `overlap` skip naming what blocked it. Optimizations
count as jobs, because they execute the DAG too.

**Optimization** rewrites a DAG and, with `--save`, stores the result as a new
version attributed to the one it came from.

```bash
dee optimize pipeline --enable hmp,pushdown --save --explain explain.html
dee dag versions pipeline
```

```
optimized pipeline v1 in 1669ms using 2 dag run(s)
runtime 634ms -> 378ms (40.4% faster)
  pays for itself after 7 run(s)
```
