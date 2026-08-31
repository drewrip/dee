-- Bookkeeping for this file's own application.
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    INTEGER PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL
);

-- Which server wrote these rows, and whether it exited cleanly. The startup
-- orphan sweep keys off this: a run still 'running' whose instance never
-- recorded a stopped_at was killed, not finished.
CREATE TABLE IF NOT EXISTS server_instances (
  instance_id VARCHAR PRIMARY KEY,
  pid         INTEGER NOT NULL,
  version     VARCHAR NOT NULL,
  bind        VARCHAR NOT NULL,
  started_at  TIMESTAMPTZ NOT NULL,
  stopped_at  TIMESTAMPTZ
);

-- `config` is the full serde-tagged `dee::connections::Connection` JSON, so
-- deserializing it is the only decode path and adding a connector variant
-- needs no migration. `kind` is denormalized out of the tag purely so listing
-- never has to parse JSON. `config_hash` keys the live connector cache, so
-- editing a connection invalidates its cached pool with no explicit eviction.
CREATE TABLE IF NOT EXISTS connections (
  name        VARCHAR PRIMARY KEY,
  kind        VARCHAR NOT NULL,
  config      VARCHAR NOT NULL,
  config_hash VARCHAR NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS dags (
  dag_id          VARCHAR PRIMARY KEY,
  name            VARCHAR NOT NULL UNIQUE,
  description     VARCHAR,
  current_version INTEGER NOT NULL,
  default_target  VARCHAR,
  created_at      TIMESTAMPTZ NOT NULL,
  updated_at      TIMESTAMPTZ NOT NULL
);

-- Content-addressed and immutable. `content_hash` is sha256 over a canonical
-- DagFile, so resubmitting an unchanged DAG is idempotent and returns the
-- existing version instead of growing history forever -- which the benchmark
-- harness, and any CI, will do constantly. `origin='optimized'` with
-- `derived_from_version` makes "which version did the optimizer produce, from
-- what" answerable without joining through reports.
CREATE TABLE IF NOT EXISTS dag_versions (
  dag_id               VARCHAR NOT NULL,
  version              INTEGER NOT NULL,
  content_hash         VARCHAR NOT NULL,
  definition           VARCHAR NOT NULL,
  sql_dialect          VARCHAR,
  node_count           INTEGER NOT NULL,
  source_count         INTEGER NOT NULL,
  origin               VARCHAR NOT NULL,
  derived_from_version INTEGER,
  optimization_id      VARCHAR,
  created_at           TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (dag_id, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dag_versions_hash
  ON dag_versions (dag_id, content_hash);

-- Exploded nodes. Exists so the structural facts the benchmark's `dag_graph`
-- table needs are computed once server-side by `dee::graph::Graph` -- the same
-- code the optimizer uses -- rather than every client re-deriving them from
-- the definition JSON.
--
-- That distinction is not academic. `paths_to_sinks` here is
-- `Graph::paths_to_sinks`, which counts the materialization points (Table and
-- TempTable nodes) reachable from a node, and is exactly what OMP filters and
-- ranks candidates by. The harness's own reimplementation counts distinct
-- paths to childless nodes instead, ignoring materialization, so the two
-- disagree on any DAG with an intermediate table. Recording the optimizer's
-- number is the point: it is what explains the optimizer's choices.
CREATE TABLE IF NOT EXISTS dag_version_nodes (
  dag_id         VARCHAR NOT NULL,
  version        INTEGER NOT NULL,
  node_id        VARCHAR NOT NULL,
  materialize    VARCHAR NOT NULL,
  query_text     VARCHAR NOT NULL,
  depends_on     VARCHAR[] NOT NULL,
  out_degree     INTEGER NOT NULL,
  paths_to_sinks INTEGER NOT NULL,
  PRIMARY KEY (dag_id, version, node_id)
);

CREATE TABLE IF NOT EXISTS dag_version_sources (
  dag_id  VARCHAR NOT NULL,
  version INTEGER NOT NULL,
  name    VARCHAR NOT NULL,
  columns VARCHAR NOT NULL,
  PRIMARY KEY (dag_id, version, name)
);

-- One schedule per DAG. `next_fire_at` is materialized in UTC so the tick is
-- one indexed range scan and never parses cron for a DAG that is not due.
-- `catchup` exists, always false, so adding backfill later is a behavior
-- change rather than a migration.
CREATE TABLE IF NOT EXISTS schedules (
  dag_id         VARCHAR PRIMARY KEY,
  cron           VARCHAR NOT NULL,
  timezone       VARCHAR NOT NULL DEFAULT 'UTC',
  enabled        BOOLEAN NOT NULL DEFAULT true,
  catchup        BOOLEAN NOT NULL DEFAULT false,
  overlap_policy VARCHAR NOT NULL DEFAULT 'skip',
  target         VARCHAR,
  next_fire_at   TIMESTAMPTZ,
  last_fire_at   TIMESTAMPTZ,
  created_at     TIMESTAMPTZ NOT NULL,
  updated_at     TIMESTAMPTZ NOT NULL
);

-- Why a window produced no run. Without this, "the schedule silently did
-- nothing" is indistinguishable from "the server was down" -- the single most
-- common operational question about a catchup-free scheduler.
CREATE TABLE IF NOT EXISTS schedule_skips (
  skip_id         VARCHAR PRIMARY KEY,
  dag_id          VARCHAR NOT NULL,
  scheduled_for   TIMESTAMPTZ NOT NULL,
  detected_at     TIMESTAMPTZ NOT NULL,
  reason          VARCHAR NOT NULL,
  blocking_run_id VARCHAR,
  windows_skipped INTEGER NOT NULL DEFAULT 1,
  detail          VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_skips_dag ON schedule_skips (dag_id, scheduled_for);

-- A group is one trigger. A scheduled fire makes a group of one; a benchmark
-- trigger with warmups=2, repetitions=9 makes eleven runs sharing a group.
-- This is how `--repeat`/`--warmups` survives the move to a server: the
-- semantics become a property of the group, executed by one driver task
-- against one cached pool -- so a benchmark repetition and a scheduled
-- execution are the same kind of object, and both accumulate history.
CREATE TABLE IF NOT EXISTS run_groups (
  run_group_id       VARCHAR PRIMARY KEY,
  dag_id             VARCHAR NOT NULL,
  dag_version        INTEGER NOT NULL,
  target             VARCHAR NOT NULL,
  trigger            VARCHAR NOT NULL,
  scheduled_for      TIMESTAMPTZ,
  warmups            INTEGER NOT NULL DEFAULT 0,
  repetitions        INTEGER NOT NULL DEFAULT 1,
  cleanup_before     BOOLEAN NOT NULL DEFAULT true,
  collect_plans      BOOLEAN NOT NULL DEFAULT false,
  sample_interval_ms INTEGER,
  status             VARCHAR NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL,
  finished_at        TIMESTAMPTZ,
  instance_id        VARCHAR NOT NULL,
  error              VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_groups_dag ON run_groups (dag_id, created_at);

-- One row per DAG execution. Columns mirror ExecStats and DagRunProfile's
-- headline fields, so the benchmark's `runs` parquet table is a projection of
-- this, and "was the optimizer's change actually faster in production" is a
-- SQL query rather than a re-run. duration_ms is an integer, never chrono's
-- TimeDelta, which serializes as a [secs, nanos] pair.
CREATE TABLE IF NOT EXISTS runs (
  run_id                VARCHAR PRIMARY KEY,
  run_group_id          VARCHAR NOT NULL,
  dag_id                VARCHAR NOT NULL,
  dag_version           INTEGER NOT NULL,
  target                VARCHAR NOT NULL,
  phase                 VARCHAR NOT NULL DEFAULT 'measure',
  rep_index             INTEGER NOT NULL DEFAULT 0,
  status                VARCHAR NOT NULL,
  queued_at             TIMESTAMPTZ NOT NULL,
  started_at            TIMESTAMPTZ,
  finished_at           TIMESTAMPTZ,
  duration_ms           BIGINT,
  node_time_ms          BIGINT,
  node_count            INTEGER,
  rows_produced         BIGINT,
  cleanup_ms            BIGINT,
  peak_engine_mem_bytes BIGINT,
  db_size_bytes         BIGINT,
  plan_time_basis       VARCHAR,
  instance_id           VARCHAR NOT NULL,
  error                 VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_runs_group ON runs (run_group_id, phase, rep_index);
CREATE INDEX IF NOT EXISTS ix_runs_dag   ON runs (dag_id, queued_at);

CREATE TABLE IF NOT EXISTS node_executions (
  run_id        VARCHAR NOT NULL,
  node_id       VARCHAR NOT NULL,
  materialize   VARCHAR NOT NULL,
  started_at    TIMESTAMPTZ NOT NULL,
  finished_at   TIMESTAMPTZ NOT NULL,
  duration_ms   BIGINT NOT NULL,
  rows_produced BIGINT,
  -- Denormalized so the common listing query never touches the fat plans table.
  has_plan      BOOLEAN NOT NULL DEFAULT false,
  PRIMARY KEY (run_id, node_id)
);

-- Split out from node_executions: raw EXPLAIN JSON is one to two orders of
-- magnitude larger than everything else and is only read on demand.
CREATE TABLE IF NOT EXISTS plans (
  run_id      VARCHAR NOT NULL,
  node_id     VARCHAR NOT NULL,
  plan_format VARCHAR NOT NULL,
  plan_json   VARCHAR NOT NULL,
  PRIMARY KEY (run_id, node_id)
);

-- ExecStats.system_samples. `seq` preserves order without trusting timestamp
-- uniqueness at sub-millisecond sampling intervals.
CREATE TABLE IF NOT EXISTS run_samples (
  run_id        VARCHAR NOT NULL,
  seq           INTEGER NOT NULL,
  timestamp     TIMESTAMPTZ NOT NULL,
  elapsed_ms    BIGINT NOT NULL,
  cpu_percent   DOUBLE,
  memory_bytes  BIGINT,
  disk_bytes    BIGINT,
  read_bytes    BIGINT,
  written_bytes BIGINT,
  PRIMARY KEY (run_id, seq)
);

-- OptimizeReport's header. `report` keeps the full typed report verbatim so no
-- future field of OptimizeReport is silently lost, while the flattened columns
-- are what anything actually filters or aggregates on. `config` is the
-- serialized OptimizerConfig, which is what makes an optimization reproducible
-- and is the key a history-seeded optimizer would look it up by.
CREATE TABLE IF NOT EXISTS optimizations (
  optimization_id       VARCHAR PRIMARY KEY,
  dag_id                VARCHAR NOT NULL,
  source_version        INTEGER NOT NULL,
  result_version        INTEGER,
  target                VARCHAR NOT NULL,
  status                VARCHAR NOT NULL,
  started_at            TIMESTAMPTZ,
  finished_at           TIMESTAMPTZ,
  wall_ms               BIGINT,
  baseline_runtime_ms   BIGINT,
  final_runtime_ms      BIGINT,
  dag_runs_used         INTEGER,
  total_changes_applied INTEGER,
  nodes_before          INTEGER,
  nodes_after           INTEGER,
  config                VARCHAR NOT NULL,
  report                VARCHAR,
  explain_html          VARCHAR,
  instance_id           VARCHAR NOT NULL,
  error                 VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_opt_dag ON optimizations (dag_id, started_at);

-- PassReport, flattened. `detail` stays as the tagged PassDetail JSON:
-- Hmp/Omp/PushdownDetail have disjoint shapes, and flattening them into one
-- table would be roughly 25 mostly-null columns.
CREATE TABLE IF NOT EXISTS optimization_passes (
  optimization_id       VARCHAR NOT NULL,
  pass_order            INTEGER NOT NULL,
  pass_name             VARCHAR NOT NULL,
  started_at            TIMESTAMPTZ NOT NULL,
  finished_at           TIMESTAMPTZ NOT NULL,
  wall_ms               BIGINT NOT NULL,
  dag_runs_used         INTEGER NOT NULL,
  changes_applied       INTEGER NOT NULL,
  candidates_considered INTEGER NOT NULL,
  working_set_size      INTEGER NOT NULL,
  detail                VARCHAR NOT NULL,
  PRIMARY KEY (optimization_id, pass_order)
);

-- IterationStat: the optimizer's search trace, one row per candidate DAG.
-- Kept relational rather than buried inside `report` because this is the table
-- a history-seeded optimizer reads -- "for this dag and target, which combos
-- have already been measured, at what runtime" is then one GROUP BY away.
CREATE TABLE IF NOT EXISTS optimization_iterations (
  optimization_id VARCHAR NOT NULL,
  pass_order      INTEGER NOT NULL,
  -- Denormalized alongside pass_order so the trace reads on its own, and so
  -- the benchmark's pass_iterations table is a direct projection.
  pass_name       VARCHAR NOT NULL,
  iteration       INTEGER NOT NULL,
  runtime_ms      BIGINT NOT NULL,
  combo           VARCHAR[] NOT NULL,
  outcome         VARCHAR,
  cpu_seconds     DOUBLE,
  peak_rss_bytes  BIGINT,
  samples         VARCHAR,
  PRIMARY KEY (optimization_id, pass_order, iteration)
);

-- Backs `dee runs logs`. Deliberately coarse: one row per lifecycle transition
-- and per node completion, not a log firehose. UUIDv7 ids sort chronologically,
-- so no separate sequence column is needed.
CREATE TABLE IF NOT EXISTS events (
  event_id        VARCHAR PRIMARY KEY,
  run_id          VARCHAR,
  run_group_id    VARCHAR,
  optimization_id VARCHAR,
  dag_id          VARCHAR,
  ts              TIMESTAMPTZ NOT NULL,
  level           VARCHAR NOT NULL,
  message         VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_run ON events (run_id, event_id);
