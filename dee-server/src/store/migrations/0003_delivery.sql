-- How a run's tables reached the warehouse, and what each half of that cost.
--
-- A continuous optimization piggybacks on real runs, and a candidate that
-- overruns its budget is now cancelled mid-run and the run finished under the
-- search's incumbent. The consumer still gets its tables, so the run succeeded
-- -- but its wall time measures neither DAG: the second half started from a
-- warm, half-built warehouse. Every query that compares run times must filter
-- `delivery = 'direct'`, or a resumed run's warm-start time is read as the
-- DAG getting faster.
--
-- 'direct' rather than NULL for runs recorded before this column existed: they
-- were all direct, and a NULL would drop out of that filter. DuckDB refuses a
-- constraint on an added column, so the default carries the backfill and the
-- writer always supplies a value.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS delivery VARCHAR DEFAULT 'direct';

-- What the cancelled candidate had spent when it was stopped, and what the
-- resume that finished the run took. NULL on a direct delivery: there was no
-- cancelled half, which is not the same as one that took no time.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS trial_elapsed_ms BIGINT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS resume_elapsed_ms BIGINT;
