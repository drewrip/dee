-- Optimizations registered on a DAG.
--
-- Before the server, an optimization was something you invoked: `dee-cli opt`
-- ran every pass to completion and exited. That is still available as
-- `dee optimize`, but it is no longer the only shape. A continuous
-- optimization is *attached* to a DAG and steps around each of its runs,
-- improving it over the DAG's lifetime. This table is what "attached" means.
--
-- The server reads it on every run to decide which optimizations to step and
-- on which side, so it holds exactly what that decision needs -- and nothing
-- about how any particular optimization searches, which lives in tables the
-- optimization creates for itself.
CREATE TABLE IF NOT EXISTS dag_optimizations (
    dag_id            VARCHAR     NOT NULL,
    -- 'hmp', 'omp', 'pushdown'.
    name              VARCHAR     NOT NULL,
    -- 'continuous' or 'once', as the optimization itself reports. Stored
    -- rather than looked up so a row explains itself, and so a listing does
    -- not have to construct every optimization to describe it.
    optimization_type VARCHAR     NOT NULL,
    -- 'before', 'after' or 'both'. The optimization author's default unless
    -- the registration overrode it.
    step_phase        VARCHAR     NOT NULL,
    -- The OptimizerConfig this optimization was registered under, as JSON.
    -- Pinned at registration: a search that changed its parameters halfway
    -- through would be comparing measurements taken under different rules.
    config            VARCHAR,
    -- Tables the optimization's `register` reported creating. Recorded so a
    -- listing can say what a registration owns without asking the pass.
    tables            VARCHAR[],
    -- Set once the optimization reports it has converged, so a finished
    -- search stops being stepped without being torn down -- its state and
    -- trial history stay readable.
    finished_at       TIMESTAMPTZ,
    -- The version a converged optimization promoted, if it promoted one.
    result_version    INTEGER,
    registered_at     TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL
);

-- One registration per optimization per DAG: registering twice is how a
-- restart re-establishes what a DAG already had, not how you get two searches
-- competing over the same tables.
CREATE UNIQUE INDEX IF NOT EXISTS ux_dag_optimizations
    ON dag_optimizations (dag_id, name);
