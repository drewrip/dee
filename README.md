# dee

An experimental SQL transformation orchestrator


## Dependencies

We have a dependency on both the `datafusion` and `duckdb` crates. However, we currently
require a recent git commit of `datafusion` that fixes a crucial bug for this project.
Related to this we also then require a slightly modified version of the `duckdb` crate.
This just updates the `duckdb-rs` version of `arrow` to 59 which is required for compatibility
with the most recent `datafusion`. These will be replaced with dependencies on versions
of the proper upstream crates once they are compatible again.
