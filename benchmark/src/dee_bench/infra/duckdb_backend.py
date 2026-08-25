"""DuckDB backend: an in-process engine over a file, so there is nothing to run.

Data preparation happens in :mod:`dee_bench.workload`, which copies the cached
warehouse for the requested scale factor into each cell's scratch directory.
The only lifecycle concern here is that DuckDB's work shows up in the dee
process tree, which the process sampler already covers.
"""

from __future__ import annotations

from .base import Backend, BackendContext


class DuckDBBackend(Backend):
    name = "duckdb"

    def setup(self) -> BackendContext:
        # DuckDB runs inside dee-cli, so `harness_process` sampling captures
        # it in full and no container sampling is needed.
        return BackendContext(name=self.name, plan_time_basis="cpu_time")

    def describe(self) -> str:
        return "duckdb (in-process)"
