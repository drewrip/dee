"""Backend infrastructure lifecycle.

A :class:`Backend` owns whatever has to exist before DAGs can run against it,
and is responsible for taking it down again. Lifecycle is *session scoped*, not
per cell: standing a Postgres instance up costs far more than a DAG run, so the
scheduler groups cells by backend and scale factor and prepares each dataset
once (see :func:`dee_bench.matrix.schedule`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..sampler import CgroupReader


@dataclass
class BackendContext:
    """What the runner needs to know about a live backend."""

    name: str
    # Connection details to write into dee's connections.json. None for
    # DuckDB, which is an in-process file.
    postgres: dict[str, Any] | None = None
    # Sampled alongside the dee process when the engine is a separate,
    # containerized server.
    cgroup: CgroupReader | None = None
    # What the backend's per-operator plan timings actually measure.
    plan_time_basis: str = "cpu_time"


class Backend:
    """Lifecycle for one benchmark backend."""

    name: str = "base"

    def setup(self) -> BackendContext:
        """Bring the backend up. Called once per run, before any cell."""
        raise NotImplementedError

    def prepare_scale(self, project: str, sf: float, prepared) -> None:
        """Load `project` at `sf` into the backend.

        Called once per (project, sf) group rather than once per cell.
        """

    def teardown(self) -> None:
        """Tear the backend down. Must be safe to call more than once."""

    def describe(self) -> str:
        return self.name
