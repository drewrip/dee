"""Backend infrastructure lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend, BackendContext
from .duckdb_backend import DuckDBBackend
from .postgres_backend import PostgresBackend

__all__ = ["Backend", "BackendContext", "DuckDBBackend", "PostgresBackend", "make_backend"]


def make_backend(
    name: str,
    config: dict[str, Any] | None = None,
    dag_bench: Path | None = None,
    fresh: bool = False,
    keep: bool = False,
    log=print,
) -> Backend:
    if name == "duckdb":
        return DuckDBBackend()
    if name == "postgres":
        return PostgresBackend(config, dag_bench=dag_bench, fresh=fresh, keep=keep, log=log)
    raise ValueError(f"unknown backend {name!r}")
