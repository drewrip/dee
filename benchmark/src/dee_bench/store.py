"""Incremental, crash-safe parquet result storage.

Each cell writes its own fragment under
``results/<table>/cell_id=<cell_id>/part-<n>.parquet``. Nothing rewrites an
existing fragment, so a crash — or a `kill -9` mid-sweep — can only ever lose
the cell that was in flight. Everything already written stays readable, which
is what lets `dee-bench viz` render a partial run and `dee-bench resume` pick
up where the worker stopped.

Rows arrive as plain dicts and are coerced against the declared
:mod:`dee_bench.schema` types, so a column that is absent or ``None`` still
lands with the right type rather than poisoning the dataset with a null column
parquet cannot merge later.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import BY_NAME, Table, Verbosity


class ResultStore:
    """Writes result tables for one benchmark run directory."""

    def __init__(self, run_dir: str | Path, verbosity: Verbosity = Verbosity.STANDARD):
        self.run_dir = Path(run_dir)
        self.results_dir = self.run_dir / "results"
        self.verbosity = verbosity
        self._counters: dict[tuple[str, str], int] = {}

    # -- writing -----------------------------------------------------------

    def records(self, table: str) -> bool:
        """Whether this run's verbosity records `table` at all."""
        t = BY_NAME[table]
        return not t.derived and t.min_verbosity <= self.verbosity

    def write(self, table: str, rows: Iterable[dict[str, Any]], *, cell_id: str | None = None) -> int:
        """Append `rows` to `table`. Returns the number of rows written.

        Silently does nothing when the run's verbosity does not record this
        table, so callers can always emit their rows without checking first.
        """
        t = BY_NAME[table]
        if t.derived:
            raise ValueError(f"{table} is derived; write it with write_derived()")
        if not self.records(table):
            return 0
        rows = list(rows)
        if not rows:
            return 0
        return self._write_fragment(t, rows, cell_id)

    def write_derived(self, table: str, rows: Iterable[dict[str, Any]]) -> int:
        """Write a derived table (produced by `analyze`, not by the runner)."""
        t = BY_NAME[table]
        if not t.derived:
            raise ValueError(f"{table} is not derived; write it with write()")
        rows = list(rows)
        if not rows:
            return 0
        # Derived tables are recomputed wholesale, so replace rather than append.
        out_dir = self.results_dir / t.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("*.parquet"):
            stale.unlink()
        self._write_table(pa.Table.from_pylist(_coerce_rows(t, rows), schema=t.arrow_schema),
                          out_dir / "part-0.parquet")
        return len(rows)

    def _write_fragment(self, t: Table, rows: list[dict[str, Any]], cell_id: str | None) -> int:
        if t.partition_by:
            if cell_id is None:
                cell_id = rows[0].get("cell_id")
            if cell_id is None:
                raise ValueError(f"{t.name} is partitioned by cell_id, but no cell_id was given")
            out_dir = self.results_dir / t.name / f"cell_id={cell_id}"
        else:
            out_dir = self.results_dir / t.name
        out_dir.mkdir(parents=True, exist_ok=True)

        key = (t.name, cell_id or "")
        n = self._counters.get(key, 0)
        self._counters[key] = n + 1
        self._write_table(
            pa.Table.from_pylist(_coerce_rows(t, rows), schema=t.arrow_schema),
            out_dir / f"part-{n}.parquet",
        )
        return len(rows)

    @staticmethod
    def _write_table(table: pa.Table, path: Path) -> None:
        """Write atomically, then fsync, so a crash never leaves a torn file.

        Results are the whole point of a long benchmark run: a fragment is
        either completely there or not there at all.
        """
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    # -- reading -----------------------------------------------------------

    def glob_for(self, table: str) -> str:
        """A duckdb-readable glob for `table`, or '' if nothing was written."""
        t = BY_NAME[table]
        root = self.results_dir / t.name
        if not root.exists():
            return ""
        pattern = "**/*.parquet" if t.partition_by else "*.parquet"
        return str(root / pattern) if any(root.glob(pattern)) else ""

    def has(self, table: str) -> bool:
        return bool(self.glob_for(table))

    def cell_ids_written(self, table: str = "runs") -> set[str]:
        """Which cells already have data in `table`."""
        root = self.results_dir / table
        if not root.exists():
            return set()
        return {
            d.name.split("=", 1)[1]
            for d in root.iterdir()
            if d.is_dir() and d.name.startswith("cell_id=") and any(d.glob("*.parquet"))
        }


def _coerce_rows(t: Table, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill missing columns with None and drop unknown ones.

    Being strict here means a schema change can never produce fragments with
    incompatible column sets that fail to merge at read time.
    """
    names = t.column_names
    known = set(names)
    out = []
    for row in rows:
        unknown = set(row) - known
        if unknown:
            raise ValueError(
                f"{t.name}: unknown column(s) {', '.join(sorted(unknown))}; "
                f"expected only {', '.join(names)}"
            )
        out.append({name: row.get(name) for name in names})
    return out


def connect(run_dir: str | Path):
    """A duckdb connection with every written result table registered as a view.

    Lets analysis and visualization query results by name::

        con = store.connect(run_dir)
        con.sql("SELECT variant, median(engine_wall_ms) FROM runs GROUP BY 1")
    """
    import duckdb

    st = ResultStore(run_dir, Verbosity.FULL)
    con = duckdb.connect()
    for name in BY_NAME:
        glob = st.glob_for(name)
        if glob:
            # CREATE VIEW cannot take a prepared parameter, so the path is
            # inlined; quotes are doubled to keep the literal well-formed.
            literal = glob.replace("'", "''")
            con.execute(
                f"CREATE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{literal}', union_by_name=true)"
            )
    return con
