"""Shared reads over a run directory's parquet, for every page builder.

A page builder never fails a dashboard: a table that a partial run has not
written yet, or a column an older run predates, comes back as an empty result
rather than an exception, and the page says what is missing instead.
"""

from __future__ import annotations

import json
from typing import Any


def tables(con) -> set[str]:
    return {r[0] for r in con.sql("SHOW TABLES").fetchall()}


def rows(con, sql: str, params: list[Any] | None = None) -> list[tuple]:
    """Run `sql`, or return no rows if the data it needs is not there.

    A missing table and a missing column are both normal states of a partial or
    older run directory, and neither is worth losing a whole dashboard over.
    """
    try:
        return con.sql(sql, params=params).fetchall() if params else con.sql(sql).fetchall()
    except Exception:  # noqa: BLE001
        return []


def columns(con, table: str) -> set[str]:
    return {r[0] for r in rows(con, f"DESCRIBE {table}")}


def has_columns(con, table: str, *names: str) -> bool:
    return set(names) <= columns(con, table)


def projection(con, table: str, names: list[str], alias: str = "") -> str:
    """A select list over `names`, with NULL standing in for what is absent.

    Result directories outlive schema additions: a run recorded before
    `optimization_mode` existed has no such column, and a query naming it fails
    outright — taking a whole page with it rather than the one value it could
    not have. Selecting NULL under the expected name keeps every reader working
    against one shape.
    """
    have = columns(con, table)
    prefix = f"{alias}." if alias else ""
    return ", ".join(f"{prefix}{name}" if name in have else f"NULL AS {name}"
                     for name in names)


def dicts(con, sql: str) -> list[dict[str, Any]]:
    """Rows as dicts, keyed by column name."""
    try:
        result = con.sql(sql)
        names = result.columns
        return [dict(zip(names, row)) for row in result.fetchall()]
    except Exception:  # noqa: BLE001
        return []


def loads(text: str | None) -> dict[str, Any]:
    """Parse a JSON detail column, tolerating null and malformed values."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def median(values: list[float]) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2


def short_node(node_id: str | None) -> str:
    """`"warehouse"."main"."churn_risk"` -> `churn_risk`.

    Node ids are fully qualified and three times longer than the part that
    distinguishes them, which is the part a tick label has room for.
    """
    if not node_id:
        return "-"
    part = str(node_id).split(".")[-1]
    return part.strip('"') or str(node_id)


def cell_label(project: str, backend: str, sf: float, variant: str | None = None,
               newline: bool = False) -> str:
    """How a cell names itself on an axis, in a legend and in a table."""
    sep = "\n" if newline else " · "
    head = f"{project}{sep}{backend} sf{sf:g}"
    return f"{head}{sep}{variant}" if variant else head


def fmt_ms(ms: float | None, digits: int = 2) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000.0:.{digits}f}s"


def fmt_num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"
