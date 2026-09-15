"""A page per optimization, built only for the optimizations a run enabled.

The seven studies ask the same questions of every sweep. These pages ask the
questions only one pass can answer — what HMP's search spent its budget on,
which rung of the parallelism ladder won and against what control, how much
duplicate work NodeFusion's rollup absorbed — and so they exist only when there
is a pass to ask them of.

Detection is from the results, not from the config: a page appears when the run
contains evidence that its pass ran. A sweep of HMP against a baseline grows an
HMP page and nothing else; adding `nodefusion` to one variant grows a second
page on the next `dee-bench viz`, with no flag to remember.
"""

from __future__ import annotations

from ..query import loads, rows, tables
from ..spec import Study
from ..theme import PASSES, PASS_BY_NAME
from . import hmp, nodefusion, omp, parallelism, pushdown

# Pipeline order, which is the order the pages appear in.
BUILDERS = {
    "parallelism": parallelism.build,
    "hmp": hmp.build,
    "omp": omp.build,
    "nodefusion": nodefusion.build,
    "pushdown": pushdown.build,
}

# Past this many cells a per-cell chart stops being a page and starts being a
# wall. Shared by every pass page so they all truncate the same way.
MAX_DETAILED_CELLS = 6


def enabled(con) -> list[str]:
    """Which optimizations this run has evidence of, in pipeline order."""
    present = tables(con)
    found: set[str] = set()
    if "pass_stats" in present:
        for (name,) in rows(con, "SELECT DISTINCT pass_name FROM pass_stats"):
            key = PASS_BY_NAME.get(name)
            if key:
                found.add(key)
        # A pass that ran but was recorded under a name this harness predates
        # still identifies itself in its own detail blob.
        for (detail,) in rows(con, "SELECT DISTINCT detail FROM pass_stats"):
            kind = str(loads(detail).get("kind", "")).replace("_", "")
            if kind in {k.replace("_", "") for k in PASSES}:
                found.add(next(k for k in PASSES if k.replace("_", "") == kind))
    if "cells" in present:
        for (passes,) in rows(con, "SELECT DISTINCT unnest(passes) FROM cells"):
            if passes in PASSES:
                found.add(passes)
    return [key for key in BUILDERS if key in found]


def build(con) -> list[Study]:
    """One page per optimization this run has evidence of.

    A pass that was detected but whose rows do not join to any cell still gets
    its page, saying so. Silently dropping it would leave the run looking as
    though the pass had never been enabled — which is exactly the thing the
    page is there to tell you.
    """
    out: list[Study] = []
    for key in enabled(con):
        meta = PASSES[key]
        page = Study(key=f"pass-{key}", number=0, group="optimizations",
                     title=meta["label"], subtitle=meta["full"],
                     question=meta["blurb"])
        try:
            built = BUILDERS[key](con)
        except Exception as e:  # noqa: BLE001 - one broken page must not lose the rest
            page.empty_reason = f"Could not build this page: {type(e).__name__}: {e}"
            built = None
        if built is not None:
            page = built
        elif not page.empty_reason:
            page.empty_reason = (
                f"This run records {meta['pass_name']} rows, but none of them belongs "
                "to a cell in `cells` — so there is nothing to attribute them to. That "
                "happens when results from separate sweeps share a directory."
            )
        out.append(page)
    return out
