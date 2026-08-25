"""Expand an experiment config into the concrete cells the runner executes.

The expansion has three steps, and the middle one matters most:

1. **Cross product** every ``matrix`` key with every list-valued ``dee_opt``.
2. **Prune** dee options the cell's enabled passes never read. Without this,
   sweeping ``hmp_strategy: [breadth, greedy]`` would double the number of
   ``unopt`` and ``omp`` cells even though the option changes nothing about
   them — inflating runtime and, worse, silently double-counting identical
   experiments in every aggregate.
3. **Deduplicate** by ``cell_id``, which is a hash of the *pruned* parameter
   set. Because pruning happens first, the duplicates collapse to one cell.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import DEE_OPT_BY_NAME, BenchConfig, Variant

# Keys the runner needs but which are not part of the swept matrix.
_RESERVED = {"project", "backend", "sf", "variant"}


@dataclass(frozen=True)
class Cell:
    """One fully-specified experiment: a DAG, a backend, a scale, a variant."""

    cell_id: str
    run_name: str
    project: str
    backend: str
    sf: float
    variant: Variant
    # dee optimizer options in effect, already pruned to those the variant's
    # passes actually read.
    dee_opt: dict[str, Any]
    backend_config: dict[str, Any]
    repetitions: int
    warmups: int
    # Any extra matrix keys, carried through for provenance.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def passes(self) -> tuple[str, ...]:
        return self.variant.passes

    @property
    def is_baseline(self) -> bool:
        return self.variant.is_baseline

    def describe(self) -> str:
        return f"{self.project}/{self.backend}/sf{self.sf:g}/{self.variant.name}"

    def identity(self) -> dict[str, Any]:
        """The parameter set `cell_id` is derived from."""
        return {
            "project": self.project,
            "backend": self.backend,
            "sf": self.sf,
            "variant": self.variant.name,
            "passes": list(self.variant.passes),
            "dee_opt": self.dee_opt,
            "backend_config": self.backend_config,
            "repetitions": self.repetitions,
            "warmups": self.warmups,
            "extra": self.extra,
        }


def _canonical(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_cell_id(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()[:16]


def prune_dee_opt(dee_opt: dict[str, Any], passes: tuple[str, ...]) -> dict[str, Any]:
    """Drop options none of `passes` reads.

    An option is kept only if at least one enabled pass consults it. HMP and
    OMP also invoke Pushdown internally on their candidates, but a cell that
    runs *only* Pushdown has no search to tune, so no options survive there.
    """
    enabled = set(passes)
    return {
        key: value
        for key, value in sorted(dee_opt.items())
        if DEE_OPT_BY_NAME[key].passes & enabled
    }


def expand(cfg: BenchConfig) -> list[Cell]:
    """Expand `cfg` into deduplicated cells, in execution order."""
    matrix_keys = list(cfg.matrix)
    opt_keys = list(cfg.dee_opt)

    axes = [cfg.matrix[k] for k in matrix_keys] + [cfg.dee_opt[k] for k in opt_keys]

    seen: dict[str, Cell] = {}
    for combo in itertools.product(*axes):
        values = dict(zip(matrix_keys + opt_keys, combo))

        variant = cfg.variants[str(values["variant"])]

        # Global dee_opt, then the variant's own overrides, then pruned to
        # what this variant's passes actually read.
        merged = {k: values[k] for k in opt_keys}
        merged.update(variant.overrides)
        dee_opt = prune_dee_opt(merged, variant.passes)

        backend = str(values["backend"])
        cell = Cell(
            cell_id="",
            run_name=cfg.name,
            project=str(values["project"]),
            backend=backend,
            sf=float(values["sf"]),
            variant=variant,
            dee_opt=dee_opt,
            backend_config=dict(cfg.backends.get(backend) or {}),
            repetitions=cfg.execution.repetitions,
            warmups=cfg.execution.warmups,
            extra={k: values[k] for k in matrix_keys if k not in _RESERVED},
        )
        cell_id = compute_cell_id(cell.identity())
        if cell_id not in seen:
            seen[cell_id] = Cell(**{**cell.__dict__, "cell_id": cell_id})

    return schedule(list(seen.values()))


def schedule(cells: list[Cell]) -> list[Cell]:
    """Order cells so infrastructure and data preparation are amortized.

    Bringing a postgres instance up, and generating or loading a scale
    factor's data, are both far more expensive than a single DAG run. Sorting
    by ``(backend, sf, project)`` means every variant and repetition sharing
    one prepared dataset runs back to back, so each dataset is prepared once
    rather than once per cell.

    Within a group, the baseline variant is ordered first so the comparison
    every other variant is measured against exists as early as possible — a
    run cut short still yields usable speedups.
    """
    return sorted(
        cells,
        key=lambda c: (
            c.backend,
            c.sf,
            c.project,
            not c.is_baseline,
            c.variant.name,
            c.cell_id,
        ),
    )


def cells_to_rows(cells: list[Cell], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    """Render cells as rows for the `cells` parquet table."""
    now = datetime.now(timezone.utc)
    return [
        {
            "cell_id": c.cell_id,
            "run_name": c.run_name,
            "project": c.project,
            "backend": c.backend,
            "sf": c.sf,
            "variant": c.variant.name,
            "passes": list(c.variant.passes),
            "dee_opt": _canonical(c.dee_opt),
            "backend_config": _canonical(c.backend_config),
            "repetitions": c.repetitions,
            "warmups": c.warmups,
            "dee_git_sha": provenance.get("dee_git_sha"),
            "dag_bench_git_sha": provenance.get("dag_bench_git_sha"),
            "harness_version": provenance.get("harness_version"),
            "host": provenance.get("host"),
            "cpu_count": provenance.get("cpu_count"),
            "mem_total_bytes": provenance.get("mem_total_bytes"),
            "created_at": now,
        }
        for c in cells
    ]


def summarize(cells: list[Cell]) -> str:
    """A short human summary of an expanded matrix."""
    if not cells:
        return "0 cells"
    backends = sorted({c.backend for c in cells})
    projects = sorted({c.project for c in cells})
    sfs = sorted({c.sf for c in cells})
    variants = sorted({c.variant.name for c in cells})
    return (
        f"{len(cells)} cells: "
        f"{len(projects)} project(s), {len(backends)} backend(s) ({', '.join(backends)}), "
        f"{len(sfs)} scale factor(s) ({', '.join(f'{s:g}' for s in sfs)}), "
        f"{len(variants)} variant(s) ({', '.join(variants)})"
    )
