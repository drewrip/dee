"""Expand an experiment config into the concrete cells the runner executes.

The expansion has three steps, and the middle one matters most:

1. **Cross product** every ``matrix`` key with every list-valued ``dee_opt``,
   and with every configuration the cell's backend is swept over.
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

from .config import DEE_OPT_BY_NAME, BenchConfig, Variant, config_labels, setup_config

# Keys the runner needs but which are not part of the swept matrix.
_RESERVED = {"project", "backend", "sf", "variant", "repeat_mode", "optimization_mode"}


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
    # The backend tuning in effect, one of the configurations expanded from
    # this backend's block. Part of the identity, so two cells that differ
    # only in it are two experiments rather than a duplicate.
    backend_config: dict[str, Any]
    repetitions: int
    warmups: int
    # "group" or "queue": whether the repetitions run as one server-side run
    # group or as that many queued ones. Part of the identity because it
    # changes what is being measured, so two cells differing only in this are
    # two experiments, not a duplicate.
    repeat_mode: str = "group"
    # "batch" or "continuous": whether the cell's optimization is run to
    # convergence up front, buying its own DAG runs, or registered on the DAG
    # and stepped around the measured runs. Part of the identity for the same
    # reason `repeat_mode` is -- it changes what the cell measures, and what
    # the optimization cost means.
    optimization_mode: str = "batch"
    # Any extra matrix keys, carried through for provenance.
    extra: dict[str, Any] = field(default_factory=dict)
    # The swept backend settings that distinguish this cell from its siblings,
    # e.g. "max_memory=8GB". Empty when the backend is not swept. Derived from
    # `backend_config`, so it is not part of the identity.
    backend_config_label: str = ""

    @property
    def passes(self) -> tuple[str, ...]:
        return self.variant.passes

    @property
    def is_baseline(self) -> bool:
        return self.variant.is_baseline

    @property
    def backend_setup_id(self) -> str:
        """Identity of the backend *instance* this cell needs.

        Cells agreeing on it can share one running backend; a cell that
        disagrees needs it brought up again, which is why the scheduler groups
        by this before anything else.
        """
        return compute_cell_id({
            "backend": self.backend,
            "setup": setup_config(self.backend, self.backend_config),
        })

    @property
    def is_continuous(self) -> bool:
        """Whether this cell's optimization steps around its measured runs.

        A baseline has no optimization at all, so it is never continuous
        regardless of the mode it was configured under -- otherwise sweeping
        `optimization_mode` would double every baseline cell with a second one
        that does exactly the same thing.
        """
        return self.optimization_mode == "continuous" and not self.is_baseline

    def describe(self) -> str:
        backend = self.backend
        if self.backend_config_label:
            backend = f"{backend}[{self.backend_config_label}]"
        return f"{self.project}/{backend}/sf{self.sf:g}/{self.variant.name}"

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
            "repeat_mode": self.repeat_mode,
            "optimization_mode": self.optimization_mode,
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

    # A backend's configurations cannot be a flat axis of their own: they only
    # mean anything alongside the backend they belong to, so they are expanded
    # per backend and iterated inside the product. Labels are computed here,
    # once, because "which setting varies" is a property of the whole sweep
    # rather than of any one configuration.
    backend_configs = {
        name: list(configs) or [{}] for name, configs in cfg.backends.items()
    }
    backend_labels = {
        name: config_labels(configs) for name, configs in backend_configs.items()
    }

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
        configs = backend_configs.get(backend) or [{}]
        labels = backend_labels.get(backend) or [""]
        for backend_config, label in zip(configs, labels):
            cell = Cell(
                cell_id="",
                run_name=cfg.name,
                project=str(values["project"]),
                backend=backend,
                sf=float(values["sf"]),
                variant=variant,
                dee_opt=dee_opt,
                backend_config=dict(backend_config),
                backend_config_label=label,
                repetitions=cfg.execution.repetitions,
                warmups=cfg.execution.warmups,
                # Swept like any other axis when the matrix names it;
                # otherwise the whole run uses one mode.
                repeat_mode=str(values.get("repeat_mode", cfg.execution.repeat_mode)),
                # A baseline runs no optimization, so there is nothing for a
                # mode to describe. Pinning it here -- rather than letting the
                # swept value through -- is what stops sweeping the mode from
                # producing two identical baseline cells and double-counting
                # them in every aggregate, the same reason `dee_opt` is pruned
                # above.
                optimization_mode=(
                    "batch"
                    if variant.is_baseline
                    else str(values.get("optimization_mode", cfg.execution.optimization_mode))
                ),
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

    ``backend_setup_id`` sorts above the dataset because a cell that changes
    the backend *instance* — a container memory ceiling, a postgres server
    setting — makes the running one unusable, and restarting it costs more
    than a preparation. Sweeping only client-side settings, which is every
    DuckDB sweep, leaves one setup id for the whole run and the order is
    exactly what it was before. The readable label sorts below the dataset
    instead, where it costs nothing: cells differing only in it share a
    preparation and need only a rewritten connection.

    Within a group, the baseline variant is ordered first so the comparison
    every other variant is measured against exists as early as possible — a
    run cut short still yields usable speedups.
    """
    return sorted(
        cells,
        key=lambda c: (
            c.backend,
            c.backend_setup_id,
            c.sf,
            c.project,
            c.backend_config_label,
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
            "repeat_mode": c.repeat_mode,
            "optimization_mode": c.optimization_mode,
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
    summary = (
        f"{len(cells)} cells: "
        f"{len(projects)} project(s), {len(backends)} backend(s) ({', '.join(backends)}), "
        f"{len(sfs)} scale factor(s) ({', '.join(f'{s:g}' for s in sfs)}), "
        f"{len(variants)} variant(s) ({', '.join(variants)})"
    )
    # Only reported when a backend is actually swept: an unswept run has one
    # unnamed configuration, which says nothing worth a line.
    configs = sorted({c.backend_config_label for c in cells if c.backend_config_label})
    if configs:
        summary += f", {len(configs)} backend config(s) ({'; '.join(configs)})"
    return summary
