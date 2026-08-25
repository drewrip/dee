"""Experiment configuration: load, validate and resolve a benchmark spec.

A config is a YAML file describing an *experiment matrix* rather than a single
run. Any key under ``matrix`` or ``dee_opt`` may be a list, and the harness
expands the cross product (see :mod:`dee_bench.matrix`).

The ``dee_opt`` keys mirror ``OptimizerConfig`` in ``dee/src/opt.rs`` one for
one. :data:`DEE_OPT_SPECS` is the single place that mapping lives: it drives
validation, CLI-flag rendering, and the pruning that stops irrelevant options
from multiplying the matrix.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema import Verbosity

HARNESS_VERSION = "0.1.0"


@dataclass(frozen=True)
class DeeOptSpec:
    """One dee optimizer option, and how to render it as a `dee-cli opt` flag."""

    name: str
    flag: str
    kind: str  # "bool" | "int" | "float" | "str"
    # Which passes actually read this option. An option whose passes are all
    # disabled for a variant is pruned from that cell, so it cannot silently
    # multiply the matrix with duplicate experiments.
    passes: frozenset[str]
    choices: tuple[str, ...] | None = None
    # True when the flag's presence *disables* something, so the config value
    # is inverted relative to the flag (e.g. hmp_use_pushdown -> --hmp-no-pushdown).
    negated: bool = False
    doc: str = ""


# Mirrors OptimizerConfig in dee/src/opt.rs. Keep in sync; `dee-bench doctor`
# checks this list against `dee-cli opt --help`.
DEE_OPT_SPECS: tuple[DeeOptSpec, ...] = (
    DeeOptSpec("omp_top", "--omp-top", "int", frozenset({"omp"}),
               doc="Consider only the top N candidate nodes in OMP."),
    DeeOptSpec("omp_node_centrality", "--omp-node-centrality", "str", frozenset({"omp"}),
               choices=("outdegree", "paths"),
               doc="How OMP ranks candidate nodes."),
    DeeOptSpec("omp_exhaust", "--omp-exhaust", "bool", frozenset({"omp"}),
               doc="Disable OMP early termination and evaluate every plan fully."),
    DeeOptSpec("omp_use_pushdown", "--omp-no-pushdown", "bool", frozenset({"omp"}), negated=True,
               doc="Run the pushdown pass on each OMP candidate before measuring it."),
    DeeOptSpec("hmp_downstream_cost", "--hmp-downstream-cost", "bool", frozenset({"hmp"}),
               doc="Rank HMP candidates by the duplicate downstream work they cause, not their own cost."),
    DeeOptSpec("hmp_max_runs", "--hmp-max-runs", "int", frozenset({"hmp"}),
               doc="DAG-run budget HMP may spend searching. The main optimization-cost dial."),
    DeeOptSpec("hmp_top_cpu_time", "--hmp-top-cpu-time", "float", frozenset({"hmp"}),
               doc="Fraction of total operator CPU time the HMP working set must cover."),
    DeeOptSpec("hmp_normalize_with_cardinality", "--hmp-normalize-with-cardinality", "bool", frozenset({"hmp"}),
               doc="Divide HMP's ranking score by the candidate's estimated cardinality."),
    DeeOptSpec("hmp_strategy", "--hmp-strategy", "str", frozenset({"hmp"}),
               choices=("breadth", "greedy"),
               doc="HMP's search strategy over the candidate ranking."),
    DeeOptSpec("hmp_beam_width", "--hmp-beam-width", "int", frozenset({"hmp"}),
               doc="Beam width for the greedy HMP strategy. Ignored by breadth."),
    DeeOptSpec("hmp_use_pushdown", "--hmp-no-pushdown", "bool", frozenset({"hmp"}), negated=True,
               doc="Run the pushdown pass on each HMP candidate before measuring it."),
    DeeOptSpec("profile_iterations", "--profile-iterations", "bool", frozenset({"hmp", "omp"}),
               doc="Capture a resource timeseries for every candidate DAG the optimizer runs."),
)

DEE_OPT_BY_NAME: dict[str, DeeOptSpec] = {s.name: s for s in DEE_OPT_SPECS}

VALID_PASSES = ("hmp", "omp", "pushdown")
VALID_BACKENDS = ("duckdb", "postgres")


class ConfigError(Exception):
    """A benchmark config could not be loaded or validated."""


@dataclass(frozen=True)
class Variant:
    """A named optimizer configuration — one rung of the ablation ladder."""

    name: str
    passes: tuple[str, ...]
    # Per-variant dee_opt overrides, applied on top of the global dee_opt.
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        """A variant with no passes runs the DAG exactly as dbt defined it."""
        return not self.passes


@dataclass(frozen=True)
class ExecutionConfig:
    repetitions: int = 5
    warmups: int = 1
    sample_interval_ms: int = 100
    timeout_s: int = 3600


@dataclass
class BenchConfig:
    name: str
    dag_bench: Path
    dee_cli: Path
    output_dir: Path
    verbosity: Verbosity
    matrix: dict[str, list[Any]]
    variants: dict[str, Variant]
    dee_opt: dict[str, list[Any]]
    backends: dict[str, dict[str, Any]]
    execution: ExecutionConfig
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` in strings, recursively through lists and dicts."""
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            var = m.group(1)
            got = os.environ.get(var)
            if got is None:
                raise ConfigError(
                    f"config references ${{{var}}} but {var} is not set in the environment"
                )
            return got

        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _as_list(value: Any) -> list[Any]:
    """Normalize a scalar-or-list matrix value to a list.

    A list is always a set of values to sweep. A scalar is a single value.
    ``None`` is a real value (dee options are frequently ``None`` = unset), so
    it is preserved rather than treated as absent.
    """
    if isinstance(value, list):
        return value
    return [value]


def _coerce(spec: DeeOptSpec, value: Any) -> Any:
    if value is None:
        return None
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"dee_opt.{spec.name} must be true/false, got {value!r}")
        return value
    if spec.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"dee_opt.{spec.name} must be an integer, got {value!r}")
        return value
    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"dee_opt.{spec.name} must be a number, got {value!r}")
        return float(value)
    value = str(value)
    if spec.choices and value not in spec.choices:
        raise ConfigError(
            f"dee_opt.{spec.name} must be one of {', '.join(spec.choices)}, got {value!r}"
        )
    return value


def load(path: str | Path, overrides: dict[str, Any] | None = None) -> BenchConfig:
    """Load and validate a benchmark config."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    raw = _expand_env(raw)
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})
    return _resolve(raw, source_path=path)


def _resolve(raw: dict[str, Any], source_path: Path | None = None) -> BenchConfig:
    base = source_path.parent if source_path else Path.cwd()

    def resolve_path(value: str) -> Path:
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    name = str(raw.get("name") or (source_path.stem if source_path else "bench"))

    if "dag_bench" not in raw and not os.environ.get("DAG_BENCH"):
        raise ConfigError(
            "config must set `dag_bench` (the dag-bench checkout), or DAG_BENCH must be set"
        )
    dag_bench = resolve_path(raw.get("dag_bench") or os.environ["DAG_BENCH"])
    if not (dag_bench / "projects").is_dir():
        raise ConfigError(f"dag_bench={dag_bench} has no projects/ directory")

    dee_cli = resolve_path(raw.get("dee_cli") or "../../target/release/dee-cli")
    if not dee_cli.exists():
        raise ConfigError(
            f"dee_cli={dee_cli} does not exist. Build it with `cargo build --release` "
            "in the dee checkout, or point `dee_cli` at the binary."
        )
    output_dir = resolve_path(raw.get("output_dir") or f"results/{name}")

    verbosity = Verbosity.parse(raw.get("verbosity", "standard"))

    # --- variants ---------------------------------------------------------
    variants_raw = raw.get("variants") or {}
    if not isinstance(variants_raw, dict) or not variants_raw:
        raise ConfigError("config must define at least one entry under `variants`")
    variants: dict[str, Variant] = {}
    for vname, vcfg in variants_raw.items():
        vcfg = vcfg or {}
        if not isinstance(vcfg, dict):
            raise ConfigError(f"variants.{vname} must be a mapping")
        passes = vcfg.get("passes", [])
        if not isinstance(passes, list):
            raise ConfigError(f"variants.{vname}.passes must be a list")
        for p in passes:
            if p not in VALID_PASSES:
                raise ConfigError(
                    f"variants.{vname}.passes: unknown pass {p!r}; "
                    f"expected one of {', '.join(VALID_PASSES)}"
                )
        overrides = {k: v for k, v in vcfg.items() if k != "passes"}
        for k in overrides:
            if k not in DEE_OPT_BY_NAME:
                raise ConfigError(
                    f"variants.{vname}: unknown dee option {k!r}; "
                    f"expected one of {', '.join(sorted(DEE_OPT_BY_NAME))}"
                )
        variants[str(vname)] = Variant(
            name=str(vname),
            passes=tuple(passes),
            overrides={k: _coerce(DEE_OPT_BY_NAME[k], v) for k, v in overrides.items()},
        )

    # --- matrix -----------------------------------------------------------
    matrix_raw = raw.get("matrix") or {}
    if not isinstance(matrix_raw, dict):
        raise ConfigError("`matrix` must be a mapping")
    required = ("project", "backend", "sf", "variant")
    missing = [k for k in required if k not in matrix_raw]
    if missing:
        raise ConfigError(f"matrix is missing required key(s): {', '.join(missing)}")
    matrix = {k: _as_list(v) for k, v in matrix_raw.items()}

    for b in matrix["backend"]:
        if b not in VALID_BACKENDS:
            raise ConfigError(
                f"matrix.backend: unknown backend {b!r}; expected one of {', '.join(VALID_BACKENDS)}"
            )
    for v in matrix["variant"]:
        if v not in variants:
            raise ConfigError(
                f"matrix.variant references {v!r}, which is not defined under `variants` "
                f"(defined: {', '.join(sorted(variants))})"
            )
    bad_sf = [s for s in matrix["sf"] if not isinstance(s, (int, float)) or isinstance(s, bool) or s <= 0]
    if bad_sf:
        raise ConfigError(f"matrix.sf must be positive numbers; got {bad_sf}")
    matrix["sf"] = [float(s) for s in matrix["sf"]]

    for project in matrix["project"]:
        if not (dag_bench / "projects" / str(project) / "dbt_project.yml").exists():
            raise ConfigError(
                f"matrix.project references {project!r}, but "
                f"{dag_bench / 'projects' / project / 'dbt_project.yml'} does not exist. "
                "dag-bench projects are the directories under projects/ that contain a "
                "dbt_project.yml."
            )

    # --- dee_opt ----------------------------------------------------------
    dee_opt_raw = raw.get("dee_opt") or {}
    if not isinstance(dee_opt_raw, dict):
        raise ConfigError("`dee_opt` must be a mapping")
    dee_opt: dict[str, list[Any]] = {}
    for key, value in dee_opt_raw.items():
        spec = DEE_OPT_BY_NAME.get(key)
        if spec is None:
            raise ConfigError(
                f"dee_opt.{key} is not a dee optimizer option; expected one of "
                f"{', '.join(sorted(DEE_OPT_BY_NAME))}"
            )
        dee_opt[key] = [_coerce(spec, v) for v in _as_list(value)]

    # --- backends ---------------------------------------------------------
    backends_raw = raw.get("backends") or {}
    if not isinstance(backends_raw, dict):
        raise ConfigError("`backends` must be a mapping")
    backends: dict[str, dict[str, Any]] = {}
    for bname in matrix["backend"]:
        backends[bname] = dict(backends_raw.get(bname) or {})

    # --- execution --------------------------------------------------------
    exec_raw = raw.get("execution") or {}
    if not isinstance(exec_raw, dict):
        raise ConfigError("`execution` must be a mapping")
    unknown = set(exec_raw) - {"repetitions", "warmups", "sample_interval_ms", "timeout_s"}
    if unknown:
        raise ConfigError(f"execution: unknown key(s): {', '.join(sorted(unknown))}")
    execution = ExecutionConfig(**exec_raw)
    if execution.repetitions < 1:
        raise ConfigError("execution.repetitions must be at least 1")
    if execution.warmups < 0:
        raise ConfigError("execution.warmups must not be negative")

    return BenchConfig(
        name=name,
        dag_bench=dag_bench,
        dee_cli=dee_cli,
        output_dir=output_dir,
        verbosity=verbosity,
        matrix=matrix,
        variants=variants,
        dee_opt=dee_opt,
        backends=backends,
        execution=execution,
        source_path=source_path,
        raw=raw,
    )
