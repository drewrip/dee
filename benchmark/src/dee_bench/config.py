"""Experiment configuration: load, validate and resolve a benchmark spec.

A config is a YAML file describing an *experiment matrix* rather than a single
run. Any key under ``matrix``, ``dee_opt`` or ``backends`` may be a list, and
the harness expands the cross product (see :mod:`dee_bench.matrix`).

The ``dee_opt`` keys mirror ``OptimizerConfig`` in ``dee/src/opt.rs`` one for
one. :data:`DEE_OPT_SPECS` is the single place that mapping lives: it drives
validation, CLI-flag rendering, and the pruning that stops irrelevant options
from multiplying the matrix.
"""

from __future__ import annotations

import itertools
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
    """One dee optimizer option, and how it reaches the server."""

    name: str
    # The equivalent `dee optimize` flag. No longer how the option is sent --
    # optimizer settings go over HTTP as an OptimizerConfig object -- but kept
    # because it is what a person types, and what `doctor` reports against.
    flag: str
    kind: str  # "bool" | "int" | "float" | "str" | "int_list"
    # Which passes actually read this option. An option whose passes are all
    # disabled for a variant is pruned from that cell, so it cannot silently
    # multiply the matrix with duplicate experiments.
    passes: frozenset[str]
    choices: tuple[str, ...] | None = None
    # True when the *flag* disables something the config enables (e.g.
    # hmp_use_pushdown is set by omitting --hmp-no-pushdown). Documentation
    # only: the JSON value is never inverted by this.
    negated: bool = False
    doc: str = ""
    # Field name in the server's OptimizerConfig, when it differs from `name`.
    field: str | None = None
    # True when the config value is the logical negation of the harness's
    # value. Only `omp_exhaust` works this way: it turns off early termination.
    invert: bool = False

    @property
    def config_field(self) -> str:
        return self.field or self.name

    def config_value(self, value: Any) -> Any:
        """The value to send under `config_field`."""
        if self.invert:
            return not bool(value)
        return value


# Mirrors OptimizerConfig in dee/src/opt.rs. `dee-bench doctor` checks this
# list against the server's own GET /v1/optimizer/options, which is a real
# contract rather than the `--help` scraping it used to do.
DEE_OPT_SPECS: tuple[DeeOptSpec, ...] = (
    DeeOptSpec("omp_top", "--omp-top", "int", frozenset({"omp"}),
               doc="Consider only the top N candidate nodes in OMP."),
    DeeOptSpec("omp_node_centrality", "--omp-node-centrality", "str", frozenset({"omp"}),
               choices=("outdegree", "paths"), field="omp_centrality",
               doc="How OMP ranks candidate nodes."),
    DeeOptSpec("omp_exhaust", "--omp-exhaust", "bool", frozenset({"omp"}),
               field="omp_early_termination", invert=True,
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
    DeeOptSpec("parallelism_ladder", "--parallelism-ladder", "int_list", frozenset({"parallelism"}),
               doc="Node-concurrency caps the parallelism ladder measures."),
    DeeOptSpec("parallelism_seed_repeats", "--parallelism-seed-repeats", "int", frozenset({"parallelism"}),
               doc="Runs spent measuring the DAG's current setting before the ladder starts."),
    DeeOptSpec("parallelism_confirm_runs", "--parallelism-confirm-runs", "int", frozenset({"parallelism"}),
               doc="Re-measurements a rung must survive after beating the incumbent's best sample."),
    DeeOptSpec("profile_iterations", "--profile-iterations", "bool", frozenset({"hmp", "omp", "parallelism"}),
               doc="Capture a resource timeseries for every candidate DAG the optimizer runs."),
)

DEE_OPT_BY_NAME: dict[str, DeeOptSpec] = {s.name: s for s in DEE_OPT_SPECS}

# How a cell's measured repetitions reach the server. See ExecutionConfig.
REPEAT_MODES = ("group", "queue")


VALID_PASSES = ("parallelism", "hmp", "omp", "pushdown")

# How a cell's optimizations are driven.
#
# ``batch`` is `dee optimize`: the optimization is run to convergence in one
# shot, buying the DAG runs its search needs, and the measured runs that follow
# execute the result. That is what every existing result in `results/` was
# produced under, and it stays the default so those numbers remain comparable.
#
# ``continuous`` registers the optimization on the DAG instead. It then steps
# around the measured runs themselves, spending no runs of its own -- the shape
# dee's server model actually makes possible, where a pipeline that runs
# nightly optimizes itself nightly. The two answer different questions: batch
# asks "how good a plan can be found, and what did finding it cost"; continuous
# asks "how quickly does a DAG converge while doing its normal work".
VALID_OPTIMIZATION_MODES = ("batch", "continuous")

# Which side of each run a registered optimization steps on. ``None`` leaves it
# to the optimization's own default, which is what a benchmark wants unless it
# is specifically studying the setting.
VALID_STEP_PHASES = ("before", "after", "both")
VALID_BACKENDS = ("duckdb", "postgres")


# Settings each backend understands, and what they mean.
#
# Declared rather than accepted loosely because these keys are sweepable: a
# typo in a swept key would otherwise expand into several cells that differ
# only in a setting nothing reads, and quietly measure the same thing several
# times under different cell ids.
BACKEND_KEYS: dict[str, frozenset[str]] = {
    "duckdb": frozenset({"threads", "num_connections", "max_memory"}),
    "postgres": frozenset({
        "provider", "image", "host", "port", "user", "password", "dbname",
        "cpus", "memory", "volume_suffix", "num_connections", "settings",
    }),
}

# Settings that only describe how the harness *connects* to a backend that is
# already up. Sweeping one of these needs no new instance: the harness rewrites
# the prepared project's connections.json and dee replaces the pool on the next
# cell. Everything else describes the instance itself -- a container's memory
# ceiling, a postgres server setting -- so a cell that changes it needs the
# backend brought up again.
#
# ``None`` means every setting is client side, which is DuckDB: it is an
# in-process engine, so its whole configuration reaches it through the
# connection and there is no instance to restart.
BACKEND_CLIENT_KEYS: dict[str, frozenset[str] | None] = {
    "duckdb": None,
    "postgres": frozenset({"num_connections"}),
}


class ConfigError(Exception):
    """A benchmark config could not be loaded or validated."""


@dataclass(frozen=True)
class Variant:
    """A named optimizer configuration — one rung of the ablation ladder."""

    name: str
    passes: tuple[str, ...]
    # Per-variant dee_opt overrides, applied on top of the global dee_opt.
    overrides: dict[str, Any] = field(default_factory=dict)
    # When a registered optimization steps, in `continuous` mode. `None` means
    # the optimization's own default. Ignored in `batch` mode, where nothing is
    # registered and the driver supplies both sides itself.
    step_phase: str | None = None

    @property
    def is_baseline(self) -> bool:
        """A variant with no passes runs the DAG exactly as dbt defined it."""
        return not self.passes


@dataclass(frozen=True)
class ServerConfig:
    """How the sweep gets a dee server.

    By default it starts its own on an ephemeral port, so concurrent sweeps on
    one machine cannot collide, and tears it down afterwards.
    """

    autostart: bool = True
    # Attach to an already-running server instead of starting one. The sweep
    # will not stop it.
    url: str | None = None
    bind: str = "127.0.0.1:0"
    startup_timeout_s: int = 60


@dataclass(frozen=True)
class ExecutionConfig:
    repetitions: int = 5
    warmups: int = 1
    sample_interval_ms: int = 100
    timeout_s: int = 3600
    # How the measured repetitions are executed on the server.
    #
    # ``group`` sends one trigger carrying every repetition, which dee runs
    # back to back inside a single driver against one already-warm engine.
    # That is the tightest measurement of the DAG itself, and the default.
    #
    # ``queue`` puts each repetition on the server's run queue as its own run
    # group. dee still runs them strictly one at a time, but each gets a fresh
    # engine and its own group in dee's history -- which is what a repetition
    # looks like in production, where nothing shares an engine with the run
    # before it. Use it to ask whether the shared engine is flattering the
    # numbers.
    #
    # ``matrix.repeat_mode`` overrides this per cell, and sweeping it there is
    # how the two modes get compared within one run.
    repeat_mode: str = "group"

    # ``batch`` or ``continuous`` -- see VALID_OPTIMIZATION_MODES. Sweepable
    # per cell through ``matrix.optimization_mode``, which is how one run
    # compares a batch optimization against the same one driven continuously.
    optimization_mode: str = "batch"

    # Measured runs to give a continuous optimization to converge in.
    #
    # A continuous optimization needs runs to learn from, and a cell that ran
    # fewer than its search needs would be recorded as "did not converge"
    # rather than as a result. This is the ceiling on how many extra runs a
    # cell will perform waiting for one; the runs still count as measurements,
    # so nothing is wasted if it converges early.
    converge_runs: int = 12


@dataclass
class BenchConfig:
    name: str
    dag_bench: Path
    dee_bin: Path
    output_dir: Path
    verbosity: Verbosity
    matrix: dict[str, list[Any]]
    variants: dict[str, Variant]
    dee_opt: dict[str, list[Any]]
    # One entry per backend named in the matrix, holding every concrete
    # configuration that backend is swept over. A backend block with no
    # list-valued setting expands to a single configuration.
    backends: dict[str, list[dict[str, Any]]]
    execution: ExecutionConfig
    server: ServerConfig = field(default_factory=ServerConfig)
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


def expand_backend_config(config: dict[str, Any], path: str = "") -> list[dict[str, Any]]:
    """Expand a backend block into one concrete configuration per combination.

    Any setting under ``backends.<name>`` may be a list, exactly as under
    ``matrix`` and ``dee_opt``, and the block expands into the cross product --
    which is how one run compares two DuckDB memory ceilings against each
    other. Nested mappings (postgres ``settings``) expand the same way, so
    ``settings.work_mem: [64MB, 512MB]`` is swept like anything else.
    """
    keys: list[str] = []
    axes: list[list[Any]] = []
    for key, value in config.items():
        keys.append(key)
        if isinstance(value, dict):
            axes.append(expand_backend_config(value, f"{path}{key}."))
        elif isinstance(value, list):
            if not value:
                raise ConfigError(
                    f"backends.{path}{key} is an empty list; give it at least one value"
                )
            axes.append(value)
        else:
            axes.append([value])
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a backend configuration to dotted leaf paths."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(flatten_config(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def config_labels(configs: list[dict[str, Any]]) -> list[str]:
    """Label each configuration by the settings that differ between them.

    Only the swept settings appear, so a label reads as the one thing that
    cell changed -- ``max_memory=8GB`` rather than the whole block. A backend
    that is not swept has a single configuration and nothing to distinguish,
    so every label is empty and cells describe themselves exactly as they did
    before backend sweeps existed.
    """
    flat = [flatten_config(c) for c in configs]
    varying = sorted(
        key for key in {k for f in flat for k in f}
        if len({str(f.get(key)) for f in flat}) > 1
    )
    return [",".join(f"{k}={f.get(k)}" for k in varying) for f in flat]


def setup_config(backend: str, config: dict[str, Any]) -> dict[str, Any]:
    """The part of `config` describing the backend *instance* rather than the
    connection to it.

    Two cells whose setup configurations agree can share one running backend;
    two that disagree cannot, and the second needs it brought up again. See
    :data:`BACKEND_CLIENT_KEYS`.
    """
    client = BACKEND_CLIENT_KEYS.get(backend, frozenset())
    if client is None:
        return {}
    return {k: v for k, v in sorted(config.items()) if k not in client}


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
    if spec.kind == "int_list":
        # A list value is a cell's setting, not a sweep. The matrix expands a
        # list into one cell per element, so a genuinely list-valued option has
        # to arrive already wrapped -- `[[1, 2], [1, 2, 4]]` sweeps two ladders,
        # `[1, 2]` is the single ladder 1,2. Rejecting a bare int here is what
        # keeps that distinction visible rather than silently sweeping.
        if not isinstance(value, (list, tuple)) or not value:
            raise ConfigError(
                f"dee_opt.{spec.name} must be a non-empty list of integers, got {value!r}; "
                f"write it wrapped -- `{spec.name}: [[1, 2, 4, 8]]` is the single ladder "
                f"1,2,4,8, while `[[1, 2], [1, 2, 4]]` sweeps two of them"
            )
        out = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigError(
                    f"dee_opt.{spec.name} must contain integers, got {item!r}"
                )
            out.append(item)
        return out
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

    # `dee_cli` is the pre-server name for the same setting; accepted so an
    # existing config keeps working.
    dee_bin = resolve_path(
        raw.get("dee_bin") or raw.get("dee_cli") or "../../target/release/dee"
    )
    if not dee_bin.exists():
        raise ConfigError(
            f"dee_bin={dee_bin} does not exist. Build it with `cargo build --release` "
            "in the dee checkout, or point `dee_bin` at the binary."
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
        step_phase = vcfg.get("step_phase")
        if step_phase is not None and step_phase not in VALID_STEP_PHASES:
            raise ConfigError(
                f"variants.{vname}.step_phase: unknown phase {step_phase!r}; "
                f"expected one of {', '.join(VALID_STEP_PHASES)}"
            )
        overrides = {k: v for k, v in vcfg.items() if k not in ("passes", "step_phase")}
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
            step_phase=step_phase,
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

    # `repeat_mode` may be swept like anything else in the matrix, which is
    # how the two measurement modes get compared inside one run rather than by
    # diffing two run directories.
    for mode in matrix.get("repeat_mode", ()):
        if mode not in REPEAT_MODES:
            raise ConfigError(
                f"matrix.repeat_mode must be one of {', '.join(REPEAT_MODES)}, got {mode!r}"
            )

    # Likewise `optimization_mode`: sweeping it is how one run answers whether
    # optimizing continuously reaches the same plan a batch optimization found.
    for mode in matrix.get("optimization_mode", ()):
        if mode not in VALID_OPTIMIZATION_MODES:
            raise ConfigError(
                f"matrix.optimization_mode must be one of "
                f"{', '.join(VALID_OPTIMIZATION_MODES)}, got {mode!r}"
            )

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
    backends: dict[str, list[dict[str, Any]]] = {}
    for bname in matrix["backend"]:
        block = backends_raw.get(bname) or {}
        if not isinstance(block, dict):
            raise ConfigError(f"backends.{bname} must be a mapping")
        known = BACKEND_KEYS.get(bname, frozenset())
        unknown_keys = sorted(set(block) - known)
        if unknown_keys:
            raise ConfigError(
                f"backends.{bname}: unknown setting(s) {', '.join(unknown_keys)}; "
                f"expected one of {', '.join(sorted(known))}"
            )
        backends[bname] = expand_backend_config(block)

    # --- execution --------------------------------------------------------
    exec_raw = raw.get("execution") or {}
    if not isinstance(exec_raw, dict):
        raise ConfigError("`execution` must be a mapping")
    unknown = set(exec_raw) - {
        "repetitions", "warmups", "sample_interval_ms", "timeout_s", "repeat_mode",
        "optimization_mode", "converge_runs",
    }
    if unknown:
        raise ConfigError(f"execution: unknown key(s): {', '.join(sorted(unknown))}")
    execution = ExecutionConfig(**exec_raw)
    if execution.repetitions < 1:
        raise ConfigError("execution.repetitions must be at least 1")
    if execution.warmups < 0:
        raise ConfigError("execution.warmups must not be negative")
    if execution.repeat_mode not in REPEAT_MODES:
        raise ConfigError(
            f"execution.repeat_mode must be one of {', '.join(REPEAT_MODES)}, "
            f"got {execution.repeat_mode!r}"
        )
    if execution.optimization_mode not in VALID_OPTIMIZATION_MODES:
        raise ConfigError(
            f"execution.optimization_mode must be one of "
            f"{', '.join(VALID_OPTIMIZATION_MODES)}, got {execution.optimization_mode!r}"
        )
    if execution.converge_runs < 1:
        raise ConfigError("execution.converge_runs must be at least 1")

    # --- server -----------------------------------------------------------
    server_raw = raw.get("server") or {}
    if not isinstance(server_raw, dict):
        raise ConfigError("server must be a mapping")
    unknown = set(server_raw) - {"autostart", "url", "bind", "startup_timeout_s"}
    if unknown:
        raise ConfigError(f"server: unknown key(s): {', '.join(sorted(unknown))}")
    server = ServerConfig(**server_raw)
    if server.url and server.autostart:
        # Attaching and starting are mutually exclusive; picking one silently
        # would leave a stray server behind or talk to the wrong one.
        server = ServerConfig(
            autostart=False,
            url=server.url,
            bind=server.bind,
            startup_timeout_s=server.startup_timeout_s,
        )

    return BenchConfig(
        name=name,
        dag_bench=dag_bench,
        dee_bin=dee_bin,
        output_dir=output_dir,
        verbosity=verbosity,
        matrix=matrix,
        variants=variants,
        dee_opt=dee_opt,
        backends=backends,
        execution=execution,
        server=server,
        source_path=source_path,
        raw=raw,
    )
