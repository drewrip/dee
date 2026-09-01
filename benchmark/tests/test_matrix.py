"""Matrix expansion, pruning and cell identity."""

from __future__ import annotations

import itertools

import pytest
from types import SimpleNamespace

from dee_bench.config import Variant
from dee_bench.matrix import Cell, compute_cell_id, expand, prune_dee_opt, schedule


def cell(variant="hmp", passes=("hmp",), dee_opt=None, **kw):
    v = Variant(name=variant, passes=tuple(passes))
    base = dict(
        cell_id="", run_name="r", project="p01_iot", backend="duckdb", sf=0.1,
        variant=v, dee_opt=dee_opt or {}, backend_config={}, repetitions=3, warmups=1,
    )
    base.update(kw)
    c = Cell(**base)
    return Cell(**{**c.__dict__, "cell_id": compute_cell_id(c.identity())})


class TestPruning:
    def test_drops_options_no_enabled_pass_reads(self):
        opts = {"hmp_strategy": "greedy", "omp_top": 5, "hmp_max_runs": 4}
        assert prune_dee_opt(opts, ("hmp",)) == {"hmp_max_runs": 4, "hmp_strategy": "greedy"}

    def test_baseline_variant_keeps_nothing(self):
        opts = {"hmp_strategy": "greedy", "omp_top": 5}
        assert prune_dee_opt(opts, ()) == {}

    def test_pushdown_alone_reads_no_search_options(self):
        # Pushdown is a static analysis with no search to tune.
        assert prune_dee_opt({"hmp_strategy": "greedy"}, ("pushdown",)) == {}

    def test_option_shared_by_two_passes_survives_either(self):
        opts = {"profile_iterations": True}
        assert prune_dee_opt(opts, ("hmp",)) == opts
        assert prune_dee_opt(opts, ("omp",)) == opts
        assert prune_dee_opt(opts, ("pushdown",)) == {}


class TestCellId:
    def test_is_stable_across_calls(self):
        a, b = cell(), cell()
        assert a.cell_id == b.cell_id

    def test_is_insensitive_to_option_ordering(self):
        a = cell(dee_opt={"hmp_max_runs": 4, "hmp_strategy": "greedy"})
        b = cell(dee_opt={"hmp_strategy": "greedy", "hmp_max_runs": 4})
        assert a.cell_id == b.cell_id

    def test_distinguishes_meaningful_differences(self):
        base = cell(dee_opt={"hmp_max_runs": 1})
        for changed in (
            cell(dee_opt={"hmp_max_runs": 4}),
            cell(project="p02_adtech", dee_opt={"hmp_max_runs": 1}),
            cell(sf=0.25, dee_opt={"hmp_max_runs": 1}),
            cell(backend="postgres", dee_opt={"hmp_max_runs": 1}),
        ):
            assert base.cell_id != changed.cell_id

    def test_pruned_options_collapse_to_one_cell(self):
        # The point of pruning: sweeping an HMP option must not silently
        # produce several identical unoptimized cells.
        a = cell(variant="unopt", passes=(), dee_opt=prune_dee_opt({"hmp_strategy": "breadth"}, ()))
        b = cell(variant="unopt", passes=(), dee_opt=prune_dee_opt({"hmp_strategy": "greedy"}, ()))
        assert a.cell_id == b.cell_id


class TestSchedule:
    def test_groups_by_backend_then_sf_then_project(self):
        cells = [
            cell(project="p02_adtech", sf=0.5, backend="postgres"),
            cell(project="p01_iot", sf=0.5, backend="duckdb"),
            cell(project="p01_iot", sf=0.1, backend="duckdb"),
            cell(project="p02_adtech", sf=0.1, backend="duckdb"),
        ]
        ordered = schedule(cells)
        keys = [(c.backend, c.sf, c.project) for c in ordered]
        assert keys == sorted(keys)

    def test_baseline_runs_before_optimized_within_a_group(self):
        opt = cell(variant="hmp", passes=("hmp",))
        base = cell(variant="unopt", passes=())
        assert schedule([opt, base])[0].variant.name == "unopt"



def _config(backends=None, **matrix):
    """The slice of a BenchConfig that `expand` actually reads."""
    return SimpleNamespace(
        name="r",
        matrix={"project": ["p01_iot"], "backend": ["duckdb"], "sf": [0.1],
                "variant": ["unopt"], **matrix},
        dee_opt={},
        variants={
            "unopt": Variant(name="unopt", passes=()),
            "hmp": Variant(name="hmp", passes=("hmp",)),
        },
        backends=backends or {"duckdb": [{}]},
        execution=SimpleNamespace(
            repetitions=3, warmups=1, repeat_mode="group",
            optimization_mode="batch", converge_runs=12,
        ),
    )


class TestRepeatMode:
    def test_the_execution_setting_applies_to_every_cell(self):
        cfg = _config()
        cfg.execution.repeat_mode = "queue"
        assert [c.repeat_mode for c in expand(cfg)] == ["queue"]

    def test_the_matrix_can_sweep_it(self):
        # Comparing the two measurement modes is the reason it is sweepable:
        # one run directory, one baseline, two ways of measuring it.
        cells = expand(_config(repeat_mode=["group", "queue"]))
        assert sorted(c.repeat_mode for c in cells) == ["group", "queue"]
        # Distinct cells, not one counted twice: they measure differently.
        assert len({c.cell_id for c in cells}) == 2

    def test_it_is_not_carried_into_extra(self):
        # `extra` holds matrix keys the runner does not understand. This one it
        # does, and duplicating it there would put it in the identity twice.
        assert expand(_config(repeat_mode=["queue"]))[0].extra == {}


# --- optimization_mode ----------------------------------------------------


def test_sweeping_optimization_mode_does_not_duplicate_the_baseline():
    """A baseline runs no optimization, so a mode says nothing about it.

    Without pinning, sweeping the mode would produce two `unopt` cells doing
    identical work, and every aggregate keyed on the baseline would count it
    twice.
    """
    cfg = _config(
        variant=["unopt", "hmp"],
        optimization_mode=["batch", "continuous"],
    )
    cells = expand(cfg)

    baselines = [c for c in cells if c.variant.name == "unopt"]
    assert len(baselines) == 1
    assert baselines[0].optimization_mode == "batch"
    assert not baselines[0].is_continuous

    # The real variant still gets both.
    hmp = sorted(c.optimization_mode for c in cells if c.variant.name == "hmp")
    assert hmp == ["batch", "continuous"]


def test_the_mode_is_part_of_a_cells_identity():
    """The same variant under the two modes is two experiments.

    They measure different things -- one buys its own runs, the other spends
    the DAG's -- so collapsing them would average two answers into one.
    """
    cfg = _config(variant=["hmp"], optimization_mode=["batch", "continuous"])
    cells = expand(cfg)
    assert len({c.cell_id for c in cells}) == 2


def test_only_a_real_variant_is_continuous():
    cfg = _config(variant=["unopt", "hmp"], optimization_mode=["continuous"])
    by_variant = {c.variant.name: c for c in expand(cfg)}
    assert by_variant["hmp"].is_continuous
    assert not by_variant["unopt"].is_continuous


# --- backend configurations ------------------------------------------------


def _duckdb(*configs):
    return {"duckdb": list(configs)}


class TestBackendConfigs:
    def test_each_configuration_becomes_its_own_cell(self):
        cells = expand(_config(backends=_duckdb(
            {"threads": 8, "max_memory": "1GB"},
            {"threads": 8, "max_memory": "8GB"},
        )))
        assert len(cells) == 2
        assert sorted(c.backend_config["max_memory"] for c in cells) == ["1GB", "8GB"]

    def test_the_configuration_is_part_of_a_cells_identity(self):
        # Two memory ceilings are two experiments over the same DAG, so they
        # must not collapse into one cell and average two answers together.
        cells = expand(_config(backends=_duckdb(
            {"max_memory": "1GB"}, {"max_memory": "8GB"},
        )))
        assert len({c.cell_id for c in cells}) == 2

    def test_an_unswept_backend_is_a_single_unlabelled_cell(self):
        cells = expand(_config(backends=_duckdb({"threads": 8})))
        assert len(cells) == 1
        assert cells[0].backend_config_label == ""
        assert cells[0].describe() == "p01_iot/duckdb/sf0.1/unopt"

    def test_the_label_names_only_what_varies(self):
        # `threads` is the same everywhere, so naming it in every label would
        # be noise that hides the one setting the sweep is about.
        cells = expand(_config(backends=_duckdb(
            {"threads": 8, "max_memory": "1GB"},
            {"threads": 8, "max_memory": "8GB"},
        )))
        assert sorted(c.backend_config_label for c in cells) == [
            "max_memory=1GB", "max_memory=8GB",
        ]
        assert "duckdb[max_memory=1GB]" in expand(_config(backends=_duckdb(
            {"threads": 8, "max_memory": "1GB"},
            {"threads": 8, "max_memory": "8GB"},
        )))[0].describe()

    def test_a_duckdb_sweep_needs_no_new_backend_instance(self):
        # Everything DuckDB reads arrives through the connection, so cells
        # differing only in tuning share one in-process engine and one
        # preparation.
        cells = expand(_config(backends=_duckdb(
            {"max_memory": "1GB"}, {"max_memory": "8GB"},
        )))
        assert len({c.backend_setup_id for c in cells}) == 1

    def test_a_postgres_server_setting_needs_a_new_instance(self):
        cells = expand(_config(
            backend=["postgres"],
            backends={"postgres": [
                {"settings": {"work_mem": "64MB"}},
                {"settings": {"work_mem": "1GB"}},
            ]},
        ))
        assert len({c.backend_setup_id for c in cells}) == 2

    def test_a_postgres_connection_setting_does_not(self):
        cells = expand(_config(
            backend=["postgres"],
            backends={"postgres": [{"num_connections": 4}, {"num_connections": 16}]},
        ))
        assert len(cells) == 2
        assert len({c.backend_setup_id for c in cells}) == 1

    def test_cells_needing_one_instance_are_scheduled_together(self):
        # Restarting a backend costs more than a preparation, so cells sharing
        # an instance must not be interleaved with cells that need another.
        cells = expand(_config(
            backend=["postgres"],
            project=["p01_iot", "p02_adtech"],
            backends={"postgres": [
                {"settings": {"work_mem": "64MB"}},
                {"settings": {"work_mem": "1GB"}},
            ]},
        ))
        setups = [c.backend_setup_id for c in cells]
        assert len(list(itertools.groupby(setups))) == len(set(setups))
