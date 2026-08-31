"""Matrix expansion, pruning and cell identity."""

from __future__ import annotations

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



def _config(**matrix):
    """The slice of a BenchConfig that `expand` actually reads."""
    return SimpleNamespace(
        name="r",
        matrix={"project": ["p01_iot"], "backend": ["duckdb"], "sf": [0.1],
                "variant": ["unopt"], **matrix},
        dee_opt={},
        variants={"unopt": Variant(name="unopt", passes=())},
        backends={"duckdb": {}},
        execution=SimpleNamespace(repetitions=3, warmups=1, repeat_mode="group"),
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
