"""Rendering optimize and run requests from a cell.

These used to assert on argv. dee is a server now, so a cell's optimizer
settings travel as an OptimizerConfig object -- which removes the whole class
of double-negation bugs the old `--hmp-no-pushdown` style flags invited.
"""

from __future__ import annotations

from dee_bench.config import DEE_OPT_BY_NAME, Variant
from dee_bench.matrix import Cell
from dee_bench.runner import build_optimize_request, build_run_request


def _cell(passes=("hmp",), dee_opt=None, repetitions=5, warmups=2):
    return Cell(
        cell_id="c", run_name="r", project="p01_iot", backend="duckdb", sf=0.1,
        variant=Variant(name="hmp", passes=tuple(passes)),
        dee_opt=dee_opt or {}, backend_config={},
        repetitions=repetitions, warmups=warmups,
    )


class TestOptimizeRequest:
    def test_enables_exactly_the_variants_passes(self):
        config = build_optimize_request(_cell(("hmp", "pushdown")))["config"]
        assert config["run_hmp_pass"] is True
        assert config["run_pushdown_pass"] is True
        # Stated explicitly rather than left to dee's defaults, so a variant's
        # pass set cannot shift when those defaults change.
        assert config["run_omp_pass"] is False

    def test_a_baseline_variant_enables_nothing(self):
        config = build_optimize_request(_cell(passes=()))["config"]
        assert not any(
            config[k] for k in ("run_hmp_pass", "run_omp_pass", "run_pushdown_pass")
        )

    def test_value_options_are_sent_under_their_config_field(self):
        config = build_optimize_request(_cell(dee_opt={"hmp_max_runs": 4}))["config"]
        assert config["hmp_max_runs"] == 4

    def test_a_renamed_option_uses_the_servers_field_name(self):
        # The harness calls it omp_node_centrality; OptimizerConfig calls it
        # omp_centrality.
        config = build_optimize_request(
            _cell(("omp",), {"omp_node_centrality": "paths"})
        )["config"]
        assert config["omp_centrality"] == "paths"
        assert "omp_node_centrality" not in config

    def test_omp_exhaust_is_sent_as_the_negation_it_actually_means(self):
        # `omp_exhaust: true` means "evaluate every plan fully", which is
        # early termination turned off.
        on = build_optimize_request(_cell(("omp",), {"omp_exhaust": True}))["config"]
        off = build_optimize_request(_cell(("omp",), {"omp_exhaust": False}))["config"]
        assert on["omp_early_termination"] is False
        assert off["omp_early_termination"] is True

    def test_pushdown_options_are_sent_verbatim_not_inverted(self):
        # These were the `--hmp-no-pushdown` style flags. Only the flag was a
        # negation; the config field says what it means, so the value goes
        # straight through.
        config = build_optimize_request(
            _cell(("hmp",), {"hmp_use_pushdown": False})
        )["config"]
        assert config["hmp_use_pushdown"] is False

    def test_a_none_valued_option_is_omitted(self):
        config = build_optimize_request(_cell(dee_opt={"omp_top": None}))["config"]
        assert "omp_top" not in config

    def test_the_result_is_saved_as_a_version(self):
        # The measure phase runs the optimizer's output, so it has to be
        # something the server can be asked to run.
        assert build_optimize_request(_cell())["save_as_version"] is True


class TestRunRequest:
    def test_carries_the_cells_repetition_shape(self):
        body = build_run_request(_cell(repetitions=9, warmups=2), 50, collect_plans=False)
        assert body["repetitions"] == 9
        assert body["warmups"] == 2
        assert body["sample_interval_ms"] == 50

    def test_cleans_up_between_repetitions(self):
        # Otherwise the second repetition would find the first's tables already
        # materialized and measure nothing.
        assert build_run_request(_cell(), 50, collect_plans=False)["cleanup_before"] is True

    def test_plan_collection_follows_verbosity(self):
        assert build_run_request(_cell(), 50, collect_plans=True)["collect_plans"] is True
        assert build_run_request(_cell(), 50, collect_plans=False)["collect_plans"] is False


class TestOptionTable:
    def test_every_spec_maps_to_a_config_field(self):
        for spec in DEE_OPT_BY_NAME.values():
            assert spec.config_field, f"{spec.name} has no config field"

    def test_only_omp_exhaust_inverts_its_value(self):
        # Inversion is a trap; the table should contain exactly one instance of
        # it, and this test is where a second one gets noticed.
        inverting = [s.name for s in DEE_OPT_BY_NAME.values() if s.invert]
        assert inverting == ["omp_exhaust"]
