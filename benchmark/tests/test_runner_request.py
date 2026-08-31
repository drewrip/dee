"""Rendering optimize and run requests from a cell.

These used to assert on argv. dee is a server now, so a cell's optimizer
settings travel as an OptimizerConfig object -- which removes the whole class
of double-negation bugs the old `--hmp-no-pushdown` style flags invited.
"""

from __future__ import annotations

from dee_bench.config import DEE_OPT_BY_NAME, Variant
from dee_bench.workload import WorkloadError
from dee_bench.matrix import Cell
import pytest

from dee_bench.runner import (
    _verify_settings,
    build_optimize_request,
    build_optimizer_config,
    build_queue_request,
    build_run_request,
)


def _cell(passes=("hmp",), dee_opt=None, repetitions=5, warmups=2,
          repeat_mode="group"):
    return Cell(
        cell_id="c", run_name="r", project="p01_iot", backend="duckdb", sf=0.1,
        variant=Variant(name="hmp", passes=tuple(passes)),
        dee_opt=dee_opt or {}, backend_config={},
        repetitions=repetitions, warmups=warmups, repeat_mode=repeat_mode,
    )


class TestOptimizerConfig:
    """The settings a cell submits with its DAG."""

    def test_enables_exactly_the_variants_passes(self):
        config = build_optimizer_config(_cell(("hmp", "pushdown")))
        assert config["run_hmp_pass"] is True
        assert config["run_pushdown_pass"] is True
        # Stated explicitly rather than left to dee's defaults, so a variant's
        # pass set cannot shift when those defaults change.
        assert config["run_omp_pass"] is False

    def test_a_baseline_variant_enables_nothing(self):
        config = build_optimizer_config(_cell(passes=()))
        assert not any(
            config[k] for k in ("run_hmp_pass", "run_omp_pass", "run_pushdown_pass")
        )

    def test_value_options_are_sent_under_their_config_field(self):
        config = build_optimizer_config(_cell(dee_opt={"hmp_max_runs": 4}))
        assert config["hmp_max_runs"] == 4

    def test_a_renamed_option_uses_the_servers_field_name(self):
        # The harness calls it omp_node_centrality; OptimizerConfig calls it
        # omp_centrality.
        config = build_optimizer_config(
            _cell(("omp",), {"omp_node_centrality": "paths"})
        )
        assert config["omp_centrality"] == "paths"
        assert "omp_node_centrality" not in config

    def test_omp_exhaust_is_sent_as_the_negation_it_actually_means(self):
        # `omp_exhaust: true` means "evaluate every plan fully", which is
        # early termination turned off.
        on = build_optimizer_config(_cell(("omp",), {"omp_exhaust": True}))
        off = build_optimizer_config(_cell(("omp",), {"omp_exhaust": False}))
        assert on["omp_early_termination"] is False
        assert off["omp_early_termination"] is True

    def test_pushdown_options_are_sent_verbatim_not_inverted(self):
        # These were the `--hmp-no-pushdown` style flags. Only the flag was a
        # negation; the config field says what it means, so the value goes
        # straight through.
        config = build_optimizer_config(
            _cell(("hmp",), {"hmp_use_pushdown": False})
        )
        assert config["hmp_use_pushdown"] is False

    def test_a_none_valued_option_is_omitted(self):
        config = build_optimizer_config(_cell(dee_opt={"omp_top": None}))
        assert "omp_top" not in config


class TestOptimizeRequest:
    def test_the_result_is_saved_as_a_version(self):
        # The measure phase runs the optimizer's output, so it has to be
        # something the server can be asked to run.
        assert build_optimize_request()["save_as_version"] is True

    def test_it_carries_no_settings_of_its_own(self):
        # The cell's settings live on the DAG. Re-sending them here would work,
        # and would mean the stored ones were never actually exercised.
        assert "config" not in build_optimize_request()


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


class TestQueueRequest:
    def test_each_entry_is_a_single_run(self):
        # Under `queue` the queue is the repetition. Repetitions inside an
        # entry as well would nest the two meanings and make rep_index
        # ambiguous.
        body = build_queue_request(
            _cell(repeat_mode="queue"), 50, collect_plans=False, count=5, warmups=0
        )
        assert body["count"] == 5
        assert body["repetitions"] == 1

    def test_warmups_ride_on_the_entry_that_asks_for_them(self):
        first = build_queue_request(
            _cell(repeat_mode="queue"), 50, collect_plans=False, count=1, warmups=2
        )
        rest = build_queue_request(
            _cell(repeat_mode="queue"), 50, collect_plans=False, count=4, warmups=0
        )
        # Warmups belong to the front of the queue only; repeating them per
        # entry would multiply the warmup work by the repetition count.
        assert first["warmups"] == 2
        assert rest["warmups"] == 0

    def test_carries_the_same_measurement_settings_as_a_trigger(self):
        cell = _cell(repeat_mode="queue")
        queued = build_queue_request(cell, 50, collect_plans=True, count=3, warmups=0)
        triggered = build_run_request(cell, 50, collect_plans=True)
        for key in ("cleanup_before", "collect_plans", "sample_interval_ms"):
            assert queued[key] == triggered[key], key


class TestRepeatMode:
    def test_the_default_is_the_single_group_trigger(self):
        # Changing this would silently change what every existing config
        # measures.
        from dee_bench.config import ExecutionConfig

        assert ExecutionConfig().repeat_mode == "group"

    def test_two_cells_differing_only_in_repeat_mode_are_two_experiments(self):
        # The dedupe key is the identity, so if repeat_mode were left out of
        # it the queue variant would collapse into the group one.
        from dee_bench.matrix import compute_cell_id

        group = compute_cell_id(_cell(repeat_mode="group").identity())
        queued = compute_cell_id(_cell(repeat_mode="queue").identity())
        assert group != queued


class TestSettingsTravelWithTheDag:
    def test_the_config_submitted_is_the_one_the_optimizer_should_run(self):
        # One builder feeds both the DAG submission and the check afterwards,
        # so the two cannot describe different experiments.
        cell = _cell(("omp",), {"omp_top": 3})
        assert build_optimizer_config(cell)["omp_top"] == 3
        assert build_optimizer_config(cell)["run_omp_pass"] is True

    def test_a_matching_resolved_config_is_accepted(self):
        cell = _cell(("omp",), {"omp_top": 3})
        resolved = dict(build_optimizer_config(cell))
        # dee resolves the whole config, so the response carries fields the
        # cell never named. Only the ones it did are its business.
        resolved.update({"hmp_max_runs": 1, "explain": True})
        _verify_settings(cell, resolved)  # does not raise

    def test_settings_the_server_ignored_fail_the_cell(self):
        # The failure this guards is silent: a dee that predates per-DAG
        # settings drops them on submit and optimizes under its own defaults,
        # producing a cell that looks like it benchmarked OMP but ran HMP too.
        cell = _cell(("omp",), {"omp_top": 3})
        stale = {"run_omp_pass": True, "run_hmp_pass": True,
                 "run_pushdown_pass": False, "omp_top": None}
        with pytest.raises(WorkloadError) as e:
            _verify_settings(cell, stale)
        message = str(e.value)
        assert "run_hmp_pass" in message and "omp_top" in message

    def test_an_empty_response_fails_rather_than_passes(self):
        # An older server may not echo the config at all; that is not evidence
        # the settings were honoured.
        with pytest.raises(WorkloadError):
            _verify_settings(_cell(("omp",), {"omp_top": 3}), {})
