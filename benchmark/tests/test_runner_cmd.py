"""Rendering dee-cli invocations from a cell."""

from __future__ import annotations

from pathlib import Path

from dee_bench.config import Variant
from dee_bench.matrix import Cell
from dee_bench.runner import build_opt_command, build_run_command
from dee_bench.workload import PreparedProject


def _prepared(tmp_path):
    return PreparedProject(
        project="p01_iot", backend="duckdb", sf=0.1, project_dir=tmp_path,
        dag_json=tmp_path / "dag.json", connections_json=tmp_path / "connections.json",
        target="dev", warehouse=None,
    )


def _cell(passes=("hmp",), dee_opt=None):
    return Cell(
        cell_id="c", run_name="r", project="p01_iot", backend="duckdb", sf=0.1,
        variant=Variant(name="hmp", passes=tuple(passes)),
        dee_opt=dee_opt or {}, backend_config={}, repetitions=5, warmups=2,
    )


class TestOptCommand:
    def test_enables_exactly_the_variants_passes(self, tmp_path):
        cmd = build_opt_command(Path("dee-cli"), _cell(("hmp", "pushdown")), _prepared(tmp_path),
                                tmp_path / "o.json", tmp_path / "r.json", None)
        assert "--enable" in cmd
        assert cmd[cmd.index("--enable") + 1] == "hmp,pushdown"
        # --disable would leave the pass set dependent on dee's defaults.
        assert "--disable" not in cmd

    def test_renders_value_options_as_flag_and_value(self, tmp_path):
        cmd = build_opt_command(Path("dee-cli"), _cell(dee_opt={"hmp_max_runs": 4}),
                                _prepared(tmp_path), tmp_path / "o.json",
                                tmp_path / "r.json", None)
        assert cmd[cmd.index("--hmp-max-runs") + 1] == "4"

    def test_emits_a_plain_bool_flag_only_when_true(self, tmp_path):
        on = build_opt_command(Path("dee-cli"), _cell(dee_opt={"hmp_downstream_cost": True}),
                               _prepared(tmp_path), tmp_path / "o", tmp_path / "r", None)
        off = build_opt_command(Path("dee-cli"), _cell(dee_opt={"hmp_downstream_cost": False}),
                                _prepared(tmp_path), tmp_path / "o", tmp_path / "r", None)
        assert "--hmp-downstream-cost" in on
        assert "--hmp-downstream-cost" not in off

    def test_emits_a_negated_flag_only_when_the_option_is_disabled(self, tmp_path):
        # hmp_use_pushdown is on by default in dee; the CLI expresses turning
        # it off as --hmp-no-pushdown, so the flag appears when the value is False.
        off = build_opt_command(Path("dee-cli"), _cell(dee_opt={"hmp_use_pushdown": False}),
                                _prepared(tmp_path), tmp_path / "o", tmp_path / "r", None)
        on = build_opt_command(Path("dee-cli"), _cell(dee_opt={"hmp_use_pushdown": True}),
                               _prepared(tmp_path), tmp_path / "o", tmp_path / "r", None)
        assert "--hmp-no-pushdown" in off
        assert "--hmp-no-pushdown" not in on

    def test_always_requests_the_machine_readable_report(self, tmp_path):
        cmd = build_opt_command(Path("dee-cli"), _cell(), _prepared(tmp_path),
                                tmp_path / "o.json", tmp_path / "r.json", None)
        assert "--report-json" in cmd


class TestRunCommand:
    def test_passes_repetitions_and_warmups_through(self, tmp_path):
        cmd = build_run_command(Path("dee-cli"), _cell(), _prepared(tmp_path),
                                tmp_path / "dag.json", tmp_path / "r.json", 100)
        assert cmd[cmd.index("--repeat") + 1] == "5"
        assert cmd[cmd.index("--warmups") + 1] == "2"
