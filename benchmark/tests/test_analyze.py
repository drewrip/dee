"""Payback derivation (study 3)."""

from __future__ import annotations

from dee_bench.analyze import _bootstrap_payback_ci, _payback, compute_payback
from dee_bench.schema import Verbosity
from dee_bench.store import ResultStore


class TestPaybackMath:
    def test_divides_cost_by_savings(self):
        assert _payback(10.0, 0.5) == 20.0

    def test_is_none_when_the_variant_is_slower(self):
        assert _payback(10.0, -0.5) is None

    def test_is_none_when_there_is_no_saving_at_all(self):
        # Exactly break-even never repays, so it must not report infinity.
        assert _payback(10.0, 0.0) is None

    def test_is_none_without_a_cost(self):
        assert _payback(None, 0.5) is None


class TestBootstrap:
    def test_brackets_the_point_estimate(self):
        baseline = [1.00, 1.02, 0.98, 1.01, 0.99]
        variant = [0.80, 0.82, 0.78, 0.81, 0.79]
        lo, hi = _bootstrap_payback_ci(baseline, variant, cost_s=10.0)
        assert lo is not None and hi is not None
        assert lo <= 10.0 / 0.2 <= hi

    def test_declines_to_report_when_improvement_is_not_reliable(self):
        # Overlapping distributions: many resamples show no improvement, so an
        # interval over only the favourable ones would be misleading.
        noisy = [1.0, 1.1, 0.9, 1.05, 0.95]
        lo, hi = _bootstrap_payback_ci(noisy, list(noisy), cost_s=10.0)
        assert lo is None and hi is None


def _seed(tmp_path, variant_wall_ms, opt_wall_ms=2000):
    """A minimal run directory with one baseline cell and one optimized cell."""
    st = ResultStore(tmp_path, Verbosity.FULL)
    for cid, variant in (("base", "unopt"), ("opt", "hmp")):
        st.write("cells", [{
            "cell_id": cid, "run_name": "t", "project": "p01_iot", "backend": "duckdb",
            "sf": 0.1, "variant": variant, "passes": [] if variant == "unopt" else ["hmp"],
        }], cell_id=cid)
    for i in range(3):
        st.write("runs", [{
            "cell_id": "base", "run_id": f"b{i}", "phase": "measure", "status": "ok",
            "engine_wall_ms": 1000, "cpu_seconds": 4.0,
        }], cell_id="base")
        st.write("runs", [{
            "cell_id": "opt", "run_id": f"o{i}", "phase": "measure", "status": "ok",
            "engine_wall_ms": variant_wall_ms, "cpu_seconds": 3.0,
        }], cell_id="opt")
    st.write("optimizations", [{
        "cell_id": "opt", "opt_wall_ms": opt_wall_ms, "opt_cpu_seconds": 8.0, "status": "ok",
    }], cell_id="opt")
    return st


class TestComputePayback:
    def test_computes_speedup_and_payback_against_the_baseline(self, tmp_path):
        _seed(tmp_path, variant_wall_ms=800)
        rows = compute_payback(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["variant"] == "hmp"
        assert row["baseline_cell_id"] == "base"
        assert row["speedup"] == 1.25
        assert abs(row["savings_per_run_wall_s"] - 0.2) < 1e-9
        # 2s of optimization repaid at 0.2s per run.
        assert abs(row["payback_runs_wall"] - 10.0) < 1e-9

    def test_reports_no_payback_when_the_variant_regressed(self, tmp_path):
        _seed(tmp_path, variant_wall_ms=1200)
        row = compute_payback(tmp_path)[0]
        assert row["payback_runs_wall"] is None
        assert row["speedup"] < 1

    def test_baseline_cells_produce_no_payback_row(self, tmp_path):
        _seed(tmp_path, variant_wall_ms=800)
        assert all(r["cell_id"] != "base" for r in compute_payback(tmp_path))

    def test_empty_results_yield_no_rows(self, tmp_path):
        assert compute_payback(tmp_path) == []
