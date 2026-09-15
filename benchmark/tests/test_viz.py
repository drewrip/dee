"""The visualization suite: which pages a run grows, and what they carry.

These build real dashboards from synthetic result directories, because that is
the contract that matters — a page appears when the results support it, every
chart is downloadable, and a partial run renders rather than raising.
"""

from __future__ import annotations

import json

import pytest

from dee_bench.schema import Verbosity
from dee_bench.store import ResultStore, connect
from dee_bench.viz import passes
from dee_bench.viz.continuous import build as build_continuous
from dee_bench.viz.dashboard import build, build_pages


def _cell(store, cell_id, variant, pass_list, *, mode="batch", project="p01_iot"):
    store.write("cells", [{
        "cell_id": cell_id, "run_name": "t", "project": project, "backend": "duckdb",
        "sf": 1.0, "variant": variant, "passes": pass_list, "dee_opt": "{}",
        "backend_config": "{}", "repetitions": 3, "warmups": 1,
        "repeat_mode": "group", "optimization_mode": mode,
    }], cell_id=cell_id)


def _runs(store, cell_id, walls, *, version=None, phase="measure", variant="unopt",
          start=0):
    for i, wall in enumerate(walls):
        store.write("runs", [{
            "cell_id": cell_id, "run_id": f"{cell_id}-{phase}-{start + i}",
            "run_group_id": cell_id, "dag_variant": variant,
            "dag_version": version[i] if isinstance(version, list) else version,
            "phase": phase, "rep_index": i, "engine_wall_ms": wall,
            "node_time_ms": wall * 2, "cpu_seconds": wall / 200.0,
            "peak_rss_bytes": 400_000_000, "delivery": "direct",
            "status": "ok", "plan_time_basis": "cpu_time",
        }], cell_id=cell_id)


@pytest.fixture
def batch_run(tmp_path):
    """A baseline against an HMP cell, with a search trace behind it."""
    store = ResultStore(tmp_path, Verbosity.FULL)
    _cell(store, "base", "unopt", [])
    _cell(store, "opt", "hmp", ["hmp"])
    _runs(store, "base", [1000, 1020, 990])
    _runs(store, "opt", [800, 810, 795], variant="optimized")
    store.write("optimizations", [{
        "cell_id": "opt", "opt_wall_ms": 4000, "opt_cpu_seconds": 12.0,
        "dag_runs_used": 3, "status": "ok", "total_changes_applied": 1,
    }], cell_id="opt")
    store.write("pass_stats", [{
        "cell_id": "opt", "pass_name": "HMPPass", "pass_order": 0, "wall_ms": 4000,
        "dag_runs_used": 3, "changes_applied": 1, "candidates_considered": 4,
        "working_set_size": 2,
        "detail": json.dumps({"kind": "hmp", "baseline_runtime_ms": 1000,
                              "final_runtime_ms": 800,
                              "new_materializations": ['"w"."m"."a"'],
                              "working_set": ['"w"."m"."a"', '"w"."m"."b"']}),
    }], cell_id="opt")
    store.write("pass_iterations", [
        {"cell_id": "opt", "pass_name": "HMPPass", "iteration": 1,
         "runtime_ms": 1000, "total_ms": 1000, "combo": [], "outcome": "baseline"},
        {"cell_id": "opt", "pass_name": "HMPPass", "iteration": 2,
         "runtime_ms": 800, "total_ms": 800, "combo": ['"w"."m"."a"'],
         "outcome": "ok", "predicted_makespan_ms": 850},
        {"cell_id": "opt", "pass_name": "HMPPass", "iteration": 3,
         "runtime_ms": 800, "total_ms": 1400, "trial_ms": 810,
         "resume_overhead_ms": 90, "resume_ms": 500,
         "combo": ['"w"."m"."b"'], "outcome": "cancelled"},
    ], cell_id="opt")
    return tmp_path


@pytest.fixture
def continuous_run(tmp_path):
    """A DAG optimized in place: the version rises at run 4 of 8."""
    store = ResultStore(tmp_path, Verbosity.FULL)
    _cell(store, "base", "unopt", [])
    _cell(store, "cont", "hmp", ["hmp"], mode="continuous")
    _runs(store, "base", [1000, 1010, 990])
    _runs(store, "cont", [1200], version=1, phase="warmup", variant="converging")
    _runs(store, "cont", [1000, 1400, 1300, 700, 710, 690, 705],
          version=[1, 1, 1, 2, 2, 2, 2], variant="converging", start=1)
    store.write("optimizations", [{
        "cell_id": "cont", "opt_wall_ms": 0, "dag_runs_used": 0,
        "status": "converged", "optimization_type": "continuous",
        "step_phase": "before", "result_version": 2,
    }], cell_id="cont")
    return tmp_path


class TestPages:
    def test_a_batch_run_grows_a_page_for_the_pass_it_ran(self, batch_run):
        pages, _ = build_pages(batch_run)
        keys = [p.key for p in pages]
        assert "overview" in keys
        assert "pass-hmp" in keys
        # Nothing ran the other passes, so they get no page at all.
        assert not {"pass-omp", "pass-nodefusion", "pass-parallelism"} & set(keys)

    def test_the_seven_studies_are_always_present(self, batch_run):
        pages, _ = build_pages(batch_run)
        numbered = sorted(p.number for p in pages if p.group == "studies")
        assert numbered == [1, 2, 3, 4, 5, 6, 7]

    def test_a_batch_run_has_no_run_by_run_page(self, batch_run):
        assert build_continuous(connect(batch_run)) is None

    def test_the_hmp_page_reads_its_search_trace(self, batch_run):
        pages, _ = build_pages(batch_run)
        hmp = next(p for p in pages if p.key == "pass-hmp")
        assert any(c.id.startswith("hmp-trace") for c in hmp.charts)
        assert any(c.id == "hmp-prediction" for c in hmp.charts)
        # The cancelled candidate's real cost, not the bound it was cut at.
        trace = next(c for c in hmp.charts if c.id.startswith("hmp-trace"))
        assert trace.series[0].y == [1.0, 0.8, 1.4]


class TestContinuous:
    def test_it_finds_the_run_the_search_promoted_on(self, continuous_run):
        page = build_continuous(connect(continuous_run))
        assert page is not None
        runs = next(c for c in page.charts if c.id.startswith("continuous-runs"))
        # Warmup first, then seven measured runs; the version rises on the
        # fifth execution overall.
        assert runs.vlines == [(5, "promoted v2")]

    def test_it_separates_converging_runs_from_settled_ones(self, continuous_run):
        page = build_continuous(connect(continuous_run))
        settled = next(k for k in page.kpis if k.label == "Settled speedup")
        # Baseline 1.0s against a settled median of ~0.7s.
        assert settled.value.startswith("1.4")

    def test_it_says_when_being_optimized_in_place_paid_for_itself(self, continuous_run):
        page = build_continuous(connect(continuous_run))
        crossover = next(c for c in page.charts if c.id.startswith("continuous-crossover"))
        assert "cross at run" in crossover.note

    def test_a_search_that_never_promoted_is_not_read_as_converged(self, tmp_path):
        store = ResultStore(tmp_path, Verbosity.FULL)
        _cell(store, "cont", "hmp", ["hmp"], mode="continuous")
        _runs(store, "cont", [1000, 1010, 1005], version=1, variant="converging")
        store.write("optimizations", [{
            "cell_id": "cont", "opt_wall_ms": 0, "status": "converging",
            "optimization_type": "continuous", "result_version": None,
        }], cell_id="cont")
        page = build_continuous(connect(tmp_path))
        converged = next(k for k in page.kpis if k.label == "Cells converged")
        assert converged.value == "0/1"


class TestDownloads:
    def test_every_chart_is_downloadable_as_png_and_pdf(self, batch_run):
        index = build(batch_run)
        charts = index.parent / "charts"
        pages, _ = build_pages(batch_run)
        drawn = [c.id for page in pages for c in page.charts if c.has_data]
        assert drawn
        for chart_id in drawn:
            assert (charts / f"{chart_id}.png").exists(), chart_id
            assert (charts / f"{chart_id}.pdf").exists(), chart_id

    def test_the_page_links_both_formats_for_every_chart(self, batch_run):
        index = build(batch_run)
        html = index.read_text()
        pages, _ = build_pages(batch_run)
        for page in pages:
            for chart in page.charts:
                if not chart.has_data:
                    continue
                assert f'href="charts/{chart.id}.png"' in html
                assert f'href="charts/{chart.id}.pdf"' in html

    def test_asking_for_html_alone_still_renders_what_it_links(self, batch_run):
        # The page always offers both formats, so building it builds them.
        index = build(batch_run, formats={"html"})
        assert (index.parent / "charts").glob("*.pdf")


class TestPartialRuns:
    def test_a_run_with_only_a_baseline_still_renders(self, tmp_path):
        store = ResultStore(tmp_path, Verbosity.FULL)
        _cell(store, "base", "unopt", [])
        _runs(store, "base", [1000, 1010])
        index = build(tmp_path)
        assert index is not None
        assert "Nothing to show yet" in index.read_text()

    def test_an_empty_directory_builds_nothing(self, tmp_path):
        assert build(tmp_path) is None

    def test_pass_detection_reads_the_results_not_the_config(self, batch_run):
        assert passes.enabled(connect(batch_run)) == ["hmp"]
