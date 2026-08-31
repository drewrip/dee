"""Measuring a cell through the server's run queue.

`repeat_mode: queue` turns one trigger carrying five repetitions into five
queued run groups carrying one each. What has to survive that change is the
shape of the results: the same number of measured runs, `rep_index` still
counting 0..n-1 within a phase, and every row still joinable to dee's own run
history.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from dee_bench.config import Variant
from dee_bench.matrix import Cell
from dee_bench.runner import CellRunner
from dee_bench.sampler import PhaseMetrics
from dee_bench.server import ServerError


def _cell(repetitions=3, warmups=1, repeat_mode="queue"):
    return Cell(
        cell_id="c1", run_name="r", project="p01_iot", backend="duckdb", sf=0.1,
        variant=Variant(name="unopt", passes=()),
        dee_opt={}, backend_config={},
        repetitions=repetitions, warmups=warmups, repeat_mode=repeat_mode,
    )


class FakeClient:
    """Records what the runner asked for and hands back plausible shapes."""

    def __init__(self, fail_on_call: int | None = None):
        self.enqueued: list[tuple[str, dict]] = []
        self.triggered: list[tuple[str, dict]] = []
        self.cleared: list[str | None] = []
        self._fail_on_call = fail_on_call
        self._groups: dict[str, list[tuple[str, int]]] = {}
        self._next = 0

    def enqueue(self, name, body, timeout):
        self.enqueued.append((name, body))
        if self._fail_on_call is not None and len(self.enqueued) >= self._fail_on_call:
            raise ServerError("the server gave up waiting")
        entries = []
        for _ in range(body["count"]):
            group_id = f"g{self._next}"
            self._next += 1
            runs = [("warmup", i) for i in range(body.get("warmups", 0))]
            runs += [("measure", i) for i in range(body["repetitions"])]
            self._groups[group_id] = runs
            entries.append({"run_group_id": group_id, "run_ids": []})
        return {"entries": entries}

    def trigger(self, name, body, timeout):
        self.triggered.append((name, body))
        group_id = "g0"
        self._groups[group_id] = (
            [("warmup", i) for i in range(body.get("warmups", 0))]
            + [("measure", i) for i in range(body["repetitions"])]
        )
        return {"run_group_id": group_id}

    def clear_queue(self, dag=None):
        self.cleared.append(dag)

    def run_group(self, group_id):
        return {
            "status": "succeeded",
            "runs": [
                {"run_id": f"{group_id}-{phase}{i}", "phase": phase, "rep_index": i}
                for phase, i in self._groups[group_id]
            ],
        }

    def group_report(self, group_id):
        return {
            "runs": [
                {
                    "phase": phase,
                    "rep_index": i,
                    "duration_ms": 100,
                    "time_executing_nodes_ms": 90,
                    "graph": {"nodes": [{"id": "a", "materialization": "table"}]},
                    "node_executions": [{"node_id": "a", "duration_ms": 90}],
                    "system_samples": [],
                }
                for phase, i in self._groups[group_id]
            ]
        }


class StubRunner(CellRunner):
    """A runner with the sampler stubbed out, so no threads or pids."""

    @contextmanager
    def _sample_phase(self):
        handle = SimpleNamespace(result=(PhaseMetrics(), 1000))
        yield handle


def _runner(client, tmp_path, records_plans=False):
    cfg = SimpleNamespace(
        execution=SimpleNamespace(sample_interval_ms=50, timeout_s=60),
    )
    store = SimpleNamespace(records=lambda table: records_plans)
    return StubRunner(cfg, store, Path(tmp_path), client)


class TestQueueing:
    def test_one_entry_per_repetition_with_warmups_at_the_front(self, tmp_path):
        client = FakeClient()
        runner = _runner(client, tmp_path)
        groups = runner._execute_queued(_cell(repetitions=3, warmups=2), "cdag", 7, tmp_path)

        # Two requests: the first entry carries the warmups, the rest do not.
        assert [b["count"] for _, b in client.enqueued] == [1, 2]
        assert [b["warmups"] for _, b in client.enqueued] == [2, 0]
        # Three repetitions, so three groups.
        assert len(groups) == 3

    def test_the_version_is_pinned_on_every_entry(self, tmp_path):
        # An entry naming no version follows the DAG to whatever is current
        # when its turn comes. That is right for watching a DAG adapt and
        # wrong here: the cell has already chosen what it measures.
        client = FakeClient()
        runner = _runner(client, tmp_path)
        runner._execute_queued(_cell(), "cdag", 7, tmp_path)
        assert all(body["version"] == 7 for _, body in client.enqueued)

    def test_a_cell_with_no_warmups_makes_one_request(self, tmp_path):
        client = FakeClient()
        runner = _runner(client, tmp_path)
        runner._execute_queued(_cell(repetitions=4, warmups=0), "cdag", 1, tmp_path)
        assert [b["count"] for _, b in client.enqueued] == [4]

    def test_giving_up_clears_what_is_still_waiting(self, tmp_path):
        # Entries left queued would run against the warehouse while the next
        # cell is being measured: not a failure, just quietly wrong numbers.
        client = FakeClient(fail_on_call=2)
        runner = _runner(client, tmp_path)
        with pytest.raises(ServerError):
            runner._execute_queued(_cell(repetitions=3, warmups=1), "cdag", 1, tmp_path)
        assert client.cleared == ["cdag"]


class TestResultShape:
    def _rows(self, tmp_path, repeat_mode):
        artifacts = Path(tmp_path)
        artifacts.mkdir(parents=True, exist_ok=True)
        client = FakeClient()
        runner = _runner(client, artifacts)
        result = SimpleNamespace(rows={}, measured_runs=0)
        ctx = SimpleNamespace(plan_time_basis="wall")
        runner._measure(
            _cell(repetitions=3, warmups=1, repeat_mode=repeat_mode),
            "cdag", 7, ctx, artifacts, "unopt", result,
        )
        return result

    def test_the_two_modes_produce_the_same_measurements(self, tmp_path):
        queued = self._rows(tmp_path / "q", "queue")
        grouped = self._rows(tmp_path / "g", "group")

        assert queued.measured_runs == grouped.measured_runs == 3
        for rows in (queued.rows["runs"], grouped.rows["runs"]):
            measured = [r for r in rows if r["phase"] == "measure"]
            # rep_index counts within a phase across the whole cell. Under
            # `queue` each group holds one run and would otherwise report
            # index 0 three times over.
            assert [r["rep_index"] for r in measured] == [0, 1, 2]
            assert [r["rep_index"] for r in rows if r["phase"] == "warmup"] == [0]

    def test_every_run_records_the_group_it_came_from(self, tmp_path):
        queued = self._rows(tmp_path / "q", "queue")
        grouped = self._rows(tmp_path / "g", "group")

        # This column is how the two modes are told apart in the data.
        assert len({r["run_group_id"] for r in queued.rows["runs"]}) == 3
        assert len({r["run_group_id"] for r in grouped.rows["runs"]}) == 1

    def test_run_ids_come_from_the_server_not_the_harness(self, tmp_path):
        # They are the join back to dee's own history; invented ones join to
        # nothing.
        result = self._rows(tmp_path, "queue")
        assert all(r["run_id"].startswith("g") for r in result.rows["runs"])
