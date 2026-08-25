"""File-backed run queue: submit, resume, inspect, cancel.

State lives in ``<run_dir>/state/<cell_id>.json``, one small file per cell,
written atomically. That choice buys three things a single state file would
not:

* **Crash safety.** A `kill -9` can lose at most the in-flight cell's status;
  every other cell's state is already durable on disk.
* **Resumability.** Resuming is a directory scan, not a replay.
* **No contention** with the parquet fragments the runner is writing.

Cells run strictly one at a time — concurrency would destroy timing fidelity,
which is the entire point of the exercise.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"

TERMINAL = {DONE, SKIPPED}


@dataclass
class CellState:
    cell_id: str
    status: str = PENDING
    describe: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    error: str | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CellState":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class RunQueue:
    """The set of cells for one run directory, and their statuses."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.state_dir = self.run_dir / "state"

    # -- state -------------------------------------------------------------

    def path_for(self, cell_id: str) -> Path:
        return self.state_dir / f"{cell_id}.json"

    def get(self, cell_id: str) -> CellState:
        path = self.path_for(cell_id)
        if not path.exists():
            return CellState(cell_id=cell_id)
        try:
            return CellState.from_dict(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            # A torn state file means the harness died mid-write; treat the
            # cell as never having run rather than failing the whole resume.
            return CellState(cell_id=cell_id)

    def put(self, state: CellState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(state.cell_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2))
        os.replace(tmp, path)

    def all_states(self) -> list[CellState]:
        if not self.state_dir.exists():
            return []
        out = []
        for p in sorted(self.state_dir.glob("*.json")):
            try:
                out.append(CellState.from_dict(json.loads(p.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    # -- transitions -------------------------------------------------------

    def initialize(self, cells: Iterable[Any]) -> None:
        """Seed state for any cell that doesn't have it yet, preserving the rest."""
        for cell in cells:
            existing = self.path_for(cell.cell_id)
            if not existing.exists():
                self.put(CellState(cell_id=cell.cell_id, describe=cell.describe()))

    def mark_running(self, cell_id: str, describe: str = "") -> None:
        state = self.get(cell_id)
        state.status = RUNNING
        state.describe = describe or state.describe
        state.started_at = datetime.now(timezone.utc).isoformat()
        state.finished_at = None
        state.error = None
        state.attempts += 1
        self.put(state)

    def mark_finished(self, cell_id: str, status: str, error: str | None = None) -> None:
        state = self.get(cell_id)
        state.status = status
        state.finished_at = datetime.now(timezone.utc).isoformat()
        state.error = error
        if state.started_at:
            started = datetime.fromisoformat(state.started_at)
            state.duration_s = (datetime.now(timezone.utc) - started).total_seconds()
        self.put(state)

    def pending(self, cells: list[Any], retry_failed: bool = True) -> list[Any]:
        """Cells still needing work, in scheduled order.

        A cell left in `running` is one the worker died during; it is retried,
        since its results were never completely written.
        """
        out = []
        for cell in cells:
            status = self.get(cell.cell_id).status
            if status in TERMINAL:
                continue
            if status == FAILED and not retry_failed:
                continue
            out.append(cell)
        return out

    def counts(self) -> dict[str, int]:
        counts = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0, SKIPPED: 0}
        for s in self.all_states():
            counts[s.status] = counts.get(s.status, 0) + 1
        return counts

    # -- worker process ----------------------------------------------------

    @property
    def pid_file(self) -> Path:
        return self.run_dir / "worker.pid"

    def write_pid(self, pid: int) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(pid))

    def worker_pid(self) -> int | None:
        """The worker's pid, if one is actually alive."""
        if not self.pid_file.exists():
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return pid

    def cancel(self, timeout: float = 15.0) -> bool:
        """Stop the worker, giving it a chance to tear infrastructure down.

        SIGTERM first, because the postgres backend's signal handler removes
        its container on the way out; SIGKILL only if it will not go.
        """
        pid = self.worker_pid()
        if pid is None:
            return False
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.worker_pid() is None:
                return True
            time.sleep(0.25)
        os.kill(pid, signal.SIGKILL)
        return True

    def estimate_remaining(self, total: int) -> float | None:
        """Seconds left, from the mean duration of completed cells."""
        durations = [s.duration_s for s in self.all_states() if s.duration_s and s.status == DONE]
        if not durations:
            return None
        counts = self.counts()
        remaining = total - counts.get(DONE, 0) - counts.get(SKIPPED, 0)
        return (sum(durations) / len(durations)) * max(remaining, 0)
