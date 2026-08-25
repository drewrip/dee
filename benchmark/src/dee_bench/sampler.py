"""External resource sampling for a benchmark phase.

Ground truth for CPU, memory and I/O comes from *outside* the process being
measured, for two reasons:

* dee's own Postgres connector reports no CPU or disk at all, and its memory
  sample reads ``pg_backend_memory_contexts`` — the monitoring connection's own
  backend, not the workers doing the work. Study 7 is unmeasurable from inside.
* CPU is read here as **counter deltas** (``/proc/<pid>/stat`` utime+stime,
  cgroup ``cpu.stat`` usage_usec) rather than sampled percentages. A sampled
  percentage integrated over time misses everything between samples; a counter
  cannot. dee's internal estimate trapezoidally integrates ``ps -o %cpu``,
  which is why the two can disagree.

Two sources are sampled, and recorded distinctly:

``harness_process``
    The dee-cli process tree, from ``/proc``. Covers DuckDB entirely, since it
    is in-process.
``harness_container``
    The postgres container's cgroup. Covers the server-side work DuckDB does
    not have.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


@dataclass
class Sample:
    source: str
    elapsed_ms: int
    timestamp: datetime
    cpu_seconds_cum: float | None = None
    rss_bytes: int | None = None
    read_bytes: int | None = None
    written_bytes: int | None = None


@dataclass
class PhaseMetrics:
    """Aggregates for one sampled phase, written onto the `runs` row."""

    cpu_seconds: float | None = None
    peak_rss_bytes: int | None = None
    read_bytes: int | None = None
    written_bytes: int | None = None
    samples: list[Sample] = field(default_factory=list)

    def as_run_columns(self) -> dict[str, Any]:
        return {
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "read_bytes": self.read_bytes,
            "written_bytes": self.written_bytes,
        }


# --------------------------------------------------------------------------
# /proc readers
# --------------------------------------------------------------------------


def _proc_children(pid: int) -> list[int]:
    """Every descendant of `pid`, via /proc/<pid>/task/*/children."""
    out: list[int] = []
    stack = [pid]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        out.append(cur)
        task_dir = Path(f"/proc/{cur}/task")
        try:
            for task in task_dir.iterdir():
                try:
                    kids = (task / "children").read_text().split()
                except (OSError, ValueError):
                    continue
                stack.extend(int(k) for k in kids)
        except OSError:
            continue
    return out


def _read_proc_stat_cpu(pid: int) -> float | None:
    """CPU seconds this process has used (utime + stime)."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # comm may contain spaces and parentheses, so split after the last ')'.
    try:
        rest = data[data.rindex(")") + 2 :].split()
        utime, stime = int(rest[11]), int(rest[12])
    except (ValueError, IndexError):
        return None
    return (utime + stime) / CLK_TCK


def _read_proc_rss(pid: int) -> int | None:
    try:
        statm = Path(f"/proc/{pid}/statm").read_text().split()
        return int(statm[1]) * PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return None


def _read_proc_io(pid: int) -> tuple[int | None, int | None]:
    try:
        text = Path(f"/proc/{pid}/io").read_text()
    except OSError:
        # Unreadable without matching credentials; not fatal.
        return None, None
    read = written = None
    for line in text.splitlines():
        if line.startswith("read_bytes:"):
            read = int(line.split()[1])
        elif line.startswith("write_bytes:"):
            written = int(line.split()[1])
    return read, written


def sample_process_tree(pid: int) -> Sample | None:
    """One sample summed over `pid` and every descendant."""
    pids = _proc_children(pid)
    if not pids:
        return None
    cpu = rss = read = written = 0.0
    any_cpu = False
    for p in pids:
        c = _read_proc_stat_cpu(p)
        if c is not None:
            cpu += c
            any_cpu = True
        r = _read_proc_rss(p)
        if r:
            rss += r
        rb, wb = _read_proc_io(p)
        if rb:
            read += rb
        if wb:
            written += wb
    if not any_cpu:
        return None
    now = datetime.now(timezone.utc)
    return Sample(
        source="harness_process",
        elapsed_ms=0,
        timestamp=now,
        cpu_seconds_cum=cpu,
        rss_bytes=int(rss),
        read_bytes=int(read),
        written_bytes=int(written),
    )


# --------------------------------------------------------------------------
# cgroup v2 readers (for a containerized postgres)
# --------------------------------------------------------------------------


def find_cgroup_path(container_id: str) -> Path | None:
    """Locate a container's cgroup v2 directory on the host.

    The layout depends on the runtime and whether it is rootful or rootless:
    docker uses ``docker-<id>.scope`` under system.slice, while rootless podman
    nests ``libpod-<id>.scope`` under the invoking user's slice. Both are tried
    before falling back to a scan, because without this the postgres
    container's CPU and memory go unsampled and study 7 loses its server-side
    half.
    """
    uid = os.getuid()
    user_slice = (
        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service"
    )
    roots = [
        # docker, rootful
        Path(f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope"),
        Path(f"/sys/fs/cgroup/docker/{container_id}"),
        # podman, rootful
        Path(f"/sys/fs/cgroup/machine.slice/libpod-{container_id}.scope"),
        # podman, rootless
        Path(f"{user_slice}/user.slice/libpod-{container_id}.scope"),
        Path(f"{user_slice}/libpod-{container_id}.scope"),
    ]
    for r in roots:
        if (r / "cpu.stat").exists():
            return r
    # Fall back to a scan; layouts vary by cgroup driver and systemd version.
    for base in (Path("/sys/fs/cgroup"),):
        if not base.exists():
            continue
        try:
            for candidate in base.glob(f"**/*{container_id[:12]}*"):
                if (candidate / "cpu.stat").exists():
                    return candidate
        except OSError:
            continue
    return None


class CgroupReader:
    """Reads cpu/memory/io counters for one container's cgroup."""

    def __init__(self, cgroup: Path):
        self.cgroup = cgroup

    def sample(self) -> Sample | None:
        cpu = self._cpu_seconds()
        if cpu is None:
            return None
        read, written = self._io_bytes()
        return Sample(
            source="harness_container",
            elapsed_ms=0,
            timestamp=datetime.now(timezone.utc),
            cpu_seconds_cum=cpu,
            rss_bytes=self._memory_bytes(),
            read_bytes=read,
            written_bytes=written,
        )

    def _cpu_seconds(self) -> float | None:
        try:
            for line in (self.cgroup / "cpu.stat").read_text().splitlines():
                if line.startswith("usage_usec"):
                    return int(line.split()[1]) / 1e6
        except (OSError, ValueError):
            return None
        return None

    def _memory_bytes(self) -> int | None:
        try:
            return int((self.cgroup / "memory.current").read_text().strip())
        except (OSError, ValueError):
            return None

    def _io_bytes(self) -> tuple[int | None, int | None]:
        read = written = 0
        found = False
        try:
            for line in (self.cgroup / "io.stat").read_text().splitlines():
                for field_ in line.split()[1:]:
                    key, _, value = field_.partition("=")
                    if key == "rbytes":
                        read += int(value)
                        found = True
                    elif key == "wbytes":
                        written += int(value)
                        found = True
        except (OSError, ValueError):
            return None, None
        return (read, written) if found else (None, None)


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------


class PhaseSampler:
    """Samples a phase in a background thread until stopped.

    Usage::

        sampler = PhaseSampler(interval_ms=100, cgroup=cg)
        sampler.start()
        pid = launch_subprocess()
        sampler.attach(pid)
        ...
        metrics = sampler.stop()

    Counters are reported *relative to the phase start*, so `cpu_seconds_cum`
    is the CPU burned by this phase alone rather than since process start.
    """

    def __init__(self, interval_ms: int = 100, cgroup: CgroupReader | None = None):
        self.interval = max(interval_ms, 10) / 1000.0
        self.cgroup = cgroup
        self._pid: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples: list[Sample] = []
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._baselines: dict[str, Sample] = {}

    def attach(self, pid: int) -> None:
        """Point the process-tree sampler at a newly launched process."""
        self._pid = pid

    def start(self) -> "PhaseSampler":
        self._t0 = time.monotonic()
        self._stop.clear()
        self._samples = []
        self._baselines = {}
        if self.cgroup is not None:
            s = self.cgroup.sample()
            if s is not None:
                self._baselines["harness_container"] = s
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _collect(self) -> list[Sample]:
        out = []
        if self._pid is not None:
            s = sample_process_tree(self._pid)
            if s is not None:
                out.append(s)
        if self.cgroup is not None:
            s = self.cgroup.sample()
            if s is not None:
                out.append(s)
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval)

    def _take(self) -> None:
        elapsed_ms = int((time.monotonic() - self._t0) * 1000)
        for s in self._collect():
            # The process tree's first sample is its own baseline: the
            # subprocess starts at zero CPU, so nothing is subtracted. The
            # container has been running since before the phase, so its
            # counters must be rebased.
            base = self._baselines.setdefault(s.source, s if s.source != "harness_process" else
                                              Sample(s.source, 0, s.timestamp, 0.0, 0, 0, 0))
            s.elapsed_ms = elapsed_ms
            s.cpu_seconds_cum = _delta(s.cpu_seconds_cum, base.cpu_seconds_cum)
            s.read_bytes = _delta(s.read_bytes, base.read_bytes)
            s.written_bytes = _delta(s.written_bytes, base.written_bytes)
            with self._lock:
                self._samples.append(s)

    def stop(self) -> PhaseMetrics:
        """Stop sampling and return this phase's aggregates and timeseries."""
        # A final sample after the work finishes, so a short phase still has
        # an endpoint carrying its full counter delta.
        self._take()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            samples = list(self._samples)

        metrics = PhaseMetrics(samples=samples)
        if not samples:
            return metrics

        # CPU and I/O are cumulative counters, so the phase total is the
        # maximum reached across every source summed together.
        for source in {s.source for s in samples}:
            group = [s for s in samples if s.source == source]
            metrics.cpu_seconds = _accumulate(
                metrics.cpu_seconds, max((s.cpu_seconds_cum or 0.0) for s in group)
            )
            metrics.read_bytes = _accumulate(
                metrics.read_bytes, max((s.read_bytes or 0) for s in group)
            )
            metrics.written_bytes = _accumulate(
                metrics.written_bytes, max((s.written_bytes or 0) for s in group)
            )
        # Peak memory is an instantaneous value, so it is the largest single
        # observation, not a sum across sources.
        rss = [s.rss_bytes for s in samples if s.rss_bytes is not None]
        metrics.peak_rss_bytes = max(rss) if rss else None
        return metrics


def _delta(current: float | int | None, base: float | int | None) -> float | int | None:
    if current is None:
        return None
    if base is None:
        return current
    return max(type(current)(0), current - base)


def _accumulate(total: float | int | None, value: float | int) -> float | int:
    return value if total is None else total + value


def samples_to_rows(
    metrics: PhaseMetrics, *, cell_id: str, run_id: str, phase: str
) -> list[dict[str, Any]]:
    """Render a phase's samples as `system_samples` rows."""
    return [
        {
            "cell_id": cell_id,
            "run_id": run_id,
            "phase": phase,
            "source": s.source,
            "elapsed_ms": s.elapsed_ms,
            "timestamp": s.timestamp,
            "cpu_seconds_cum": s.cpu_seconds_cum,
            "rss_bytes": s.rss_bytes,
            "engine_mem_bytes": None,
            "read_bytes": s.read_bytes,
            "written_bytes": s.written_bytes,
        }
        for s in metrics.samples
    ]
