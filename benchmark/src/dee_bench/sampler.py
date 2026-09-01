"""External resource sampling for a benchmark phase.

Ground truth for CPU, memory and I/O comes from *outside* the process being
measured, for two reasons:

* dee's own Postgres connector reports no CPU or disk at all, and its memory
  sample reads ``pg_backend_memory_contexts`` — the monitoring connection's own
  backend, not the workers doing the work. Study 7 is unmeasurable from inside.
* CPU is read here as **counter deltas** (process cumulative CPU time, cgroup
  ``cpu.stat`` usage_usec) rather than sampled percentages. A sampled
  percentage integrated over time misses everything between samples; a counter
  cannot. dee's internal estimate trapezoidally integrates ``ps -o %cpu``,
  which is why the two can disagree.

Two sources are sampled, and recorded distinctly:

``harness_process``
    The dee-cli process tree, via ``psutil``. Covers DuckDB entirely, since it
    is in-process. ``psutil`` (rather than ``/proc`` directly) is what makes
    this work on macOS dev machines as well as Linux.
``harness_container``
    The postgres container's cgroup, read through ``exec`` rather than the
    host's ``/sys/fs/cgroup``: on Docker Desktop (macOS) containers run inside
    a Linux VM the host filesystem never exposes, so there is no cgroup path
    to find on the host at all. Reading it from inside the container works
    the same way regardless of host OS or cgroup driver.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import psutil


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
# process-tree reader (psutil, so it works on both Linux and macOS)
# --------------------------------------------------------------------------


def sample_process_tree(pid: int) -> Sample | None:
    """One sample summed over `pid` and every descendant."""
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return None

    cpu = rss = read = written = 0.0
    any_cpu = any_io = False
    for p in procs:
        try:
            times = p.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        cpu += times.user + times.system
        any_cpu = True
        try:
            rss += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        # io_counters() doesn't exist on macOS at all (not just unimplemented),
        # so the attribute itself is probed rather than assumed present.
        io_counters = getattr(p, "io_counters", None)
        if io_counters is None:
            continue
        try:
            io = io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        read += io.read_bytes
        written += io.write_bytes
        any_io = True
    if not any_cpu:
        return None
    now = datetime.now(timezone.utc)
    return Sample(
        source="harness_process",
        elapsed_ms=0,
        timestamp=now,
        cpu_seconds_cum=cpu,
        rss_bytes=int(rss),
        read_bytes=int(read) if any_io else None,
        written_bytes=int(written) if any_io else None,
    )


# --------------------------------------------------------------------------
# cgroup v2 reader (for a containerized postgres), via `exec`
# --------------------------------------------------------------------------


class CgroupReader:
    """Reads cpu/memory/io counters for one container's own cgroup v2 root.

    Read through ``<runtime> exec ... cat ...`` rather than the host's
    ``/sys/fs/cgroup``: with cgroup namespaces (the default since Docker
    20.10), a container's own root cgroup is mounted at ``/sys/fs/cgroup``
    inside it, so this needs no knowledge of the host's cgroup driver or
    layout, and works the same on a bare-metal Linux host or a macOS/Docker
    Desktop VM.
    """

    def __init__(self, runtime: str, container_id: str):
        self.runtime = runtime
        self.container_id = container_id

    def _cat(self, path: str) -> str | None:
        try:
            proc = subprocess.run(
                [self.runtime, "exec", self.container_id, "cat", path],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout if proc.returncode == 0 else None

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
        text = self._cat("/sys/fs/cgroup/cpu.stat")
        if text is None:
            return None
        for line in text.splitlines():
            if line.startswith("usage_usec"):
                try:
                    return int(line.split()[1]) / 1e6
                except (ValueError, IndexError):
                    return None
        return None

    def _memory_bytes(self) -> int | None:
        text = self._cat("/sys/fs/cgroup/memory.current")
        if text is None:
            return None
        try:
            return int(text.strip())
        except ValueError:
            return None

    def _io_bytes(self) -> tuple[int | None, int | None]:
        text = self._cat("/sys/fs/cgroup/io.stat")
        if text is None:
            return None, None
        read = written = 0
        found = False
        for line in text.splitlines():
            for field_ in line.split()[1:]:
                key, _, value = field_.partition("=")
                if key == "rbytes":
                    read += int(value)
                    found = True
                elif key == "wbytes":
                    written += int(value)
                    found = True
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
        """Point the process-tree sampler at the process to measure.

        The baseline for that source is dropped, so the first sample taken
        after attaching becomes it. Attaching to a different process therefore
        rebases rather than carrying the previous one's counters.
        """
        self._pid = pid
        self._baselines.pop("harness_process", None)

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
            # Every source's first sample of the phase is its own baseline.
            # Neither source starts at zero: the container has been running
            # since before the phase, and the dee server is long-lived across
            # the whole sweep, so both carry counters from work this phase did
            # not do.
            # A *copy*, because the sample itself is rebased in place below
            # and a baseline aliasing it would zero itself out.
            base = self._baselines.setdefault(s.source, replace(s))
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
