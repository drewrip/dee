"""The phase sampler's counter baselining.

The dee server is long-lived across a whole sweep, so its CPU and IO counters
are far from zero when a phase starts. A phase must report what *it* burned.
"""

from datetime import datetime, timezone

import dee_bench.sampler as sampler_mod
from dee_bench.sampler import PhaseSampler, Sample


class FakeTree:
    """A process tree whose cumulative counters advance on every sample."""

    def __init__(self, cpu_at_attach: float, per_sample: float = 1.0):
        self.cpu = cpu_at_attach
        self.per_sample = per_sample
        self.io = 1_000_000

    def __call__(self, pid: int) -> Sample:
        s = Sample(
            source="harness_process",
            elapsed_ms=0,
            timestamp=datetime.now(timezone.utc),
            cpu_seconds_cum=self.cpu,
            rss_bytes=4096,
            read_bytes=self.io,
            written_bytes=self.io,
        )
        self.cpu += self.per_sample
        self.io += 500
        return s


def _run_phase(monkeypatch, tree, samples=3):
    monkeypatch.setattr(sampler_mod, "sample_process_tree", tree)
    s = PhaseSampler(interval_ms=10).start()
    s.attach(1234)
    for _ in range(samples):
        s._take()
    return s.stop()


class TestProcessTreeBaseline:
    def test_reports_the_phase_delta_not_the_process_total(self, monkeypatch):
        """A server that has already burned 300 CPU-seconds across previous
        cells must not charge them to this phase."""
        metrics = _run_phase(monkeypatch, FakeTree(cpu_at_attach=300.0, per_sample=1.0))
        # 3 explicit takes plus stop()'s final one: the first is the baseline,
        # so the phase is credited with the advance after it.
        assert metrics.cpu_seconds == 3.0

    def test_a_fresh_process_is_unaffected(self, monkeypatch):
        """The old zero-baseline case still reports the same total."""
        metrics = _run_phase(monkeypatch, FakeTree(cpu_at_attach=0.0, per_sample=1.0))
        assert metrics.cpu_seconds == 3.0

    def test_io_counters_are_rebased_too(self, monkeypatch):
        metrics = _run_phase(monkeypatch, FakeTree(cpu_at_attach=300.0))
        assert metrics.read_bytes == 1500
        assert metrics.written_bytes == 1500

    def test_peak_rss_stays_absolute(self, monkeypatch):
        """Memory is an instantaneous reading, not a counter: nothing is
        subtracted from it."""
        metrics = _run_phase(monkeypatch, FakeTree(cpu_at_attach=300.0))
        assert metrics.peak_rss_bytes == 4096

    def test_reattaching_rebases_on_the_new_process(self, monkeypatch):
        tree = FakeTree(cpu_at_attach=10.0, per_sample=1.0)
        monkeypatch.setattr(sampler_mod, "sample_process_tree", tree)
        s = PhaseSampler(interval_ms=10).start()
        s.attach(1)
        s._take()
        tree.cpu = 900.0            # a different process, far along its own counter
        s.attach(2)
        s._take()
        metrics = s.stop()
        assert metrics.cpu_seconds == 1.0
