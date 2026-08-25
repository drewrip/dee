# dee

An experimental SQL transformation orchestrator

## Benchmarking

The benchmark harness lives in [`benchmark/`](benchmark/) and runs `dee`
against [dag-bench](https://github.com/drewrip/dag-bench). See
[`benchmark/README.md`](benchmark/README.md) for full details; the two most
common runs:

```bash
cd benchmark
uv venv && uv pip install -e .
export DAG_BENCH=/path/to/dag-bench
cargo build --release --manifest-path ../Cargo.toml

# Smoke test: one project, one scale factor, runs in a couple of minutes.
.venv/bin/dee-bench run -c configs/smoke.yaml

# Full evaluation: every project, both backends, a scale-factor and HMP
# tuning sweep. This is long-running, so submit it as a background worker
# rather than running it in the foreground.
.venv/bin/dee-bench submit -c configs/full.yaml
.venv/bin/dee-bench status results/full-eval
```
