"""`dee-bench` command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import queue as q
from .config import VALID_PASSES, ConfigError, load
from .schema import Verbosity, render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dee-bench",
        description="Benchmark dee's optimizer passes against dag-bench.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", required=True, help="Path to the experiment YAML")
        p.add_argument("-o", "--output-dir", help="Override the config's output_dir")
        p.add_argument("--verbosity", choices=[v.label() for v in Verbosity],
                       help="Override how much detail is recorded")
        p.add_argument("--fresh", action="store_true",
                       help="Discard any persisted backend data volume before starting")
        p.add_argument("--keep-infra", action="store_true",
                       help="Leave backend infrastructure running after the sweep")

    p_run = sub.add_parser("run", help="Run a benchmark in the foreground")
    with_config(p_run)
    p_run.add_argument("--no-viz", action="store_true", help="Skip the dashboard at the end")

    p_submit = sub.add_parser("submit", help="Run a benchmark in a detached background worker")
    with_config(p_submit)

    p_status = sub.add_parser("status", help="Show progress for a run directory")
    p_status.add_argument("run_dir")
    p_status.add_argument("--failed", action="store_true", help="List failed cells and their errors")

    p_resume = sub.add_parser("resume", help="Continue an interrupted or partially failed run")
    p_resume.add_argument("run_dir")
    p_resume.add_argument("--background", action="store_true", help="Resume in a detached worker")
    p_resume.add_argument("--skip-failed", action="store_true",
                          help="Retry only pending cells, leaving previously failed ones alone")

    p_cancel = sub.add_parser("cancel", help="Stop a running background worker")
    p_cancel.add_argument("run_dir")

    p_viz = sub.add_parser("viz", help="Build the dashboard and charts from results")
    p_viz.add_argument("run_dir")
    p_viz.add_argument("--only", help="Render a single study (e.g. scaling, payback, ablation)")
    p_viz.add_argument("--format", default="html,png,pdf",
                       help="Comma-separated outputs to produce (default: html,png,pdf)")
    p_viz.add_argument("--open", action="store_true", dest="open_browser",
                       help="Open the dashboard when it is built")

    p_analyze = sub.add_parser("analyze", help="Recompute derived tables (payback) from results")
    p_analyze.add_argument("run_dir")

    p_schema = sub.add_parser("schema", help="Print the result schemas as markdown")
    p_schema.add_argument("-o", "--output", help="Write to a file instead of stdout")

    p_doctor = sub.add_parser("doctor", help="Check the environment and clean up strays")
    p_doctor.add_argument("--dee-bin", dest="dee_bin", default=None,
                          help="dee binary to check against (default ../../target/release/dee)")
    p_doctor.add_argument("--clean", action="store_true",
                          help="Remove leftover dee-bench containers")

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        return cmd_run(args)
    if args.command == "submit":
        return cmd_submit(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "resume":
        return cmd_resume(args)
    if args.command == "cancel":
        return cmd_cancel(args)
    if args.command == "viz":
        return cmd_viz(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "schema":
        return cmd_schema(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    raise AssertionError(args.command)


def _load(args: argparse.Namespace):
    overrides = {}
    if getattr(args, "output_dir", None):
        overrides["output_dir"] = args.output_dir
    if getattr(args, "verbosity", None):
        overrides["verbosity"] = args.verbosity
    return load(args.config, overrides)


def cmd_run(args: argparse.Namespace) -> int:
    from .sweep import Sweep, write_run_meta

    cfg = _load(args)
    sweep = Sweep(cfg, fresh=args.fresh, keep_infra=args.keep_infra)
    sweep.initialize()
    write_run_meta(sweep.run_dir, cfg, len(sweep.cells))
    counts = sweep.run()

    _finalize(sweep.run_dir, skip_viz=args.no_viz)
    return 1 if counts.get(q.FAILED, 0) else 0


def cmd_submit(args: argparse.Namespace) -> int:
    from .sweep import Sweep, write_run_meta

    cfg = _load(args)
    sweep = Sweep(cfg, fresh=args.fresh, keep_infra=args.keep_infra)
    sweep.initialize()
    write_run_meta(sweep.run_dir, cfg, len(sweep.cells))

    queue = q.RunQueue(sweep.run_dir)
    if queue.worker_pid() is not None:
        print(f"a worker is already running for {sweep.run_dir}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "dee_bench.worker", str(sweep.run_dir), args.config]
    if args.fresh:
        cmd.append("--fresh")
    if args.keep_infra:
        cmd.append("--keep-infra")

    log_path = sweep.run_dir / "worker.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,  # survives the launching shell
            cwd=Path.cwd(),
        )
    queue.write_pid(proc.pid)
    print(f"submitted {len(sweep.cells)} cell(s) as pid {proc.pid}")
    print(f"  run dir : {sweep.run_dir}")
    print(f"  log     : {log_path}")
    print(f"  progress: dee-bench status {sweep.run_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 2
    queue = q.RunQueue(run_dir)
    states = queue.all_states()
    counts = queue.counts()
    total = len(states)

    meta = {}
    if (run_dir / "run.json").exists():
        meta = json.loads((run_dir / "run.json").read_text())

    pid = queue.worker_pid()
    print(f"run      : {meta.get('name', run_dir.name)}  ({run_dir})")
    print(f"worker   : {'running, pid ' + str(pid) if pid else 'not running'}")
    print(f"progress : {counts.get(q.DONE,0)}/{total} done, "
          f"{counts.get(q.FAILED,0)} failed, {counts.get(q.RUNNING,0)} running, "
          f"{counts.get(q.PENDING,0)} pending")

    eta = queue.estimate_remaining(total)
    if eta:
        print(f"eta      : ~{_duration(eta)} remaining")

    for s in states:
        if s.status == q.RUNNING:
            print(f"current  : {s.describe} ({s.cell_id})")

    if args.failed:
        failed = [s for s in states if s.status == q.FAILED]
        if not failed:
            print("\nno failed cells")
        for s in failed:
            print(f"\nFAILED {s.describe} ({s.cell_id})\n  {s.error}")
    elif counts.get(q.FAILED, 0):
        print(f"\nrun `dee-bench status {run_dir} --failed` to see failures")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from .sweep import Sweep

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"{run_dir} has no frozen config.yaml; cannot resume", file=sys.stderr)
        return 2
    queue = q.RunQueue(run_dir)
    if queue.worker_pid() is not None:
        print(f"a worker is already running for {run_dir}", file=sys.stderr)
        return 1

    cfg = load(config_path, {"output_dir": str(run_dir)})
    if args.background:
        cmd = [sys.executable, "-m", "dee_bench.worker", str(run_dir), str(config_path)]
        with open(run_dir / "worker.log", "ab") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        queue.write_pid(proc.pid)
        print(f"resumed as pid {proc.pid}")
        return 0

    sweep = Sweep(cfg, run_dir=run_dir)
    counts = sweep.run(retry_failed=not args.skip_failed)
    _finalize(run_dir)
    return 1 if counts.get(q.FAILED, 0) else 0


def cmd_cancel(args: argparse.Namespace) -> int:
    queue = q.RunQueue(Path(args.run_dir))
    if queue.cancel():
        print("worker stopped")
        return 0
    print("no worker running", file=sys.stderr)
    return 1


def cmd_viz(args: argparse.Namespace) -> int:
    from .viz.dashboard import build

    formats = {f.strip() for f in args.format.split(",") if f.strip()}
    out = build(Path(args.run_dir), only=args.only, formats=formats)
    if out is None:
        print("no results to visualize yet", file=sys.stderr)
        return 1
    print(f"dashboard: {out}")
    if args.open_browser:
        import webbrowser

        webbrowser.open(out.as_uri())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze

    n = analyze(Path(args.run_dir))
    print(f"payback: {n} row(s)")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    text = render_markdown()
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .infra.postgres_backend import remove_stray_containers, stray_containers

    ok = True
    from .workload import dbt_executable

    print("tooling:")
    for tool in ("psql", "cargo", "git"):
        path = shutil.which(tool)
        print(f"  {tool:8} {path or 'NOT FOUND'}")
        if path is None and tool == "git":
            ok = False
    try:
        from .infra.postgres_backend import container_runtime

        print(f"  {'container':8} {container_runtime()}")
    except Exception as e:  # noqa: BLE001
        print(f"  {'container':8} NOT FOUND ({e})")
    try:
        print(f"  {'dbt':8} {dbt_executable()}")
    except Exception as e:  # noqa: BLE001
        print(f"  {'dbt':8} NOT FOUND ({e})")
        ok = False

    dag_bench = os.environ.get("DAG_BENCH")
    print(f"\nDAG_BENCH: {dag_bench or 'not set'}")
    if dag_bench:
        from .workload import discover_projects

        projects = discover_projects(Path(dag_bench))
        print(f"  {len(projects)} project(s): {', '.join(projects)}")

    ok = _check_server(args) and ok

    strays = stray_containers()
    print(f"\nleftover dee-bench containers: {len(strays)}")
    for c in strays:
        print(f"  {c.get('Names')} {c.get('State')} {c.get('Status')}")
    if strays and args.clean:
        print(f"  removed {remove_stray_containers()}")
    elif strays:
        print("  run with --clean to remove them")

    return 0 if ok else 1


def _check_server(args: argparse.Namespace) -> bool:
    """Check the dee binary against what the harness assumes about it.

    Two contracts, one server start. `DEE_OPT_SPECS` mirrors `OptimizerConfig`
    in dee/src/opt.rs, and a mirror that drifts silently is worse than no
    mirror: a sweep would keep sending an option the optimizer no longer reads.
    Separately, the run queue behind `execution.repeat_mode: queue` is newer
    than the rest of the API, so an older dee would fail a sweep on its first
    cell rather than here. Both are answered by asking the server, not by
    parsing `--help`.
    """
    import tempfile

    from .config import DEE_OPT_SPECS
    from .server import ApiError, DeeServer, ServerError

    dee_bin = Path(getattr(args, "dee_bin", None) or "../../target/release/dee").resolve()
    print(f"\ndee binary: {dee_bin if dee_bin.exists() else f'{dee_bin} NOT FOUND'}")
    if not dee_bin.exists():
        print("  build it with `cargo build --release`, or pass --dee-bin")
        return False

    try:
        with tempfile.TemporaryDirectory() as tmp:
            with DeeServer(dee_bin, Path(tmp)) as client:
                info = client.info()
                server_options = {o["name"]: o for o in client.optimizer_options()}
                # Probed, not inferred from a version: dee keeps one schema
                # rather than a migration chain, so nothing it reports about
                # itself says whether the queue is there.
                try:
                    client.queue()
                    has_queue = None
                except (ServerError, OSError) as e:
                    has_queue = str(e)
                # Likewise for the registration endpoint, which is what
                # `optimization_mode: continuous` needs.
                try:
                    available = {
                        o["name"]: o for o in client.available_optimizations()
                    }
                    no_registration = None
                except (ApiError, ServerError, OSError) as e:
                    available, no_registration = {}, str(e)
    except (ServerError, OSError) as e:
        print(f"  could not start a server to check options: {e}")
        return False

    print(f"  version {info['version']}, metadata schema v{info['schema_version']}")
    if has_queue is None:
        print("  run queue: available (execution.repeat_mode may be 'group' or 'queue')")
    else:
        print(f"  run queue: NOT AVAILABLE ({has_queue})")
        print("    execution.repeat_mode must be 'group' with this dee")

    ours = {spec.config_field: spec for spec in DEE_OPT_SPECS}
    missing = sorted(set(ours) - set(server_options))
    extra = sorted(set(server_options) - set(ours))

    print(f"\noptimizer options: {len(ours)} known here, {len(server_options)} on the server")
    for name in missing:
        print(f"  {name}: in dee-bench but not in the server -- remove it from DEE_OPT_SPECS")
    for name in extra:
        # Not every server option needs to be sweepable; the pass toggles are
        # set from a variant's pass list rather than from dee_opt.
        if name in (
            "run_hmp_pass",
            "run_omp_pass",
            "run_pushdown_pass",
            "run_parallelism_pass",
        ):
            continue
        print(f"  {name}: on the server but not in dee-bench -- consider adding it")

    mismatched = []
    for name, spec in ours.items():
        server_option = server_options.get(name)
        if not server_option:
            continue
        if spec.choices and server_option.get("choices"):
            if tuple(spec.choices) != tuple(server_option["choices"]):
                mismatched.append(
                    f"  {name}: choices {spec.choices} here, "
                    f"{tuple(server_option['choices'])} on the server"
                )
    for line in mismatched:
        print(line)

    if not missing and not mismatched:
        print("  in sync")

    # Which optimizations the server has, and how each is driven. `continuous`
    # mode only means anything for an optimization that steps around runs, so a
    # config asking for it on a `once` one is a mistake worth catching here
    # rather than mid-sweep.
    unknown_passes: list[str] = []
    if no_registration is not None:
        print(f"\noptimizations: this dee server cannot list them ({no_registration});")
        print("  execution.optimization_mode must be 'batch' with this dee")
    else:
        print(f"\noptimizations: {len(available)} available")
        for name, o in sorted(available.items()):
            print(
                f"  {name}: {o['optimization_type']}, "
                f"steps {o['default_step_phase']} by default"
            )
        unknown_passes = [p for p in VALID_PASSES if p not in available]
        for name in unknown_passes:
            print(f"  {name}: dee-bench knows it but the server does not")

    return (
        not missing
        and not mismatched
        and not unknown_passes
        and has_queue is None
    )


def _finalize(run_dir: Path, skip_viz: bool = False) -> None:
    """Derive payback and build the dashboard after a sweep."""
    from .analyze import analyze

    try:
        analyze(run_dir)
    except Exception as e:  # noqa: BLE001 - never lose a sweep to reporting
        print(f"warning: could not compute payback: {e}", file=sys.stderr)
    if skip_viz:
        return
    try:
        from .viz.dashboard import build

        out = build(run_dir)
        if out:
            print(f"dashboard: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not build dashboard: {e}", file=sys.stderr)
        print(f"  results are intact; retry with `dee-bench viz {run_dir}`", file=sys.stderr)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


if __name__ == "__main__":
    sys.exit(main())
