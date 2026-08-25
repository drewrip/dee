"""Detached worker entry point used by `dee-bench submit` and `resume`.

Runs in its own session so it survives the launching shell, writing progress to
``<run_dir>/worker.log`` and cell state to ``<run_dir>/state/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import load
from .sweep import Sweep


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: python -m dee_bench.worker <run_dir> <config> [--fresh] [--keep-infra]",
              file=sys.stderr)
        return 2
    run_dir, config_path = Path(argv[0]), argv[1]
    flags = set(argv[2:])

    cfg = load(config_path, {"output_dir": str(run_dir)})

    def log(*parts) -> None:
        # Unbuffered, so `tail -f worker.log` tracks a long sweep live.
        print(*parts, flush=True)

    sweep = Sweep(
        cfg, run_dir=run_dir,
        fresh="--fresh" in flags, keep_infra="--keep-infra" in flags, log=log,
    )
    counts = sweep.run()

    from .cli import _finalize

    _finalize(run_dir)
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
