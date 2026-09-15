"""Chart the three-shape emission experiment.

    .venv/bin/python scripts/nodefusion_shape_chart.py \
        results/p05-nodefusion-pg-w12/charts/shape-experiment.json

Reads what `nodefusion_shape_experiment.py` wrote and draws the one figure the
experiment exists to produce: what each emission costs in wall clock, against
the query work it does and the concurrency it leaves the DAG with.

Absolute times here come from the experiment's own mini-driver, not from
dee-bench, so they are lower across the board than a harness run of the same
DAGs. The comparison between shapes is the result; the absolute numbers are
not comparable to `runs.engine_wall_ms`.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SRC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(
    __file__).resolve().parents[1] / "results" / "p05-nodefusion-pg-w12" / "charts" / "shape-experiment.json"
OUT = SRC.parent
PNG_DPI = 400

BASE = "#2a78d6"
OPT = "#eb6834"
PROPOSED = "#008300"
WARN = "#eda100"
CRIT = "#e34948"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#78776f"
GRID = "#e6e6e2"
BORDER = "#dcdcd6"
TARGET = 1.3

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": BORDER,
    "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "grid.color": GRID, "axes.axisbelow": True,
})


def style(ax, ygrid=True, xgrid=False):
    for axis, on in (("y", ygrid), ("x", xgrid)):
        if on:
            ax.grid(True, axis=axis, color=GRID, linewidth=1.0)
        else:
            ax.grid(False, axis=axis)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(labelsize=8, length=0)


def figtitle(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.24 / h, title, fontsize=14, color=INK,
             fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.47 / h, subtitle, fontsize=9, color=INK_2, va="top")


data = json.load(open(SRC))
SHAPES = [
    ("unopt", BASE, "as authored"),
    ("fused", OPT, "NodeFusion today\n(one rollup query)"),
    ("nodes", PROPOSED, "proposed\n(shared models as nodes)"),
]
med = {k: statistics.median(v["samples"]) for k, v in data.items()}
lo = {k: min(v["samples"]) for k, v in data.items()}
hi = {k: max(v["samples"]) for k, v in data.items()}
work = {k: sum(v["nodes"].values()) for k, v in data.items()}
reps = len(data["unopt"]["samples"])

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
ax1, ax2, ax3 = axes

# -- 1: makespan
for i, (k, color, _label) in enumerate(SHAPES):
    ax1.bar(i, med[k], width=0.5, color=color, zorder=3, linewidth=0)
    ax1.plot([i, i], [lo[k], hi[k]], color=SURFACE, lw=3.0, zorder=4)
    ax1.plot([i, i], [lo[k], hi[k]], color=INK_3, lw=1.2, zorder=5)
    ax1.annotate(f"{med[k] / 1000:.1f}s", xy=(i, med[k]), xytext=(0, 7),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=11, color=INK, fontweight="bold")
ax1.set_xticks(range(3))
ax1.set_xticklabels([s[2] for s in SHAPES], fontsize=9, color=INK_2)
ax1.set_ylim(0, max(hi.values()) * 1.22)
ax1.set_ylabel("makespan (ms)", fontsize=9.5)
ax1.set_title("DAG wall clock", fontsize=10.5, color=INK, loc="left",
              pad=8, fontweight="bold")
style(ax1)

# -- 2: speedup vs the DAG as authored
for i, (k, color, _label) in enumerate(SHAPES):
    sp = med["unopt"] / med[k]
    ax2.bar(i, sp, width=0.5, color=color, zorder=3, linewidth=0)
    ax2.annotate(f"{sp:.2f}x", xy=(i, sp), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=11.5, color=INK, fontweight="bold")
ax2.axhline(1.0, color=INK_3, lw=1.0, zorder=2)
ax2.axhline(TARGET, color=CRIT, lw=1.3, ls=(0, (4, 3)), zorder=6)
ax2.annotate(f"target {TARGET:g}x", xy=(2.45, TARGET), xytext=(0, 4),
             textcoords="offset points", ha="right", va="bottom",
             fontsize=9, color=CRIT, fontweight="bold")
ax2.set_xticks(range(3))
ax2.set_xticklabels([s[2] for s in SHAPES], fontsize=9, color=INK_2)
ax2.set_ylim(0, max(med["unopt"] / m for m in med.values()) * 1.3)
ax2.set_ylabel("speedup over the DAG as authored (x)", fontsize=9.5)
ax2.set_title("Only the proposed shape clears the bar", fontsize=10.5,
              color=INK, loc="left", pad=8, fontweight="bold")
style(ax2)

# -- 3: work done vs concurrency it runs at -- the whole argument in one panel
w = 0.36
for i, (k, color, _label) in enumerate(SHAPES):
    ax3.bar(i - w / 2 - 0.01, work[k] / 1000, width=w, color=color,
            zorder=3, linewidth=0)
    ax3.annotate(f"{work[k] / 1000:.0f}s", xy=(i - w / 2 - 0.01, work[k] / 1000),
                 xytext=(0, 4), textcoords="offset points", ha="center",
                 va="bottom", fontsize=9, color=INK_2, fontweight="bold")
ax3.set_xticks(range(3))
ax3.set_xticklabels([s[2] for s in SHAPES], fontsize=9, color=INK_2)
ax3.set_ylabel("summed node time (s)", fontsize=9.5)
ax3.set_ylim(0, max(work.values()) / 1000 * 1.25)
style(ax3)

axr = ax3.twinx()
for i, (k, _color, _label) in enumerate(SHAPES):
    ov = work[k] / med[k]
    axr.bar(i + w / 2 + 0.01, ov, width=w, color=INK_3, zorder=3, linewidth=0)
    axr.annotate(f"{ov:.2f}x", xy=(i + w / 2 + 0.01, ov), xytext=(0, 4),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=9, color=INK_2, fontweight="bold")
axr.set_ylabel("nodes executing at once (overlap)", fontsize=9.5, color=INK_2)
axr.set_ylim(0, max(work[k] / med[k] for k in med) * 1.25)
axr.spines["top"].set_visible(False)
axr.grid(False)
axr.tick_params(labelsize=8, length=0)
ax3.set_title("The rewrite keeps the work saved, and gets the overlap back",
              fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")

handles = [Patch(facecolor=BASE, label="as authored"),
           Patch(facecolor=OPT, label="NodeFusion today"),
           Patch(facecolor=PROPOSED, label="proposed emission"),
           Patch(facecolor=INK_3, label="concurrency (right axis, panel 3)")]
fig.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_2,
           loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.02))
figtitle(fig,
         "Emitting the Shared Models as Nodes, Not as CTEs",
         f"p05_hr · PostgreSQL 18 · sf=1 · 8 cpus · {reps} measured runs each · "
         f"prototype driver, so absolute times are not dee-bench numbers")
fig.tight_layout(rect=(0, 0.07, 1, 1 - 0.62 / fig.get_figheight()))
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"5_emission.{fmt}", format=fmt, dpi=PNG_DPI,
                facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
print(f"  wrote {OUT.name}/5_emission.png / .pdf")
for k in ("unopt", "fused", "nodes"):
    print(f"  {k:<6} {med[k] / 1000:6.1f}s  work {work[k] / 1000:6.1f}s  "
          f"overlap {work[k] / med[k]:.2f}x  speedup {med['unopt'] / med[k]:.2f}x")
