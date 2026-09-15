"""p05 against p06: when promoting the shared models beats the rollup, and when it does not.

    .venv/bin/python scripts/nodefusion_emission_compare.py

Both projects were measured at sf=1 on the same PostgreSQL tuning by the same
mini-driver, over three shapes:

  as authored   the DAG as dbt wrote it
  rollup        what NodeFusion emits today -- one fused node, a WITH chain
  promoted      the models the pass chose to share, each given its own table
                node instead of an `AS MATERIALIZED` CTE

The two projects disagree, and the third panel is why: promoting only wins when
storing the shared models costs less than the duplication it removes. On p05 it
is nearly free and the rollup's work reduction is matched exactly; on p06 the
shared models are expensive to store, so the promoted shape does far more work
and its better concurrency cannot pay for it.

Absolute times come from the prototype driver, not dee-bench.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P05 = ROOT / "results" / "p05-nodefusion-pg-w12" / "charts" / "shape-experiment.json"
P06 = ROOT / "results" / "p06-prepare" / "promote_experiment.json"
OUT = ROOT / "results" / "p05-nodefusion-pg-w12" / "charts"

BASE = "#2a78d6"
ROLLUP = "#eb6834"
PROMOTED = "#008300"
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


def style(ax, ygrid=True):
    ax.grid(bool(ygrid), axis="y", color=GRID, linewidth=1.0) if ygrid else ax.grid(False, axis="y")
    ax.grid(False, axis="x")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BORDER)
    ax.tick_params(labelsize=8, length=0)


def figtitle(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.24 / h, title, fontsize=14.5, color=INK,
             fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.47 / h, subtitle, fontsize=9, color=INK_2, va="top")


# ---------------------------------------------------------------- data
def load():
    out = {}
    d5 = json.load(open(P05))
    out["p05_hr"] = {
        "shared": 3,
        "shapes": {k: {"samples": d5[src]["samples"], "nodes": d5[src]["nodes"]}
                   for k, src in (("unopt", "unopt"), ("rollup", "fused"),
                                  ("promoted", "nodes"))},
    }
    d6 = json.load(open(P06))["p06_logistics"]
    out["p06_logistics"] = {
        "shared": len(d6["shared"]),
        "shapes": {k: d6["shapes"][src]
                   for k, src in (("unopt", "unopt"), ("rollup", "fused"),
                                  ("promoted", "promote"))},
    }
    for p in out.values():
        for s in p["shapes"].values():
            s["med"] = statistics.median(s["samples"])
            s["work"] = sum(s["nodes"].values())
    return out


data = load()
PROJECTS = ["p05_hr", "p06_logistics"]
SHAPES = [("unopt", BASE, "as authored"),
          ("rollup", ROLLUP, "rollup (today)"),
          ("promoted", PROMOTED, "promoted models")]

fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))
ax1, ax2, ax3 = axes
w = 0.26

# -- 1: makespan
for j, (key, color, _lab) in enumerate(SHAPES):
    for i, proj in enumerate(PROJECTS):
        v = data[proj]["shapes"][key]
        x = i + (j - 1) * (w + 0.02)
        ax1.bar(x, v["med"] / 1000, width=w, color=color, zorder=3, linewidth=0)
        ax1.plot([x, x], [min(v["samples"]) / 1000, max(v["samples"]) / 1000],
                 color=SURFACE, lw=2.6, zorder=4)
        ax1.plot([x, x], [min(v["samples"]) / 1000, max(v["samples"]) / 1000],
                 color=INK_3, lw=1.1, zorder=5)
        ax1.annotate(f"{v['med'] / 1000:.0f}", xy=(x, v["med"] / 1000),
                     xytext=(0, 4), textcoords="offset points", ha="center",
                     va="bottom", fontsize=9, color=INK_2, fontweight="bold")
ax1.set_xticks(range(len(PROJECTS)))
ax1.set_xticklabels([f"{p}\n({data[p]['shared']} models shared)" for p in PROJECTS],
                    fontsize=9.5, color=INK_2)
ax1.set_ylabel("makespan (s)", fontsize=9.5)
ax1.set_title("Wall clock", fontsize=10.5, color=INK, loc="left", pad=8,
              fontweight="bold")
style(ax1)

# -- 2: speedup
for j, (key, color, _lab) in enumerate(SHAPES):
    for i, proj in enumerate(PROJECTS):
        sp = data[proj]["shapes"]["unopt"]["med"] / data[proj]["shapes"][key]["med"]
        x = i + (j - 1) * (w + 0.02)
        ax2.bar(x, sp, width=w, color=color, zorder=3, linewidth=0)
        ax2.annotate(f"{sp:.2f}x", xy=(x, sp), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=9.5, color=INK, fontweight="bold")
ax2.axhline(1.0, color=INK_3, lw=1.0, zorder=2)
ax2.axhline(TARGET, color=CRIT, lw=1.3, ls=(0, (4, 3)), zorder=6)
ax2.annotate(f"target {TARGET:g}x", xy=(1.42, TARGET), xytext=(0, 4),
             textcoords="offset points", ha="right", va="bottom", fontsize=9,
             color=CRIT, fontweight="bold")
ax2.set_xticks(range(len(PROJECTS)))
ax2.set_xticklabels(PROJECTS, fontsize=9.5, color=INK_2)
ax2.set_ylabel("speedup over the DAG as authored (x)", fontsize=9.5)
ax2.set_ylim(0, 1.72)
ax2.set_title("The two projects disagree about which emission wins",
              fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
style(ax2)

# -- 3: the query work each shape actually performs -- the explanation
for j, (key, color, _lab) in enumerate(SHAPES):
    for i, proj in enumerate(PROJECTS):
        v = data[proj]["shapes"][key]
        x = i + (j - 1) * (w + 0.02)
        ax3.bar(x, v["work"] / 1000, width=w, color=color, zorder=3, linewidth=0)
        ax3.annotate(f"{v['work'] / 1000:.0f}", xy=(x, v["work"] / 1000),
                     xytext=(0, 4), textcoords="offset points", ha="center",
                     va="bottom", fontsize=9, color=INK_2, fontweight="bold")
# Call out the one comparison the whole finding turns on.
for i, proj in enumerate(PROJECTS):
    r = data[proj]["shapes"]["rollup"]["work"] / 1000
    p = data[proj]["shapes"]["promoted"]["work"] / 1000
    ax3.annotate(f"{p / r:.2f}x the rollup's work",
                 xy=(i + 0.5 * (w + 0.02), max(r, p)), xytext=(0, 20),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold",
                 color=PROMOTED if p <= r * 1.05 else CRIT)
ax3.set_xticks(range(len(PROJECTS)))
ax3.set_xticklabels(PROJECTS, fontsize=9.5, color=INK_2)
ax3.set_ylabel("summed node time (s)", fontsize=9.5)
ax3.set_ylim(0, 148)
ax3.set_title("Why: promoting is only free when storing the models is",
              fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
style(ax3)

handles = [Patch(facecolor=c, label=lab) for _k, c, lab in SHAPES]
fig.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_2,
           loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.02))
figtitle(fig, "Promoting the Shared Models Is a Per-DAG Bet, Not a Rule",
         "p05_hr and p06_logistics · PostgreSQL 18 · sf=1 · 8 cpus · "
         "3 measured runs per bar · prototype driver")
fig.tight_layout(rect=(0, 0.07, 1, 1 - 0.62 / fig.get_figheight()))
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"6_emission_p05_p06.{fmt}", format=fmt, dpi=400,
                facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
print(f"  wrote charts/6_emission_p05_p06.png / .pdf")
for proj in PROJECTS:
    d = data[proj]
    base = d["shapes"]["unopt"]["med"]
    bits = "  ".join(
        f"{k} {v['med'] / 1000:5.1f}s ({base / v['med']:.2f}x, work {v['work'] / 1000:5.1f}s)"
        for k, v in d["shapes"].items())
    print(f"  {proj:<15} shared={d['shared']}  {bits}")
