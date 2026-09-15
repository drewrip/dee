"""Charts for NodeFusion on PostgreSQL: what the rollup bought, and why.

    .venv/bin/python scripts/nodefusion_pg_charts.py results/p05-nodefusion-pg-w12

Reads only the parquet a sweep wrote, so it can be re-run at any time --
including on a partial or cancelled run -- and writes png and pdf into
`<run_dir>/charts/`. It expects a sweep pairing an `unopt` variant against
`nf_rule`, and reads the tables `detailed` verbosity and above record.

A second run directory may be passed as `--profile <dir>`: a `full`-verbosity
companion whose `plans` table holds PostgreSQL's EXPLAIN (ANALYZE, BUFFERS)
JSON. Its wall clock is inflated by that instrumentation and is never plotted
as a time; only its *shape* -- workers launched, per-operator time -- is read.

  1_makespan     the DAG as authored against the rollup, and the gap to target
  2_nodes        where each variant's node time goes, and what the rollup moved
  3_cpu          CPU actually busy through a run of each variant
  4_parallelism  what the twelve-worker grant actually bought (profile only)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------- arguments
args = sys.argv[1:]
PROFILE_DIR = None
if "--profile" in args:
    i = args.index("--profile")
    PROFILE_DIR = Path(args[i + 1]).resolve()
    del args[i:i + 2]
DEFAULT_RUN = Path(__file__).resolve().parents[1] / "results" / "p05-nodefusion-pg-w12"
RUN_DIRS = [Path(a).resolve() for a in args] or [DEFAULT_RUN]
for _d in RUN_DIRS:
    if not (_d / "results").is_dir():
        sys.exit(f"{_d} is not a dee-bench run directory (no results/ inside)")
RUN_DIR = RUN_DIRS[0]
OUT = RUN_DIR / "charts"
PNG_DPI = 400

# ---------------------------------------------------------------- theme
# The same palette and slot order the rest of the harness's charts use, so a
# variant keeps its color across every artifact of this study.
BASE = "#2a78d6"          # slot 1 -- the DAG as authored
OPT = "#eb6834"           # slot 2 -- the DAG NodeFusion produced
GOOD = "#008300"
WARN = "#eda100"
CRIT = "#e34948"
SURFACE = "#ffffff"
SURFACE_2 = "#f4f4f2"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#78776f"
GRID = "#e6e6e2"
BORDER = "#dcdcd6"

# The speedup this run was asked to clear. Drawn as a reference, not a
# threshold the chart passes judgement on.
TARGET = 1.3

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "grid.color": GRID,
    "axes.axisbelow": True,
})


def style(ax, ygrid=True, xgrid=False):
    # Line properties passed alongside `False` turn the grid back on, so an
    # axis being switched off is switched off on its own.
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
    if subtitle:
        fig.text(0.012, 1 - 0.47 / h, subtitle, fontsize=9, color=INK_2, va="top")


def title_rect(fig, bottom=0.0):
    return (0, bottom, 1, 1 - 0.62 / fig.get_figheight())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{fmt}", format=fmt, dpi=PNG_DPI,
                    facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote charts/{name}.png / .pdf")


def short(node_id: str) -> str:
    """`"benchmark"."synth_multi_p05_hr"."rpt_x"` -> `rpt_x`."""
    return node_id.replace('"', "").split(".")[-1]


# ---------------------------------------------------------------- data
con = duckdb.connect()


def attach(tbl: str, dirs: list[Path], view: str | None = None) -> bool:
    globs = [f"'{d / 'results' / tbl}/**/*.parquet'"
             for d in dirs
             if list((d / "results" / tbl).glob("**/*.parquet"))]
    if not globs:
        return False
    con.execute(
        f"create view {view or tbl} as select * from read_parquet("
        f"[{', '.join(globs)}], hive_partitioning=1, union_by_name=1)"
    )
    return True


for _t in ("cells", "runs", "node_executions", "dag_graph", "pass_stats",
           "optimizations", "system_samples"):
    attach(_t, RUN_DIRS)

HAVE_PLANS = False
if PROFILE_DIR is not None:
    for _t in ("cells", "runs", "node_executions", "plans"):
        attach(_t, [PROFILE_DIR], view=f"p_{_t}")
    HAVE_PLANS = "p_plans" in {r[0] for r in con.execute("show tables").fetchall()}


def q(sql):
    return con.execute(sql).fetchall()


# One row per variant: the makespan of each measured repetition, summarized.
# Only `direct` deliveries are comparable -- a resumed run measures neither DAG.
PERF_SQL = """
select c.variant,
       median(r.engine_wall_ms) as ms,
       min(r.engine_wall_ms)    as lo,
       max(r.engine_wall_ms)    as hi,
       median(r.node_time_ms)   as node_ms,
       count(*)                 as n
from runs r join cells c using (cell_id)
where r.phase = 'measure' and r.status = 'ok' and r.delivery = 'direct'
group by 1
"""
perf = {r[0]: r[1:] for r in q(PERF_SQL)}
if "unopt" not in perf or "nf_rule" not in perf:
    sys.exit(f"need both `unopt` and `nf_rule` measured; have {sorted(perf)}")

B_MS, B_LO, B_HI, B_NODE, REPS = perf["unopt"]
O_MS, O_LO, O_HI, O_NODE, _ = perf["nf_rule"]
SPEEDUP = B_MS / O_MS
# The same ratio on summed node time: how much query work the rewrite removed,
# as against how much wall clock it saved. The gap between the two is the
# subject of the whole profiling half of this script.
WORK_SPEEDUP = (B_NODE / O_NODE) if (B_NODE and O_NODE) else None

_sf = q("select distinct sf from cells")[0][0]
SUBTITLE = (f"p05_hr · PostgreSQL 18 · sf={_sf:g} · 8 cpus · "
            f"max_parallel_workers_per_gather=12")
REPS_NOTE = f"{REPS} measured run" + ("s" if REPS != 1 else "") + " per bar"


# ---------------------------------------------------- 1. makespan
def chart_makespan():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax, ax2 = axes

    # -- left: the two makespans, as measured
    for i, (ms, lo, hi, color) in enumerate((
            (B_MS, B_LO, B_HI, BASE),
            (O_MS, O_LO, O_HI, OPT))):
        ax.bar(i, ms, width=0.46, color=color, zorder=3, linewidth=0)
        if REPS > 1:
            ax.plot([i, i], [lo, hi], color=SURFACE, lw=3.0, zorder=4)
            ax.plot([i, i], [lo, hi], color=INK_3, lw=1.2, zorder=5)
        ax.annotate(f"{ms / 1000:.1f}s", xy=(i, lo), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=10, color=SURFACE, fontweight="bold")

    delta = (O_MS - B_MS) / B_MS * 100.0
    faster = delta < -1.0
    ax.annotate(f"{SPEEDUP:.2f}x  ({delta:+.0f}%)",
                xy=(0.5, max(B_HI, O_HI)), xytext=(0, 16),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color=GOOD if faster else CRIT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["as authored", "NodeFusion\n(rule)"], fontsize=9.5,
                       color=INK_2)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, max(B_HI, O_HI) * 1.26)
    ax.set_ylabel("makespan (ms)", fontsize=9.5)
    ax.set_title("End-to-end DAG wall clock", fontsize=10.5, color=INK,
                 loc="left", pad=8, fontweight="bold")
    style(ax)

    # -- right: the speedup actually achieved against the work removed and the
    # target asked for. Three numbers on one axis is the whole finding.
    bars = [("wall clock\n(makespan)", SPEEDUP, OPT)]
    if WORK_SPEEDUP:
        bars.append(("query work\n(summed node time)", WORK_SPEEDUP, WARN))
    for i, (_label, val, color) in enumerate(bars):
        ax2.bar(i, val, width=0.46, color=color, zorder=3, linewidth=0)
        ax2.annotate(f"{val:.2f}x", xy=(i, val), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=11, fontweight="bold", color=INK)
    ax2.axhline(1.0, color=INK_3, lw=1.0, zorder=2)
    ax2.axhline(TARGET, color=CRIT, lw=1.3, ls=(0, (4, 3)), zorder=6)
    ax2.annotate(f"target {TARGET:g}x", xy=(len(bars) - 0.45, TARGET),
                 xytext=(0, 4), textcoords="offset points", ha="right",
                 va="bottom", fontsize=9, color=CRIT, fontweight="bold")
    ax2.set_xticks(range(len(bars)))
    ax2.set_xticklabels([b[0] for b in bars], fontsize=9.5, color=INK_2)
    ax2.set_xlim(-0.6, len(bars) - 0.4)
    ax2.set_ylim(0, max([b[1] for b in bars] + [TARGET]) * 1.22)
    ax2.set_ylabel("speedup over the DAG as authored (x)", fontsize=9.5)
    ax2.set_title("What the rewrite removed vs. what it saved", fontsize=10.5,
                  color=INK, loc="left", pad=8, fontweight="bold")
    style(ax2)

    handles = [Patch(facecolor=BASE, label="as authored"),
               Patch(facecolor=OPT, label="NodeFusion (rule)")]
    if REPS > 1:
        handles.append(Line2D([], [], color=INK_3, lw=1.2,
                              label=f"min–max over {REPS} runs"))
    fig.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_2,
               loc="lower center", ncols=len(handles), bbox_to_anchor=(0.5, -0.02))
    figtitle(fig, "NodeFusion on PostgreSQL: Work Removed Is Not Time Saved",
             f"{SUBTITLE} · {REPS_NOTE}")
    fig.tight_layout(rect=title_rect(fig, 0.07))
    save(fig, "1_makespan")


# ---------------------------------------------------- 2. where node time goes
NODES_SQL = """
select c.variant, n.node_id, n.materialization,
       median(n.duration_ms) as ms, median(n.rows_produced) as rows
from node_executions n join cells c using (cell_id)
join runs r on r.run_id = n.run_id
where r.phase = 'measure' and r.status = 'ok'
group by 1, 2, 3
-- Views materialize nothing and land at a millisecond or two. Listing them
-- pads the chart with empty bars and buys nothing.
having median(n.duration_ms) >= 100
order by ms desc
"""


def chart_nodes(top=10):
    rows = q(NODES_SQL)
    if not rows:
        print("  (no node_executions; skipping 2_nodes)")
        return
    per = {"unopt": [], "nf_rule": []}
    for variant, node_id, mat, ms, nrows in rows:
        if variant in per:
            per[variant].append((short(node_id), mat, ms, nrows))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.34 * top + 3.0), sharex=True)
    for ax, (variant, color, label) in zip(axes, (
            ("unopt", BASE, "as authored"), ("nf_rule", OPT, "NodeFusion (rule)"))):
        items = per[variant][:top]
        ypos = range(len(items))
        for y, (name, _mat, ms, _rows) in zip(ypos, items):
            # The fused rollup is the node the whole pass exists to create, so
            # it is marked rather than left to be found by reading labels.
            fused = name.startswith("dee_fused")
            ax.barh(y, ms, height=0.62, zorder=3, linewidth=0,
                    color=CRIT if fused else color)
            ax.annotate(f"{ms / 1000:.1f}s", xy=(ms, y), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8, color=INK_2)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([n for n, *_ in items], fontsize=8)
        ax.invert_yaxis()
        total = sum(r[2] for r in per[variant])
        ax.set_title(f"{label} — {total / 1000:.0f}s of node time in "
                     f"{len(per[variant])} node(s) over 100ms",
                     fontsize=10.5, color=INK, loc="left", pad=8,
                     fontweight="bold")
        ax.set_xlabel("median node duration (ms)", fontsize=9.5)
        style(ax, ygrid=False, xgrid=True)

    handles = [Patch(facecolor=CRIT, label="the fused rollup node")]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_2,
               loc="lower center", ncols=1, bbox_to_anchor=(0.5, -0.03))
    figtitle(fig, "Where the DAG's Time Goes, Node by Node",
             f"{SUBTITLE} · median over {REPS} runs · top {top} nodes each")
    fig.tight_layout(rect=title_rect(fig, 0.06))
    save(fig, "2_nodes")


# ---------------------------------------------------- 3. cpu over the run
# Cores busy, as the delta of the container's cumulative CPU seconds over the
# sampling interval. This is the harness's external cgroup sampling, not dee's
# own -- on PostgreSQL the work happens in the server, so anything measured
# inside dee would describe the orchestrator.
CPU_SQL = """
with d as (
  select c.variant, s.run_id, s.elapsed_ms,
         s.cpu_seconds_cum - lag(s.cpu_seconds_cum)
             over (partition by s.run_id order by s.elapsed_ms) as dcpu,
         (s.elapsed_ms - lag(s.elapsed_ms)
             over (partition by s.run_id order by s.elapsed_ms)) / 1000.0 as dt
  from system_samples s join cells c using (cell_id)
  where s.phase = 'measure' and s.source = 'harness_container'
)
select variant, run_id, elapsed_ms, dcpu / dt as cores
from d where dt > 0 and dcpu >= 0
order by variant, run_id, elapsed_ms
"""


def chart_cpu():
    rows = q(CPU_SQL)
    if not rows:
        print("  (no container CPU samples; skipping 3_cpu)")
        return
    series = defaultdict(lambda: defaultdict(list))
    for variant, run_id, elapsed, cores in rows:
        series[variant][run_id].append((elapsed / 1000.0, cores))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for ax, (variant, color, label) in zip(axes, (
            ("unopt", BASE, "as authored"),
            ("nf_rule", OPT, "NodeFusion (rule)"))):
        runs = series.get(variant, {})
        # `elapsed_ms` is measured from the start of the cell's measure phase,
        # which the repetitions share, so the series read as one continuous
        # timeline of all of them rather than as overlaid single runs.
        for run_id, pts in runs.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, lw=0.9, alpha=0.32, zorder=3)
        if runs:
            best = max(runs.values(), key=len)
            ax.plot([p[0] for p in best], [p[1] for p in best],
                    color=color, lw=1.9, zorder=4)
        allc = [c for pts in runs.values() for _t, c in pts]
        med = sorted(allc)[len(allc) // 2] if allc else 0
        ax.axhline(med, color=INK_3, lw=1.1, ls=(0, (4, 3)), zorder=5)
        ax.annotate(f"median {med:.1f} cores", xy=(0.995, med),
                    xycoords=("axes fraction", "data"), xytext=(0, -13),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=9, color=INK_2, fontweight="bold")
        ax.axhline(8, color=CRIT, lw=1.2, ls=(0, (2, 2)), zorder=5)
        ax.annotate("8 cpus granted to the container", xy=(0.99, 8),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8.5, color=CRIT)
        ax.set_title(label, fontsize=10.5, color=INK, loc="left", pad=8,
                     fontweight="bold")
        ax.set_xlabel("seconds into the measured phase "
                      "(repetitions run back to back)", fontsize=9.5)
        if ax is axes[0]:
            ax.set_ylabel("CPU cores busy in the PostgreSQL container",
                          fontsize=9.5)
        ax.set_ylim(0, 9.6)
        style(ax)

    figtitle(fig, "The Rollup Removes Work and the Parallelism That Hid It",
             f"{SUBTITLE} · every measured repetition drawn")
    fig.tight_layout(rect=title_rect(fig))
    save(fig, "3_cpu")


# ---------------------------------------------------- 4. parallelism
def walk(plan, out):
    """Every Plan node in a PostgreSQL EXPLAIN tree, depth first."""
    out.append(plan)
    for child in plan.get("Plans", []) or []:
        walk(child, out)
    return out


def executed_plans():
    """The profile run's plans, restricted to nodes that actually executed.

    A View is created with EXPLAIN (VERBOSE) and no ANALYZE -- it materializes
    nothing, so there is nothing to analyze -- and its gathers therefore report
    `Workers Launched: 0` for the trivial reason that the plan never ran.
    Counting those reads the DAG's twenty-odd views as a starved server.
    """
    rows = q("""
        select c.variant, p.node_id, p.plan_json
        from p_plans p
        join p_cells c using (cell_id)
        join p_node_executions n on n.run_id = p.run_id and n.node_id = p.node_id
        where p.plan_format = 'postgres_json'
          and n.materialization in ('table', 'temp_table')
    """)
    out = []
    for variant, node_id, plan_json in rows:
        try:
            doc = json.loads(plan_json)
        except (json.JSONDecodeError, TypeError):
            continue
        root = doc[0]["Plan"] if isinstance(doc, list) else doc.get("Plan")
        if root is not None:
            out.append((variant, short(node_id), root))
    return out


def self_time(node) -> float:
    """A node's own time: its total, less the children it waited on."""
    total = node.get("Actual Total Time", 0.0) * max(node.get("Actual Loops", 1), 1)
    for child in node.get("Plans", []) or []:
        total -= child.get("Actual Total Time", 0.0) * max(child.get("Actual Loops", 1), 1)
    return max(total, 0.0)


def chart_parallelism():
    if not HAVE_PLANS:
        print("  (no profile run given; skipping 4_parallelism)")
        return
    plans = executed_plans()
    if not plans:
        print("  (no executed-node plans captured; skipping 4_parallelism)")
        return

    # Per variant: every gather in an executed node, and the fused node's split
    # between CTE-scan time (which no grant can parallelize) and the rest.
    g = defaultdict(list)
    fused_cte = fused_total = 0.0
    for variant, name, root in plans:
        for node in walk(root, []):
            typ = str(node.get("Node Type", ""))
            if "Gather" in typ:
                g[variant].append((node.get("Workers Planned", 0),
                                   node.get("Workers Launched", 0)))
        if name.startswith("dee_fused"):
            # The node's own elapsed time is the denominator. Summing every
            # operator's self-time instead would double-count a subtree with
            # `Actual Loops` above one and put the total well past the wall
            # clock the node actually took.
            fused_total += root.get("Actual Total Time", 0.0)
            for node in walk(root, []):
                if str(node.get("Node Type", "")) == "CTE Scan":
                    fused_cte += self_time(node)
    fused_rest = max(fused_total - fused_cte, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax, ax2 = axes

    # -- left: what the planner asked for, against what it was granted
    order = [v for v in ("unopt", "nf_rule") if v in g]
    w = 0.34
    for i, variant in enumerate(order):
        items = g[variant]
        planned = max(x[0] for x in items)
        launched = max(x[1] for x in items)
        for off, val, color in ((-w / 2 - 0.02, planned, INK_3),
                                (w / 2 + 0.02, launched,
                                 OPT if variant == "nf_rule" else BASE)):
            ax.bar(i + off, val, width=w, color=color, zorder=3, linewidth=0)
            ax.annotate(f"{val:.0f}", xy=(i + off, val), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color=INK_2, fontweight="bold")
    ax.axhline(12, color=CRIT, lw=1.4, ls=(0, (4, 3)), zorder=6)
    ax.annotate("the grant: 12 workers per gather — never asked for",
                xy=(len(order) - 0.42, 12), xytext=(0, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=9, color=CRIT, fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["as authored" if v == "unopt" else "NodeFusion"
                        for v in order], fontsize=9.5, color=INK_2)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 14.4)
    ax.set_ylabel("workers on a gather (most seen)", fontsize=9.5)
    ax.set_title("PostgreSQL sizes a gather from the table, not from the cap",
                 fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
    style(ax)

    # -- right: the fused node's own time, split by what a worker could touch
    total = fused_total
    if total:
        ax2.barh(0, fused_cte, height=0.5, color=CRIT, zorder=3, linewidth=0)
        ax2.barh(0, fused_rest, left=fused_cte, height=0.5, color=OPT,
                 zorder=3, linewidth=0)
        ax2.annotate(f"CTE Scan — {fused_cte / 1000:.0f}s "
                     f"({fused_cte / total * 100:.0f}%)",
                     xy=(fused_cte / 2, 0), ha="center", va="center",
                     fontsize=10, color=SURFACE, fontweight="bold")
        # A narrow remainder cannot hold its own label, so it is set beside
        # the bar in ink rather than overflowing the segment in white.
        if fused_rest / total > 0.28:
            ax2.annotate(f"the rest of the node\n{fused_rest / 1000:.0f}s",
                         xy=(fused_cte + fused_rest / 2, 0), ha="center",
                         va="center", fontsize=9.5, color=SURFACE,
                         fontweight="bold")
        else:
            ax2.annotate(f"the rest of the node — {fused_rest / 1000:.0f}s",
                         xy=(total, 0), xytext=(8, 0),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=9.5, color=INK_2, fontweight="bold")
            ax2.set_xlim(0, total * 1.42)
    ax2.set_yticks([])
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_xlabel("dee_fused elapsed time (ms)", fontsize=9.5)
    ax2.set_title("A CTE Scan is not parallel-safe, whatever the grant",
                  fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
    style(ax2, ygrid=False, xgrid=True)

    handles = [Patch(facecolor=INK_3, label="workers the planner asked for"),
               Patch(facecolor=BASE, label="workers actually launched"),
               Patch(facecolor=CRIT, label="time no worker can take")]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_2,
               loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.02))
    figtitle(fig, "Why Twelve Workers Changed Nothing",
             f"{SUBTITLE} · from the full-verbosity profiling run")
    fig.tight_layout(rect=title_rect(fig, 0.07))
    save(fig, "4_parallelism")


if __name__ == "__main__":
    print(f"reading {', '.join(str(d.name) for d in RUN_DIRS)}"
          + (f" (+ profile {PROFILE_DIR.name})" if PROFILE_DIR else ""))
    print(f"  unopt   {B_MS / 1000:7.1f}s   nf_rule {O_MS / 1000:7.1f}s   "
          f"speedup {SPEEDUP:.3f}x"
          + (f"   work {WORK_SPEEDUP:.2f}x" if WORK_SPEEDUP else ""))
    chart_makespan()
    chart_nodes()
    chart_cpu()
    chart_parallelism()
