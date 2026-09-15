"""Charts for an HMP-vs-baseline sweep: what the pass bought, and what it changed.

    .venv/bin/python scripts/hmp_makespan_charts.py results/hmp-makespan-eval

Reads only the parquet a sweep wrote, so it can be re-run at any time --
including on a partial or cancelled run -- and writes png and pdf into
`<run_dir>/charts/`. It expects a sweep pairing an `unopt` variant against an
`hmp` one, and reads the tables `standard` verbosity and above record.

Four figures, each answering one question:

  1_makespan       what the optimized DAG costs against the DAG as authored
  2_changes        what HMP actually changed, and how much of the DAG that is
  3_search         how the search spent its run budget, and what cancelling bought
  4_cost           what the search cost, and how many runs repay it
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DEFAULT_RUN = Path(__file__).resolve().parents[1] / "results" / "hmp-makespan-eval"
# Several run directories may be passed, and are read as one dataset: every
# table is keyed by cell_id, so the union of two sweeps is the same as having
# run their cells together.
RUN_DIRS = [Path(a).resolve() for a in sys.argv[1:]] or [DEFAULT_RUN]
for _d in RUN_DIRS:
    if not (_d / "results").is_dir():
        sys.exit(f"{_d} is not a dee-bench run directory (no results/ inside)")
RUN_DIR = RUN_DIRS[0]
OUT = RUN_DIR / "charts"
# Raster resolution. The pdf is vector and ignores this; the png is what lands
# on a slide, where it gets scaled up and wants the headroom.
PNG_DPI = 400

# ---------------------------------------------------------------- theme
# The validated default categorical palette, in the same slot order the
# dashboard uses, so a variant keeps its color across every artifact of this
# sweep. Status hues are reserved and never reused as a series.
BASE = "#2a78d6"          # slot 1 — the DAG as authored
OPT = "#eb6834"           # slot 2 — the DAG the pass produced
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

def project_label(name: str) -> str:
    """`p06_logistics` -> `p06` over `logistics`, for any dag-bench project."""
    head, _, rest = name.partition("_")
    return f"{head}\n{rest}" if rest else name
BACKEND_LABEL = {"duckdb": "DuckDB", "postgres": "PostgreSQL"}

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
    ax.grid(ygrid, axis="y", color=GRID, linewidth=1.0)
    ax.grid(xgrid, axis="x", color=GRID, linewidth=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(labelsize=8, length=0)


def figtitle(fig, title, subtitle):
    """Title block pinned a constant number of inches from the top.

    Offsets in figure fractions would shrink with a taller figure and put the
    subtitle back under the title on the tallest one, so they are converted
    from inches each time instead.
    """
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.24 / h, title, fontsize=14, color=INK,
             fontweight="bold", va="top")
    if subtitle:
        fig.text(0.012, 1 - 0.47 / h, subtitle, fontsize=9, color=INK_2, va="top")


def title_rect(fig, bottom=0.0):
    """`tight_layout` rect that clears the title block above the axes."""
    return (0, bottom, 1, 1 - 0.62 / fig.get_figheight())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{fmt}", format=fmt, dpi=PNG_DPI,
                    facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote charts/{name}.png / .pdf")


# ---------------------------------------------------------------- data
def q(sql):
    return con.execute(sql).fetchall()


con = duckdb.connect()
for tbl in ("cells", "runs", "optimizations", "pass_stats",
            "pass_iterations", "dag_graph", "node_executions"):
    globs = [f"'{d / 'results' / tbl}/**/*.parquet'"
             for d in RUN_DIRS
             if list((d / "results" / tbl).glob("**/*.parquet"))]
    if not globs:
        continue
    con.execute(
        f"create view {tbl} as select * from read_parquet([{', '.join(globs)}], "
        f"hive_partitioning=1, union_by_name=1)"
    )

# One row per project x backend: the baseline's makespan and the optimized
# DAG's, both as the median of the measured repetitions. Only `direct`
# deliveries are comparable — a resumed run measures neither DAG.
PERF_SQL = """
with m as (
  select c.project, c.backend, c.variant,
         median(r.engine_wall_ms) as ms,
         min(r.engine_wall_ms) as lo,
         max(r.engine_wall_ms) as hi,
         median(r.node_time_ms) as node_ms,
         count(*) as n
  from runs r join cells c using (cell_id)
  where r.phase = 'measure' and r.status = 'ok' and r.delivery = 'direct'
  group by 1, 2, 3
)
select b.project, b.backend, b.ms, b.lo, b.hi, b.node_ms,
       o.ms, o.lo, o.hi, o.node_ms, b.n
from m b join m o on b.project = o.project and b.backend = o.backend
where b.variant = 'unopt' and o.variant = 'hmp'
order by b.backend, b.project
"""

CHANGE_SQL = """
select c.project, c.backend, p.changes_applied, p.candidates_considered,
       p.working_set_size, o.nodes_before, o.nodes_after,
       o.dag_runs_used, o.opt_wall_ms
from pass_stats p
join cells c using (cell_id)
join optimizations o using (cell_id)
where p.pass_name = 'HMPPass' and o.status = 'ok'
order by c.backend, c.project
"""

# What the pass actually did to the graph, by diffing the recorded structure of
# a cell's two DAG variants. HMP does not flip a view to a table in place: it
# inserts a *landing pad* -- a temp table holding that view's result, computed
# once -- and repoints the view's consumers at the pad, so the node that changes
# and the nodes that change their mind about where to read are different sets.
ADDED_SQL = """
select c.project, c.backend, o.node_id, o.materialization, o.out_degree
from dag_graph o
join cells c on c.cell_id = o.cell_id
left join dag_graph u on u.cell_id = o.cell_id and u.node_id = o.node_id
                     and u.dag_variant = 'unopt'
where o.dag_variant = 'optimized' and u.node_id is null
order by c.backend, c.project, o.node_id
"""

REWIRE_SQL = """
select c.project, c.backend, u.node_id
from dag_graph u
join dag_graph o on u.cell_id = o.cell_id and u.node_id = o.node_id
                and u.dag_variant = 'unopt' and o.dag_variant = 'optimized'
join cells c on c.cell_id = u.cell_id
where u.depends_on::varchar is distinct from o.depends_on::varchar
order by c.backend, c.project, u.node_id
"""

ITER_SQL = """
select c.project, c.backend, i.iteration, i.outcome, i.runtime_ms,
       i.trial_ms, i.resume_ms, i.resume_overhead_ms, i.total_ms, i.combo
from pass_iterations i join cells c using (cell_id)
where i.pass_name = 'HMPPass'
order by c.backend, c.project, i.iteration
"""


def short(node_id: str) -> str:
    """`"warehouse"."main"."order_line_facts"` -> `order_line_facts`."""
    return node_id.replace('"', "").split(".")[-1]


REPS = max((n for (n,) in q("select distinct repetitions from cells")), default=1)
# One short provenance line, the same on every figure, so a slide carries its
# conditions without a paragraph explaining them.
_sfs = sorted({sf for (sf,) in q("select distinct sf from cells")})
SUBTITLE = " · ".join([
    ", ".join(BACKEND_LABEL.get(b, b) for b in
              (bb for (bb,) in q("select distinct backend from cells order by 1"))),
    "sf=" + "/".join(f"{s_:g}" for s_ in _sfs),
])
REPS_NOTE = f"{REPS} measured run" + ("s" if REPS != 1 else "") + " per bar"
perf = q(PERF_SQL)
changes = q(CHANGE_SQL)
added = q(ADDED_SQL)
rewired = q(REWIRE_SQL)
iters = q(ITER_SQL)
if not perf:
    sys.exit("no measured runs yet")

BACKENDS = [b for b in ("duckdb", "postgres") if any(r[1] == b for r in perf)]
# The run budget the search was given, read off the cells rather than fixed
# here, so the "runs never needed" shading means the right thing for whatever
# `hmp_max_runs` the sweep was configured with.
_budgets = {json.loads(d).get("hmp_max_runs") for (d,) in
            q("select distinct dee_opt from cells where list_contains(passes, 'hmp')")}
_budgets.discard(None)
BUDGET = max(_budgets) if _budgets else 8

# Cells where the pass applied nothing ran the *same* DAG under both variants.
# Their difference is run-to-run variance, not an effect, and every chart has
# to say so rather than report a speedup the optimizer did not produce.
TOUCHED = {}
for _p, _b, _ch, *_ in changes:
    TOUCHED[(_p, _b)] = _ch
for _p, _b, _node, *_ in added:
    TOUCHED[(_p, _b)] = max(TOUCHED.get((_p, _b), 0), 1)


def unchanged(project, backend):
    return TOUCHED.get((project, backend), 0) == 0


# ---------------------------------------------------- 1. makespan
def chart_makespan():
    fig, axes = plt.subplots(1, len(BACKENDS), figsize=(6.4 * len(BACKENDS) + 1.2, 4.4))
    axes = [axes] if len(BACKENDS) == 1 else list(axes)

    for ax, backend in zip(axes, BACKENDS):
        rows = [r for r in perf if r[1] == backend]
        pos = range(len(rows))
        w = 0.23

        for i, r in enumerate(rows):
            _, _, b_ms, b_lo, b_hi, _, o_ms, o_lo, o_hi, _, _ = r
            for off, ms, lo, hi, color in ((-w / 2 - 0.02, b_ms, b_lo, b_hi, BASE),
                                           (w / 2 + 0.02, o_ms, o_lo, o_hi, OPT)):
                ax.bar(i + off, ms, width=w, color=color, zorder=3, linewidth=0)
                # One repetition has no spread to draw, and a zero-length
                # whisker would imply a measurement that was repeated.
                if REPS > 1:
                    ax.plot([i + off, i + off], [lo, hi], color=SURFACE, lw=3.0, zorder=4)
                    ax.plot([i + off, i + off], [lo, hi], color=INK_3, lw=1.2, zorder=5)

            top = max(b_hi, o_hi)
            if unchanged(r[0], backend):
                ax.annotate("unchanged", xy=(i, top), xytext=(0, 9),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=9, color=INK_3)
            else:
                delta = (o_ms - b_ms) / b_ms * 100.0
                faster, slower = delta < -1.0, delta > 1.0
                ax.annotate(f"{delta:+.0f}%" if (faster or slower) else "±0%",
                            xy=(i, top), xytext=(0, 9), textcoords="offset points",
                            ha="center", va="bottom", fontsize=11,
                            color=GOOD if faster else (CRIT if slower else INK_3),
                            fontweight="bold")

        ax.set_xticks(list(pos))
        ax.set_xticklabels([project_label(r[0]) for r in rows],
                           fontsize=9.5, color=INK_2)
        ax.set_xlim(-0.55, len(rows) - 0.45)
        ax.set_ylim(0, max(max(r[4], r[8]) for r in rows) * 1.22)
        ax.set_ylabel("makespan (ms)" if ax is axes[0] else "", fontsize=9.5)
        if len(BACKENDS) > 1:
            ax.set_title(BACKEND_LABEL.get(backend, backend), fontsize=10.5,
                         color=INK, loc="left", pad=8, fontweight="bold")
        style(ax)

    handles = [Patch(facecolor=BASE, label="original"),
               Patch(facecolor=OPT, label="optimized")]
    if REPS > 1:
        handles.append(Line2D([], [], color=INK_3, lw=1.2,
                              label=f"min–max over {REPS} runs"))
    fig.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_2,
               loc="lower center", ncols=len(handles), bbox_to_anchor=(0.5, -0.02))

    figtitle(fig, "Adaptive Materialization Reduces End-to-End Runtime",
             f"{SUBTITLE} · {REPS_NOTE}")
    fig.tight_layout(rect=title_rect(fig, 0.06))
    save(fig, "1_makespan")


def _entry(ax, y, color, text):
    """One line of the change list: a colored mark, then text in ink.

    Identity rides on the mark rather than the type, because a sentence set in
    a series color reads as decoration and fails contrast besides.
    """
    ax.plot([0.022], [y - 0.008], marker="s", markersize=5.5, color=color,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.042, y, text, fontsize=8.5, color=INK_2,
            transform=ax.transAxes, va="top")


# ---------------------------------------------------- 2. what changed
def chart_changes():
    rows = [(c[0], c[1], c[2], c[5], c[6]) for c in changes]
    if not rows:
        return
    by_added, by_rewired = {}, {}
    for project, backend, node, mat, out_degree in added:
        by_added.setdefault((project, backend), []).append((node, mat, out_degree))
    for project, backend, node in rewired:
        by_rewired.setdefault((project, backend), []).append(node)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 0.58 * len(rows) + 3.1),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    ax, ax2 = axes

    labels = [project_label(p).replace("\n", " ") + (
        f" · {BACKEND_LABEL.get(b, b)}" if len(BACKENDS) > 1 else "")
        for p, b, *_ in rows]
    n_added = [len(by_added.get((p, b), [])) for p, b, *_ in rows]
    n_rewired = [len(by_rewired.get((p, b), [])) for p, b, *_ in rows]
    totals = [r[3] for r in rows]
    y = list(range(len(rows)))[::-1]

    untouched = [t - a - r_ for t, a, r_ in zip(totals, n_added, n_rewired)]
    ax.barh(y, n_added, height=0.5, color=OPT, zorder=3, linewidth=0,
            label="tables inserted")
    ax.barh(y, n_rewired, height=0.5, left=n_added, color=BASE, zorder=3,
            linewidth=0, label="nodes repointed")
    ax.barh(y, untouched, height=0.5,
            left=[a + r_ for a, r_ in zip(n_added, n_rewired)],
            color=SURFACE_2, edgecolor=BORDER, linewidth=1.0, zorder=2,
            label="unchanged")

    for yi, a, r_, nb in zip(y, n_added, n_rewired, totals):
        touched = a + r_
        ax.annotate("nothing changed" if touched == 0 else f"{touched} of {nb}",
                    xy=(nb, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9,
                    color=INK_3 if touched == 0 else INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_2)
    ax.set_xlabel("nodes in the DAG", fontsize=9.5)
    ax.set_xlim(0, max(totals) * 1.30)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncols=3, columnspacing=1.3,
              handlelength=1.2)
    style(ax, ygrid=False, xgrid=True)

    # Naming them is the half that makes the counts mean anything.
    ax2.axis("off")
    lines = []
    for project, backend, ch, nb, na in rows:
        key = (project, backend)
        head = project_label(project).replace("\n", " ")
        if len(BACKENDS) > 1:
            head += f" · {BACKEND_LABEL.get(backend, backend)}"
        lines.append(("head", head))
        a_rows, r_rows = by_added.get(key, []), by_rewired.get(key, [])
        if not a_rows and not r_rows:
            lines.append((None, "left exactly as authored"))
            continue
        for node, mat, out_degree in a_rows:
            lines.append((OPT, f"+ {short(node)}  ({out_degree} readers)"))
        if r_rows:
            # A DAG with four repointed consumers runs off the panel on one
            # line, so the list wraps and continues unmarked underneath.
            wrapped = textwrap.wrap(", ".join(short(n) for n in r_rows), width=58)
            lines.append((BASE, "→ " + wrapped[0]))
            lines.extend((False, w) for w in wrapped[1:])
    step = 1.0 / (len(lines) + 1)
    line_y = 1.0
    for color, text in lines:
        if color == "head":
            ax2.text(0.0, line_y, text, fontsize=9.5, color=INK,
                     fontweight="bold", transform=ax2.transAxes, va="top")
        elif color is None:
            ax2.text(0.03, line_y, text, fontsize=9, color=INK_3,
                     transform=ax2.transAxes, va="top")
        elif color is False:
            ax2.text(0.042, line_y, text, fontsize=9, color=INK_2,
                     transform=ax2.transAxes, va="top")
        else:
            _entry(ax2, line_y, color, text)
        line_y -= step

    figtitle(fig, "Changes Made to Each DAG", SUBTITLE)
    fig.tight_layout(rect=title_rect(fig, 0.02))
    save(fig, "2_changes")


# ---------------------------------------------------- 3. the search
def chart_search():
    cells = []
    for project, backend, *_ in iters:
        if (project, backend) not in cells:
            cells.append((project, backend))
    if not cells:
        return
    ncol = min(3, len(cells))
    nrow = (len(cells) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.9 * nrow + 0.9),
                             squeeze=False)
    flat = [a for row in axes for a in row]

    for ax, (project, backend) in zip(flat, cells):
        rows = [r for r in iters if r[0] == project and r[1] == backend]
        best = None
        for r in rows:
            it, outcome, runtime, trial, resume, overhead, total, combo = r[2:]
            if outcome == "baseline":
                color, hatch = INK_3, None
            elif outcome == "ok":
                color, hatch = GOOD, None
            else:
                color, hatch = CRIT, "///"
            ax.bar(it, total, width=0.62, color=color, zorder=3, linewidth=0,
                   hatch=hatch, edgecolor=SURFACE)
            if outcome == "cancelled" and trial is not None:
                ax.bar(it, trial, width=0.62, color=CRIT, zorder=4, linewidth=0)
                ax.plot([it - 0.31, it + 0.31], [trial, trial], color=SURFACE,
                        lw=2.0, zorder=5)
            if outcome in ("baseline", "ok"):
                best = total if best is None else min(best, total)

        if best is not None:
            ax.axhline(best, color=INK_3, lw=1.0, ls="--", zorder=2)

        used = max(r[2] for r in rows)
        ax.set_xticks(list(range(1, BUDGET + 1)))
        ax.set_xticklabels([str(i) for i in range(1, BUDGET + 1)], fontsize=8)
        ax.set_xlim(0.4, BUDGET + 0.6)
        if used < BUDGET:
            ax.axvspan(used + 0.5, BUDGET + 0.6, color=SURFACE_2, zorder=1)
            ax.annotate(f"{BUDGET - used} unused",
                        xy=((used + 0.5 + BUDGET) / 2, 0.5),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="center", fontsize=8.5, color=INK_3)
        ax.set_xlabel("DAG run", fontsize=9)
        ax.set_ylabel("wall time (ms)" if ax is flat[0] else "", fontsize=9)
        ax.set_title(project_label(project).replace("\n", " "), fontsize=10,
                     color=INK, loc="left", pad=6, fontweight="bold")
        style(ax)

    for ax in flat[len(cells):]:
        ax.axis("off")

    handles = [Patch(facecolor=INK_3, label="baseline"),
               Patch(facecolor=GOOD, label="kept"),
               Patch(facecolor=CRIT, label="cancelled at the cut"),
               Patch(facecolor=CRIT, hatch="///", edgecolor=SURFACE,
                     label="resumed under the incumbent"),
               Line2D([], [], color=INK_3, ls="--", lw=1.0, label="best so far")]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2,
               loc="lower center", ncols=5, bbox_to_anchor=(0.5, 0.004),
               columnspacing=1.4, handlelength=1.3)

    figtitle(fig, "The Results of Each Attempt",
             f"{SUBTITLE} · {BUDGET}-run budget")
    fig.tight_layout(rect=title_rect(fig, 0.075))
    save(fig, "3_search")


# ---------------------------------------------------- 4. cost and payback
def chart_cost():
    rows = []
    cost = {(c[0], c[1]): c for c in changes}
    for project, backend, b_ms, b_lo, b_hi, _, o_ms, o_lo, o_hi, _, _ in perf:
        c = cost.get((project, backend))
        if not c:
            continue
        saved = b_ms - o_ms
        opt_ms = c[8]
        if unchanged(project, backend):
            runs, why = None, "nothing changed"
        elif saved <= 0:
            runs, why = None, "never repays"
        else:
            runs, why = opt_ms / saved, None
        rows.append((project, backend, opt_ms, c[7], saved, runs, why))
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(8.6, 1.6 * len(rows) + 4.8), 4.3))
    ax, ax2 = axes
    labels = [project_label(p).replace("\n", " ") + (
        f"\n{BACKEND_LABEL.get(b, b)}" if len(BACKENDS) > 1 else "")
        for p, b, *_ in rows]
    pos = list(range(len(rows)))

    ax.bar(pos, [r[2] / 1000.0 for r in rows], width=0.34, color=BASE,
           zorder=3, linewidth=0)
    for i, r in enumerate(rows):
        ax.annotate(f"{r[3]} runs", xy=(i, r[2] / 1000.0), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK_2)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK_2)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, max(r[2] for r in rows) / 1000.0 * 1.20)
    ax.set_ylabel("search wall time (s)", fontsize=9.5)
    ax.set_title("Cost, paid once", fontsize=10.5, color=INK, loc="left",
                 pad=8, fontweight="bold")
    style(ax)

    finite = [r[5] for r in rows if r[5] is not None]
    cap = (max(finite) * 1.35) if finite else 10.0
    for i, r in enumerate(rows):
        if r[5] is None:
            # No bar: a placeholder height would read off the axis as that
            # many runs, which is the one thing this slot must not say.
            ax2.axvspan(i - 0.3, i + 0.3, color=SURFACE_2, zorder=1)
            ax2.annotate(r[6], xy=(i, cap * 0.5), ha="center", va="center",
                         fontsize=9,
                         color=INK_3 if r[6].startswith("nothing") else CRIT)
        else:
            ax2.bar(i, r[5], width=0.34, color=OPT, zorder=3, linewidth=0)
            ax2.annotate(f"{r[5]:.0f}", xy=(i, r[5]), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=10,
                         color=INK_2, fontweight="bold")
    ax2.set_xticks(pos)
    ax2.set_xticklabels(labels, fontsize=9.5, color=INK_2)
    ax2.set_xlim(-0.6, len(rows) - 0.4)
    ax2.set_ylim(0, cap * 1.15)
    ax2.set_ylabel("scheduled runs to break even", fontsize=9.5)
    ax2.set_title("Repaid after", fontsize=10.5, color=INK, loc="left",
                  pad=8, fontweight="bold")
    style(ax2)

    figtitle(fig, "Seconds to search, repaid within tens of scheduled runs",
             SUBTITLE)
    fig.tight_layout(rect=title_rect(fig, 0.02))
    save(fig, "4_cost")


print("rendering charts from " + ", ".join(d.name for d in RUN_DIRS))
chart_makespan()
chart_changes()
chart_search()
chart_cost()
