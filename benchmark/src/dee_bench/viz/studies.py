"""Build each study's charts and table from the parquet results.

One function per study, each taking a duckdb connection over the results and
returning a :class:`Study`. A study with no data returns an explicit
`empty_reason` rather than an empty chart, so a partial run explains itself.
"""

from __future__ import annotations

from typing import Any

from .spec import ChartSpec, Series, Study
from .theme import ordered


def _tables(con) -> set[str]:
    return {r[0] for r in con.sql("SHOW TABLES").fetchall()}


def _rows(con, sql: str) -> list[tuple]:
    try:
        return con.sql(sql).fetchall()
    except Exception:  # noqa: BLE001 - a missing table is a normal partial-run state
        return []


def _drop_empty_labels(labels: list[str], by_variant: dict[str, list]) -> tuple[list[str], dict[str, list]]:
    """Drop label positions where every series is None.

    A partial run measures the unoptimized baseline for a cell well before its
    optimized variants finish, which would otherwise show that cell as a bare
    x-axis label with no bars at all -- a category advertising data that isn't
    there yet.
    """
    keep = [i for i in range(len(labels)) if any(v[i] is not None for v in by_variant.values())]
    return ([labels[i] for i in keep],
            {name: [v[i] for i in keep] for name, v in by_variant.items()})


# Measured runtime per cell, the basis of most studies. Medians rather than
# means: a single slow repetition from an unrelated system hiccup should not
# move the headline number.
_MEASURED = """
    SELECT c.project, c.backend, c.sf, c.variant, c.cell_id,
           median(r.engine_wall_ms) / 1000.0 AS wall_s,
           median(r.cpu_seconds)              AS cpu_s,
           median(r.peak_rss_bytes) / 1048576.0 AS rss_mb,
           count(*)                           AS n
    FROM runs r JOIN cells c USING (cell_id)
    WHERE r.phase = 'measure' AND r.status = 'ok'
    GROUP BY ALL
"""


def study_scaling(con) -> Study:
    s = Study(
        key="scaling", number=1,
        title="Runtime vs scale factor",
        question="As dag-bench's scale factor grows, how does DAG runtime change?",
    )
    rows = _rows(con, f"SELECT * FROM ({_MEASURED}) ORDER BY sf")
    if not rows:
        s.empty_reason = "No measured runs recorded yet."
        return s

    sfs = sorted({r[2] for r in rows})
    single_sf = len(sfs) < 2
    if single_sf:
        s.empty_reason = (
            f"Only one scale factor ({sfs[0]:g}) was benchmarked. "
            "Sweep `matrix.sf` over several values to see scaling behaviour."
        )

    for backend in sorted({r[1] for r in rows}):
        for project in sorted({r[0] for r in rows if r[1] == backend}):
            names = ordered(list({r[3] for r in rows if r[0] == project and r[1] == backend}))
            if single_sf:
                # A single scale factor has no trend to draw as a line -- it
                # would just be disconnected dots on an arbitrary axis. Compare
                # variants directly instead, which is what the data actually is.
                by_name = {r[3]: r[5] for r in rows if r[0] == project and r[1] == backend}
                s.charts.append(ChartSpec(
                    id=f"scaling-{backend}-{project}",
                    kind="grouped_bar",
                    title=f"{project} on {backend}",
                    subtitle=f"Median measured runtime at scale factor {sfs[0]:g}",
                    x_label="", y_label="Runtime (s)",
                    series=[Series(name=n, x=[f"sf{sfs[0]:g}"], y=[by_name.get(n)])
                            for n in names],
                    note="Only one scale factor was benchmarked, so this shows runtime by "
                         "variant rather than a trend across scale.",
                ))
                continue
            series = []
            for name in names:
                pts = sorted(
                    (r for r in rows if r[0] == project and r[1] == backend and r[3] == name),
                    key=lambda r: r[2],
                )
                series.append(Series(name=name, x=[p[2] for p in pts], y=[p[5] for p in pts],
                                     meta={"n": [p[8] for p in pts]}))
            s.charts.append(ChartSpec(
                id=f"scaling-{backend}-{project}",
                kind="line",
                title=f"{project} on {backend}",
                subtitle="Median measured runtime at each scale factor",
                x_label="Scale factor", y_label="Runtime (s)",
                x_type="linear", series=series,
                note=("Both axes are linear, so a straight line means runtime grows in "
                      "proportion to data volume."),
            ))

    s.table_columns = ["Project", "Backend", "SF", "Variant", "Runtime (s)", "Reps"]
    s.table_rows = [[r[0], r[1], f"{r[2]:g}", r[3], f"{r[5]:.3f}", r[8]] for r in rows]
    return s


def study_optimization(con) -> Study:
    s = Study(
        key="optimization", number=2,
        title="Runtime response to optimization",
        question="How does DAG runtime change in response to dee's optimizations?",
    )
    rows = _rows(con, f"SELECT * FROM ({_MEASURED})")
    if not rows:
        s.empty_reason = "No measured runs recorded yet."
        return s

    # Normalize to each group's unoptimized baseline so projects of very
    # different absolute runtimes can be read on one axis.
    baselines = {(r[0], r[1], r[2]): r[5] for r in rows if r[3] == "unopt"}
    variants = ordered([v for v in {r[3] for r in rows} if v != "unopt"])
    if not variants:
        s.empty_reason = "Only the unoptimized variant was benchmarked; nothing to compare."
        return s

    labels, by_variant = [], {v: [] for v in variants}
    for key in sorted({(r[0], r[1], r[2]) for r in rows}):
        base = baselines.get(key)
        if not base:
            continue
        labels.append(f"{key[0]}\n{key[1]} sf{key[2]:g}")
        for v in variants:
            match = [r[5] for r in rows if (r[0], r[1], r[2]) == key and r[3] == v]
            by_variant[v].append(base / match[0] if match and match[0] else None)
    labels, by_variant = _drop_empty_labels(labels, by_variant)
    if not labels:
        s.empty_reason = "No optimized variant has a measured baseline to compare against yet."
        return s

    s.charts.append(ChartSpec(
        id="optimization-speedup", kind="grouped_bar",
        title="Speedup over the unoptimized DAG",
        subtitle="Median runtime of the unoptimized DAG divided by the variant's",
        x_label="", y_label="Speedup (x)",
        series=[Series(name=v, x=labels, y=by_variant[v]) for v in variants],
        hline=1.0, hline_label="unoptimized baseline",
        note="Above the baseline line is faster than unoptimized; below it is a regression.",
    ))

    s.table_columns = ["Project", "Backend", "SF", "Variant", "Runtime (s)", "Speedup"]
    for r in sorted(rows):
        base = baselines.get((r[0], r[1], r[2]))
        speedup = f"{base / r[5]:.2f}x" if base and r[5] else "-"
        s.table_rows.append([r[0], r[1], f"{r[2]:g}", r[3], f"{r[5]:.3f}", speedup])
    return s


def study_payback(con) -> Study:
    s = Study(
        key="payback", number=3,
        title="Optimization payback",
        question="How many DAG runs does it take to pay back the cost of optimizing?",
    )
    if "payback" not in _tables(con):
        s.empty_reason = "Run `dee-bench analyze <run_dir>` to compute the payback table."
        return s
    rows = _rows(con, """
        SELECT project, backend, sf, variant, opt_cost_wall_s, opt_cost_cpu_s,
               savings_per_run_wall_s, payback_runs_wall, payback_runs_cpu,
               payback_runs_wall_lo, payback_runs_wall_hi, speedup
        FROM payback ORDER BY backend, project, sf, variant
    """)
    if not rows:
        s.empty_reason = "No optimized cells had a matching unoptimized baseline to compare against."
        return s

    repaid = [r for r in rows if r[7] is not None]
    if repaid:
        labels = [f"{r[0]}\n{r[1]} sf{r[2]:g}" for r in repaid]
        by_variant: dict[str, list] = {}
        for v in ordered(list({r[3] for r in repaid})):
            by_variant[v] = [
                next((r[7] for r in repaid if r[3] == v and f"{r[0]}\n{r[1]} sf{r[2]:g}" == lab), None)
                for lab in dict.fromkeys(labels)
            ]
        uniq = list(dict.fromkeys(labels))
        s.charts.append(ChartSpec(
            id="payback-runs", kind="grouped_bar",
            title="Runs to repay the optimization",
            subtitle="Optimization wall time divided by the wall time saved per run",
            x_label="", y_label="DAG runs to break even",
            series=[Series(name=v, x=uniq, y=y) for v, y in by_variant.items()],
            note=("Lower is better: the optimization pays for itself sooner. Cells where the "
                  "variant was not faster never break even and are omitted here — see the table."),
        ))

    never = len(rows) - len(repaid)
    if never:
        s.charts and s.charts[0].__setattr__(
            "note", s.charts[0].note + f" {never} of {len(rows)} cells never break even."
        )

    s.table_columns = ["Project", "Backend", "SF", "Variant", "Opt cost (s)",
                       "Saved/run (s)", "Speedup", "Payback (runs)", "95% CI"]
    for r in rows:
        ci = f"{r[9]:.1f}–{r[10]:.1f}" if r[9] is not None and r[10] is not None else "-"
        s.table_rows.append([
            r[0], r[1], f"{r[2]:g}", r[3],
            f"{r[4]:.1f}" if r[4] is not None else "-",
            f"{r[6]:.3f}" if r[6] is not None else "-",
            f"{r[11]:.2f}x" if r[11] else "-",
            f"{r[7]:.1f}" if r[7] is not None else "never",
            ci,
        ])
    return s


def study_ablation(con) -> Study:
    s = Study(
        key="ablation", number=4,
        title="Ablation: progressively more aggressive optimization",
        question="How does runtime change as each optimization is layered on?",
    )
    rows = _rows(con, f"SELECT * FROM ({_MEASURED})")
    if not rows:
        s.empty_reason = "No measured runs recorded yet."
        return s
    ladder = [v for v in ["unopt", "hmp", "hmp_pushdown", "full"] if v in {r[3] for r in rows}]
    if len(ladder) < 2:
        s.empty_reason = (
            "The ablation needs at least two rungs of the ladder "
            "(unopt -> hmp -> hmp_pushdown -> full). Add them to `matrix.variant`."
        )
        return s

    series = []
    for backend in sorted({r[1] for r in rows}):
        for project in sorted({r[0] for r in rows if r[1] == backend}):
            for sf in sorted({r[2] for r in rows if r[0] == project and r[1] == backend}):
                base = next((r[5] for r in rows
                             if (r[0], r[1], r[2], r[3]) == (project, backend, sf, "unopt")), None)
                if not base:
                    continue
                y = []
                for v in ladder:
                    match = [r[5] for r in rows
                             if (r[0], r[1], r[2], r[3]) == (project, backend, sf, v)]
                    y.append(match[0] / base if match else None)
                series.append(Series(name=f"{project} · {backend} sf{sf:g}", x=ladder, y=y))

    s.charts.append(ChartSpec(
        id="ablation-ladder", kind="line",
        title="Runtime as optimizations are layered on",
        subtitle="Runtime relative to the unoptimized DAG; each step adds another optimization",
        x_label="", y_label="Relative runtime", series=series,
        hline=1.0, hline_label="unoptimized",
        note="Downward is faster. A line that rises between rungs means that optimization hurt.",
    ))
    s.table_columns = ["Project", "Backend", "SF"] + [f"{v} (rel.)" for v in ladder]
    for ser in series:
        name = ser.name.replace(" · ", "|").replace(" sf", "|")
        s.table_rows.append(name.split("|") + [f"{v:.3f}" if v else "-" for v in ser.y])
    return s


def study_pass_changes(con) -> Study:
    s = Study(
        key="pass_changes", number=5,
        title="Changes made by each optimizer pass",
        question="How many changes did each optimization pass make to each DAG?",
    )
    rows = _rows(con, """
        SELECT c.project, c.backend, c.sf, c.variant, p.pass_name,
               p.changes_applied, p.candidates_considered, p.working_set_size,
               p.wall_ms / 1000.0 AS wall_s, p.dag_runs_used
        FROM pass_stats p JOIN cells c USING (cell_id)
        ORDER BY c.backend, c.project, c.sf, p.pass_order
    """)
    if not rows:
        s.empty_reason = "No optimizer passes ran. Add a variant with passes to `matrix.variant`."
        return s

    labels = list(dict.fromkeys(f"{r[0]}\n{r[3]} sf{r[2]:g}" for r in rows))
    passes = sorted({r[4] for r in rows})
    series = []
    for p in passes:
        series.append(Series(
            name=p, x=labels,
            y=[next((r[5] for r in rows if r[4] == p and f"{r[0]}\n{r[3]} sf{r[2]:g}" == lab), None)
               for lab in labels],
        ))
    s.charts.append(ChartSpec(
        id="pass-changes", kind="grouped_bar",
        title="Changes applied per pass",
        subtitle="Materializations for HMP and OMP; query rewrites for Pushdown",
        x_label="", y_label="Changes applied", series=series,
        note=("The unit differs by pass — a materialization is a much larger structural change "
              "than a rewrite — so compare a pass against itself across DAGs, not against "
              "another pass."),
    ))

    s.table_columns = ["Project", "Backend", "SF", "Variant", "Pass",
                       "Changes", "Candidates", "Working set", "Pass time (s)", "DAG runs"]
    s.table_rows = [[r[0], r[1], f"{r[2]:g}", r[3], r[4], r[5], r[6], r[7],
                     f"{r[8]:.2f}", r[9]] for r in rows]
    return s


def study_system(con) -> Study:
    s = Study(
        key="system", number=6,
        title="System resource usage during runs",
        question="What do CPU, memory and I/O look like while DAGs run and optimize?",
    )
    if "system_samples" not in _tables(con):
        s.empty_reason = (
            "System samples are recorded at `detailed` verbosity and above. "
            "Re-run with `verbosity: detailed`."
        )
        return s
    rows = _rows(con, """
        SELECT c.variant, c.project, c.backend, s.phase, s.source,
               s.elapsed_ms, s.cpu_seconds_cum, s.rss_bytes
        FROM system_samples s JOIN cells c USING (cell_id)
        WHERE s.source <> 'engine_internal' AND s.phase = 'measure'
        ORDER BY s.elapsed_ms
    """)
    if not rows:
        s.empty_reason = "No external system samples were captured."
        return s

    for metric, idx, label, scale in (
        ("cpu", 6, "Cumulative CPU (s)", 1.0),
        ("mem", 7, "Resident memory (MB)", 1 / 1048576.0),
    ):
        series = []
        for variant in ordered(list({r[0] for r in rows})):
            pts = [r for r in rows if r[0] == variant and r[idx] is not None]
            if pts:
                series.append(Series(
                    name=variant,
                    x=[p[5] / 1000.0 for p in pts],
                    y=[p[idx] * scale for p in pts],
                ))
        if series:
            s.charts.append(ChartSpec(
                id=f"system-{metric}", kind="scatter",
                title=label.split(" (")[0] + " over a run",
                subtitle="Sampled externally from /proc and cgroup counters, by variant",
                x_label="Elapsed (s)", y_label=label,
                x_type="linear", series=series,
            ))
    return s


def study_resource_response(con) -> Study:
    s = Study(
        key="resource_response", number=7,
        title="How runtime, memory and CPU respond to optimization",
        question="Do dee's optimizations trade one resource for another?",
    )
    rows = _rows(con, f"SELECT * FROM ({_MEASURED})")
    if not rows:
        s.empty_reason = "No measured runs recorded yet."
        return s
    baselines = {(r[0], r[1], r[2]): r for r in rows if r[3] == "unopt"}
    variants = ordered([v for v in {r[3] for r in rows} if v != "unopt"])
    if not variants or not baselines:
        s.empty_reason = "Needs both an unoptimized baseline and at least one optimized variant."
        return s

    # Three panels rather than one chart with three axes: these are different
    # units and a shared axis would be meaningless.
    for metric, idx, label in (("runtime", 5, "Runtime"), ("cpu", 6, "CPU seconds"),
                               ("memory", 7, "Peak memory")):
        labels, by_variant = [], {v: [] for v in variants}
        for key in sorted(baselines):
            base = baselines[key][idx]
            if not base:
                continue
            labels.append(f"{key[0]}\n{key[1]} sf{key[2]:g}")
            for v in variants:
                match = [r[idx] for r in rows if (r[0], r[1], r[2]) == key and r[3] == v]
                by_variant[v].append(match[0] / base if match and match[0] else None)
        labels, by_variant = _drop_empty_labels(labels, by_variant)
        if not labels:
            continue
        s.charts.append(ChartSpec(
            id=f"response-{metric}", kind="grouped_bar",
            title=f"{label} relative to unoptimized",
            subtitle=f"{label} of each variant divided by the unoptimized DAG's",
            x_label="", y_label=f"Relative {label.lower()}",
            series=[Series(name=v, x=labels, y=by_variant[v]) for v in variants],
            hline=1.0, hline_label="unoptimized",
            note="Below 1.0 uses less of this resource than the unoptimized DAG; above 1.0 uses more.",
        ))

    s.table_columns = ["Project", "Backend", "SF", "Variant",
                       "Runtime (s)", "CPU (s)", "Peak mem (MB)"]
    for r in sorted(rows):
        s.table_rows.append([
            r[0], r[1], f"{r[2]:g}", r[3],
            f"{r[5]:.3f}" if r[5] else "-",
            f"{r[6]:.2f}" if r[6] else "-",
            f"{r[7]:.0f}" if r[7] else "-",
        ])
    return s


BUILDERS = [
    study_scaling,
    study_optimization,
    study_payback,
    study_ablation,
    study_pass_changes,
    study_system,
    study_resource_response,
]


def build_all(con) -> list[Study]:
    out = []
    for fn in BUILDERS:
        try:
            out.append(fn(con))
        except Exception as e:  # noqa: BLE001 - one broken study must not lose the rest
            key = fn.__name__.replace("study_", "")
            out.append(Study(key=key, number=0, title=key, question="",
                             empty_reason=f"Could not build this study: {type(e).__name__}: {e}"))
    return out
