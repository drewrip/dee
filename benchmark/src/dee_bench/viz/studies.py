"""Build each study's charts and table from the parquet results.

One function per study, each taking a duckdb connection over the results and
returning a :class:`Study`. A study with no data returns an explicit
`empty_reason` rather than an empty chart, so a partial run explains itself.
"""

from __future__ import annotations

import json

from ..config import config_labels
from .query import median
from .spec import ChartSpec, Kpi, Series, Study
from .theme import STATUS, ordered


def _tables(con) -> set[str]:
    return {r[0] for r in con.sql("SHOW TABLES").fetchall()}


def _rows(con, sql: str) -> list[tuple]:
    try:
        return con.sql(sql).fetchall()
    except Exception:  # noqa: BLE001 - a missing table is a normal partial-run state
        return []


def label_backends(con) -> None:
    """Fold swept backend tuning into the `backend` column of every view.

    A study identifies a measurement by project, backend, scale factor and
    variant. When a run sweeps backend settings, two cells agree on all four
    and differ only in their tuning, so one would silently overwrite the other
    in every chart and table. Rewriting `backend` as ``duckdb[max_memory=8GB]``
    separates them everywhere at once, and reads as what actually differs --
    rather than teaching each study about backend configurations one by one.

    A run that sweeps nothing has one configuration per backend, no label to
    add, and its views are left exactly as they were.
    """
    tables = _tables(con)
    if "cells" not in tables:
        return

    by_backend: dict[str, set[str]] = {}
    for backend, config in _rows(con, "SELECT DISTINCT backend, backend_config FROM cells"):
        by_backend.setdefault(backend, set()).add(config or "{}")

    mapping: dict[tuple[str, str], str] = {}
    for backend, configs in by_backend.items():
        ordered_configs = sorted(configs)
        labels = config_labels([json.loads(c) for c in ordered_configs])
        for config, label in zip(ordered_configs, labels):
            mapping[(backend, config)] = f"{backend}[{label}]" if label else backend
    if all(label == backend for (backend, _), label in mapping.items()):
        return

    con.execute(
        "CREATE OR REPLACE TEMP TABLE backend_labels"
        "(backend VARCHAR, backend_config VARCHAR, label VARCHAR)"
    )
    con.executemany(
        "INSERT INTO backend_labels VALUES (?, ?, ?)",
        [[backend, config, label] for (backend, config), label in mapping.items()],
    )
    for table in ("cells", "payback"):
        if table not in tables:
            continue
        columns = {c[0] for c in _rows(con, f"DESCRIBE {table}")}
        # A payback table computed before backend sweeps existed has no
        # tuning to join on. Leaving it alone is right: it cannot have swept
        # one either.
        if not {"backend", "backend_config"} <= columns:
            continue
        con.execute(f"CREATE OR REPLACE TEMP TABLE _{table}_raw AS SELECT * FROM {table}")
        con.execute(f"DROP VIEW {table}")
        con.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT r.* REPLACE (COALESCE(l.label, r.backend) AS backend) "
            f"FROM _{table}_raw r "
            f"LEFT JOIN backend_labels l USING (backend, backend_config)"
        )


def label_variants(con) -> None:
    """Fold a swept optimizer parameter into the `variant` column of every view.

    The counterpart of :func:`label_backends`, for the other axis a cell can be
    swept along. An option that belongs to a pass is written under `dee_opt`
    and swept there rather than copied into a variant of its own, so two cells
    agree on project, backend, scale factor *and* variant and differ only in
    that parameter -- and one would silently overwrite the other in every chart
    and table. Rewriting `variant` as
    ``hmp[hmp_objective=makespan]`` separates them
    everywhere at once, and reads as the one thing that cell changed.

    Only settings that vary *within a variant* appear, so a run that sweeps
    nothing has one configuration per variant, no label to add, and its views
    are left exactly as they were.
    """
    tables = _tables(con)
    if "cells" not in tables:
        return

    by_variant: dict[str, set[str]] = {}
    for variant, dee_opt in _rows(con, "SELECT DISTINCT variant, dee_opt FROM cells"):
        by_variant.setdefault(variant, set()).add(dee_opt or "{}")

    labels: dict[tuple[str, str], str] = {}
    for variant, configs in by_variant.items():
        ordered_configs = sorted(configs)
        for config, label in zip(ordered_configs, config_labels(
            [json.loads(c) for c in ordered_configs]
        )):
            labels[(variant, config)] = f"{variant}[{label}]" if label else variant
    if all(label == variant for (variant, _), label in labels.items()):
        return

    # Keyed by cell, because `payback` carries the cell it priced but not the
    # `dee_opt` that distinguishes it.
    by_cell = {
        cell_id: labels[(variant, dee_opt or "{}")]
        for cell_id, variant, dee_opt in _rows(
            con, "SELECT cell_id, variant, dee_opt FROM cells"
        )
    }
    con.execute("CREATE OR REPLACE TEMP TABLE variant_labels(cell_id VARCHAR, label VARCHAR)")
    con.executemany(
        "INSERT INTO variant_labels VALUES (?, ?)", [[c, l] for c, l in by_cell.items()]
    )
    for table in ("cells", "payback"):
        if table not in tables:
            continue
        columns = {c[0] for c in _rows(con, f"DESCRIBE {table}")}
        if not {"cell_id", "variant"} <= columns:
            continue
        con.execute(f"CREATE OR REPLACE TEMP TABLE _{table}_variant_raw AS SELECT * FROM {table}")
        con.execute(f"DROP VIEW IF EXISTS {table}")
        con.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT r.* REPLACE (COALESCE(l.label, r.variant) AS variant) "
            f"FROM _{table}_variant_raw r "
            f"LEFT JOIN variant_labels l USING (cell_id)"
        )


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
           count(*)                           AS n,
           min(r.engine_wall_ms) / 1000.0     AS lo_s,
           max(r.engine_wall_ms) / 1000.0     AS hi_s
    FROM runs r JOIN cells c USING (cell_id)
    WHERE r.phase = 'measure' AND r.status = 'ok'
    GROUP BY ALL
"""


def _arms(rows: list[tuple]) -> list[tuple[float, float]]:
    """Error bars as the full range of a cell's repetitions.

    With a handful of repetitions the range is the honest width; a standard
    error over five samples would draw a precision the measurement does not
    have.
    """
    return [(max((r[5] or 0) - (r[9] or 0), 0.0), max((r[10] or 0) - (r[5] or 0), 0.0))
            for r in rows]


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
                here = [r for r in rows if r[0] == project and r[1] == backend]
                by_name = {r[3]: r for r in here}
                s.charts.append(ChartSpec(
                    id=f"scaling-{backend}-{project}",
                    kind="grouped_bar",
                    title=f"{project} on {backend}",
                    subtitle=f"Median measured runtime at scale factor {sfs[0]:g}",
                    x_label="", y_label="Runtime (s)",
                    value_labels=True, value_fmt="{:.2f}",
                    series=[Series(name=n, x=[f"sf{sfs[0]:g}"],
                                   y=[by_name[n][5]] if n in by_name else [None],
                                   error=_arms([by_name[n]]) if n in by_name else None)
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
                                     error=_arms(pts),
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

    s.table_columns = ["Project", "Backend", "SF", "Variant", "Runtime (s)",
                       "Fastest", "Slowest", "Reps"]
    s.table_rows = [[r[0], r[1], f"{r[2]:g}", r[3], f"{r[5]:.3f}",
                     f"{r[9]:.3f}" if r[9] is not None else "-",
                     f"{r[10]:.3f}" if r[10] is not None else "-", r[8]] for r in rows]
    s.kpis = [
        Kpi("Scale factors", str(len(sfs)), ", ".join(f"{v:g}" for v in sfs)),
        Kpi("Fastest cell", f"{min(r[5] for r in rows):.2f}s"),
        Kpi("Slowest cell", f"{max(r[5] for r in rows):.2f}s"),
    ]
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
        value_labels=True,
        note="Above the baseline line is faster than unoptimized; below it is a "
             "regression. A missing bar is a cell that has not been measured yet, not "
             "a speedup of zero.",
    ))
    best = [v for values in by_variant.values() for v in values if v]
    if best:
        s.kpis = [
            Kpi("Best speedup", f"{max(best):.2f}x",
                "The single best cell across every optimized variant.",
                tone="good" if max(best) > 1 else "warning"),
            Kpi("Median speedup", f"{median(best):.2f}x",
                tone="good" if (median(best) or 0) > 1.02 else "warning"),
            Kpi("Cells that regressed",
                f"{sum(1 for v in best if v < 1)}/{len(best)}",
                "Optimized cells that ran slower than their baseline.",
                tone="critical" if any(v < 1 for v in best) else "good"),
        ]

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
    if never and s.charts:
        s.charts[0].note += f" {never} of {len(rows)} cells never break even."

    if repaid:
        s.charts.append(ChartSpec(
            id="payback-tradeoff", kind="points",
            title="What the optimization cost, against what it saves per run",
            subtitle="Each mark is a cell; the diagonals are the runs it takes to "
                     "break even",
            x_label="Wall seconds saved per run", y_label="Wall seconds spent optimizing",
            x_type="linear",
            series=([Series(name=v,
                            x=[r[6] for r in repaid if r[3] == v],
                            y=[r[4] for r in repaid if r[3] == v],
                            meta={"payback": [f"{r[7]:.1f} runs"
                                              for r in repaid if r[3] == v]})
                     for v in ordered(list({r[3] for r in repaid}))]
                    + _payback_guides(repaid)),
            note="A mark below a guide line repays within that many runs. Down and to "
                 "the right is a cheap optimization that saves a lot.",
        ))

    s.kpis = [
        Kpi("Cells priced", str(len(rows))),
        Kpi("Fastest payback",
            f"{min(r[7] for r in repaid):.1f} runs" if repaid else "-",
            "Runs of the DAG before the optimization has paid for itself.",
            tone="good" if repaid else ""),
        Kpi("Median payback",
            f"{median([r[7] for r in repaid]):.1f} runs" if repaid else "-"),
        Kpi("Never repaid", f"{never}/{len(rows)}",
            "Cells whose variant was no faster than its baseline.",
            tone="critical" if never else "good"),
    ]

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


def _payback_guides(repaid: list[tuple]) -> list[Series]:
    """Iso-payback diagonals: cost = runs x savings, for a few round run counts.

    The chart's real content is a ratio, and a ratio read off two axes is hard.
    The guides turn it back into the number the study is about — how many runs
    — without collapsing the two quantities that produced it.
    """
    savings = [r[6] for r in repaid if r[6] and r[6] > 0]
    costs = [r[4] for r in repaid if r[4]]
    if not savings or not costs:
        return []
    hi = max(savings) * 1.05
    out = []
    for runs in (1, 10, 100):
        if runs * hi < min(costs) / 5:
            continue  # a guide entirely below the data says nothing
        out.append(Series(name=f"{runs} run{'s' if runs > 1 else ''}",
                          x=[0.0, hi], y=[0.0, runs * hi],
                          kind="line", dash="dot", color=STATUS["neutral"]))
    return out


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
        note=("The unit differs by pass — a materialization is a much larger structural "
              "change than a rewrite — so compare a pass against itself across DAGs, not "
              "against another pass. Each pass's own page breaks its changes down into "
              "what they actually were."),
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

    # One representative run per cell, rather than every run of every cell
    # overlaid. Twenty repetitions of the same shape is not twenty findings,
    # and drawing them all buries the one difference between two variants.
    representatives = _rows(con, """
        WITH measured AS (
            SELECT cell_id, run_id, engine_wall_ms,
                   row_number() OVER (PARTITION BY cell_id
                                      ORDER BY engine_wall_ms) AS rank,
                   count(*)    OVER (PARTITION BY cell_id)      AS total
            FROM runs WHERE phase = 'measure' AND status = 'ok'
                      AND engine_wall_ms IS NOT NULL
        )
        SELECT c.project, c.backend, c.sf, c.variant, m.run_id
        FROM measured m JOIN cells c USING (cell_id)
        WHERE m.rank = (m.total + 1) / 2
    """)
    if not representatives:
        s.empty_reason = "No measured run has system samples attached to it yet."
        return s
    by_run = {r[4]: r for r in representatives}

    samples = _rows(con, """
        SELECT s.run_id, s.elapsed_ms, s.cpu_seconds_cum, s.rss_bytes,
               s.read_bytes, s.written_bytes
        FROM system_samples s
        WHERE s.source <> 'engine_internal' AND s.phase = 'measure'
        ORDER BY s.run_id, s.elapsed_ms
    """)
    if not samples:
        s.empty_reason = "No external system samples were captured."
        return s

    traces: dict[str, list[tuple]] = {}
    for row in samples:
        if row[0] in by_run:
            traces.setdefault(row[0], []).append(row)
    if not traces:
        s.empty_reason = (
            "System samples exist, but none belong to a measured run — they were "
            "captured during optimization only."
        )
        return s

    def label(run_id: str) -> str:
        project, backend, sf, variant, _ = by_run[run_id]
        return f"{variant} · {project} {backend} sf{sf:g}"

    names = ordered([label(run_id) for run_id in traces])
    order = {name: i for i, name in enumerate(names)}

    def build_series(value, scale=1.0, rate=False):
        out = []
        for run_id, points in traces.items():
            xs, ys, previous = [], [], None
            for point in points:
                current = point[value]
                if current is None:
                    continue
                elapsed = point[1] / 1000.0
                if rate:
                    if previous is not None and elapsed > previous[0]:
                        xs.append(elapsed)
                        ys.append((current - previous[1]) / (elapsed - previous[0]) * scale)
                    previous = (elapsed, current)
                else:
                    xs.append(elapsed)
                    ys.append(current * scale)
            if xs:
                out.append(Series(name=label(run_id), x=xs, y=ys))
        return sorted(out, key=lambda ser: order.get(ser.name, 99))

    panels = [
        ("cpu", build_series(2, rate=True), "CPU in use (cores)",
         "CPU seconds consumed per second of wall clock — how many cores were busy",
         "Differentiated from the cumulative counter, so this is real consumption "
         "rather than a sampled percentage. Above 1.0 means more than one core was "
         "working."),
        ("mem", build_series(3, scale=1 / 1048576.0), "Resident memory (MB)",
         "Resident set size of the sampled process tree",
         "The dee server is long-lived across a sweep, so a previous cell's buffer "
         "pool is still resident here. Use `peak_engine_mem_bytes` for memory studies."),
        ("io", build_series(5, scale=1 / 1048576.0), "Bytes written (MB)",
         "Cumulative bytes written since the run started",
         "A step is a materialization landing. A variant that writes more is paying "
         "for the tables its optimizer chose to create."),
    ]
    for key, series, y_label, subtitle, note in panels:
        if not series:
            continue
        s.charts.append(ChartSpec(
            id=f"system-{key}", kind="scatter",
            title=y_label.split(" (")[0] + " over a run",
            subtitle=subtitle + ", for the median run of each cell",
            x_label="Elapsed (s)", y_label=y_label,
            x_type="linear", series=series, note=note, height=340,
        ))

    s.table_columns = ["Variant", "Project", "Backend", "SF", "Samples",
                       "Peak CPU (cores)", "Peak RSS (MB)", "Written (MB)"]
    for run_id, points in traces.items():
        project, backend, sf, variant, _ = by_run[run_id]
        rates = [ser for ser in build_series(2, rate=True) if ser.name == label(run_id)]
        peak_cpu = max(rates[0].y) if rates and rates[0].y else None
        rss = [p[3] for p in points if p[3] is not None]
        written = [p[5] for p in points if p[5] is not None]
        s.table_rows.append([
            variant, project, backend, f"{sf:g}", len(points),
            f"{peak_cpu:.2f}" if peak_cpu else "-",
            f"{max(rss) / 1048576.0:.0f}" if rss else "-",
            f"{max(written) / 1048576.0:.1f}" if written else "-",
        ])
    s.table_rows.sort(key=lambda row: (order.get(f"{row[0]} · {row[1]} {row[2]} sf{row[3]}", 99),))
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
            value_labels=True,
            note="Below 1.0 uses less of this resource than the unoptimized DAG; "
                 "above 1.0 uses more. A pass that buys wall time with memory shows "
                 "up as a runtime panel below the line and a memory panel above it.",
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


def label_all(con) -> None:
    """Apply both labellings, once.

    Called before any page is built — the optimization pages read `cells` too,
    and a variant that names its swept setting on one page and not on another
    would be two different things with one name.

    Not idempotent: each labelling rewrites the view it labels, so calling it
    twice would wrap an already-wrapped view. :func:`build_all` therefore takes
    a flag rather than calling it again.
    """
    for label in (label_backends, label_variants):
        try:
            label(con)
        except Exception:  # noqa: BLE001 - labelling must never lose the studies
            pass


def build_all(con, labelled: bool = False) -> list[Study]:
    if not labelled:
        label_all(con)
    out = []
    for fn in BUILDERS:
        try:
            out.append(fn(con))
        except Exception as e:  # noqa: BLE001 - one broken study must not lose the rest
            key = fn.__name__.replace("study_", "")
            out.append(Study(key=key, number=0, title=key, question="",
                             empty_reason=f"Could not build this study: {type(e).__name__}: {e}"))
    return out
