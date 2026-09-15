"""Parallelism tuning: the ladder, rung by rung, against its own control.

The pass walks a ladder of node-concurrency caps and keeps a rung only if it
beat a control measured beside it. Every rung's samples are recorded, including
the rungs that lost and the ones the search never reached — so the ladder can
be drawn as what it is: a measurement with a verdict, not a single chosen
number.
"""

from __future__ import annotations

from ..query import fmt_num, median
from ..spec import ChartSpec, Kpi, Series, Study, Table
from ..theme import STATUS
from ._common import MAX_DETAILED_CELLS, page, pass_cells, settings_table

SETTINGS = ["parallelism_ladder", "parallelism_seed_repeats", "parallelism_confirm_runs",
            "parallelism_after_materialization", "parallelism_paired",
            "parallelism_cpu_guard", "parallelism_adaptive_order",
            "parallelism_stop_after_failures", "parallelism_min_effect",
            "parallelism_stop_on_narrowest_failure"]

# How a rung's verdict reads as a status. The recorded strings carry a reason
# in parentheses, so they are matched by prefix.
VERDICTS = {
    "accepted": "good",
    "baseline": "neutral",
    "rejected": "critical",
    "not measured": "",
}


def build(con) -> Study | None:
    cells = pass_cells(con, "parallelism")
    if not cells:
        return None

    study = page("parallelism", question=(
        "Capping how many nodes run at once trades contention against idle cores. "
        "Which rung of the ladder won, by how much, and did the rungs that lost lose "
        "to contention or to noise?"
    ))
    study.kpis = _kpis(cells)
    study.charts.append(_outcome(cells))
    for cell in cells[:MAX_DETAILED_CELLS]:
        ladder = _ladder_chart(cell)
        if ladder:
            study.charts.append(ladder)
    verdicts = _verdict_heatmap(cells)
    if verdicts:
        study.charts.append(verdicts)

    study.tables = [t for t in (
        _summary_table(cells),
        settings_table(cells, SETTINGS, key="parallelism-settings"),
        _rung_table(cells),
    ) if t]
    return study


def _rungs(cell: dict) -> list[dict]:
    return [r for r in (cell["detail"].get("rungs") or []) if isinstance(r, dict)]


def _cap_label(value) -> str:
    return "uncapped" if value is None else str(value)


def _kpis(cells: list[dict]) -> list[Kpi]:
    changed = [c for c in cells if c["detail"].get("chosen_parallelism") is not None]
    gains = [-(c["detail"].get("opt_change") or 0.0) for c in cells
             if c["detail"].get("opt_change")]
    measured, total = 0, 0
    for cell in cells:
        for rung in _rungs(cell):
            total += 1
            if rung.get("samples"):
                measured += 1
    return [
        Kpi("Cells running the ladder", str(len(cells))),
        Kpi("Cells that changed the cap", f"{len(changed)}/{len(cells)}",
            "A cell that kept its cap found no rung that beat its control.",
            tone="good" if changed else "warning"),
        Kpi("Best improvement",
            f"{max(gains):.1%}" if gains else "-",
            "Against the DAG's parallelism before the ladder ran.",
            tone="good" if gains and max(gains) > 0 else ""),
        Kpi("Rungs actually measured", f"{measured}/{total}" if total else "-",
            "The rest were skipped: the search stopped early, or a guard refused them "
            "before they ran."),
    ]


def _outcome(cells: list[dict]) -> ChartSpec:
    labels = [c["short"] for c in cells]
    return ChartSpec(
        id="parallelism-outcome", kind="grouped_bar",
        title="What the ladder found",
        subtitle="The DAG's runtime before the ladder ran, and at the rung it kept",
        y_label="Runtime (s)",
        series=[
            Series(name="before the ladder", x=labels,
                   y=[(c["detail"].get("baseline_runtime_ms") or 0) / 1000.0 or None
                      for c in cells]),
            Series(name="at the chosen cap", x=labels,
                   y=[(c["detail"].get("best_runtime_ms") or 0) / 1000.0 or None
                      for c in cells]),
        ],
        value_labels=True,
        note="Equal bars mean the ladder measured every rung it reached and kept none — "
             "a result, not a failure to run.",
    )


def _ladder_chart(cell: dict) -> ChartSpec | None:
    """Every rung's samples, with the control each was judged against."""
    rungs = _rungs(cell)
    measured = [r for r in rungs if r.get("samples") or r.get("control_samples")]
    if not measured:
        return None

    labels = [_cap_label(r.get("parallelism")) for r in measured]
    trial, control, spread = [], [], []
    for rung in measured:
        samples = [s / 1000.0 for s in (rung.get("samples") or [])]
        controls = [s / 1000.0 for s in (rung.get("control_samples") or [])]
        best = median(samples)
        trial.append(best)
        control.append(median(controls))
        # Every sample, not a model of them: with two or three runs per rung the
        # range is the honest width, and a standard error would invent precision.
        spread.append((best - min(samples), max(samples) - best) if best and samples
                      else (0.0, 0.0))

    chosen = _cap_label(cell["detail"].get("chosen_parallelism"))
    start = _cap_label(cell["detail"].get("baseline_parallelism"))
    baseline = (cell["detail"].get("baseline_runtime_ms") or 0) / 1000.0 or None

    def role(label: str) -> tuple[str, str]:
        if label == chosen:
            # A ladder that kept the cap it started with found nothing better,
            # which is a different finding from having moved to a new rung.
            return STATUS["good"], "kept, unchanged" if chosen == start else "kept"
        if label == start:
            return STATUS["neutral"], "where the DAG started"
        return "#2a78d6", "measured and rejected"

    colors = [role(label)[0] for label in labels]
    series = [Series(name="rung samples", x=labels, y=trial, error=spread,
                     colors=colors,
                     meta={"legend": [role(label)[1] for label in labels],
                           "verdict": [r.get("verdict") or "-" for r in measured]})]
    if any(c is not None for c in control):
        series.append(Series(name="control measured beside it", x=labels, y=control,
                             color=STATUS["neutral"]))
    return ChartSpec(
        id=f"parallelism-ladder-{cell['slug']}", kind="grouped_bar",
        title=f"The ladder — {cell['label']}",
        subtitle="Median runtime at each node-concurrency cap, with the range of its "
                 "samples",
        x_label="Node concurrency cap", y_label="Runtime (s)",
        series=series,
        hlines=[(baseline, "before the ladder")] if baseline else [],
        note="A paired ladder judges each rung against a control measured beside it, so "
             "warehouse drift over a long sweep cannot be read as a parallelism effect. "
             "Bars are medians; the whisker is the full range of that rung's samples.",
        height=360,
    )


def _verdict_heatmap(cells: list[dict]) -> ChartSpec | None:
    """Every rung of every cell, and what the search decided about it.

    The verdicts are categories, so this encodes them as an ordered scale from
    "never reached" to "kept" — which is what makes a ladder that stopped early
    visible as a row that goes blank partway across.
    """
    caps: list[str] = []
    for cell in cells:
        for rung in _rungs(cell):
            label = _cap_label(rung.get("parallelism"))
            if label not in caps:
                caps.append(label)
    if len(cells) < 2 or len(caps) < 2:
        return None

    scale = {"not measured": 0.0, "rejected": 1.0, "baseline": 2.0, "accepted": 3.0}
    shown = {"not measured": "skipped", "rejected": "rejected",
             "baseline": "start", "accepted": "kept"}
    matrix, text = [], []
    for cell in cells:
        by_cap = {_cap_label(r.get("parallelism")): r for r in _rungs(cell)}
        row, labels = [], []
        for cap in caps:
            rung = by_cap.get(cap)
            verdict = str((rung or {}).get("verdict") or "").split(" (")[0]
            row.append(scale.get(verdict) if rung is not None else None)
            labels.append(shown.get(verdict, "") if rung is not None else "")
        matrix.append(row)
        text.append(labels)
    return ChartSpec(
        id="parallelism-verdicts", kind="heatmap",
        title="What the search decided about each rung",
        subtitle="Each rung the ladder defined, and what the search decided about it",
        x_label="Node concurrency cap", y_label="",
        matrix=matrix, x_ticks=caps,
        y_ticks=[c["short"].replace("\n", " · ") for c in cells],
        z_text=text, z_fmt="{:.0f}",
        note="A row that goes blank partway across is a ladder that stopped early — "
             "`parallelism_stop_after_failures` and "
             "`parallelism_stop_on_narrowest_failure` are what stop it.",
    )


def _summary_table(cells: list[dict]) -> Table:
    table = Table(
        key="parallelism-summary", title="What the ladder did, per cell",
        columns=["Project", "Backend", "SF", "Variant", "Ladder", "Paired",
                 "Before", "Chosen cap", "At chosen", "Change", "DAG runs", "Pass time"],
        numeric=(6, 8, 9, 10, 11),
    )
    for c in cells:
        d = c["detail"]
        change = d.get("opt_change")
        table.rows.append([
            c["project"], c["backend"], f"{float(c['sf']):g}", c["variant"],
            ", ".join(str(v) for v in (d.get("ladder") or [])) or "-",
            "yes" if d.get("paired") else "no",
            _cap_label(d.get("baseline_parallelism")),
            _cap_label(d.get("chosen_parallelism")),
            fmt_num((d.get("best_runtime_ms") or 0) / 1000.0 or None, 3, "s"),
            f"{change:+.1%}" if change is not None else "-",
            c["dag_runs_used"],
            fmt_num((c["wall_ms"] or 0) / 1000.0, 2, "s"),
        ])
    return table


def _rung_table(cells: list[dict]) -> Table | None:
    table = Table(
        key="parallelism-rungs", title="Every rung, and every sample behind it",
        columns=["Cell", "Cap", "Verdict", "Samples", "Median", "Control samples",
                 "Control median"],
        numeric=(4, 6),
    )
    for cell in cells:
        for rung in _rungs(cell):
            samples = [s / 1000.0 for s in (rung.get("samples") or [])]
            controls = [s / 1000.0 for s in (rung.get("control_samples") or [])]
            table.rows.append([
                cell["label"], _cap_label(rung.get("parallelism")),
                rung.get("verdict") or "-",
                ", ".join(f"{s:.3f}" for s in samples) or "-",
                fmt_num(median(samples), 3, "s"),
                ", ".join(f"{s:.3f}" for s in controls) or "-",
                fmt_num(median(controls), 3, "s"),
            ])
    return table if table.rows else None
