//! Shared HTML rendering helpers used by every `Explain` implementation, plus
//! the top-level combiner that turns each pass's explain snippet into a tab
//! on one report page (styled like the `run --profile-viz` report).

use crate::report::render_report_shell;

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#039;")
}

/// A row of stat tiles, reusing the `.summary`/`.card` classes from the
/// shared report CSS.
pub(crate) fn render_card_grid(cards: &[(&str, String)]) -> String {
    let cards_html: String = cards
        .iter()
        .map(|(label, value)| {
            format!(
                r#"<div class="card"><div class="label">{}</div><div class="value">{}</div></div>"#,
                escape_html(label),
                escape_html(value)
            )
        })
        .collect();
    format!(r#"<div class="summary">{cards_html}</div>"#)
}

/// A plain data table, styled like the profiling report's `.compare-table`.
pub(crate) fn render_ranked_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let header_html: String = headers
        .iter()
        .map(|h| format!("<th>{}</th>", escape_html(h)))
        .collect();
    let rows_html: String = rows
        .iter()
        .map(|row| {
            let cells: String = row
                .iter()
                .map(|cell| format!("<td>{}</td>", escape_html(cell)))
                .collect();
            format!(r#"<tr class="compare-row">{cells}</tr>"#)
        })
        .collect();
    format!(
        r#"<div class="svg-wrap" style="padding: 0; overflow-x: auto;">
      <table class="compare-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"#
    )
}

/// A labeled horizontal bar row, reusing the `.plan-*` visual pattern from
/// the profiling report's plan-tree rendering.
pub(crate) fn render_bar_row(label: &str, value_label: &str, pct: f64) -> String {
    let pct = pct.clamp(0.0, 100.0);
    format!(
        r#"<div class="plan-header" style="cursor: default;">
      <span class="plan-type">{}</span>
      <div class="plan-impact-wrap"><div class="plan-impact-bar" style="width: {:.1}%"></div></div>
      <span class="plan-timing">{}</span>
    </div>"#,
        escape_html(label),
        pct,
        escape_html(value_label)
    )
}

/// Combine each pass's `(label, html)` explain section into one report,
/// one tab per pass, in the same visual style as `run --profile-viz`.
pub fn render_explain_html(sections: &[(String, String)]) -> String {
    let tabs_html: String = sections
        .iter()
        .enumerate()
        .map(|(i, (label, _))| {
            format!(
                r#"<button class="tab{}" data-index="{i}">{}</button>"#,
                if i == 0 { " active" } else { "" },
                escape_html(label)
            )
        })
        .collect();

    let pages_html: String = sections
        .iter()
        .enumerate()
        .map(|(i, (_, html))| {
            format!(
                r#"<section class="page{}" data-page="{i}">{html}</section>"#,
                if i == 0 { " active" } else { "" }
            )
        })
        .collect();

    let subtitle_html = format!(
        "<p>{} optimizer pass(es) explained below.</p>",
        sections.len()
    );

    render_report_shell(
        "dee optimizer explain",
        "dee optimizer",
        "optimization report",
        &subtitle_html,
        &tabs_html,
        &pages_html,
        "",
        "",
    )
}
