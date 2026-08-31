pub mod connections;
pub mod dags;
pub mod meta;
pub mod optimize;
pub mod runs;
pub mod schedules;

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(meta::healthz))
        .route("/v1/info", get(meta::info))
        .route("/v1/optimizer/options", get(meta::optimizer_options))
        .route(
            "/v1/connections",
            get(connections::list).post(connections::create),
        )
        .route(
            "/v1/connections/{name}",
            get(connections::get).delete(connections::delete),
        )
        .route("/v1/connections/{name}/test", post(connections::test))
        .route("/v1/dags", get(dags::list).post(dags::submit))
        .route("/v1/dags/{name}", get(dags::get).delete(dags::delete))
        .route("/v1/dags/{name}/versions", get(dags::versions))
        .route("/v1/dags/{name}/versions/{version}", get(dags::version))
        .route("/v1/dags/{name}/graph", get(dags::graph))
        .route("/v1/dags/{name}/runs", post(runs::trigger))
        .route("/v1/runs", get(runs::list))
        .route("/v1/runs/{run_id}", get(runs::get))
        .route("/v1/runs/{run_id}/nodes", get(runs::nodes))
        .route("/v1/runs/{run_id}/plans", get(runs::plans))
        .route("/v1/runs/{run_id}/samples", get(runs::samples))
        .route("/v1/runs/{run_id}/logs", get(runs::events))
        .route("/v1/runs/{run_id}/report", get(runs::run_report))
        .route("/v1/runs/{run_id}/cancel", post(runs::cancel))
        .route("/v1/run-groups/{group_id}", get(runs::group))
        .route("/v1/run-groups/{group_id}/report", get(runs::group_report))
        .route("/v1/run-groups/{group_id}/report.html", get(runs::group_report_html))
        .route("/v1/run-groups/{group_id}/cancel", post(runs::cancel_group_route))
        .route("/v1/dags/{name}/optimize", post(optimize::start))
        .route("/v1/optimizations", get(optimize::list))
        .route("/v1/optimizations/{id}", get(optimize::get))
        .route("/v1/optimizations/{id}/report", get(optimize::report))
        .route("/v1/optimizations/{id}/explain.html", get(optimize::explain))
        .route("/v1/optimizations/{id}/dag", get(optimize::result_dag))
        .route(
            "/v1/dags/{name}/schedule",
            get(schedules::get).put(schedules::set).delete(schedules::delete),
        )
        .route("/v1/dags/{name}/schedule/pause", post(schedules::pause))
        .route("/v1/dags/{name}/schedule/resume", post(schedules::resume))
        .route("/v1/dags/{name}/schedule/skips", get(schedules::skips))
        .route("/v1/schedules", get(schedules::list))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
