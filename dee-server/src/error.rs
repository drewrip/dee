use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use thiserror::Error;

use crate::store::StoreError;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error(transparent)]
    Store(#[from] StoreError),

    #[error("{0} '{1}' not found")]
    NotFound(&'static str, String),

    #[error("{0}")]
    BadRequest(String),

    /// The requested work collides with something already in flight. This is
    /// the same rule the scheduler's skip policy enforces, surfaced to a
    /// client that asked directly.
    #[error("{0}")]
    Conflict(String),

    #[error("{0}")]
    Internal(String),
}

impl ServerError {
    fn parts(&self) -> (StatusCode, &'static str) {
        match self {
            ServerError::NotFound(..) => (StatusCode::NOT_FOUND, "not_found"),
            ServerError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            ServerError::Conflict(_) => (StatusCode::CONFLICT, "conflict"),
            ServerError::Store(_) | ServerError::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal")
            }
        }
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    code: &'static str,
    message: String,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, code) = self.parts();
        let message = self.to_string();
        // A 5xx is a bug or an outage; it must reach the log even though the
        // client also sees it. 4xx is the caller's problem and stays quiet.
        if status.is_server_error() {
            log::error!("{code}: {message}");
        }
        (
            status,
            Json(ErrorBody {
                error: ErrorDetail { code, message },
            }),
        )
            .into_response()
    }
}
