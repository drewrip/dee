//! Thin HTTP client for the dee server.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

pub struct Client {
    base: String,
    http: reqwest::Client,
}

#[derive(Debug)]
pub struct ApiError {
    pub status: reqwest::StatusCode,
    pub code: String,
    pub message: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ApiError {}

impl Client {
    pub fn new(base: &str) -> Self {
        Client {
            base: base.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, Box<dyn std::error::Error>> {
        let response = self.http.get(self.url(path)).send().await.map_err(explain)?;
        decode(response).await
    }

    /// For endpoints that return a rendered document (SVG, DOT, HTML) rather
    /// than JSON.
    pub async fn get_text(&self, path: &str) -> Result<String, Box<dyn std::error::Error>> {
        let response = self.http.get(self.url(path)).send().await.map_err(explain)?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| Some(v.get("error")?.get("message")?.as_str()?.to_string()))
                .unwrap_or_else(|| text.clone());
            return Err(Box::new(ApiError {
                status,
                code: status.as_str().to_string(),
                message,
            }));
        }
        Ok(text)
    }

    pub async fn post<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let response = self
            .http
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(explain)?;
        decode(response).await
    }

    pub async fn put<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let response = self
            .http
            .put(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(explain)?;
        decode(response).await
    }

    pub async fn patch<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let response = self
            .http
            .patch(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(explain)?;
        decode(response).await
    }

    /// A DELETE whose response body matters -- deregistering an optimization
    /// reports what it tore down.
    pub async fn delete_for<T: DeserializeOwned>(
        &self,
        path: &str,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let response = self
            .http
            .delete(self.url(path))
            .send()
            .await
            .map_err(explain)?;
        decode(response).await
    }

    pub async fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let response = self
            .http
            .delete(self.url(path))
            .send()
            .await
            .map_err(explain)?;
        let _: Option<Value> = decode_optional(response).await?;
        Ok(())
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }
}

/// A connection refusal is by far the most common failure, and "connection
/// refused" alone does not tell the user what to do about it.
fn explain(e: reqwest::Error) -> Box<dyn std::error::Error> {
    if e.is_connect() {
        return format!(
            "cannot reach the dee server ({e}).\n\
             Start one with `dee serve`, or point at another with --server / $DEE_SERVER."
        )
        .into();
    }
    Box::new(e)
}

async fn decode<T: DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, Box<dyn std::error::Error>> {
    match decode_optional(response).await? {
        Some(value) => Ok(value),
        None => Err("server returned an empty body where one was expected".into()),
    }
}

async fn decode_optional<T: DeserializeOwned>(
    response: reqwest::Response,
) -> Result<Option<T>, Box<dyn std::error::Error>> {
    let status = response.status();
    let text = response.text().await?;

    if !status.is_success() {
        // The server's error envelope is {"error":{"code","message"}}. Fall
        // back to the raw body for anything that did not come from it.
        let (code, message) = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                let e = v.get("error")?;
                Some((
                    e.get("code")?.as_str()?.to_string(),
                    e.get("message")?.as_str()?.to_string(),
                ))
            })
            .unwrap_or_else(|| (status.as_str().to_string(), text.clone()));
        return Err(Box::new(ApiError {
            status,
            code,
            message,
        }));
    }

    if text.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&text)?))
}
