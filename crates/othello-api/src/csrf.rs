use axum::body::Body;
use axum::extract::{FromRequestParts, Request};
use axum::http::{HeaderMap, Method, Response, StatusCode};
use axum::middleware::Next;
use axum::response::IntoResponse;
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use rand::{distributions::Alphanumeric, Rng};
use serde_json::json;
use time::Duration;

pub async fn init_csrf(jar: CookieJar) -> impl IntoResponse {
    if jar.get("csrf").is_none() {
        let token = generate_csrf();
        let cookie: Cookie = if cfg!(debug_assertions) {
            Cookie::build(("csrf", token.clone()))
                .path("/")
                .same_site(SameSite::None)
                .secure(true)
                .http_only(false)
                .build()
        } else {
            Cookie::build(("csrf", token.clone()))
                .path("/")
                .domain("jhqcat.com")
                .same_site(SameSite::None)
                .secure(true)
                .http_only(false)
                .max_age(Duration::hours(1))
                .build()
        };

        return jar.add(cookie).into_response();
    }

    "no".into_response()
}

fn extract_csrf_header(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-csrf-token")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
}

pub async fn csrf_protect(
    req: Request,
    next: Next,
) -> Result<Response<Body>, impl IntoResponse> {
    // Safe methods don't require CSRF, that is handled by our CORS policy
    if matches!(*req.method(), Method::GET | Method::HEAD | Method::OPTIONS) {
        return Ok(next.run(req).await);
    }

    let (mut parts, body) = req.into_parts();

    let jar = CookieJar::from_request_parts(&mut parts, &()).await.unwrap();

    let req = Request::from_parts(parts, body);

    let cookie_token = jar.get("csrf").map(|c| c.value().to_string());
    let header_token = extract_csrf_header(req.headers());

    // Validate
    if cookie_token.is_none()
        || header_token.is_none()
        || cookie_token != header_token
    {
        let body = axum::Json(json!({ "error": "CSRF validation failed" }));
        return Err((StatusCode::FORBIDDEN, body));
    }

    Ok(next.run(req).await)
}

pub fn generate_csrf() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}
