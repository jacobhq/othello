use rand::{distributions::Alphanumeric, Rng};
use axum::{extract::Query, response::IntoResponse, Router};
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};

pub async fn init_csrf(jar: CookieJar) -> impl IntoResponse {
    if jar.get("csrf").is_none() {
        let token = generate_csrf();
        let cookie = Cookie::build(("csrf", token.clone()))
            .path("/")
            .domain("a.com") // important!
            .same_site(SameSite::None)
            .secure(true)
            .http_only(false)
            .build();

        return jar.add(cookie).into_response();
    }

    ().into_response()
}

pub fn generate_csrf() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}
