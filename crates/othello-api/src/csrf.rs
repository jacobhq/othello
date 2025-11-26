use axum::response::IntoResponse;
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use rand::{distributions::Alphanumeric, Rng};

pub async fn init_csrf(jar: CookieJar) -> impl IntoResponse {
    println!("{:?}", jar.get("csrf"));
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
                .build()
        };

        println!("{cookie:?}");

        return jar.add(cookie).into_response();
    }

    "no".into_response()
}

pub fn generate_csrf() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}
