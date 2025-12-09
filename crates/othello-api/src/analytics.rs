use axum::Extension;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::IntoResponse;
use posthog_rs::Client;
use crate::auth::Account;

pub async fn track_authenticated(
    State(posthog_client): State<Client>,
    Extension(account): Extension<Account>,
    req: Request,
    next: Next,
) -> impl IntoResponse {
    let event = posthog_rs::Event::new("api_hit", req.uri().path());

    posthog_client.capture(event).await.unwrap();
}

pub async fn track_anon(
    State(posthog_client): State<Client>,
    Extension(account): Extension<Account>,
    req: Request,
    next: Next,
) -> impl IntoResponse {
    let event = posthog_rs::Event::new("api_hit", req.uri().path());

    posthog_client.capture(event).await.unwrap();
}

// pub fn user_signed_up(id: &str, email: &str, username: String) {
//     let mut event = posthog_rs::Event::new("user_signed_up", id);
//     event.insert_prop("email", email).unwrap();
//     event.insert_prop("email", username).unwrap();
//
//     client.capture(event).unwrap();
// }
