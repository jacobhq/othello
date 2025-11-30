use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use crate::{auth, services};

pub async fn auth(pool: sqlx::PgPool) -> Router {
    Router::new()
        .route("/auth/sign-in", post(auth::sign_in))
        .route("/auth/sign-up", post(auth::sign_up))
        .route(
            "/protected",
            get(services::hello).layer(middleware::from_fn_with_state(pool.clone(), auth::authorise)),
        )
        .with_state(pool)
}
