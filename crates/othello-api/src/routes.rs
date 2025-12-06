use crate::csrf::csrf_protect;
use crate::auth;
use axum::middleware::from_fn;
use axum::{
    middleware, routing::{get, post},
    Router,
};

pub async fn auth(pool: sqlx::PgPool) -> Router {
    Router::new()
        .route("/auth/sign-in", post(auth::sign_in))
        .route("/auth/sign-up", post(auth::sign_up))
        .route(
            "/auth/logout",
            get(auth::logout)
                .layer(middleware::from_fn_with_state(
                    pool.clone(),
                    auth::authorise,
                ))
                .layer(from_fn(csrf_protect)),
        )
        .with_state(pool)
}
