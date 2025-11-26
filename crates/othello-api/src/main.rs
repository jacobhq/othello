mod routes;
mod services;
mod auth;
mod db;
mod csrf;

use axum::{routing::get, Router};
use dotenvy::dotenv;
use std::net::SocketAddr;
use axum::http::HeaderValue;
use axum::middleware::from_fn;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, CorsLayer};
use crate::auth::authorize;
use crate::csrf::{csrf_protect, init_csrf};

#[tokio::main]
async fn main() {
    dotenv().ok();
    let pool = db::init_db().await;

    // CORS Setup
    let allowed_origins = [
        HeaderValue::from_static("http://localhost:5173"),
        HeaderValue::from_static("https://othello.jhqcat.com"),
    ];

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(allowed_origins))
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST, axum::http::Method::OPTIONS])
        .allow_headers([axum::http::header::CONTENT_TYPE, axum::http::header::AUTHORIZATION])
        .allow_credentials(true);

    // Protected Routes
    let protected = Router::new()
        .route("/protected", get(|| async { "OK, Protected" }))
        .layer(from_fn(csrf_protect))    // CSRF check (only POST, PUT, PATCH, DELETE)
        .layer(from_fn(authorize));       // JWT check

    // Public Routes
    let app = Router::new()
        // Database
        .with_state(pool.clone())

        // Basic routes
        .route("/health", get(|| async { "OK" }))
        .route("/csrf/init", get(init_csrf))

        // Auth routes
        .merge(routes::auth(pool.clone()).await)

        // Prefix protected routes with /api
        .nest("/api", protected)

        // CORS applies to all routes
        .layer(cors);

    // Run Server
    #[cfg(debug_assertions)]
    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

    #[cfg(not(debug_assertions))]
    let addr: SocketAddr = "0.0.0.0:80".parse().unwrap();

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    println!("🚀 Server running at http://{}", addr);
}
