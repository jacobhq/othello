mod routes;
mod services;
mod auth;

use axum::{routing::get, Router};
use dotenvy::dotenv;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::cors::{CorsLayer};

#[tokio::main]
async fn main() {
    dotenv().ok();

    // CORS Setup
    let cors = CorsLayer::new()
        .allow_origin("http://localhost:5173".parse::<axum::http::HeaderValue>().unwrap())
        .allow_origin("https://othello.jhqcat.com".parse::<axum::http::HeaderValue>().unwrap())
        .allow_methods([axum::http::Method::GET])
        .allow_credentials(true);

    let auth = routes::auth().await;

    // Router
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .merge(auth)
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
