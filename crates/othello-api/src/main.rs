mod analytics;
mod auth;
mod csrf;
mod db;
mod env_macro;
mod routes;
mod services;

use crate::auth::{authorise, get_me};
use crate::csrf::{csrf_protect, init_csrf};
use crate::services::{get_all_games, get_in_play_game, new_game, set_in_play_game};
use axum::http::HeaderValue;
use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::post;
use axum::{Router, routing::get};
use dotenvy::dotenv;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{EnvFilter, Registry};
use tracing_subscriber::prelude::*;

const FRONTEND_URL: &str = env_or_dotenv!("FRONTEND_URL");
const POSTHOG_API_KEY: &str = env_or_dotenv!("POSTHOG_API_KEY");
const AXIOM_DATASET: &str = env_or_dotenv!("AXIOM_DATASET");
const AXIOM_API_TOKEN: &str = env_or_dotenv!("AXIOM_API_TOKEN");

#[tokio::main]
async fn main() {
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(false);

    let axiom_layer = tracing_axiom::builder("othello-api")
        .with_dataset(AXIOM_DATASET)
        .expect("Error setting AXIOM_DATASET")
        .with_token(AXIOM_API_TOKEN)
        .expect("Error setting AXIOM_API_TOKEN")
        .build()
        .expect("Error building Axiom layer");

    let filter = EnvFilter::new("othello_api=info,tower_http=info");

    Registry::default()
        .with(filter)
        .with(fmt_layer)
        .with(axiom_layer)
        .try_init()
        .unwrap();

    dotenv().ok();
    let pool = db::init_db().await;

    // CORS Setup
    let cors = CorsLayer::new()
        .allow_origin(FRONTEND_URL.parse::<HeaderValue>().unwrap())
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::HeaderName::from_static("x-csrf-token"),
        ])
        .allow_credentials(true);

    // Protected Routes
    let protected = Router::new()
        .route("/user", get(get_me))
        .route("/games", get(get_all_games))
        .route("/games/new", post(new_game))
        .route(
            "/games/{game_id}",
            get(get_in_play_game).post(set_in_play_game),
        )
        .with_state(pool.clone())
        .layer(from_fn_with_state(pool.clone(), authorise)) // JWT check
        .layer(from_fn(csrf_protect)); // CSRF check (only POST, PUT, PATCH, DELETE)

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
        // Tracing
        .layer(TraceLayer::new_for_http())
        // CORS applies to all routes
        .layer(cors);

    // Run Server
    #[cfg(debug_assertions)]
    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

    #[cfg(not(debug_assertions))]
    let addr: SocketAddr = "0.0.0.0:80".parse().unwrap();

    let listener = TcpListener::bind(addr).await.unwrap();

    tracing::info!("🚀 Server running at http://{}", addr);

    axum::serve(listener, app).await.unwrap();
}
