use crate::env_or_dotenv;
use sqlx::{postgres::PgPoolOptions, PgPool};

const DATABASE_URL: &str = env_or_dotenv!("DATABASE_URL");

pub async fn init_db() -> PgPool {
    PgPoolOptions::new()
        .max_connections(5)
        .connect(DATABASE_URL)
        .await
        .expect("Failed to connect to database")
}
