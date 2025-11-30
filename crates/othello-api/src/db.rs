use dotenvy_macro::dotenv;
use sqlx::{PgPool, postgres::PgPoolOptions};

const DATABASE_URL: &str = dotenv!("DATABASE_URL");

pub async fn init_db() -> PgPool {
    PgPoolOptions::new()
        .max_connections(5)
        .connect(DATABASE_URL)
        .await
        .expect("Failed to connect to database")
}
