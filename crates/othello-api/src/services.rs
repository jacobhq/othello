use crate::auth::Account;
use crate::env_or_dotenv;
use axum::extract::State;
use axum::response::Redirect;
use axum::{Extension, Json, response::IntoResponse};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

const FRONTEND_URL: &str = env_or_dotenv!("FRONTEND_URL");

#[derive(Serialize, Deserialize)]
struct UserResponse {
    email: String,
    username: String,
}

pub async fn hello(Extension(currentUser): Extension<Account>) -> impl IntoResponse {
    Json(UserResponse {
        email: currentUser.email,
        username: currentUser.username,
    })
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameType {
    PlayOnline,
    PlayBots,
    PlayFriend,
    PassAndPlay,
}

#[derive(Serialize, Deserialize)]
pub struct GameInfo {
    pub game_type: GameType,
}

pub async fn new_game(
    State(pool): State<PgPool>,
    Extension(account): Extension<Account>,
    Json(game_info): Json<GameInfo>,
) -> impl IntoResponse {
    match game_info.game_type {
        GameType::PassAndPlay => {
            let id = uuid::Uuid::new_v4().to_string();
            let insert_result = sqlx::query("INSERT INTO Game (id, player_one_id, type) VALUES ($1, $2, 'user_anon')")
                .bind(&id)
                .bind(&account.id)
                .execute(&pool)
                .await;

            if insert_result.is_err() {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }

            Redirect::to(&format!("{}/play/{}", FRONTEND_URL, &id)).into_response()
        },
        _ => Redirect::to(&format!("{}/play?error=not_implemented", FRONTEND_URL)).into_response(),
    }
}
