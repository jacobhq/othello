use crate::auth::Account;
use crate::env_or_dotenv;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Redirect;
use axum::{response::IntoResponse, Extension, Json};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;
use othello::bitboard::BitBoard;
use othello::othello_game::Color;

const FRONTEND_URL: &str = env_or_dotenv!("FRONTEND_URL");

#[derive(Serialize, Deserialize)]
struct UserResponse {
    email: String,
    username: String,
}

pub async fn hello(Extension(current_user): Extension<Account>) -> impl IntoResponse {
    Json(UserResponse {
        email: current_user.email,
        username: current_user.username,
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
) -> Result<impl IntoResponse, StatusCode> {
    match game_info.game_type {
        GameType::PassAndPlay => {
            let id = Uuid::new_v4().to_string();

            let mut black_board = BitBoard(0);
            let mut white_board = BitBoard(0);

            black_board.set(3, 4);
            black_board.set(4, 3);
            white_board.set(3, 3);
            white_board.set(4, 4);

            let insert_result = sqlx::query(
                "INSERT INTO Game (id, player_one_id, type, bitboard_black, bitboard_white) VALUES ($1, $2, 'user_anon', $3, $4)",
            )
                .bind(&id)
                .bind(&account.id)
                .bind(black_board.slices())
                .bind(white_board.slices())
                .execute(&pool)
                .await;

            if insert_result.is_err() {
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }

            Ok(StatusCode::CREATED)
        }
        _ => Err(StatusCode::NOT_IMPLEMENTED),
    }
}

#[derive(Serialize, FromRow)]
struct InPlayField {
    current_turn: String,
    bitboard_white: Vec<u8>,
    bitboard_black: Vec<u8>,
}

#[derive(Serialize, FromRow)]
pub struct InPlayResponse {
    current_turn: String,
    bitboard_white: u64,
    bitboard_black: u64,
}

pub async fn get_in_play_game(
    State(pool): State<PgPool>,
    Extension(account): Extension<Account>,
    Path(game_id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    match sqlx::query_as::<sqlx::Postgres, InPlayField>("SELECT current_turn, bitboard_white, bitboard_black FROM Game WHERE id = $1 AND (player_one_id = $2 OR player_two_id = $2)")
        .bind(game_id.to_string())
        .bind(account.id)
        .fetch_one(&pool)
        .await {
        Ok(row) => {
            let w: [u8; 8] = row.bitboard_white.as_slice().try_into().unwrap();
            let b: [u8; 8] = row.bitboard_black.as_slice().try_into().unwrap();

            let response = InPlayResponse {
                current_turn: row.current_turn,
                bitboard_white: u64::from_be_bytes(w),
                bitboard_black: u64::from_be_bytes(b),
            };

            Ok(Json(response))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR)
    }
}

pub async fn get_game(
    State(pool): State<PgPool>,
    Extension(account): Extension<Account>,
    Path(game_id): Path<Uuid>,
) -> impl IntoResponse {
    let game_result = sqlx::query(
        "SELECT * FROM Game WHERE id = $1 AND (player_one_id = $2 OR player_two_id = $2)",
    )
        .bind(game_id.to_string())
        .bind(account.id)
        .execute(&pool)
        .await;
}
