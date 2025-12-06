use crate::auth::Account;
use crate::env_or_dotenv;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::{response::IntoResponse, Extension, Json};
use chrono::NaiveDateTime;
use othello::bitboard::BitBoard;
use othello::othello_game::{Color as OthelloColor, OthelloGame};
use serde::{Deserialize, Serialize};
use sqlx::error::BoxDynError;
use sqlx::{Database, Decode, Encode, Error, FromRow, PgPool, Postgres, Type};
use std::fmt::Display;
use uuid::Uuid;

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

#[derive(FromRow, Debug)]
struct PlayerResult {
    id: String,
}

#[derive(Serialize)]
pub struct NewGameResponse {
    id: String,
}

impl IntoResponse for NewGameResponse {
    fn into_response(self) -> Response {
        (StatusCode::CREATED, Json(self)).into_response()
    }
}

pub async fn new_game(
    State(pool): State<PgPool>,
    Extension(account): Extension<Account>,
    Json(game_info): Json<GameInfo>,
) -> Result<impl IntoResponse, StatusCode> {
    match game_info.game_type {
        GameType::PassAndPlay => {
            let game_id = Uuid::new_v4().to_string();
            let new_player_id = Uuid::new_v4().to_string();

            let new_player_result = sqlx::query(
                "INSERT INTO Player (id, type, user_id) VALUES ($1, 'user', $2) ON CONFLICT DO NOTHING"
            )
                .bind(&new_player_id)
                .bind(&account.id)
                .execute(&pool)
                .await;

            if new_player_result.is_err() {
                println!("{:?}", new_player_result);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }

            let player_result = sqlx::query_as::<sqlx::Postgres, PlayerResult>(
                "SELECT (id) FROM Player WHERE user_id = $1",
            )
                .bind(&account.id)
                .fetch_one(&pool)
                .await;

            if player_result.is_err() {
                println!("{:?}", player_result);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }

            let mut black_board = BitBoard(0);
            let mut white_board = BitBoard(0);

            black_board.set(3, 4);
            black_board.set(4, 3);
            white_board.set(3, 3);
            white_board.set(4, 4);

            let insert_result = sqlx::query(
                "INSERT INTO Game (id, player_one_id, type, bitboard_black, bitboard_white) VALUES ($1, $2, 'pass_and_play', $3, $4)",
            )
                .bind(&game_id)
                .bind(&player_result.unwrap().id)
                .bind(black_board.slices())
                .bind(white_board.slices())
                .execute(&pool)
                .await;

            if insert_result.is_err() {
                println!("{:?}", insert_result);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }

            Ok(NewGameResponse { id: game_id })
        }
        _ => Err(StatusCode::NOT_IMPLEMENTED),
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Color {
    Black,
    White,
}

impl From<Color> for OthelloColor {
    fn from(c: Color) -> Self {
        match c {
            Color::Black => OthelloColor::Black,
            Color::White => OthelloColor::White,
        }
    }
}

impl From<OthelloColor> for Color {
    fn from(c: OthelloColor) -> Self {
        match c {
            OthelloColor::Black => Color::Black,
            OthelloColor::White => Color::White,
        }
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Color::Black => String::from("black"),
            Color::White => String::from("white"),
        };
        write!(f, "{}", str)
    }
}

impl<'r> Decode<'r, sqlx::Postgres> for Color {
    fn decode(value: <Postgres as Database>::ValueRef<'r>) -> Result<Self, BoxDynError> {
        let s = <&str as Decode<Postgres>>::decode(value)?;

        match s {
            "black" => Ok(Color::Black),
            "white" => Ok(Color::White),
            other => Err(format!("invalid Color variant: {}", other).into()),
        }
    }
}

impl<'q> Encode<'q, Postgres> for Color {
    fn encode_by_ref(&self, buf: &mut sqlx::postgres::PgArgumentBuffer) -> Result<sqlx::encode::IsNull, Box<(dyn std::error::Error + Send + Sync + 'static)>> {
        let s = match self {
            Color::Black => "black",
            Color::White => "white",
        };
        <&str as Encode<Postgres>>::encode_by_ref(&s, buf)
    }

    fn size_hint(&self) -> usize {
        let s = match self {
            Color::Black => "black",
            Color::White => "white",
        };
        <&str as Encode<Postgres>>::size_hint(&s)
    }
}

impl sqlx::Type<sqlx::Postgres> for Color {
    fn type_info() -> <sqlx::Postgres as sqlx::Database>::TypeInfo {
        <&str as sqlx::Type<sqlx::Postgres>>::type_info()
    }
}

#[derive(Serialize, FromRow)]
struct MinimalGameFromDb {
    current_turn: Color,
    bitboard_white: Vec<u8>,
    bitboard_black: Vec<u8>,
}

#[derive(Serialize, FromRow)]
pub struct InPlayResponse {
    current_turn: String,
    bitboard_white: u64,
    bitboard_black: u64,
}

#[derive(FromRow)]
struct Player {
    id: String,
}

pub async fn get_in_play_game(
    State(pool): State<PgPool>,
    Extension(account): Extension<Account>,
    Path(game_id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let player_id =
        match sqlx::query_as::<sqlx::Postgres, Player>("SELECT id FROM Player WHERE user_id = $1")
            .bind(&account.id)
            .fetch_one(&pool)
            .await
        {
            Ok(player) => player.id,
            Err(_) => return Err(StatusCode::NOT_FOUND),
        };

    match sqlx::query_as::<sqlx::Postgres, MinimalGameFromDb>("SELECT current_turn, bitboard_white, bitboard_black FROM Game WHERE id = $1 AND (player_one_id = $2 OR player_two_id = $2)")
        .bind(game_id.to_string())
        .bind(&player_id)
        .fetch_one(&pool)
        .await {
        Ok(row) => {
            let w: [u8; 8] = row.bitboard_white.as_slice().try_into().unwrap();
            let b: [u8; 8] = row.bitboard_black.as_slice().try_into().unwrap();

            let response = InPlayResponse {
                current_turn: row.current_turn.to_string(),
                bitboard_white: u64::from_le_bytes(w),
                bitboard_black: u64::from_le_bytes(b),
            };

            Ok(Json(response))
        }
        Err(err) => match err {
            Error::RowNotFound => Err(StatusCode::NOT_FOUND),
            _ => Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[derive(Deserialize)]
pub struct Move {
    row: u8,
    col: u8,
    color: Color,
}

pub async fn set_in_play_game(
    State(pool): State<PgPool>,
    Extension(account): Extension<Account>,
    Path(game_id): Path<String>,
    Json(new_move): Json<Move>,
) -> Result<impl IntoResponse, StatusCode> {
    let player_id =
        match sqlx::query_as::<sqlx::Postgres, Player>("SELECT id FROM Player WHERE user_id = $1")
            .bind(&account.id)
            .fetch_one(&pool)
            .await
        {
            Ok(player) => player.id,
            Err(err) => {
                return match err {
                    Error::RowNotFound => Err(StatusCode::NOT_FOUND),
                    _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
                };
            }
        };

    let game_from_db = match sqlx::query_as::<sqlx::Postgres, MinimalGameFromDb>("SELECT current_turn, bitboard_white, bitboard_black FROM Game WHERE id = $1 AND (player_one_id = $2 OR player_two_id = $2)")
        .bind(game_id.to_string())
        .bind(&player_id)
        .fetch_one(&pool)
        .await {
        Ok(game) => game,
        Err(err) => return match err {
            Error::RowNotFound => Err(StatusCode::NOT_FOUND),
            _ => Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    };

    let mut game = OthelloGame::new_with_state(
        u64::from_le_bytes(game_from_db.bitboard_black.try_into().unwrap()),
        u64::from_le_bytes(game_from_db.bitboard_white.try_into().unwrap()),
        game_from_db.current_turn.into(),
    );

    if game.legal_moves(new_move.color.clone().into()).contains(&(new_move.row as usize, new_move.col as usize)) {
        match game.play(new_move.row as usize, new_move.col as usize, new_move.color.into()) {
            Ok(_) => match sqlx::query("UPDATE Game SET current_turn = $1, bitboard_black = $2, bitboard_white = $3 WHERE id = $4")
                .bind::<&Color>(&game.current_turn.into())
                .bind(game.black.slices())
                .bind(game.white.slices())
                .bind(&game_id)
                .execute(&pool)
                .await {
                Ok(_) => Ok(StatusCode::CREATED),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
            Err(_) => Err(StatusCode::UNPROCESSABLE_ENTITY)
        }
    } else {
        println!("{:?} {:?}", game.legal_moves(game.current_turn), game);
        Err(StatusCode::UNPROCESSABLE_ENTITY)
    }
}


#[derive(FromRow)]
struct GameRow {
    id: String,
    timestamp: NaiveDateTime,
    #[sqlx(rename = "type")]
    game_type: String,
    status: String,
    bitboard_white: Vec<u8>,
    bitboard_black: Vec<u8>,
    current_turn: Color,
    player_one_id: String,
    player_one_color: Color,
    player_two_id: Option<String>,
    player_two_color: Color,
}

#[derive(Serialize)]
struct GameResponse {
    id: String,
    timestamp: NaiveDateTime,
    #[serde(rename = "type")]
    game_type: String,
    status: String,
    bitboard_white: u64,
    bitboard_black: u64,
    current_turn: Color,
    player_one_color: Color,
    player_two_color: Color,
    current_user_player: u8,
    white_score: u8,
    black_score: u8,
}

pub async fn get_all_games(
    State(pool): State<PgPool>,
    Extension(account): Extension<Account>,
) -> impl IntoResponse {
    let player_id =
        match sqlx::query_as::<sqlx::Postgres, Player>("SELECT id FROM Player WHERE user_id = $1")
            .bind(&account.id)
            .fetch_one(&pool)
            .await
        {
            Ok(player) => player.id,
            Err(err) => {
                return match err {
                    Error::RowNotFound => Err(StatusCode::NOT_FOUND),
                    _ => {
                        println!("err");
                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                    },
                };
            }
        };

    let games: Vec<GameResponse> = match sqlx::query_as::<sqlx::Postgres, GameRow>(
        "SELECT id, timestamp, type, status, bitboard_white, bitboard_black, player_one_id, player_one_color, player_two_id, player_two_color, current_turn FROM Game WHERE (player_one_id = $1 OR player_two_id = $1)",
    )
        .bind(&player_id)
        .fetch_all(&pool)
        .await {
        Ok(games) => games.into_iter().map(|game| {
            let othello_game = OthelloGame::new_with_state(
                u64::from_le_bytes(game.bitboard_black.try_into().unwrap()),
                u64::from_le_bytes(game.bitboard_white.try_into().unwrap()),
                game.current_turn.clone().into(),
            );

            GameResponse {
                id: game.id,
                timestamp: game.timestamp,
                game_type: game.game_type,
                status: game.status,
                current_turn: game.current_turn,
                player_one_color: game.player_one_color,
                player_two_color: game.player_two_color,
                bitboard_white: othello_game.white.0,
                bitboard_black: othello_game.black.0,
                current_user_player: if player_id == game.player_one_id {
                    1
                } else if player_id == game.player_two_id.unwrap_or("".to_string()) {
                    2
                } else {
                    0
                },
                // It will always be fine to truncate the score from u32 to u8 here, because the score cannot be higher than 64, and max value of u8 is 255
                white_score: othello_game.score().0 as u8,
                black_score: othello_game.score().1 as u8,
            }
        }).collect(),
        Err(err) => return match err {
            Error::RowNotFound => Err(StatusCode::NOT_FOUND),
            err => {
                println!("{err:?}");
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            },
        }
    };

    Ok(Json(games))
}
