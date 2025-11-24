use axum::{Extension, Json, response::IntoResponse};
use serde::{Serialize, Deserialize};

use crate::auth::Account;

#[derive(Serialize, Deserialize)]
struct UserResponse {
    email: String,
    username: String
}

pub async fn hello(Extension(currentUser): Extension<Account>) -> impl IntoResponse {
    Json(UserResponse {
        email: currentUser.email,
        username: currentUser.username
    })
}
