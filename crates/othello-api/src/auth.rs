use axum::extract::State;
use axum::http::header::COOKIE;
use axum::http::HeaderMap;
use axum::response::Redirect;
use axum::{body::Body, extract::{Json, Request}, http::{Response, StatusCode}, middleware::Next, response::IntoResponse, Form};
use bcrypt::{hash, verify, DEFAULT_COST};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::PgPool;


#[derive(Serialize, Deserialize)]
pub struct Claims {
    pub exp: usize,
    pub iat: usize,
    pub email: String,
}

pub struct AuthError {
    message: String,
    status_code: StatusCode,
}

pub fn verify_password(password: &str, hash: &str) -> Result<bool, bcrypt::BcryptError> {
    verify(password, hash)
}

pub fn hash_password(password: &str) -> Result<String, bcrypt::BcryptError> {
    let hash = hash(password, DEFAULT_COST)?;
    Ok(hash)
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response<Body> {
        let body = Json(json!({
            "error": self.message,
        }));

        (self.status_code, body).into_response()
    }
}

pub fn encode_jwt(email: String) -> Result<String, StatusCode> {
    let jwt_secret = std::env::var("JWT_SECRET")
        .expect("JWT_SECRET must be set in .env file");

    let now = Utc::now();
    let expire: chrono::TimeDelta = Duration::hours(24);
    let exp: usize = (now + expire).timestamp() as usize;
    let iat: usize = now.timestamp() as usize;

    let claim = Claims { iat, exp, email };
    let secret = jwt_secret.clone();

    encode(
        &Header::default(),
        &claim,
        &EncodingKey::from_secret(secret.as_ref()),
    )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

pub fn decode_jwt(jwt: String) -> Result<TokenData<Claims>, StatusCode> {
    let jwt_secret = std::env::var("JWT_SECRET")
        .expect("JWT_SECRET must be set in .env file");

    let result: Result<TokenData<Claims>, StatusCode> = decode(
        &jwt,
        &DecodingKey::from_secret(jwt_secret.as_ref()),
        &Validation::default(),
    )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    result
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Account {
    pub id: String,
    pub username: String,
    pub email: String,
    pub password_hash: String,
    pub elo_rating: i32,
    pub date_joined: chrono::NaiveDateTime,
    pub last_login: Option<chrono::NaiveDateTime>,
    pub is_admin: bool,
}

fn extract_jwt_from_cookie(headers: &HeaderMap) -> Option<String> {
    let cookie_header = headers.get(COOKIE)?.to_str().ok()?;

    cookie_header
        .split(';')
        .map(|c| c.trim())
        .find(|c| c.starts_with("auth_token="))
        .map(|c| c.trim_start_matches("auth_token=").to_string())
}

pub async fn authorize(
    mut req: Request,
    next: Next,
) -> impl IntoResponse {
    let token = match extract_jwt_from_cookie(req.headers()) {
        Some(t) => t,
        None => {
            return AuthError {
                message: "Missing auth cookie".into(),
                status_code: StatusCode::FORBIDDEN,
            }.into_response();
        }
    };

    let token_data = match decode_jwt(token) {
        Ok(t) => t,
        Err(_) => {
            return AuthError {
                message: "Invalid or expired token".into(),
                status_code: StatusCode::UNAUTHORIZED,
            }
                .into_response();
        }
    };

    let pool = match req.extensions().get::<PgPool>() {
        Some(pool) => pool.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "missing database pool in request state",
            )
                .into_response();
        }
    };


    let account = match retrieve_user_by_email(&pool, &token_data.claims.email).await {
        Ok(Some(acc)) => acc,
        Ok(None) => {
            return AuthError {
                message: "User not found".into(),
                status_code: StatusCode::UNAUTHORIZED,
            }
                .into_response();
        }
        Err(_) => {
            return AuthError {
                message: "Database error".into(),
                status_code: StatusCode::INTERNAL_SERVER_ERROR,
            }
                .into_response();
        }
    };

    req.extensions_mut().insert(account);

    next.run(req).await
}


#[derive(Deserialize)]
pub struct SignInData {
    pub email: String,
    pub password: String,
    pub csrf: String,
}

pub async fn sign_in(
    State(pool): State<PgPool>,
    headers: HeaderMap,
    Form(user_data): Form<SignInData>,
) -> impl IntoResponse {
    let login_url = if cfg!(debug_assertions) {
        "http://localhost:5173/auth/login?error=incorrect_credentials"
    } else {
        "https://othello.jhqcat.com/auth/login?error=incorrect_credentials"
    };

    let invalid_csrf_redirect = if cfg!(debug_assertions) {
        "http://localhost:5173/auth/login?error=csrf"
    } else {
        "https://othello.jhqcat.com/auth/login?error=csrf"
    };

    // Validate CSRF
    if !validate_csrf(&headers, &user_data.csrf) {
        return Redirect::to(invalid_csrf_redirect).into_response();
    }

    // 1. Look up account
    let account = match retrieve_user_by_email(&pool, &user_data.email).await {
        Ok(acc) => acc,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let account = match account {
        Some(acc) => acc,
        None => return Redirect::to(login_url).into_response(),
    };

    // 2. Verify password
    let valid = match verify_password(&user_data.password, &account.password_hash) {
        Ok(v) => v,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    if !valid {
        return Redirect::to(login_url).into_response();
    }

    // 3. Generate JWT
    let token = match encode_jwt(account.email.clone()) {
        Ok(t) => t,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    // 4. Optionally update last_login
    sqlx::query("UPDATE Account SET last_login = CURRENT_TIMESTAMP WHERE id = $1")
        .bind(&account.id)
        .execute(&pool)
        .await
        .ok();

    // 5. Build the cookie
    let cookie_value = if cfg!(debug_assertions) {
        // Development: no Domain attribute (localhost behaves better)
        format!(
            "auth_token={}; Secure; HttpOnly; SameSite=None; Path=/",
            token
        )
    } else {
        // Production: include Domain
        let backend_domain = std::env::var("BACKEND_COOKIE_DOMAIN")
            .expect("BACKEND_COOKIE_DOMAIN must be set in production");

        format!(
            "auth_token={}; Secure; HttpOnly; SameSite=None; Path=/; Domain={}",
            token,
            backend_domain
        )
    };

    let mut headers = HeaderMap::new();
    headers.insert("Set-Cookie", cookie_value.parse().unwrap());

    // 6. Send JSON back with cookie set
    let redirect_target = if cfg!(debug_assertions) {
        // Dev
        "http://localhost:5173/"
    } else {
        // Production
        "https://othello.jhqcat.com/"
    };

    let response = (headers, Redirect::to(redirect_target)).into_response();
    response
}

pub fn validate_csrf(headers: &HeaderMap, form_value: &str) -> bool {
    // 1. Read Cookie header
    let cookie_header = match headers.get(COOKIE).and_then(|h| h.to_str().ok()) {
        Some(c) => c,
        None => return false,
    };

    // 2. Parse cookies (super simple)
    let csrf_cookie = cookie_header
        .split(';')
        .map(|c| c.trim())
        .find(|c| c.starts_with("csrf="))
        .map(|c| c.trim_start_matches("csrf="));

    match csrf_cookie {
        Some(cookie_val) => cookie_val == form_value,
        None => false,
    }
}


pub async fn retrieve_user_by_email(
    pool: &PgPool,
    email: &str,
) -> Result<Option<Account>, sqlx::Error> {
    let account = sqlx::query_as::<_, Account>(
        r#"
        SELECT id, username, email, password_hash, elo_rating, date_joined, last_login, is_admin
        FROM Account
        WHERE email = $1
        "#,
    )
        .bind(email)
        .fetch_optional(pool)
        .await?;

    Ok(account)
}
