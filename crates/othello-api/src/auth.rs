use crate::env_or_dotenv;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Redirect;
use axum::{
    body::Body,
    extract::{Json, Request},
    http::{Response, StatusCode},
    middleware::Next,
    response::IntoResponse,
    Form,
};
use axum_extra::extract::cookie::{Cookie, SameSite};
use axum_extra::extract::CookieJar;
use bcrypt::{hash, verify, DEFAULT_COST};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::PgPool;
use time::Duration as TimeDuration;
use time::OffsetDateTime;

const JWT_SECRET: &str = env_or_dotenv!("JWT_SECRET");
const FRONTEND_URL: &str = env_or_dotenv!("FRONTEND_URL");
const BACKEND_COOKIE_DOMAIN: &str = env_or_dotenv!("BACKEND_COOKIE_DOMAIN");


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
    let now = Utc::now();
    let expire: chrono::TimeDelta = Duration::hours(24);
    let exp: usize = (now + expire).timestamp() as usize;
    let iat: usize = now.timestamp() as usize;

    let claim = Claims { iat, exp, email };

    encode(
        &Header::default(),
        &claim,
        &EncodingKey::from_secret(JWT_SECRET.as_ref()),
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

pub fn decode_jwt(jwt: String) -> Result<TokenData<Claims>, StatusCode> {
    let result: Result<TokenData<Claims>, StatusCode> = decode(
        &jwt,
        &DecodingKey::from_secret(JWT_SECRET.as_ref()),
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
    let jar = CookieJar::from_headers(headers);

    jar.get("auth_token")
        .map(|cookie| cookie.value().to_string())
}

pub async fn authorise(State(pool): State<PgPool>, mut req: Request, next: Next) -> impl IntoResponse {
    let token = match extract_jwt_from_cookie(req.headers()) {
        Some(t) => t,
        None => {
            return AuthError {
                message: "Missing auth cookie".into(),
                status_code: StatusCode::FORBIDDEN,
            }
            .into_response();
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
    let login_url = format!("{}/auth/login?error=incorrect_credentials", FRONTEND_URL);
    let invalid_csrf_redirect = format!("{}/auth/login?error=csrf", FRONTEND_URL);

    // Validate CSRF
    if !validate_csrf(&headers, &user_data.csrf) {
        return Redirect::to(&invalid_csrf_redirect).into_response();
    }

    // 1. Look up account
    let account = match retrieve_user_by_email(&pool, &user_data.email).await {
        Ok(acc) => acc,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let account = match account {
        Some(acc) => acc,
        None => return Redirect::to(&login_url).into_response(),
    };

    // 2. Verify password
    let valid = match verify_password(&user_data.password, &account.password_hash) {
        Ok(v) => v,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    if !valid {
        return Redirect::to(&login_url).into_response();
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
    let cookie = generate_auth_cookie(token);

    let mut headers = HeaderMap::new();
    headers.insert("Set-Cookie", cookie.to_string().parse().unwrap());

    // 6. Send JSON back with cookie set
    (headers, Redirect::to(FRONTEND_URL)).into_response()
}

fn generate_auth_cookie(token: String) -> Cookie<'static> {
    // JWT expires in 24 hours
    let max_age = TimeDuration::hours(24);
    let expires = OffsetDateTime::now_utc() + max_age;

    let mut cookie = Cookie::build(("auth_token", token.clone()))
        .path("/")
        .secure(true)
        .http_only(true)
        .same_site(SameSite::None)
        .max_age(max_age)
        .expires(expires);

    if !cfg!(debug_assertions) {
        cookie = cookie.domain(BACKEND_COOKIE_DOMAIN);
    }

    cookie.build()
}

#[derive(Deserialize)]
pub struct SignUpData {
    pub username: String,
    pub email: String,
    pub password: String,
    pub csrf: String,
}

pub async fn sign_up(
    State(pool): State<PgPool>,
    headers: HeaderMap,
    Form(data): Form<SignUpData>,
) -> impl IntoResponse {
    let redirect =
        |suffix: &str| Redirect::to(&format!("{}/auth/signup?error={}", FRONTEND_URL, suffix)).into_response();

    // CSRF validation
    if !validate_csrf(&headers, &data.csrf) {
        return redirect("csrf");
    }

    // Input validation
    if !validate_username(&data.username) {
        return redirect("username_invalid");
    }

    if !validate_email(&data.email) {
        return redirect("email_invalid");
    }

    if !validate_password(&data.password) {
        return redirect("password_invalid");
    }

    // Check if email already exists
    if let Ok(Some(_)) = retrieve_user_by_email(&pool, &data.email).await {
        return redirect("email_taken");
    }

    // Check if username already exists
    if let Ok(Some(_)) = retrieve_user_by_username(&pool, &data.username).await {
        return redirect("username_taken");
    }

    // Hash password
    let password_hash = match hash_password(&data.password) {
        Ok(h) => h,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    // Insert user
    let id = uuid::Uuid::new_v4().to_string();
    let insert_result = sqlx::query(
        r#"
        INSERT INTO Account (id, username, email, password_hash, elo_rating, date_joined, is_admin)
        VALUES ($1, $2, $3, $4, 800, CURRENT_TIMESTAMP, false)
        "#,
    )
    .bind(&id)
    .bind(&data.username)
    .bind(&data.email)
    .bind(&password_hash)
    .execute(&pool)
    .await;

    if insert_result.is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    // Generate JWT
    let token = match encode_jwt(data.email.clone()) {
        Ok(t) => t,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    // Set Cookie
    let cookie = generate_auth_cookie(token.clone());

    let mut out_headers = HeaderMap::new();
    out_headers.insert("Set-Cookie", cookie.to_string().parse().unwrap());

    // Final redirect (login successful)
    (out_headers, Redirect::to(FRONTEND_URL)).into_response()
}

pub fn validate_csrf(headers: &HeaderMap, form_value: &str) -> bool {
    let jar = CookieJar::from_headers(headers);

    let csrf_cookie = jar
        .get("csrf")
        .map(|cookie| cookie.value().to_string());

    match csrf_cookie {
        Some(cookie_val) => cookie_val == form_value,
        None => false,
    }
}

pub async fn retrieve_user_by_username(
    pool: &PgPool,
    username: &str,
) -> Result<Option<Account>, sqlx::Error> {
    sqlx::query_as::<_, Account>(
        r#"
        SELECT id, username, email, password_hash, elo_rating, date_joined, last_login, is_admin
        FROM Account
        WHERE username = $1
        "#,
    )
    .bind(username)
    .fetch_optional(pool)
    .await
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

fn validate_username(username: &str) -> bool {
    let valid_len = username.len() >= 4 && username.len() <= 20;
    let valid_chars = username.chars().all(|c| c.is_ascii_alphanumeric());

    valid_len && valid_chars
}

fn validate_email(email: &str) -> bool {
    if email.len() < 3 || email.len() > 254 {
        return false;
    }

    let re = regex::Regex::new(r"(?i)^[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z0-9-.]+$").unwrap();

    re.is_match(email)
}

fn validate_password(password: &str) -> bool {
    let correct_length = password.len() >= 8 && password.len() <= 72;
    let has_lower = password.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = password.chars().any(|c| c.is_ascii_uppercase());
    let has_digit = password.chars().any(|c| c.is_ascii_digit());

    correct_length && has_lower && has_upper && has_digit
}
