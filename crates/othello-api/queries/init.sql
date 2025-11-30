CREATE TABLE Account
(
    id            TEXT PRIMARY KEY,
    username      TEXT      NOT NULL UNIQUE,
    email         TEXT      NOT NULL UNIQUE,
    password_hash TEXT      NOT NULL,
    elo_rating    INTEGER   NOT NULL DEFAULT 800 CHECK (elo_rating >= 0),
    date_joined   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login    TIMESTAMP,
    is_admin      BOOLEAN   NOT NULL DEFAULT FALSE
);

CREATE TABLE Model
(
    id         TEXT PRIMARY KEY,
    name       TEXT      NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    elo_rating INTEGER   NOT NULL DEFAULT 800 CHECK (elo_rating >= 0),
    url        TEXT      NOT NULL
);

-- Unified Player abstraction
CREATE TABLE Player
(
    id       TEXT PRIMARY KEY,
    type     TEXT NOT NULL CHECK (type IN ('user', 'model')),
    user_id  TEXT,
    model_id TEXT,
    FOREIGN KEY (user_id) REFERENCES Account (id),
    FOREIGN KEY (model_id) REFERENCES Model (id),
    CHECK (
        (type = 'user' AND user_id IS NOT NULL AND model_id IS NULL)
            OR
        (type = 'model' AND model_id IS NOT NULL AND user_id IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS Player_type_idx ON Player (type);

-- Game table
CREATE TABLE Game
(
    id                      TEXT PRIMARY KEY,
    player_one_id           TEXT      NOT NULL,
    player_two_id           TEXT      NOT NULL,
    winner_id               TEXT,
    moves_json              JSONB,
    timestamp               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    player_one_allowed_time REAL      NOT NULL,
    player_two_allowed_time REAL      NOT NULL,
    player_one_elapsed_time REAL      NOT NULL,
    player_two_elapsed_time REAL      NOT NULL,
    FOREIGN KEY (player_one_id) REFERENCES Player (id),
    FOREIGN KEY (player_two_id) REFERENCES Player (id),
    FOREIGN KEY (winner_id) REFERENCES Player (id),
    CHECK (player_one_id <> player_two_id)
);

CREATE INDEX IF NOT EXISTS Game_timestamp_idx ON Game (timestamp);
CREATE INDEX IF NOT EXISTS Game_player_idx ON Game (player_one_id, player_two_id);