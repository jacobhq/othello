ALTER TABLE game
    ALTER COLUMN player_two_id DROP NOT NULL;

ALTER TABLE game
    ALTER COLUMN player_one_allowed_time DROP NOT NULL,
    ALTER COLUMN player_two_allowed_time DROP NOT NULL,
    ALTER COLUMN player_one_elapsed_time DROP NOT NULL,
    ALTER COLUMN player_two_elapsed_time DROP NOT NULL;
