ALTER TABLE player
    DROP COLUMN color;

ALTER TABLE game
    ADD COLUMN player_one_color TEXT NOT NULL CHECK (player_one_color IN ('white', 'black')) DEFAULT 'black',
    ADD COLUMN player_two_color TEXT NOT NULL CHECK (player_two_color IN ('white', 'black')) DEFAULT 'white',
    ADD CHECK ( player_one_color <> player_two_color );