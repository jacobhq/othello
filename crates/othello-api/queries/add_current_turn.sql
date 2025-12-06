ALTER TABLE Game
    ADD COLUMN current_turn TEXT NOT NULL CHECK ( current_turn in ('white', 'black') ) DEFAULT 'black';