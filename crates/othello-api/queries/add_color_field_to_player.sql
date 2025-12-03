ALTER TABLE Player
    ADD COLUMN color TEXT NOT NULL CHECK (color IN ('white', 'black')) DEFAULT 'black'