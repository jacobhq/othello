ALTER TABLE Game ADD COLUMN status TEXT NOT NULL CHECK (status IN ('in_play', 'won', 'drew')) DEFAULT 'in_play';
ALTER TABLE Game DROP COLUMN moves_json;
ALTER TABLE Game ADD COLUMN bitboard_white BIGINT;
ALTER TABLE Game ADD COLUMN bitboard_black BIGINT;

ALTER TABLE Game DROP COLUMN bitboard_white;
ALTER TABLE Game DROP COLUMN bitboard_black;
ALTER TABLE Game ADD COLUMN bitboard_white bytea;
ALTER TABLE Game Add COLUMN bitboard_black bytea;