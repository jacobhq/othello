use crate::bitboard::BitBoard;

/// A color enum to represent white and black
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

// Masks to prevent wrapping when shifting
const NOT_A_FILE: BitBoard = BitBoard(0xfefefefefefefefe);
const NOT_H_FILE: BitBoard = BitBoard(0x7f7f7f7f7f7f7f7f);

/// Represents a full Othello/Reversi board using two BitBoards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OthelloGame {
    pub black: BitBoard,
    pub white: BitBoard,
    pub current_turn: Color
}

impl OthelloGame {
    /// Creates a new Othello board with the standard starting position.
    pub fn new() -> Self {
        let mut board = OthelloGame {
            black: BitBoard(0),
            white: BitBoard(0),
            current_turn: Color::Black
        };
        board.set(3, 4, Color::Black);
        board.set(4, 3, Color::Black);
        board.set(3, 3, Color::White);
        board.set(4, 4, Color::White);
        board
    }

    /// Returns the color at a given square, if any.
    pub fn get(&self, row: usize, col: usize) -> Option<Color> {
        if self.black.get(row, col) {
            Some(Color::Black)
        } else if self.white.get(row, col) {
            Some(Color::White)
        } else {
            None
        }
    }

    /// Sets a square to a given color.
    pub fn set(&mut self, row: usize, col: usize, color: Color) {
        match color {
            Color::Black => {
                self.black.set(row, col);
                self.white.clear(row, col);
            }
            Color::White => {
                self.white.set(row, col);
                self.black.clear(row, col);
            }
        }
    }

    /// Return a bitmask of all legal moves for the given player
    fn legal_moves_mask(&self, player: Color) -> BitBoard {
        let (me, opp) = match player {
            Color::Black => (self.black, self.white),
            Color::White => (self.white, self.black),
        };

        let empty = !(me | opp);
        let mut moves = BitBoard(0);

        let directions: [(i64, BitBoard); 8] = [
            (8, BitBoard(0xffffffffffffffff)),  // north
            (-8, BitBoard(0xffffffffffffffff)), // south
            (1, NOT_A_FILE),                    // east
            (-1, NOT_H_FILE),                   // west
            (9, NOT_A_FILE),                    // northeast
            (7, NOT_H_FILE),                    // northwest
            (-7, NOT_A_FILE),                   // southeast
            (-9, NOT_H_FILE),                   // southwest
        ];

        for (shift, mask) in directions {
            let mut candidates = match shift {
                s if s > 0 => (me << s) & opp & mask,
                s if s < 0 => (me >> -s) & opp & mask,
                _ => BitBoard(0),
            };

            let mut flips = candidates;
            while candidates != BitBoard(0) {
                candidates = match shift {
                    s if s > 0 => (candidates << s) & opp & mask,
                    s if s < 0 => (candidates >> -s) & opp & mask,
                    _ => BitBoard(0),
                };
                flips |= candidates;
            }

            moves |= match shift {
                s if s > 0 => (flips << s) & empty,
                s if s < 0 => (flips >> -s) & empty,
                _ => BitBoard(0),
            };
        }

        moves
    }

    /// Return a vector of legal moves as (row, col)
    pub fn legal_moves(&self, player: Color) -> Vec<(usize, usize)> {
        let mut mask = self.legal_moves_mask(player).0;
        let mut moves = Vec::new();

        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            moves.push((i / 8, i % 8));
            mask &= mask - 1;
        }

        moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_setup() {
        let game = OthelloGame::new();

        // The starting four discs
        assert_eq!(game.get(3, 3), Some(Color::White));
        assert_eq!(game.get(4, 4), Some(Color::White));
        assert_eq!(game.get(3, 4), Some(Color::Black));
        assert_eq!(game.get(4, 3), Some(Color::Black));

        // Everything else is empty
        for r in 0..8 {
            for c in 0..8 {
                if !((r == 3 && c == 3)
                    || (r == 4 && c == 4)
                    || (r == 3 && c == 4)
                    || (r == 4 && c == 3))
                {
                    assert_eq!(game.get(r, c), None);
                }
            }
        }
    }

    #[test]
    fn test_set_and_get() {
        let mut game = OthelloGame::new();

        // Set and overwrite positions
        game.set(0, 0, Color::Black);
        assert_eq!(game.get(0, 0), Some(Color::Black));

        game.set(0, 0, Color::White);
        assert_eq!(game.get(0, 0), Some(Color::White));

        // Clearing happens automatically
        assert!(!game.black.get(0, 0));
    }

    #[test]
    fn test_legal_moves_black_starting_position() {
        let game = OthelloGame::new();
        let mut moves = game.legal_moves(Color::Black);
        moves.sort();

        // Black to play at start: (2,3), (3,2), (4,5), (5,4)
        let expected = vec![(2, 3), (3, 2), (4, 5), (5, 4)];
        assert_eq!(moves, expected);
    }

    #[test]
    fn test_legal_moves_white_starting_position() {
        let game = OthelloGame::new();
        let mut moves = game.legal_moves(Color::White);
        moves.sort();

        // White to play at start: (2,4), (3,5), (4,2), (5,3)
        let expected = vec![(2, 4), (3, 5), (4, 2), (5, 3)];
        assert_eq!(moves, expected);
    }

    #[test]
    fn test_legal_moves_mask_matches_moves_vector() {
        let game = OthelloGame::new();
        let mask = game.legal_moves_mask(Color::Black).0;
        let from_mask: Vec<(usize, usize)> = (0..64)
            .filter(|i| (mask >> i) & 1 == 1)
            .map(|i| (i / 8, i % 8))
            .collect();

        let mut moves = game.legal_moves(Color::Black);
        moves.sort();

        assert_eq!(moves, from_mask);
    }
}
