use crate::bitboard::{BitBoard};

/// A color enum to represent white and black
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

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
}
