use std::fmt::{Display, Formatter};
use crate::bitboard::BitBoard;

/// An error enum to represent which error has occurred in the game
#[derive(Debug, PartialEq)]
pub enum OthelloError {
    NoMovesForPlayer,
    NotYourTurn,
    IllegalMove,
}

/// A color enum to represent white and black
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::White => write!(f, "● White")?,
            Self::Black => write!(f, "○ Black")?
        }

        Ok(())
    }
}

// Masks to prevent wrapping when shifting
const NOT_A_FILE: BitBoard = BitBoard(0xfefefefefefefefe);
const NOT_H_FILE: BitBoard = BitBoard(0x7f7f7f7f7f7f7f7f);
const DIRECTIONS: [(i64, BitBoard); 8] = [
    (8, BitBoard(0xffffffffffffffff)),  // north
    (-8, BitBoard(0xffffffffffffffff)), // south
    (1, NOT_H_FILE),                    // east
    (-1, NOT_A_FILE),                   // west
    (9, NOT_H_FILE),                    // northeast
    (7, NOT_A_FILE),                    // northwest
    (-7, NOT_H_FILE),                   // southeast
    (-9, NOT_A_FILE),                   // southwest
];

/// Represents a full Othello/Reversi board using two BitBoards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OthelloGame {
    pub black: BitBoard,
    pub white: BitBoard,
    pub current_turn: Color,
}

impl Default for OthelloGame {
    fn default() -> Self {
        Self::new()
    }
}

impl OthelloGame {
    /// Creates a new Othello board with the standard starting position.
    pub fn new() -> Self {
        let mut board = OthelloGame {
            black: BitBoard(0),
            white: BitBoard(0),
            current_turn: Color::Black,
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

        for (shift, mask) in DIRECTIONS {
            let mut candidates = Self::step_along(me, shift, mask) & opp;

            let mut flips = candidates;
            while candidates != BitBoard(0) {
                candidates = Self::step_along(candidates, shift, mask) & opp;
                flips |= candidates;
            }

            moves |= match shift {
                s if s > 0 => ((flips & mask) << s) & empty,
                s if s < 0 => ((flips & mask) >> -s) & empty,
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

    /// Play a move for the given player. Returns false if illegal.
    pub fn play(&mut self, row: usize, col: usize, player: Color) -> Result<(), OthelloError> {
        if self.legal_moves_mask(self.current_turn) == BitBoard(0) {
            let next = match self.current_turn {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
            self.current_turn = next;
            return Err(OthelloError::NoMovesForPlayer);
        }
        let move_mask: BitBoard = BitBoard(BitBoard::mask(row, col));
        if self.current_turn != player {
            return Err(OthelloError::NotYourTurn);
        }
        if self.legal_moves_mask(player) & move_mask == BitBoard(0) {
            self.current_turn = player;
            return Err(OthelloError::IllegalMove);
        }

        let (mut me, mut opp) = match player {
            Color::Black => (self.black, self.white),
            Color::White => (self.white, self.black),
        };

        let mut flips: BitBoard = BitBoard(0);

        for (shift, mask) in DIRECTIONS {
            let mut candidate = if shift > 0 {
                (move_mask & mask) << shift
            } else if shift < 0 {
                (move_mask & mask) >> -shift
            } else {
                BitBoard(0)
            };

            let mut flips_in_dir = BitBoard(0);

            while candidate != BitBoard(0) && (candidate & opp) != BitBoard(0) {
                flips_in_dir |= candidate;

                candidate = if shift > 0 {
                    (candidate & mask) << shift
                } else {
                    (candidate & mask) >> -shift
                };
            }

            if candidate & me != BitBoard(0) {
                flips |= flips_in_dir;
            }
        }

        me |= move_mask | flips;
        opp &= !flips;

        match player {
            Color::Black => {
                self.black = me;
                self.white = opp;
            }
            Color::White => {
                self.white = me;
                self.black = opp;
            }
        }

        let next_player = match player {
            Color::Black => Color::White,
            Color::White => Color::Black,
        };
        if self.legal_moves_mask(next_player) != BitBoard(0) {
            // Opponent has a move: switch turn
            self.current_turn = next_player;
        } else if self.legal_moves_mask(player) != BitBoard(0) {
            // Opponent has no moves but current player does: stay on current player
            self.current_turn = player;
        }

        Ok(())
    }

    pub fn score(&self) -> (u32, u32) {
        (self.white.0.count_ones(), self.black.0.count_ones())
    }

    pub fn game_over(&self) -> bool {
        self.legal_moves_mask(Color::Black) == BitBoard(0)
            && self.legal_moves_mask(Color::White) == BitBoard(0)
    }

    pub fn encode(&self, player: Color) -> [[[i32; 8]; 8]; 2] {
        let mut state = [[[0; 8]; 8]; 2];
        for row in 0..8 {
            for col in 0..8 {
                if let Some(c) = self.get(row, col) {
                    match (player, c) {
                        (Color::White, Color::White) | (Color::Black, Color::Black) => {
                            state[0][row][col] = 1; // me
                        }
                        (Color::White, Color::Black) | (Color::Black, Color::White) => {
                            state[1][row][col] = 1; // opponent
                        }
                    }
                }
            }
        }
        state
    }

    #[inline]
    fn step_along(start: BitBoard, shift: i64, mask: BitBoard) -> BitBoard {
        match shift {
            s if s > 0 => (start & mask) << s,
            s if s < 0 => (start & mask) >> -s,
            _ => BitBoard(0),
        }
    }
}

impl Display for OthelloGame {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  0 1 2 3 4 5 6 7")?;
        for row in 0..8 {
            write!(f, "{} ", row)?;
            for col in 0..8 {
                let m = BitBoard::mask(row, col);
                if self.black & BitBoard(m) != BitBoard(0) {
                    write!(f, "○ ")?;
                } else if self.white & BitBoard(m) != BitBoard(0) {
                    write!(f, "● ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
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
                if r != 3 && r != 4 || c != 3 && c != 4
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

    #[test]
    fn test_play_valid_move_and_flips() {
        let mut game = OthelloGame::new();
        // Black plays at (2,3) — should flip (3,3)
        assert!(game.play(2, 3, Color::Black).is_ok());
        assert_eq!(game.get(2, 3), Some(Color::Black));
        assert_eq!(game.get(3, 3), Some(Color::Black));
    }

    #[test]
    fn test_play_illegal_move_outside_legal_moves() {
        let mut game = OthelloGame::new();
        // (0,0) is not a legal move initially
        assert!(game.play(0, 0, Color::Black).is_err());
    }

    #[test]
    fn test_play_illegal_wrong_turn() {
        let mut game = OthelloGame::new();
        // It’s Black’s turn initially, so White can’t play
        assert_eq!(
            game.play(2, 4, Color::White).unwrap_err(),
            OthelloError::NotYourTurn
        );
    }

    #[test]
    fn test_play_no_flips_does_not_change_board() {
        let mut game = OthelloGame::new();
        let before_black = game.black;
        let before_white = game.white;
        // Even if it’s Black’s turn, (0,0) won’t flip any pieces
        assert_eq!(
            game.play(0, 0, Color::Black).unwrap_err(),
            OthelloError::IllegalMove
        );
        assert_eq!(game.black, before_black);
        assert_eq!(game.white, before_white);
    }

    #[test]
    fn test_play_flips_in_multiple_directions() {
        let mut game = OthelloGame::new();
        // Set up a custom board for a multi-direction flip test
        game.black = BitBoard(0);
        game.white = BitBoard(0);
        // Form a cross of White pieces with a Black center
        game.black.set(3, 3);
        for &(r, c) in &[(2, 3), (4, 3), (3, 2), (3, 4)] {
            game.white.set(r, c);
        }
        // Black plays (3,5) → should flip (3,4)
        assert!(game.play(3, 5, Color::Black).is_ok());
        assert_eq!(game.get(3, 4), Some(Color::Black));
        assert_eq!(game.get(3, 5), Some(Color::Black));
    }

    #[test]
    fn test_play_fills_board_and_detects_game_over() {
        let mut game = OthelloGame::new();
        // Artificially fill the board
        game.black = BitBoard(u64::MAX);
        game.white = BitBoard(0);
        assert!(game.game_over());
    }

    #[test]
    fn test_score_counts_correctly() {
        let mut game = OthelloGame::new();
        // Initially both have 2 pieces
        assert_eq!(game.score(), (2, 2));
        // Add more black pieces manually
        game.black.set(0, 0);
        game.black.set(0, 1);
        assert_eq!(game.score(), (2, 4));
    }

    #[test]
    fn test_encode_shape_and_values_for_black() {
        let game = OthelloGame::new();
        let encoded = game.encode(Color::Black);
        // Shape should be [2][8][8]
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0].len(), 8);
        assert_eq!(encoded[0][0].len(), 8);
        // Check that (3,4) is "me" for black
        assert_eq!(encoded[0][3][4], 1);
        // Check that (3,3) is "opponent"
        assert_eq!(encoded[1][3][3], 1);
    }

    #[test]
    fn test_encode_shape_and_values_for_white() {
        let game = OthelloGame::new();
        let encoded = game.encode(Color::White);
        // (3,3) is "me" for white
        assert_eq!(encoded[0][3][3], 1);
        // (3,4) is "opponent" for white
        assert_eq!(encoded[1][3][4], 1);
    }

    #[test]
    fn test_no_moves_for_both_players_game_over() {
        let mut game = OthelloGame::new();
        // Fill board fully to disable all moves
        game.black = BitBoard(u64::MAX);
        game.white = BitBoard(0);
        assert!(game.game_over());
    }

    #[test]
    fn test_partial_game_not_over() {
        let game = OthelloGame::new();
        assert!(!game.game_over());
    }

    #[test]
    fn test_turn_pass_when_opponent_has_no_moves() {
        let mut game = OthelloGame {
            black: BitBoard(
                0b10111111_10000001_10000001_10000001_10000001_10000001_10000001_01111111,
            ),
            white: BitBoard(
                0b01000000_01111110_01111110_01111110_01111110_01111110_01111110_00000000,
            ),
            current_turn: Color::White,
        };

        assert_eq!(game.current_turn, Color::White);
        let turn_one = game.play(0, 7, Color::White);
        assert_eq!(turn_one.unwrap_err(), OthelloError::NoMovesForPlayer);

        assert_eq!(game.current_turn, Color::Black);
        let turn_two = game.play(0, 7, Color::Black);
        assert!(turn_two.is_ok(), "Move should be legal");

        assert_eq!(game.get(0, 7), Some(Color::Black));
        assert_eq!(
            game.current_turn,
            Color::Black,
            "White has no legal moves, turn should remain Black"
        );

        assert!(game.legal_moves(Color::White).is_empty());
        assert!(game.legal_moves(Color::Black).is_empty());
    }

    #[test]
    fn test_board_state_flips_tokens_correctly_1() {
        // Set up the specific board:
        // It is ○ Black's turn.
        //   0 1 2 3 4 5 6 7
        // 0 . . . ● . . ○ .
        // 1 . . . ● ● ○ . .
        // 2 . . ● ● ● ● . .
        // 3 . . ● ○ ● ● ● ○
        // 4 ○ ○ ○ ○ ○ ● ● ○
        // 5 . . ○ ● ● ● ● ○
        // 6 . . ○ ○ . ● . .
        // 7 . . ○ ○ . . . .
        let mut game = OthelloGame {
            black: BitBoard(0),
            white: BitBoard(0),
            current_turn: Color::Black,
        };

        // Row 0: `. . . ● . . ○ .`
        game.set(0, 3, Color::White);
        game.set(0, 6, Color::Black);

        // Row 1: `. . . ● ● ○ . .`
        game.set(1, 3, Color::White);
        game.set(1, 4, Color::White);
        game.set(1, 5, Color::Black);

        // Row 2: `. . ● ● ● ● . .`
        game.set(2, 2, Color::White);
        game.set(2, 3, Color::White);
        game.set(2, 4, Color::White);
        game.set(2, 5, Color::White);

        // Row 3: `. . ● ○ ● ● ● ○`
        game.set(3, 2, Color::White);
        game.set(3, 3, Color::Black);
        game.set(3, 4, Color::White);
        game.set(3, 5, Color::White);
        game.set(3, 6, Color::White);
        game.set(3, 7, Color::Black);

        // Row 4: `○ ○ ○ ○ ○ ● ● ○`
        game.set(4, 0, Color::Black);
        game.set(4, 1, Color::Black);
        game.set(4, 2, Color::Black);
        game.set(4, 3, Color::Black);
        game.set(4, 4, Color::Black);
        game.set(4, 5, Color::White);
        game.set(4, 6, Color::White);
        game.set(4, 7, Color::Black);

        // Row 5: `. . ○ ● ● ● ● ○`
        game.set(5, 2, Color::Black);
        game.set(5, 3, Color::White);
        game.set(5, 4, Color::White);
        game.set(5, 5, Color::White);
        game.set(5, 6, Color::White);
        game.set(5, 7, Color::Black);

        // Row 6: `. . ○ ○ . ● . .`
        game.set(6, 2, Color::Black);
        game.set(6, 3, Color::Black);
        game.set(6, 5, Color::White);

        // Row 7: `. . ○ ○ . . . .`
        game.set(7, 2, Color::Black);
        game.set(7, 3, Color::Black);

        // Verify initial state - row 7 should have Black at (7,2) and (7,3)
        assert_eq!(game.get(7, 2), Some(Color::Black));
        assert_eq!(game.get(7, 3), Some(Color::Black));
        assert_eq!(game.get(7, 4), None);
        assert_eq!(game.current_turn, Color::Black);

        // Verify (7,4) is a legal move
        let legal_moves = game.legal_moves(Color::Black);
        assert!(
            legal_moves.contains(&(7, 4)),
            "Position (7,4) should be a legal move for Black"
        );

        // Play the move at (7,4)
        let result = game.play(7, 4, Color::Black);
        assert!(result.is_ok(), "Move should be legal and successful");

        // After:
        // It is ● White's turn.
        //   0 1 2 3 4 5 6 7
        // 0 . . . ● . . ○ .
        // 1 . . . ● ● ○ . .
        // 2 . . ● ● ● ● . .
        // 3 . . ● ○ ● ● ● ○
        // 4 ○ ○ ○ ○ ○ ● ● ○
        // 5 . . ○ ● ● ● ○ ○
        // 6 . . ○ ○ . ○ . .
        // 7 . . ○ ○ ○ . . .
        // After the move, positions (7,4), (6,5), and (5,6) should all be Black
        assert_eq!(
            game.get(7, 4),
            Some(Color::Black),
            "Position (7,4) should be flipped to Black"
        );
        assert_eq!(
            game.get(6, 5),
            Some(Color::Black),
            "Position (6,5) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 6),
            Some(Color::Black),
            "Position (5,6) should be Black after the move"
        );
    }

    #[test]
    fn test_board_state_flips_tokens_correctly_2() {
        // Set up the specific board:
        // It is ○ Black's turn.
        //   0 1 2 3 4 5 6 7
        // 0 . . . ● . . ○ .
        // 1 . . . ● ● ○ . .
        // 2 . . ● ● ● ● . .
        // 3 . . ● ● ● ● ● ○
        // 4 ○ ○ ● ○ ○ ● ● ○
        // 5 . ● ● ● ● ● ● ○
        // 6 . . ○ ○ . ● . .
        // 7 . . ○ ○ ○ . . .
        let mut game = OthelloGame {
            black: BitBoard(0),
            white: BitBoard(0),
            current_turn: Color::Black,
        };

        // Row 0: `. . . ● . . ○ .`
        game.set(0, 3, Color::White);
        game.set(0, 6, Color::Black);

        // Row 1: `. . . ● ● ○ . .`
        game.set(1, 3, Color::White);
        game.set(1, 4, Color::White);
        game.set(1, 5, Color::Black);

        // Row 2: `. . ● ● ● ● . .`
        game.set(2, 2, Color::White);
        game.set(2, 3, Color::White);
        game.set(2, 4, Color::White);
        game.set(2, 5, Color::White);

        // Row 3: `. . ● ● ● ● ● ○`
        game.set(3, 2, Color::White);
        game.set(3, 3, Color::White);
        game.set(3, 4, Color::White);
        game.set(3, 5, Color::White);
        game.set(3, 6, Color::White);
        game.set(3, 7, Color::Black);

        // Row 4: `○ ○ ● ○ ○ ● ● ○`
        game.set(4, 0, Color::Black);
        game.set(4, 1, Color::Black);
        game.set(4, 2, Color::White);
        game.set(4, 3, Color::Black);
        game.set(4, 4, Color::Black);
        game.set(4, 5, Color::White);
        game.set(4, 6, Color::White);
        game.set(4, 7, Color::Black);

        // Row 5: `. ● ● ● ● ● ● ○`
        game.set(5, 1, Color::White);
        game.set(5, 2, Color::White);
        game.set(5, 3, Color::White);
        game.set(5, 4, Color::White);
        game.set(5, 5, Color::White);
        game.set(5, 6, Color::White);
        game.set(5, 7, Color::Black);

        // Row 6: `. . ○ ○ . ● . .`
        game.set(6, 2, Color::Black);
        game.set(6, 3, Color::Black);
        game.set(6, 5, Color::White);

        // Row 7: `. . ○ ○ ○ . . .`
        game.set(7, 2, Color::Black);
        game.set(7, 3, Color::Black);
        game.set(7, 4, Color::Black);

        // Verify initial state - row 7 should have Black at (7,2), (7,3) and (7,4)
        assert_eq!(game.get(7, 2), Some(Color::Black));
        assert_eq!(game.get(7, 3), Some(Color::Black));
        assert_eq!(game.get(7, 4), Some(Color::Black));
        assert_eq!(game.current_turn, Color::Black);

        // Verify (5,0) is a legal move
        let legal_moves = game.legal_moves(Color::Black);
        assert!(
            legal_moves.contains(&(5, 0)),
            "Position (5,0) should be a legal move for Black"
        );

        // Play the move at (7,4)
        let result = game.play(5, 0, Color::Black);
        assert!(result.is_ok(), "Move should be legal and successful");

        // After:
        // It is ● White's turn.
        //   0 1 2 3 4 5 6 7
        // 0 . . . ● . . ○ .
        // 1 . . . ● ● ○ . .
        // 2 . . ● ● ● ● . .
        // 3 . . ● ● ● ● ● ○
        // 4 ○ ○ ● ○ ○ ● ● ○
        // 5 ○ ○ ○ ○ ○ ○ ○ ○
        // 6 . . ○ ○ . ● . .
        // 7 . . ○ ○ ○ . . .
        // After the move, all of row 5 should be black
        assert_eq!(
            game.get(5, 0),
            Some(Color::Black),
            "Position (5,0) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 1),
            Some(Color::Black),
            "Position (5,1) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 2),
            Some(Color::Black),
            "Position (5,2) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 3),
            Some(Color::Black),
            "Position (5,3) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 4),
            Some(Color::Black),
            "Position (5,4) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 5),
            Some(Color::Black),
            "Position (5,5) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 6),
            Some(Color::Black),
            "Position (5,6) should be flipped to Black"
        );
        assert_eq!(
            game.get(5, 7),
            Some(Color::Black),
            "Position (5,7) should be flipped to Black"
        );
    }

    #[test]
    fn test_board_state_flips_tokens_correctly_3() {
        // Set up the specific board:
        // It is ● White's turn.
        //   0 1 2 3 4 5 6 7
        // 0 . . ○ ○ ○ ● ● ●
        // 1 . ● ● ● ● ● . .
        // 2 . ● ● ● ○ ● . .
        // 3 ○ ● ○ ● ● ● ● ○
        // 4 ○ ● ○ ● ● ● ● ○
        // 5 ○ ○ ○ ● ○ ● ● ○
        // 6 ○ ○ ○ ○ ○ ○ . .
        // 7 ○ ● ○ ○ ○ ○ ○ .
        let mut game = OthelloGame {
            black: BitBoard(0),
            white: BitBoard(0),
            current_turn: Color::White,
        };

        // Row 0: `. . ○ ○ ○ ● ● ●`
        game.set(0, 2, Color::Black);
        game.set(0, 3, Color::Black);
        game.set(0, 4, Color::Black);
        game.set(0, 5, Color::White);
        game.set(0, 6, Color::White);
        game.set(0, 7, Color::White);

        // Row 1: `. ● ● ● ● ● . .`
        game.set(1, 1, Color::White);
        game.set(1, 2, Color::White);
        game.set(1, 3, Color::White);
        game.set(1, 4, Color::White);
        game.set(1, 5, Color::White);

        // Row 2: `. ● ● ● ○ ● . .`
        game.set(2, 1, Color::White);
        game.set(2, 2, Color::White);
        game.set(2, 3, Color::White);
        game.set(2, 4, Color::Black);
        game.set(2, 5, Color::White);

        // Row 3: `○ ● ○ ● ● ● ● ○`
        game.set(3, 0, Color::Black);
        game.set(3, 1, Color::White);
        game.set(3, 2, Color::Black);
        game.set(3, 3, Color::White);
        game.set(3, 4, Color::White);
        game.set(3, 5, Color::White);
        game.set(3, 6, Color::White);
        game.set(3, 7, Color::Black);

        // Row 4: `○ ● ○ ● ● ● ● ○`
        game.set(4, 0, Color::Black);
        game.set(4, 1, Color::White);
        game.set(4, 2, Color::Black);
        game.set(4, 3, Color::White);
        game.set(4, 4, Color::White);
        game.set(4, 5, Color::White);
        game.set(4, 6, Color::White);
        game.set(4, 7, Color::Black);

        // Row 5: `○ ○ ○ ● ○ ● ● ○`
        game.set(5, 0, Color::Black);
        game.set(5, 1, Color::Black);
        game.set(5, 2, Color::Black);
        game.set(5, 3, Color::White);
        game.set(5, 4, Color::Black);
        game.set(5, 5, Color::White);
        game.set(5, 6, Color::White);
        game.set(5, 7, Color::Black);

        // Row 6: `○ ○ ○ ○ ○ ○ . .`
        game.set(6, 0, Color::Black);
        game.set(6, 1, Color::Black);
        game.set(6, 2, Color::Black);
        game.set(6, 3, Color::Black);
        game.set(6, 5, Color::Black);

        // Row 7: `○ ● ○ ○ ○ ○ ○ .`
        game.set(7, 0, Color::Black);
        game.set(7, 1, Color::White);
        game.set(7, 2, Color::Black);
        game.set(7, 3, Color::Black);
        game.set(7, 4, Color::Black);
        game.set(7, 5, Color::Black);
        game.set(7, 6, Color::Black);

        // Verify initial state - row 7
        assert_eq!(game.get(7, 0), Some(Color::Black));
        assert_eq!(game.get(7, 1), Some(Color::White));
        assert_eq!(game.get(7, 2), Some(Color::Black));
        assert_eq!(game.get(7, 3), Some(Color::Black));
        assert_eq!(game.get(7, 4), Some(Color::Black));
        assert_eq!(game.get(7, 5), Some(Color::Black));
        assert_eq!(game.get(7, 6), Some(Color::Black));
        assert_eq!(game.get(7, 7), None);
        assert_eq!(game.current_turn, Color::White);

        // Verify (7,7) is a legal move
        let legal_moves = game.legal_moves(Color::White);
        assert!(
            legal_moves.contains(&(7, 7)),
            "Position (7,7) should be a legal move for White"
        );

        // Play the move at (7,7)
        let result = game.play(7, 7, Color::White);
        assert!(result.is_ok(), "Move should be legal and successful");

        // After:
        // It is ● White's turn.
        //   0 1 2 3 4 5 6 7
        // 0 . . ○ ○ ○ ● ● ●
        // 1 . ● ● ● ● ● . .
        // 2 . ● ● ● ○ ● . .
        // 3 ○ ● ○ ● ● ● ● ○
        // 4 ○ ● ○ ● ● ● ● ○
        // 5 ○ ○ ● ● ○ ● ● ○
        // 6 ○ ● ○ ○ ○ ○ . .
        // 7 ○ ● ● ● ● ● ● ●
        // After the move, row 7 should be white except from at 7 0
        assert_eq!(
            game.get(7, 0),
            Some(Color::Black),
            "Position (7,0) should be flipped to Black"
        );
        assert_eq!(
            game.get(7, 1),
            Some(Color::White),
            "Position (7,1) should be flipped to White"
        );
        assert_eq!(
            game.get(7, 2),
            Some(Color::White),
            "Position (7,1) should be flipped to White"
        );
        assert_eq!(
            game.get(7, 3),
            Some(Color::White),
            "Position (7,3) should be flipped to White"
        );
        assert_eq!(
            game.get(7, 4),
            Some(Color::White),
            "Position (7,4) should be flipped to White"
        );
        assert_eq!(
            game.get(7, 5),
            Some(Color::White),
            "Position (7,5) should be flipped to White"
        );
        assert_eq!(
            game.get(7, 6),
            Some(Color::White),
            "Position (7,6) should be flipped to White"
        );
        assert_eq!(
            game.get(7, 7),
            Some(Color::White),
            "Position (7,7) should be flipped to White"
        );
    }

    #[test]
    fn test_legal_moves_error() {
        // Set up the specific board: https://eu.posthog.com/project/99250/replay/019a54ed-0f42-7442-afa0-2ec6d5e167a9?t=401
        let mut game = OthelloGame {
            black: BitBoard(0),
            white: BitBoard(0),
            current_turn: Color::White,
        };

        // Row 0: `○ ○ ● ● . . . .`
        game.set(0, 0, Color::Black);
        game.set(0, 1, Color::Black);
        game.set(0, 2, Color::White);
        game.set(0, 3, Color::White);

        // Row 1: `○ ○ ○ ● ● . . .`
        game.set(0, 0, Color::Black);
        game.set(0, 1, Color::Black);
        game.set(0, 2, Color::Black);
        game.set(0, 3, Color::White);
        game.set(0, 4, Color::White);

        // Row 2: `○ ○ ○ ○ ○ ● . .`
        game.set(0, 0, Color::Black);
        game.set(0, 1, Color::Black);
        game.set(0, 2, Color::Black);
        game.set(0, 3, Color::Black);
        game.set(0, 4, Color::Black);
        game.set(0, 5, Color::White);

        // Row 3: `○ ○ ○ ○ ○ ○ ● ●`
        game.set(3, 0, Color::Black);
        game.set(3, 1, Color::Black);
        game.set(3, 2, Color::Black);
        game.set(3, 3, Color::Black);
        game.set(3, 4, Color::Black);
        game.set(3, 5, Color::Black);
        game.set(3, 6, Color::White);
        game.set(3, 7, Color::White);

        // Row 4: `○ ○ ○ ● ○ ○ ○ ●`
        game.set(4, 0, Color::Black);
        game.set(4, 1, Color::Black);
        game.set(4, 2, Color::Black);
        game.set(4, 3, Color::White);
        game.set(4, 4, Color::Black);
        game.set(4, 5, Color::Black);
        game.set(4, 6, Color::Black);
        game.set(4, 7, Color::White);

        // Row 5: `○ ○ ○ ○ ○ ○ ○ ●`
        game.set(5, 0, Color::Black);
        game.set(5, 1, Color::Black);
        game.set(5, 2, Color::Black);
        game.set(5, 3, Color::Black);
        game.set(5, 4, Color::Black);
        game.set(5, 5, Color::Black);
        game.set(5, 6, Color::Black);
        game.set(5, 7, Color::White);

        // Row 6: `○ ○ ○ ○ ○ ○ . ●`
        game.set(6, 0, Color::Black);
        game.set(6, 1, Color::Black);
        game.set(6, 2, Color::Black);
        game.set(6, 3, Color::Black);
        game.set(6, 4, Color::Black);
        game.set(6, 5, Color::Black);
        game.set(6, 7, Color::White);

        // Row 7: `○ . . . . . . .`
        game.set(7, 0, Color::Black);

        // Verify initial state
        assert_eq!(game.current_turn, Color::White);

        // Verify (7,1) is a legal move
        let legal_moves = game.legal_moves(Color::White);
        assert!(
            !legal_moves.contains(&(7, 1)),
            "Position (7,1) should not be a legal move for White"
        );
    }
}
