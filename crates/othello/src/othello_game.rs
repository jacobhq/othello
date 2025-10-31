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
            (1, NOT_H_FILE),                    // east
            (-1, NOT_A_FILE),                   // west
            (9, NOT_H_FILE),                    // northeast
            (7, NOT_A_FILE),                    // northwest
            (-7, NOT_H_FILE),                   // southeast
            (-9, NOT_A_FILE),                   // southwest
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

    /// Play a move for the given player. Returns false if illegal.
    pub fn play(&mut self, row: usize, col: usize, player: Color) -> bool {
        if self.legal_moves_mask(self.current_turn) == BitBoard(0) {
            let next = match self.current_turn {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
            self.current_turn = next;
            return false;
        }
        let move_mask: BitBoard = BitBoard(BitBoard::mask(row, col));
        if self.current_turn != player {
            return false;
        }
        if self.legal_moves_mask(player) & move_mask == BitBoard(0) {
            self.current_turn = player;
            return false;
        }

        let (mut me, mut opp) = match player {
            Color::Black => (self.black, self.white),
            Color::White => (self.white, self.black),
        };

        let mut flips: BitBoard = BitBoard(0);

        let directions: [(i64, BitBoard); 8] = [
            (8, BitBoard(0xffffffffffffffffu64)),
            (-8, BitBoard(0xffffffffffffffffu64)),
            (1, NOT_H_FILE),
            (-1, NOT_A_FILE),
            (9, NOT_H_FILE),
            (7, NOT_A_FILE),
            (-7, NOT_H_FILE),
            (-9, NOT_A_FILE),
        ];

        for (shift, mask) in directions {
            let mut captured = BitBoard(0);
            let mut candidate = match shift {
                s if s > 0 => (move_mask << s) & opp & mask,
                s if s < 0 => (move_mask >> -s) & opp & mask,
                _ => BitBoard(0),
            };

            while candidate != BitBoard(0) {
                captured |= candidate;
                let next = match shift {
                    s if s > 0 => (candidate << s) & mask,
                    s if s < 0 => (candidate >> -s) & mask,
                    _ => BitBoard(0),
                };

                if next & opp != BitBoard(0) {
                    candidate = next & opp;
                    continue;
                } else if next & me != BitBoard(0) {
                    flips |= captured;
                    break;
                } else {
                    break;
                }
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

        // *** New code to update current_turn ***
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

        true
    }

    pub fn score(&self) -> (u32, u32) {
        (self.white.0.count_ones(), self.black.0.count_ones())
    }

    pub fn game_over(&self) -> bool {
        self.legal_moves_mask(Color::Black) == BitBoard(0) && self.legal_moves_mask(Color::White) == BitBoard(0)
    }

    pub fn encode(&self, player: Color) -> [[ [i32; 8]; 8]; 2] {
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
}

impl std::fmt::Display for OthelloGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  0 1 2 3 4 5 6 7")?;
        for row in 0..8 {
            write!(f, "{} ", row)?;
            for col in 0..8 {
                let m = BitBoard::mask(row, col);
                if self.black & BitBoard(m) != BitBoard(0) {
                    write!(f, "● ")?;
                } else if self.white & BitBoard(m) != BitBoard(0) {
                    write!(f, "○ ")?;
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

    #[test]
    fn test_play_valid_move_and_flips() {
        let mut game = OthelloGame::new();
        // Black plays at (2,3) — should flip (3,3)
        assert!(game.play(2, 3, Color::Black));
        assert_eq!(game.get(2, 3), Some(Color::Black));
        assert_eq!(game.get(3, 3), Some(Color::Black));
    }

    #[test]
    fn test_play_illegal_move_outside_legal_moves() {
        let mut game = OthelloGame::new();
        // (0,0) is not a legal move initially
        assert!(!game.play(0, 0, Color::Black));
    }

    #[test]
    fn test_play_illegal_wrong_turn() {
        let mut game = OthelloGame::new();
        // It’s Black’s turn initially, so White can’t play
        assert!(!game.play(2, 4, Color::White));
    }

    #[test]
    fn test_play_no_flips_does_not_change_board() {
        let mut game = OthelloGame::new();
        let before_black = game.black;
        let before_white = game.white;
        // Even if it’s Black’s turn, (0,0) won’t flip any pieces
        assert!(!game.play(0, 0, Color::Black));
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
        assert!(game.play(3, 5, Color::Black));
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
            black: BitBoard(0b10111111_10000001_10000001_10000001_10000001_10000001_10000001_01111111),
            white: BitBoard(0b01000000_01111110_01111110_01111110_01111110_01111110_01111110_00000000),
            current_turn: Color::White,
        };

        println!("{}", game);

        assert_eq!(game.current_turn, Color::White);
        let turn_one = game.play(0, 0, Color::White);
        assert!(!turn_one, "Move should be illegal");

        assert_eq!(game.current_turn, Color::Black);
        let turn_two = game.play(0, 7, Color::Black);
        assert!(turn_two, "Move should be legal");

        assert_eq!(game.get(0, 7), Some(Color::Black));
        assert_eq!(
            game.current_turn,
            Color::Black,
            "White has no legal moves, turn should remain Black"
        );

        assert!(game.legal_moves(Color::White).is_empty());
    }

}
