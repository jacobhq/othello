use othello::othello_game::{Color, Move, OthelloError, OthelloGame};

/// A trait was used which will allow this system to work very easily for other game
/// implementations.
pub trait Game: Clone + Send + Sync + 'static {
    /// Player to move
    fn current_player(&self) -> Color;

    /// Legal moves for the current player
    fn legal_moves(&self) -> Vec<(usize, usize)>;

    /// Apply a move for the current player
    fn play_move(&mut self, m: Move) -> Result<(), OthelloError>;

    /// Is the game over?
    fn is_terminal(&self) -> bool;

    /// Terminal value from the *root player*'s perspective
    fn terminal_value(&self, root_player: Color) -> f32;

    /// Encode state for NN from a given player's perspective
    fn encode(&self, player: Color) -> Vec<f32>;
}

impl Game for OthelloGame {
    fn current_player(&self) -> Color {
        self.current_turn
    }

    fn legal_moves(&self) -> Vec<(usize, usize)> {
        self.legal_moves(self.current_turn)
    }

    fn play_move(&mut self, m: Move) -> Result<(), OthelloError> {
        self.mcts_play(m, self.current_turn)
    }

    fn is_terminal(&self) -> bool {
        self.game_over()
    }

    /// 'If the game ended now, who would win?'
    fn terminal_value(&self, root_player: Color) -> f32 {
        // Note: score() returns (white_count, black_count)
        let (white, black) = self.score();

        let diff = match root_player {
            Color::Black => black as i32 - white as i32,
            Color::White => white as i32 - black as i32,
        };

        // Convert the score difference into a terminal value from the root player's perspective.
        //  1.0  -> root player is winning
        // -1.0  -> root player is losing
        //  0.0  -> draw (equal score)
        if diff > 0 {
            1.0
        } else if diff < 0 {
            -1.0
        } else {
            0.0
        }
    }

    /// Flatten the game into a vector
    fn encode(&self, player: Color) -> Vec<f32> {
        let planes = self.encode(player);

        // Flatten [[[i32;8];8];2] → Vec<f32>
        let mut out = Vec::with_capacity(2 * 8 * 8);
        for p in 0..2 {
            for r in 0..8 {
                for c in 0..8 {
                    out.push(planes[p][r][c] as f32);
                }
            }
        }
        out
    }
}
