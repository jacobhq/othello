mod neural_net;

use othello::othello_game::{Color, OthelloError, OthelloGame};
use wasm_bindgen::prelude::*;

/// Game config exposed to wasm
#[wasm_bindgen]
pub struct GameConfig {
    mode: GameMode,
}

#[wasm_bindgen]
impl GameConfig {
    pub fn pass_and_play() -> GameConfig {
        GameConfig {
            mode: GameMode::PassAndPlay,
        }
    }

    pub fn neural_net(player: Player) -> GameConfig {
        GameConfig {
            mode: GameMode::NeuralNet(player.into()),
        }
    }

    pub fn is_neural_net(&self) -> bool {
        matches!(self.mode, GameMode::NeuralNet(_))
    }

    pub fn neural_net_player(&self) -> Option<Player> {
        match self.mode {
            GameMode::NeuralNet(p) => Some(p.into()),
            _ => None,
        }
    }
}


/// Player represents the color as a u8
#[wasm_bindgen]
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Player {
    Black = 1,
    White = 2,
}

impl From<Player> for Color {
    // Doing this gives us the into method on Player
    fn from(value: Player) -> Self {
        match value {
            Player::Black => Color::Black,
            Player::White => Color::White
        }
    }
}

impl From<Color> for Player {
    fn from(value: Color) -> Self {
        match value {
            Color::Black => Player::Black,
            Color::White => Player::White
        }
    }
}

/// Enum to represent the game mode
pub enum GameMode {
    PassAndPlay,
    NeuralNet(Color),
}

/// JS-facing wrapper around the core Rust OthelloGame.
#[wasm_bindgen]
pub struct WasmGame {
    inner: OthelloGame,
    game_mode: GameMode,
}

#[wasm_bindgen]
impl WasmGame {
    /// Create a new standard Othello board.
    ///
    /// The player is not passed as a string due to performance reasons
    /// Use player = 1 for Black, player 2 for White
    #[wasm_bindgen(constructor)]
    pub fn new(game_config: GameConfig) -> WasmGame {
        WasmGame {
            inner: OthelloGame::new(),
            game_mode: game_config.mode,
        }
    }

    /// Creates a new Othello board from a black and a white bitboard, and sets the current turn.
    #[wasm_bindgen]
    pub fn new_from_state(
        black: u64,
        white: u64,
        player: Player,
        game_config: GameConfig,
    ) -> Result<WasmGame, JsValue> {
        Ok(WasmGame {
            inner: OthelloGame::new_with_state(black, white, player.into()),
            game_mode: game_config.mode,
        })
    }

    /// Play a move (row, col, player = 1 for Black, 2 for White).
    pub fn play_turn(&mut self, row: usize, col: usize, player: Player) -> Result<(), JsValue> {
        match self.game_mode {
            GameMode::PassAndPlay => self.human_play_turn(row, col, player),
            GameMode::NeuralNet(net_color) => match self.inner.current_turn {
                color if color == net_color => Err(JsValue::from_str("It's not your turn")),
                _ => self.human_play_turn(row, col, player)
            }
        }
    }

    /// Human player to move, should only be used when we are taking a move from the user
    fn human_play_turn(&mut self, row: usize, col: usize, player: Player) -> Result<(), JsValue> {
        match self.inner.play(row, col, player.into()) {
            Ok(()) => Ok(()),
            Err(OthelloError::NoMovesForPlayer) => Err(JsValue::from_str("You have no moves")),
            Err(OthelloError::NotYourTurn) => Err(JsValue::from_str("It's not your turn")),
            Err(OthelloError::IllegalMove) => Err(JsValue::from_str("Illegal move")),
        }
    }

    /// Get the board as a 2D array:
    /// 0 = empty, 1 = black, 2 = white.
    pub fn board(&self) -> JsValue {
        let board: Vec<Vec<u8>> = (0..8)
            .map(|row| {
                (0..8)
                    .map(|col| match self.inner.get(row, col) {
                        Some(Color::Black) => 1,
                        Some(Color::White) => 2,
                        None => 0,
                    })
                    .collect()
            })
            .collect();

        serde_wasm_bindgen::to_value(&board).unwrap()
    }

    /// Return all legal moves as Vec<[row, col]> pairs.
    pub fn legal_moves(&self) -> JsValue {
        let moves: Vec<[u8; 2]> = self
            .inner
            .legal_moves(self.inner.current_turn)
            .iter()
            .map(|(r, c)| [*r as u8, *c as u8])
            .collect();

        serde_wasm_bindgen::to_value(&moves).unwrap()
    }

    /// Return 1 for Black, 2 for White (whose turn it is).
    pub fn current_player(&self) -> u8 {
        match self.inner.current_turn {
            Color::Black => 1,
            Color::White => 2,
        }
    }

    /// Return true if no legal moves remain.
    pub fn game_over(&self) -> bool {
        self.inner.game_over()
    }

    /// Return the current (white, black) score as [white, black].
    pub fn score(&self) -> Vec<u32> {
        let (w, b) = self.inner.score();
        vec![w, b]
    }
}
