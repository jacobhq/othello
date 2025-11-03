use wasm_bindgen::prelude::*;
use serde::Serialize;
use othello::othello_game::{OthelloGame,Color,OthelloError};

/// JS-facing wrapper around the core Rust OthelloGame.
#[wasm_bindgen]
pub struct WasmGame {
    inner: OthelloGame,
}

#[wasm_bindgen]
impl WasmGame {
    /// Create a new standard Othello board.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGame {
        WasmGame { inner: OthelloGame::new() }
    }

    /// Play a move (row, col, player = 1 for Black, 2 for White).
    pub fn play_turn(&mut self, row: usize, col: usize, player: u8) -> Result<(), JsValue> {
        let color = match player {
            1 => Color::Black,
            2 => Color::White,
            _ => return Err(JsValue::from_str("Player out of range"))
        };

        match self.inner.play(row, col, color) {
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