mod mcts;
mod model;
mod neural_net;

use crate::mcts::mcts_search;
use crate::neural_net::{ModelType, NeuralNet};
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Device;
use othello::othello_game::{Color, Move, OthelloError, OthelloGame};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    // print pretty errors in wasm https://github.com/rustwasm/console_error_panic_hook
    // This is not needed for tracing_wasm to work, but it is a common tool for getting proper error line numbers for panics.
    console_error_panic_hook::set_once();

    wasm_tracing::set_as_global_default();

    Ok(())
}

#[wasm_bindgen]
#[repr(u8)]
#[derive(Eq, PartialEq)]
pub enum GameType {
    PassAndPlay = 1,
    PlayerVsModel = 2,
}

#[wasm_bindgen]
#[repr(u8)]
pub enum DeviceType {
    Ndarray = 1,
    WebGpu = 2,
}

/// JS-facing wrapper around the core Rust OthelloGame.
#[wasm_bindgen]
pub struct WasmGame {
    pub(crate) inner: OthelloGame,
    game_type: GameType,
    pub(crate) model: Option<ModelType>,
    human_player: Option<Color>,
}

#[wasm_bindgen]
impl WasmGame {
    /// Create a new standard Othello board.
    #[wasm_bindgen(constructor)]
    pub fn new(game_type: u8, device_type: Option<u8>) -> Result<WasmGame, JsValue> {
        let game_type = match game_type {
            1 => GameType::PassAndPlay,
            2 => GameType::PlayerVsModel,
            _ => return Err(JsValue::from_str("GameType out of range")),
        };

        let device_type = match device_type {
            Some(1) => Some(DeviceType::Ndarray),
            Some(2) => Some(DeviceType::WebGpu),
            Some(_) => return Err(JsValue::from_str("DeviceType out of range")),
            None => None,
        };

        if device_type.is_none() && game_type == GameType::PlayerVsModel {
            return Err(JsValue::from_str(
                "You must specify a DeviceType when playing in PlayerVsModel mode",
            ));
        }

        let model = if game_type == GameType::PlayerVsModel {
            Some(match &device_type.unwrap() {
                DeviceType::Ndarray => {
                    ModelType::WithNdArrayBackend(NeuralNet::new(&Default::default()))
                }
                DeviceType::WebGpu => {
                    ModelType::WithWgpuBackend(NeuralNet::new(&WgpuDevice::default()))
                }
            })
        } else {
            None
        };

        Ok(WasmGame {
            inner: OthelloGame::new(),
            model,
            game_type,
            human_player: Some(Color::Black),
        })
    }

    /// Creates a new Othello board from a black and a white bitboard, and sets the current turn.
    #[wasm_bindgen]
    pub fn new_from_state(
        black: u64,
        white: u64,
        player: u8,
        game_type: u8,
        device_type: Option<u8>,
    ) -> Result<WasmGame, JsValue> {
        let color = match player {
            1 => Color::Black,
            2 => Color::White,
            _ => return Err(JsValue::from_str("Player out of range")),
        };

        let game_type = match game_type {
            1 => GameType::PassAndPlay,
            2 => GameType::PlayerVsModel,
            _ => return Err(JsValue::from_str("GameType out of range")),
        };

        let device_type = match device_type {
            Some(1) => Some(DeviceType::Ndarray),
            Some(2) => Some(DeviceType::WebGpu),
            Some(_) => return Err(JsValue::from_str("DeviceType out of range")),
            None => None,
        };

        if device_type.is_none() && game_type == GameType::PlayerVsModel {
            return Err(JsValue::from_str(
                "You must specify a DeviceType when playing in PlayerVsModel mode",
            ));
        }

        Ok(WasmGame {
            inner: OthelloGame::new_with_state(black, white, color),
            model: None,
            game_type,
            human_player: Some(color),
        })
    }

    /// Play a move (row, col, player = 1 for Black, 2 for White).
    pub fn play_turn(&mut self, row: usize, col: usize, player: u8) -> Result<(), JsValue> {
        let color = match player {
            1 => Color::Black,
            2 => Color::White,
            _ => return Err(JsValue::from_str("Player out of range")),
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

    pub async fn play_ai_move(&mut self) -> Result<(), JsValue> {
        if self.game_type != GameType::PlayerVsModel || self.model.is_none() {
            return Err(JsValue::from_str("This is not an AI game"));
        }

        if self.inner.current_turn == self.human_player.unwrap() {
            return Err(JsValue::from_str("It's not the AI's turn"));
        }

        let ai_move = match self.model.as_ref().unwrap() {
            ModelType::WithNdArrayBackend(model) => {
                mcts_search(model, &self.inner, self.inner.current_turn, 800).await
            }
            ModelType::WithWgpuBackend(model) => {
                mcts_search(model, &self.inner, self.inner.current_turn, 800).await
            }
        };

        let player = self.inner.current_turn;

        let mv = match ai_move {
            Some((r, c)) => Move::Move(r, c),
            None => Move::Pass,
        };

        self.inner.mcts_play(mv, player).map_err(|e| match e {
            OthelloError::NoMovesForPlayer => JsValue::from_str("AI has no legal moves"),
            OthelloError::NotYourTurn => JsValue::from_str("It's not the AI's turn"),
            OthelloError::IllegalMove => JsValue::from_str("AI attempted an illegal move"),
        })
    }
}
