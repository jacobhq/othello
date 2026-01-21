//! Neural network inference utilities for Othello self-play.
//!
//! This module provides helpers for loading an ONNX neural network model
//! and evaluating Othello positions using that model.
//!
//! The neural network is assumed to follow an AlphaZero-style interface:
//!
//! - **Input**: a `(1, 2, 8, 8)` tensor representing the board state from the
//!   current player's perspective
//!     - Channel 0: current player's stones
//!     - Channel 1: opponent's stones
//! - **Outputs**:
//!     1. A policy vector of length 64, giving a score or probability for
//!        placing a stone on each board square
//!     2. A scalar value estimating the position outcome from the current
//!        player's perspective
//!
//! The policy output is filtered to include only legal moves before being
//! returned.

use ort::ep::CUDA;
use ort::session::Session;
use ort::value::Tensor;
use ort::Error;
use othello::othello_game::{Color, OthelloGame};

/// Standardised way to load the model during self-play iterations
pub(crate) fn load_model(path: &str) -> Result<Session, Error> {
    ort::init().with_execution_providers([CUDA::default().build().error_on_failure()]).commit();

    let model = Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_file(path)?;

    Ok(model)
}

/// An element in the policy vector in form ((row, col), probability)
pub(crate) type PolicyElement = ((usize, usize), f32);

/// A tuple containing the policy vector, and an evaluation scalar
pub(crate) type PolicyVectorWithEvaluation = (Vec<PolicyElement>, f32);

/// Evaluates an Othello position using a neural network.
///
/// This function encodes the given game state into a neural network input
/// tensor, runs inference using the provided ONNX model, and returns:
///
/// - A policy over *legal moves only*, pairing each legal `(row, col)` move
///   with its corresponding probability or score from the network
/// - A scalar value estimating the position from the current player's
///   perspective
///
/// # Arguments
///
/// * `model` - A mutable reference to an initialised ONNX `Session` used to
///   perform inference
/// * `game` - The current `OthelloGame` position to evaluate
/// * `player` - The player for whom the evaluation is performed; the board
///   is encoded relative to this player
///
/// # Returns
///
/// On success, returns a `PolicyVectorWithEvaluation`
///
/// If inference or tensor extraction fails, an `ort::Error` is returned.
///
/// # Neural Network Assumptions
///
/// The model is expected to output:
///
/// 1. A policy tensor with 64 elements, corresponding to board positions
///    indexed by `row * 8 + col`
/// 2. A scalar value tensor representing the position evaluation
pub(crate) fn nn_eval(
    model: &mut Session,
    game: &OthelloGame,
    player: Color,
) -> Result<PolicyVectorWithEvaluation, Error> {
    // Encode the board state into a (1, 2, 8, 8) tensor from the current player's
    // perspective:
    //   - Channel 0: current player's stones
    //   - Channel 1: opponent's stones
    let mut input: Tensor<f32> = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 2, 8, 8)))?;

    // Build the input tensor by iterating over the board, and marking squares in correct channel
    for row in 0..8 {
        for col in 0..8 {
            if let Some(c) = game.get(row, col) {
                // Map stones to channels relative to the evaluating player
                match (player, c) {
                    (Color::White, Color::White) | (Color::Black, Color::Black) => {
                        // Safe to dangerously cast here, because i64 can represent all of 0..8
                        input[[0, 0, row as i64, col as i64]] = 1.0;
                    }
                    (Color::White, Color::Black) | (Color::Black, Color::White) => {
                        // Safe to dangerously cast here, because i64 can represent all of 0..8
                        input[[0, 1, row as i64, col as i64]] = 1.0;
                    }
                }
            }
        }
    }

    // Run neural network inference; expects policy and value outputs
    let outputs = model.run(ort::inputs!(input))?;
    let policy: Vec<f32> = outputs[0]
        .try_extract_array::<f32>()?
        .iter()
        .copied()
        .collect();
    let value_array = outputs[1].try_extract_tensor::<f32>()?;
    let value = value_array.1[0]; // Access the first element of the [1, 1] shape

    // Filter the policy to include only legal moves
    let legal = game.legal_moves(player);
    let mut move_probs = Vec::new();
    for (row, col) in legal {
        // Convert (row, col) into a flat policy index
        let idx = row * 8 + col;
        move_probs.push(((row, col), policy[idx]));
    }

    Ok((move_probs, value))
}

/// Batched neural network evaluation for multiple Othello positions
///
/// `games` - slice of game states to evaluate
/// `players` - slice of corresponding players
/// Returns a Vec of (policy vector, value) tuples
pub fn nn_eval_batch(
    model: &mut Session,
    states: &[Vec<f32>], // flattened state vectors
) -> Result<Vec<(Vec<PolicyElement>, f32)>, Error> {
    let batch_size = states.len();

    let mut input_tensor: Tensor<f32> =
        Tensor::from_array(ndarray::Array4::<f32>::zeros((batch_size, 2, 8, 8)))?;

    for (i, state) in states.iter().enumerate() {
        assert_eq!(state.len(), 128, "state must be 2*8*8 flattened");

        for row in 0..8 {
            for col in 0..8 {
                let idx_black = row * 8 + col;
                let idx_white = 64 + idx_black;
                input_tensor[[i as i64, 0, row as i64, col as i64]] = state[idx_black];
                input_tensor[[i as i64, 1, row as i64, col as i64]] = state[idx_white];
            }
        }
    }

    let outputs = model.run(ort::inputs!(input_tensor))?;

    let policy_tensor_dyn = outputs[0].try_extract_array::<f32>()?;
    let value_tensor_dyn = outputs[1].try_extract_array::<f32>()?;

    let policy_tensor = policy_tensor_dyn.to_shape((batch_size, 64)).unwrap();
    let value_tensor = value_tensor_dyn.to_shape((batch_size, 1)).unwrap();

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let policy_flat: Vec<f32> = policy_tensor
            .row(i)
            .iter()
            .map(|&logp| logp.exp())
            .collect();
        
        let move_probs: Vec<PolicyElement> = policy_flat
            .into_iter()
            .enumerate()
            .map(|(idx, p)| ( (idx / 8, idx % 8), p ))
            .collect();

        let value = value_tensor[[i, 0]];
        results.push((move_probs, value));
    }

    Ok(results)
}
