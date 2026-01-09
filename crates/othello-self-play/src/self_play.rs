use crate::mcts::mcts_search;
use crate::neural_net::load_model;
use ort::session::Session;
use othello::othello_game::{Color, OthelloGame};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Represents a single self-play game
#[derive(Clone)]
pub struct Sample {
    pub state: [[[i32; 8]; 8]; 2], // encoded board
    pub policy: Vec<f32>,          // length 64
    pub value: f32,                // final game result
}

/// Plays a single self-play game and collects training samples.
///
/// This function simulates a complete game of Othello using Monte Carlo Tree Search (MCTS)
/// to select moves for both players. At each turn, the current game state is encoded and
/// a training [`Sample`] is recorded containing the state, the MCTS policy, and a placeholder
/// value. After the game ends, the final outcome is used to assign a value to each sample
/// from the perspective of the player who made the move.
///
/// # Parameters
///
/// * `iterations` - The number of MCTS iterations to perform per move.
/// * `model` - Optional mutable reference to a neural network [`Session`] used to guide
///   the MCTS. If `None`, MCTS runs without a model.
///
/// # Returns
///
/// Returns a vector of [`Sample`] values corresponding to each move played in the game.
/// The `value` field of each sample is assigned as:
/// - `1.0` if the player eventually won the game
/// - `-1.0` if the player eventually lost the game
/// - `0.0` in the case of a draw
///
/// # Notes
///
/// * The provided model (if any) is mutated during search and reused across moves.
/// * Passed turns (when no legal moves are available) are handled explicitly.
pub fn self_play_game(iterations: u32, mut model: Option<&mut Session>) -> Vec<Sample> {
    let mut game = OthelloGame::new();
    let mut samples: Vec<(Sample, Color)> = Vec::new();

    while !game.game_over() {
        let player = game.current_turn;

        let encoded = game.encode(player);

        let (best_move, policy) = mcts_search(game, player, iterations, model.as_deref_mut());

        // Store sample with placeholder value
        samples.push((
            Sample {
                state: encoded,
                policy,
                value: 0.0,
            },
            player,
        ));

        match best_move {
            Some((row, col)) => {
                // Normal move
                let _ = game.play(row, col, player);
            }
            None => {
                // PASS TURN
                game.current_turn = match player {
                    Color::White => Color::Black,
                    Color::Black => Color::White,
                };
            }
        }
    }

    // Game ended — compute final outcome
    let (white, black) = game.score();
    let outcome = if white > black {
        Color::White
    } else if black > white {
        Color::Black
    } else {
        // Draw
        // value = 0 for all
        for (sample, _) in samples.iter_mut() {
            sample.value = 0.0;
        }
        return samples.into_iter().map(|(s, _)| s).collect();
    };

    // Assign values from each player's perspective
    for (sample, player) in samples.iter_mut() {
        sample.value = if *player == outcome { 1.0 } else { -1.0 };
    }

    samples.into_iter().map(|(s, _)| s).collect()
}

/// Generates training samples by running multiple self-play games in parallel.
///
/// This function executes `games` self-play matches using Monte Carlo Tree Search (MCTS),
/// optionally backed by a neural network model. Games are distributed across Rayon worker
/// threads, and if a model path is provided, the model is loaded once per worker thread
/// to avoid repeated initialisation overhead.
///
/// # Parameters
///
/// * `games` - The number of self-play games to run.
/// * `mcts_iters` - The number of MCTS iterations to perform per move.
/// * `model_path` - Optional path to a serialized model used to guide MCTS. If `None`,
///   games are played without a model.
///
/// # Returns
///
/// Returns a vector of [`Sample`] values collected from all self-play games on success.
/// If an error occurs during execution, an [`anyhow::Error`] is returned.
///
/// # Parallelism
///
/// This function uses Rayon to parallelise self-play across worker threads. Each worker
/// loads its own copy of the model (if provided) and reuses it for all games assigned
/// to that thread.
pub fn generate_self_play_data(
    games: usize,
    mcts_iters: u32,
    model_path: Option<PathBuf>,
) -> anyhow::Result<Vec<Sample>> {
    let samples = (0..games)
        .into_par_iter()
        .map_init(
            || {
                // This runs ONCE per Rayon worker thread
                match &model_path {
                    Some(path) => {
                        info!("Loading model on worker from {:?}", path);
                        Some(load_model(path.to_str().unwrap()).unwrap())
                    }
                    None => None,
                }
            },
            |model, g| {
                debug!("Starting self-play game {}", g);

                let samples = self_play_game(mcts_iters, model.as_mut());

                debug!("Completed {} self-play games", g);

                samples
            },
        )
        .flatten()
        .collect();

    Ok(samples)
}
