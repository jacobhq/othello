use crate::mcts::mcts_search;
use crate::neural_net::load_model;
use othello::othello_game::{Color, OthelloGame};
use rayon::prelude::*;
use std::path::PathBuf;
use serde::Serialize;
use tracing::{debug, info};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use crate::eval_queue::{gpu_worker, EvalQueue};
use crate::symmetry::get_symmetries;

/// Represents a single self-play game
#[derive(Clone)]
pub struct Sample {
    pub state: [[[i32; 8]; 8]; 2], // encoded board
    pub policy: Vec<f32>,          // length 64
    pub value: f32,                // final game result
}

#[derive(Clone, Serialize)]
pub struct MctsStats {
    pub game_id: usize,
    pub move_idx: usize,
    pub entropy: f32,
    pub max_visit_frac: f32,
    pub q_selected: f32,
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
pub fn self_play_game(game_id: usize, iterations: u32, eval_queue: Option<Arc<EvalQueue>>) -> (Vec<Sample>, Vec<MctsStats>) {
    let mut game = OthelloGame::new();
    let mut samples: Vec<(Sample, Color)> = Vec::new();
    let mut stats: Vec<MctsStats> = Vec::new();
    let mut move_idx = 0;

    while !game.game_over() {
        let player = game.current_turn;

        let encoded = game.encode(player);

        let (best_move, policy, mcts_stats_opt) =
            mcts_search(game, player, iterations, eval_queue.clone());

        if let Some((entropy, max_visit_frac, q_selected)) = mcts_stats_opt {
            stats.push(MctsStats {
                game_id,
                move_idx,
                entropy,
                max_visit_frac,
                q_selected,
            });
        }

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

        move_idx += 1;
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
        return (samples.into_iter().map(|(s, _)| s).collect(), stats);
    };

    // Assign values from each player's perspective
    for (sample, player) in samples.iter_mut() {
        sample.value = if *player == outcome { 1.0 } else { -1.0 };
    }

    (samples.into_iter().map(|(s, _)| s).collect(), stats)
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
    prefix: &String,
    games: usize,
    mcts_iters: u32,
    model_path: Option<PathBuf>,
) -> anyhow::Result<Vec<Sample>> {
    // Create shared evaluation queue if we are using a model
    let eval_queue = model_path.as_ref().map(|_| Arc::new(EvalQueue::new()));

    // Spawn GPU worker thread(s)
    if let (Some(queue), Some(path)) = (eval_queue.clone(), model_path.as_ref()) {
        let num_gpu_workers = 1; // start with 1, increase if GPU allows
        let batch_size = 256;     // tune for your GPU

        for worker_id in 0..num_gpu_workers {
            let q = queue.clone();
            let model_path = path.clone();

            std::thread::spawn(move || {
                info!("Starting GPU worker {}", worker_id);

                let model = load_model(model_path.to_str().unwrap())
                    .expect("Failed to load model");

                gpu_worker(q, model, batch_size);
            });
        }
    }

    // Run self-play games in parallel (Rayon)
    let results: Vec<(Vec<Sample>, Vec<MctsStats>)> = (0..games)
        .into_par_iter()
        .map(|g| {
            debug!("Starting self-play game {}", g);

            let (samples, stats) =
                self_play_game(g, mcts_iters, eval_queue.clone());

            debug!("Finished self-play game {}", g);

            (
                samples
                    .into_iter()
                    .flat_map(get_symmetries)
                    .collect::<Vec<Sample>>(),
                stats,
            )
        })
        .collect();

    // Merge results
    let (all_samples, all_stats): (Vec<_>, Vec<_>) =
        results.into_iter().unzip();

    let samples = all_samples.into_iter().flatten().collect();
    let stats: Vec<MctsStats> = all_stats.into_iter().flatten().collect();

    // Write MCTS stats to disk
    {
        let file = File::create(format!("{}_mcts_stats.jsonl", prefix))?;
        let mut writer = BufWriter::new(file);

        for stat in &stats {
            serde_json::to_writer(&mut writer, stat)?;
            writer.write_all(b"\n")?;
        }

        writer.flush()?;
    }

    Ok(samples)
}
