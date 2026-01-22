//! Model evaluation through head-to-head matches.
//!
//! Plays games between two models to measure relative playing strength.

use std::path::PathBuf;
use std::thread;

use crate::async_mcts::search::SearchWorker;
use crate::async_mcts::tree::Tree;
use crate::eval_queue::{EvalQueue, GpuHandle};
use crate::neural_net::{load_model, nn_eval_batch};
use anyhow::Result;
use othello::othello_game::{Color, Move, OthelloGame};
use tracing::{debug, info};

/// Result of an evaluation match
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub new_wins: u32,
    pub old_wins: u32,
    pub draws: u32,
    pub total_games: u32,
}

impl MatchResult {
    pub fn win_rate(&self) -> f64 {
        if self.total_games == 0 {
            return 0.5;
        }
        (self.new_wins as f64 + 0.5 * self.draws as f64) / self.total_games as f64
    }
}

impl std::fmt::Display for MatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "New: {} wins, Old: {} wins, Draws: {} (win rate: {:.1}%)",
            self.new_wins,
            self.old_wins,
            self.draws,
            self.win_rate() * 100.0
        )
    }
}

/// Play a single evaluation game between two models.
/// Returns 1 if new_model wins, -1 if old_model wins, 0 for draw.
fn play_eval_game(
    new_queue: EvalQueue,
    old_queue: EvalQueue,
    sims_per_move: u32,
    new_plays_black: bool,
) -> Result<i32> {
    let mut game = OthelloGame::new();

    while !game.game_over() {
        let current_player = game.current_turn;

        // Determine which model to use based on color
        let is_new_model = (current_player == Color::Black) == new_plays_black;
        let eval_queue = if is_new_model { &new_queue } else { &old_queue };

        // Run MCTS
        let tree = Tree::new();
        let search_handle = eval_queue.search_handle();

        // Expand root
        {
            let mut init_worker = SearchWorker::new(tree.clone(), search_handle.clone());
            init_worker.simulate(&game);
            while init_worker.has_pending() {
                init_worker.poll_results();
                std::thread::yield_now();
            }
        }

        // Run remaining simulations (no Dirichlet noise for evaluation)
        {
            let mut worker = SearchWorker::new(tree.clone(), search_handle.clone());
            for _ in 1..sims_per_move {
                worker.simulate(&game);
                worker.poll_results();
            }
            while worker.has_pending() {
                worker.poll_results();
                std::thread::yield_now();
            }
        }

        // Select best move (greedy, not sampled)
        let visits = tree.child_visits(tree.root());
        let legal_moves = game.legal_moves(current_player);

        if legal_moves.is_empty() {
            game.mcts_play(Move::Pass, current_player).unwrap();
        } else {
            // Find move with most visits
            let best_move = visits
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(action, _)| (action / 8, action % 8));

            if let Some((r, c)) = best_move {
                game.mcts_play(Move::Move(r, c), current_player).unwrap();
            } else {
                // Fallback to first legal move
                let (r, c) = legal_moves[0];
                game.mcts_play(Move::Move(r, c), current_player).unwrap();
            }
        }
    }

    // Determine winner
    let (black_score, white_score) = game.score();
    let black_wins = black_score > white_score;
    let white_wins = white_score > black_score;

    // Convert to new model perspective
    let result = if black_wins {
        if new_plays_black { 1 } else { -1 }
    } else if white_wins {
        if new_plays_black { -1 } else { 1 }
    } else {
        0
    };

    Ok(result)
}

/// Start a GPU worker thread for evaluation
fn start_eval_gpu_worker(
    gpu: GpuHandle,
    model_path: PathBuf,
    max_batch_size: usize,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut model = load_model(model_path.to_str().unwrap()).expect("failed to load model");

        loop {
            let batch = gpu.pop_batch(max_batch_size);

            if batch.is_empty() {
                std::thread::yield_now();
                continue;
            }

            let states: Vec<Vec<f32>> = batch.iter().map(|req| req.state.clone()).collect();
            let ids: Vec<u64> = batch.iter().map(|req| req.id).collect();

            let evals = match nn_eval_batch(&mut model, &states) {
                Ok(evals) => evals,
                Err(e) => {
                    debug!("Eval GPU worker: batch evaluation failed: {:?}", e);
                    return;
                }
            };

            let mut results = Vec::with_capacity(evals.len());
            for ((policy_map, value), id) in evals.into_iter().zip(ids) {
                let policy_flat: Vec<(usize, f32)> = policy_map
                    .into_iter()
                    .map(|((r, c), p)| (r * 8 + c, p))
                    .collect();

                results.push(crate::eval_queue::EvalResult {
                    id,
                    policy: policy_flat,
                    value,
                });
            }

            gpu.push_results(results);
        }
    })
}

/// Run evaluation matches between two models.
///
/// Plays `num_games` games, alternating which model plays black.
/// Uses fewer simulations than training for speed.
pub fn evaluate_models(
    new_model: PathBuf,
    old_model: PathBuf,
    num_games: u32,
    sims_per_move: u32,
) -> Result<MatchResult> {
    info!(
        "Evaluating models: {} vs {} ({} games, {} sims/move)",
        new_model.display(),
        old_model.display(),
        num_games,
        sims_per_move
    );

    // Create separate eval queues for each model
    let new_queue = EvalQueue::new();
    let old_queue = EvalQueue::new();

    // Start GPU workers for both models
    let _new_gpu = start_eval_gpu_worker(new_queue.gpu_handle(), new_model, 64);
    let _old_gpu = start_eval_gpu_worker(old_queue.gpu_handle(), old_model, 64);

    // Play games sequentially (simpler than parallel for eval)
    let mut new_wins = 0u32;
    let mut old_wins = 0u32;
    let mut draws = 0u32;

    for game_idx in 0..num_games {
        let new_plays_black = game_idx % 2 == 0;

        let result = play_eval_game(
            new_queue.clone(),
            old_queue.clone(),
            sims_per_move,
            new_plays_black,
        )?;

        match result {
            1 => new_wins += 1,
            -1 => old_wins += 1,
            _ => draws += 1,
        }

        if (game_idx + 1) % 10 == 0 || game_idx + 1 == num_games {
            info!(
                "Eval progress: {}/{} games (new: {}, old: {}, draws: {})",
                game_idx + 1,
                num_games,
                new_wins,
                old_wins,
                draws
            );
        }
    }

    let result = MatchResult {
        new_wins,
        old_wins,
        draws,
        total_games: num_games,
    };

    info!("Evaluation complete: {}", result);

    Ok(result)
}
