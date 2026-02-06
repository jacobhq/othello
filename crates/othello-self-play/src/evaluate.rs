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
use serde::Serialize;
use tracing::{debug, info};

/// Result of an evaluation match
#[derive(Debug, Clone, Serialize)]
pub struct MatchResult {
    pub new_wins: u32,
    pub old_wins: u32,
    pub draws: u32,
    pub total_games: u32,
    pub win_rate: f64,
}

impl std::fmt::Display for MatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "New: {} wins, Old: {} wins, Draws: {} (win rate: {:.1}%)",
            self.new_wins,
            self.old_wins,
            self.draws,
            self.win_rate * 100.0
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
                thread::yield_now();
            }
        }

        // Run remaining simulations with pipelining
        // Keep up to MAX_IN_FLIGHT simulations queued, adding more as results come back
        // This keeps GPU fed while tree gets updated from completed simulations
        {
            let mut worker = SearchWorker::new(tree.clone(), search_handle.clone());
            const MAX_IN_FLIGHT: usize = 48;
            let remaining_sims = sims_per_move.saturating_sub(1);
            
            let mut queued = 0u32;
            while queued < remaining_sims || worker.has_pending() {
                // Poll for any completed results first
                worker.poll_results();
                
                // Queue more simulations if we have room and more to do
                while worker.pending_count() < MAX_IN_FLIGHT && queued < remaining_sims {
                    worker.simulate(&game);
                    queued += 1;
                }
                
                // Yield if we're waiting for results
                if worker.has_pending() && queued >= remaining_sims {
                    thread::yield_now();
                }
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
    // Note: score() returns (white_count, black_count)
    let (white_score, black_score) = game.score();

    if black_score > white_score {
        // Black wins
        Ok(if new_plays_black { 1 } else { -1 })
    } else if white_score > black_score {
        // White wins
        Ok(if new_plays_black { -1 } else { 1 })
    } else {
        // Draw
        Ok(0)
    }
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
                thread::sleep(std::time::Duration::from_micros(50));
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
/// Games are played in parallel for faster evaluation.
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
    let _new_gpu = start_eval_gpu_worker(new_queue.gpu_handle(), new_model, 256);
    let _old_gpu = start_eval_gpu_worker(old_queue.gpu_handle(), old_model, 256);

    // Determine parallelism - use available CPU cores
    let num_parallel = num_cpus::get_physical();
    info!("Running {} parallel evaluation games", num_parallel);

    // Use atomic counters for thread-safe updates
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let new_wins = Arc::new(AtomicU32::new(0));
    let old_wins = Arc::new(AtomicU32::new(0));
    let draws = Arc::new(AtomicU32::new(0));
    let completed = Arc::new(AtomicU32::new(0));

    // Spawn game threads
    let mut handles = Vec::new();
    let (tx, rx) = std::sync::mpsc::channel();

    for game_idx in 0..num_games {
        let new_queue = new_queue.clone();
        let old_queue = old_queue.clone();
        let new_wins = Arc::clone(&new_wins);
        let old_wins = Arc::clone(&old_wins);
        let draws = Arc::clone(&draws);
        let completed = Arc::clone(&completed);
        let tx = tx.clone();
        let total = num_games;

        let new_plays_black = game_idx % 2 == 0;

        let handle = thread::spawn(move || {
            let result = play_eval_game(
                new_queue,
                old_queue,
                sims_per_move,
                new_plays_black,
            );

            match result {
                Ok(1) => { new_wins.fetch_add(1, Ordering::Relaxed); }
                Ok(-1) => { old_wins.fetch_add(1, Ordering::Relaxed); }
                Ok(_) => { draws.fetch_add(1, Ordering::Relaxed); }
                Err(e) => { debug!("Eval game error: {:?}", e); }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == total {
                info!(
                    "Eval progress: {}/{} games (new: {}, old: {}, draws: {})",
                    done,
                    total,
                    new_wins.load(Ordering::Relaxed),
                    old_wins.load(Ordering::Relaxed),
                    draws.load(Ordering::Relaxed)
                );
            }
            let _ = tx.send(());
        });

        handles.push(handle);

        // Limit parallelism - wait if we've spawned enough threads
        while handles.len() >= num_parallel {
            let _ = rx.recv();
            handles.retain(|h| !h.is_finished());
        }
    }

    // Wait for all remaining games
    for handle in handles {
        let _ = handle.join();
    }

    let new_w = new_wins.load(Ordering::Relaxed);
    let old_w = old_wins.load(Ordering::Relaxed);
    let draw_count = draws.load(Ordering::Relaxed);
    let win_rate = if num_games == 0 {
        0.5
    } else {
        (new_w as f64 + 0.5 * draw_count as f64) / num_games as f64
    };
    let result = MatchResult {
        new_wins: new_w,
        old_wins: old_w,
        draws: draw_count,
        total_games: num_games,
        win_rate,
    };

    info!("Evaluation complete: {}", result);

    Ok(result)
}

/// Play a single game: model vs true random player.
/// Returns 1 if model wins, -1 if random wins, 0 for draw.
fn play_vs_random_game(
    model_queue: EvalQueue,
    sims_per_move: u32,
    model_plays_black: bool,
) -> Result<i32> {
    use rand::prelude::*;

    let mut game = OthelloGame::new();
    let mut rng = rand::rng();

    while !game.game_over() {
        let current_player = game.current_turn;
        let is_model = (current_player == Color::Black) == model_plays_black;

        let legal_moves = game.legal_moves(current_player);

        if legal_moves.is_empty() {
            game.mcts_play(Move::Pass, current_player).unwrap();
            continue;
        }

        if is_model {
            // Model plays using MCTS
            let tree = Tree::new();
            let search_handle = model_queue.search_handle();

            // Expand root
            {
                let mut init_worker = SearchWorker::new(tree.clone(), search_handle.clone());
                init_worker.simulate(&game);
                while init_worker.has_pending() {
                    init_worker.poll_results();
                    thread::yield_now();
                }
            }

            // Run remaining simulations with pipelining
            {
                let mut worker = SearchWorker::new(tree.clone(), search_handle.clone());
                const MAX_IN_FLIGHT: usize = 48;
                let remaining_sims = sims_per_move.saturating_sub(1);
                
                let mut queued = 0u32;
                while queued < remaining_sims || worker.has_pending() {
                    worker.poll_results();
                    
                    while worker.pending_count() < MAX_IN_FLIGHT && queued < remaining_sims {
                        worker.simulate(&game);
                        queued += 1;
                    }
                    
                    if worker.has_pending() && queued >= remaining_sims {
                        thread::yield_now();
                    }
                }
            }

            // Select best move (greedy)
            let visits = tree.child_visits(tree.root());
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
        } else {
            // Random player: pick uniformly at random from legal moves
            let (r, c) = *legal_moves.choose(&mut rng).unwrap();
            game.mcts_play(Move::Move(r, c), current_player).unwrap();
        }
    }

    // Determine winner
    // Note: score() returns (white_count, black_count)
    let (white_score, black_score) = game.score();

    if black_score > white_score {
        // Black wins
        Ok(if model_plays_black { 1 } else { -1 })
    } else if white_score > black_score {
        // White wins
        Ok(if model_plays_black { -1 } else { 1 })
    } else {
        // Draw
        Ok(0)
    }
}

/// Evaluate a model against a true random player.
///
/// The random player selects uniformly at random from legal moves.
/// This is a sanity check - any trained model should win >90% of games.
pub fn evaluate_vs_random(
    model_path: PathBuf,
    num_games: u32,
    sims_per_move: u32,
) -> Result<MatchResult> {
    info!(
        "Evaluating model {} vs random ({} games, {} sims/move)",
        model_path.display(),
        num_games,
        sims_per_move
    );

    let model_queue = EvalQueue::new();
    let _gpu_worker = start_eval_gpu_worker(model_queue.gpu_handle(), model_path, 256);

    let num_parallel = num_cpus::get_physical();
    info!("Running {} parallel evaluation games", num_parallel);

    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let model_wins = Arc::new(AtomicU32::new(0));
    let random_wins = Arc::new(AtomicU32::new(0));
    let draws = Arc::new(AtomicU32::new(0));
    let completed = Arc::new(AtomicU32::new(0));

    let mut handles = Vec::new();
    let (tx, rx) = std::sync::mpsc::channel();

    for game_idx in 0..num_games {
        let model_queue = model_queue.clone();
        let model_wins = Arc::clone(&model_wins);
        let random_wins = Arc::clone(&random_wins);
        let draws = Arc::clone(&draws);
        let completed = Arc::clone(&completed);
        let tx = tx.clone();
        let total = num_games;

        let model_plays_black = game_idx % 2 == 0;

        let handle = thread::spawn(move || {
            let result = play_vs_random_game(model_queue, sims_per_move, model_plays_black);

            match result {
                Ok(1) => { model_wins.fetch_add(1, Ordering::Relaxed); }
                Ok(-1) => { random_wins.fetch_add(1, Ordering::Relaxed); }
                Ok(_) => { draws.fetch_add(1, Ordering::Relaxed); }
                Err(e) => { debug!("Eval game error: {:?}", e); }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == total {
                info!(
                    "Eval progress: {}/{} games (model: {}, random: {}, draws: {})",
                    done,
                    total,
                    model_wins.load(Ordering::Relaxed),
                    random_wins.load(Ordering::Relaxed),
                    draws.load(Ordering::Relaxed)
                );
            }
            let _ = tx.send(());
        });

        handles.push(handle);

        if handles.len() >= num_parallel {
            let _ = rx.recv();
            handles.retain(|h| !h.is_finished());
        }
    }

    for handle in handles {
        let _ = handle.join();
    }

    let model_w = model_wins.load(Ordering::Relaxed);
    let random_w = random_wins.load(Ordering::Relaxed);
    let draw_count = draws.load(Ordering::Relaxed);
    let win_rate = if num_games == 0 {
        0.5
    } else {
        (model_w as f64 + 0.5 * draw_count as f64) / num_games as f64
    };
    let result = MatchResult {
        new_wins: model_w,
        old_wins: random_w,
        draws: draw_count,
        total_games: num_games,
        win_rate,
    };

    info!("Evaluation vs random complete: {}", result);

    Ok(result)
}
