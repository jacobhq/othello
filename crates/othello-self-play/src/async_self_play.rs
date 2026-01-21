use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;

use crate::async_mcts::search::SearchWorker;
use crate::async_mcts::tree::Tree;
use crate::eval_queue::{EvalQueue, EvalResult, GpuHandle};
use crate::neural_net::{load_model, nn_eval_batch};
use anyhow::{Result, anyhow};
use othello::othello_game::{Color, Move, OthelloGame};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use tracing::debug;

/// Represents a single self-play training sample
#[derive(Clone)]
pub struct Sample {
    pub state: [[[i32; 8]; 8]; 2], // encoded board
    pub policy: Vec<f32>,          // length 64
    pub value: f32,                // final game result
}

pub fn generate_self_play_data(
    prefix: &str,
    games: usize,
    sims_per_move: u32,
    model: PathBuf,
) -> Result<Vec<Sample>> {
    let mut all_samples = Vec::new();

    // ------------------------------------------------------------
    // Shared NN eval infrastructure
    // ------------------------------------------------------------

    let eval_queue = EvalQueue::new();

    // Start GPU worker once
    let _gpu_thread = start_gpu_worker(eval_queue.gpu_handle(), model, 128);

    // ------------------------------------------------------------
    // Self-play games
    // ------------------------------------------------------------

    for game_idx in 0..games {
        debug!("Entering game {game_idx}");
        let mut game = OthelloGame::new();
        let mut current_player = Color::Black;

        // Store (sample, player_at_time)
        let mut game_samples: Vec<(Sample, Color)> = Vec::new();

        while !game.game_over() {
            // -----------------------------------------------------
            // MCTS setup
            // -----------------------------------------------------

            let tree = Tree::new();
            let search_handle = eval_queue.search_handle();

            let num_threads = 1;
            let barrier = Arc::new(Barrier::new(num_threads));

            let mut workers = Vec::new();

            for _ in 0..num_threads {
                let tree = tree.clone();
                let handle = search_handle.clone();
                let barrier = Arc::clone(&barrier);
                let root_game = game;

                workers.push(thread::spawn(move || {
                    let mut worker = SearchWorker::new(tree, handle);

                    barrier.wait();

                    let sims_per_thread = (sims_per_move / num_threads as u32).max(1);

                    for _ in 0..sims_per_thread {
                        worker.simulate(&root_game);
                        worker.poll_results();
                    }

                    while worker.has_pending() {
                        worker.poll_results();
                        std::thread::yield_now();
                    }

                    // Drain any outstanding evaluations to ensure the shared tree
                    // gets expanded before we read visit counts.
                    // Finish all evals this worker issued
                    while worker.has_pending() {
                        worker.poll_results();
                        std::thread::yield_now();
                    }

                    debug!(
                        "After draining iterations: pending={}, has_results={}",
                        worker.has_pending(),
                        worker.eval_queue.has_results()
                    )
                }));
            }

            for w in workers {
                w.join().unwrap();
            }

            // -----------------------------------------------------
            // Extract policy from visit counts
            // -----------------------------------------------------
            let mut policy = vec![0.0f32; 64];

            let visits = tree.child_visits(tree.root());
            debug!(
                "Tree visits: {} actions with total {} visits",
                visits.len(),
                visits.iter().map(|(_, v)| *v).sum::<u32>()
            );
            let total_visits: u32 = visits.iter().map(|(_, v)| *v).sum::<u32>().max(1);
            for (action, count) in visits {
                policy[action] = count as f32 / total_visits as f32;
            }

            // -----------------------------------------------------
            // Store training sample
            // -----------------------------------------------------

            let sample = Sample {
                state: game.encode(current_player),
                policy: policy.clone(),
                value: 0.0,
            };

            game_samples.push((sample, current_player));

            // -----------------------------------------------------
            // Play move (sample from policy or pass)
            // -----------------------------------------------------
            let legal_moves = game.legal_moves(current_player);

            if legal_moves.is_empty() {
                game.mcts_play(Move::Pass, current_player)
                    .map_err(|e| anyhow!("pass move failed: {e:?}"))?;
            } else {
                // Pull probabilities only for legal moves; if the tree ended up
                // with zero visits (e.g. sims_per_move too small), fall back to uniform.
                let mut legal_probs: Vec<f32> =
                    legal_moves.iter().map(|(r, c)| policy[r * 8 + c]).collect();

                debug!(
                    "Legal moves: {} total, policy sum: {}",
                    legal_moves.len(),
                    legal_probs.iter().sum::<f32>()
                );

                let sum_probs: f32 = legal_probs.iter().copied().sum();
                if sum_probs <= f32::EPSILON {
                    let uniform = 1.0 / legal_probs.len() as f32;
                    for p in &mut legal_probs {
                        *p = uniform;
                    }
                }

                let dist = WeightedIndex::new(&legal_probs)
                    .map_err(|e| anyhow!("failed to build policy dist: {e}"))?;
                let mut rng = rng();
                let choice = dist.sample(&mut rng);
                let (row, col) = legal_moves[choice];

                game.mcts_play(Move::Move(row, col), current_player)
                    .map_err(|e| anyhow!("mcts_play failed: {e:?}"))?;
            }

            current_player = match current_player {
                Color::Black => Color::White,
                Color::White => Color::Black,
            };
        }

        // ---------------------------------------------------------
        // Backfill values
        // ---------------------------------------------------------

        let (black_score, white_score) = game.score();
        let outcome = match black_score.cmp(&white_score) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
        };

        for (mut sample, player) in game_samples {
            sample.value = match player {
                Color::Black => outcome,
                Color::White => -outcome,
            };
            all_samples.push(sample);
        }

        println!(
            "[{}] finished game {} (total samples {})",
            prefix,
            game_idx,
            all_samples.len()
        );
    }

    Ok(all_samples)
}

pub fn start_gpu_worker(
    gpu: GpuHandle,
    model_path: PathBuf,
    max_batch_size: usize,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        debug!(
            "GPU worker: starting, attempting to load model from {:?}",
            model_path
        );
        let mut model = load_model(model_path.to_str().unwrap()).expect("failed to load model");
        debug!("GPU worker: model loaded successfully");

        loop {
            // Pop a batch of requests
            let batch = gpu.pop_batch(max_batch_size);

            if batch.is_empty() {
                // debug!("No batch available, yielding...");
                std::thread::yield_now();
                continue;
            }
            debug!("GPU worker: processing batch of {} requests", batch.len());

            // Extract states and IDs from requests
            let states: Vec<Vec<f32>> = batch.iter().map(|req| req.state.clone()).collect();
            let ids: Vec<u64> = batch.iter().map(|req| req.id).collect();

            // Evaluate the batch with the neural network
            let evals = match nn_eval_batch(&mut model, &states) {
                Ok(evals) => {
                    debug!("GPU worker: batch evaluation succeeded");
                    evals
                }
                Err(e) => {
                    debug!("GPU worker: batch evaluation failed: {:?}", e);
                    return;
                }
            };

            // Convert NN output to EvalResults
            let mut results = Vec::with_capacity(evals.len());
            for ((policy_map, value), id) in evals.into_iter().zip(ids) {
                // Flatten the policy map to (action_index, probability) pairs
                let policy_flat: Vec<(usize, f32)> = policy_map
                    .into_iter()
                    .map(|((r, c), p)| (r * 8 + c, p))
                    .collect();

                results.push(EvalResult {
                    id,
                    policy: policy_flat,
                    value,
                });
            }

            // Push results back to the queue
            debug!("GPU worker: pushing {} results back", results.len());
            gpu.push_results(results);
        }
    })
}
