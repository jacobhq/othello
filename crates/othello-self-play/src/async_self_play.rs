use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;

use anyhow::Result;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use tracing::debug;
use crate::async_mcts::search::SearchWorker;
use crate::async_mcts::tree::Tree;
use crate::eval_queue::{EvalQueue, EvalResult, GpuHandle};
use crate::neural_net::{load_model, nn_eval_batch};

use othello::othello_game::{Color, OthelloGame};

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
    debug!("Starting worker");
    let _gpu_thread = start_gpu_worker(eval_queue.gpu_handle(), model, 128);
    debug!("Worker started");

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
            debug!("Entering MCTS loop");
            // -----------------------------------------------------
            // MCTS setup
            // -----------------------------------------------------

            let tree = Arc::new(Tree::new());
            let search_handle = eval_queue.search_handle();

            let num_threads = num_cpus::get().max(1);
            let barrier = Arc::new(Barrier::new(num_threads));

            let mut workers = Vec::new();

            for _ in 0..num_threads {
                let tree = Arc::clone(&tree);
                let handle = search_handle.clone();
                let barrier = Arc::clone(&barrier);
                let root_game = game.clone();

                workers.push(thread::spawn(move || {
                    let mut worker = SearchWorker::new((*tree).clone(), handle);

                    barrier.wait();

                    let sims_per_thread = sims_per_move / num_threads as u32;

                    for _ in 0..sims_per_thread {
                        worker.simulate(&root_game);
                        worker.poll_results();
                    }
                }));
            }

            for w in workers {
                w.join().unwrap();
            }

            // -----------------------------------------------------
            // Extract policy from visit counts
            // -----------------------------------------------------
            debug!("About to extract policy");
            let mut policy = vec![0.0f32; 64];
            let visits = tree.child_visits(tree.root());

            if visits.is_empty() {
                // Fallback: uniform over legal moves
                let legal_moves = game.legal_moves(current_player);

                let p = 1.0 / legal_moves.len() as f32;
                for (row, col) in legal_moves {
                    policy[row * 8 + col] = p;
                }
            } else {
                let visits_clone = visits.clone();
                let total_visits: u32 = visits_clone.iter().map(|(_, v)| *v).sum::<u32>().max(1);
                for (action, count) in visits_clone {
                    policy[action] = count as f32 / total_visits as f32;
                }
            }

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
            // Play move (sample from policy)
            // -----------------------------------------------------
            debug!("Making a dist over policy");
            debug!("{policy:?}");
            let dist = WeightedIndex::new(&policy)?;
            debug!("Made a dist over policy");
            let mut rng = rng();
            let action = dist.sample(&mut rng);

            let row = action / 8;
            let col = action % 8;

            game.play(row, col, current_player);
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
        let mut model =
            load_model(model_path.to_str().unwrap()).expect("failed to load model");

        loop {
            // Pop a batch of requests
            let batch = gpu.pop_batch(max_batch_size);

            if batch.is_empty() {
                std::thread::yield_now();
                continue;
            }

            // Extract states and IDs from requests
            let states: Vec<Vec<f32>> = batch.iter().map(|req| req.state.clone()).collect();
            let ids: Vec<u64> = batch.iter().map(|req| req.id).collect();

            // Evaluate the batch with the neural network
            let evals = nn_eval_batch(&mut model, &states)
                .expect("nn_eval_batch failed");

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
            gpu.push_results(results);
        }
    })
}
