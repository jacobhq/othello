use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;

use crate::async_mcts::search::SearchWorker;
use crate::async_mcts::tree::Tree;
use crate::eval_queue::{EvalQueue, EvalResult, GpuHandle};
use crate::neural_net::{load_model, nn_eval_batch};
use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use othello::othello_game::{Color, Move, OthelloGame};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use tracing::{debug, info, info_span};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use crate::symmetry::get_symmetries;

/// Represents a single self-play training sample
#[derive(Clone)]
pub struct Sample {
    pub state: [[[i32; 8]; 8]; 2], // encoded board
    pub policy: Vec<f32>,          // length 64
    pub value: f32,                // final game result
}

fn play_one_game(
    game_idx: usize,
    prefix: &str,
    sims_per_move: u32,
    eval_queue: EvalQueue,
    tree_threads: usize,
    iteration: u32,
) -> Result<Vec<Sample>> {
    debug!("Entering game {game_idx}");
    let mut game = OthelloGame::new();

    let mut game_samples: Vec<(Sample, Color)> = Vec::new();
    let mut move_number = 0u32;

    while !game.game_over() {
        let current_player = game.current_turn;

        // ---------------- MCTS ----------------
        let tree = Tree::new();
        let search_handle = eval_queue.search_handle();

        // --- First, expand the root with a single simulation ---
        {
            let mut init_worker = SearchWorker::new(tree.clone(), search_handle.clone());
            init_worker.simulate(&game);
            while init_worker.has_pending() {
                init_worker.poll_results();
                std::thread::yield_now();
            }
        }

        // --- Add Dirichlet noise to the root node for exploration ---
        // Use less noise in early iterations (0-2) when the network is weak/random
        // This lets MCTS concentrate visits on promising moves
        let dirichlet_epsilon = if iteration <= 2 { 0.15 } else { 0.25 };
        tree.add_dirichlet_noise(tree.root(), 0.3, dirichlet_epsilon);

        // --- Run the rest of the simulations in parallel ---
        let num_threads = tree_threads;
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

                // Subtract 1 because we already did one simulation for root expansion
                let remaining_sims = sims_per_move.saturating_sub(1);
                let sims_per_thread = (remaining_sims / num_threads as u32).max(1);

                for _ in 0..sims_per_thread {
                    worker.simulate(&root_game);
                    worker.poll_results();
                }

                while worker.has_pending() {
                    worker.poll_results();
                    std::thread::yield_now();
                }
            }));
        }

        for w in workers {
            w.join().unwrap();
        }

        // ---------------- Policy extraction ----------------
        let mut policy = vec![0.0f32; 64];
        let visits = tree.child_visits(tree.root());
        let total_visits: u32 = visits.iter().map(|(_, v)| *v).sum::<u32>().max(1);

        for (action, count) in visits {
            policy[action] = count as f32 / total_visits as f32;
        }

        // ---------------- Store sample ----------------
        game_samples.push((
            Sample {
                state: game.encode(current_player),
                policy: policy.clone(),
                value: 0.0,
            },
            current_player,
        ));

        // ---------------- Play move ----------------
        let legal_moves = game.legal_moves(current_player);

        if legal_moves.is_empty() {
            game.mcts_play(Move::Pass, current_player).unwrap();
        } else {
            // Temperature annealing: use temp=1.0 for first 10 moves (exploration),
            // then temp=0.05 for remaining moves (exploitation)
            let temperature = if move_number < 10 { 1.0f32 } else { 0.05f32 };
            
            let mut probs: Vec<f32> = legal_moves
                .iter()
                .map(|(r, c)| {
                    let p = policy[r * 8 + c];
                    if temperature < 1.0 {
                        // Apply temperature: p^(1/T), then re-normalize
                        p.powf(1.0 / temperature)
                    } else {
                        p
                    }
                })
                .collect();

            let sum: f32 = probs.iter().sum();
            if sum <= f32::EPSILON {
                let u = 1.0 / probs.len() as f32;
                probs.iter_mut().for_each(|p| *p = u);
            } else {
                // Normalize after temperature scaling
                probs.iter_mut().for_each(|p| *p /= sum);
            }

            let dist = WeightedIndex::new(&probs)?;
            let choice = dist.sample(&mut rng());
            let (r, c) = legal_moves[choice];
            game.mcts_play(Move::Move(r, c), current_player).unwrap();
        }
        move_number += 1;
    }

    // ---------------- Backfill values ----------------
    let (b, w) = game.score();
    let outcome = match b.cmp(&w) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => -1.0,
        _ => 0.0,
    };

    let samples = game_samples
        .into_iter()
        .map(|(mut s, p)| {
            s.value = if p == Color::Black { outcome } else { -outcome };
            s
        })
        .flat_map(get_symmetries)
        .collect::<Vec<_>>();

    info!(
        "Finished game {} (prefix: {}, samples {})",
        game_idx,
        prefix,
        samples.len()
    );

    Ok(samples)
}

pub fn generate_self_play_data(
    prefix: &str,
    games: usize,
    sims_per_move: u32,
    model: PathBuf,
    game_threads: usize,
    tree_threads: usize,
    iteration: u32,
) -> Result<Vec<Sample>> {
    // ---------------- Shared GPU infra ----------------
    let eval_queue = EvalQueue::new();
    let _gpu_thread = start_gpu_worker(eval_queue.gpu_handle(), model, 128);

    let span = info_span!("self_play");
    span.pb_set_length(games as u64);
    span.pb_set_style(
        &ProgressStyle::with_template(
            "{span_child_prefix}[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} games"
        )?
            .progress_chars("##-"),
    );
    let _guard = span.enter();

    // ---------------- Parallel self-play ----------------
    let pool = ThreadPoolBuilder::new()
        .num_threads(game_threads) // Set N = 4
        .build()?;

    let all_samples: Result<Vec<Sample>> = pool.install(|| {
        (0..games)
            .into_par_iter()
            .map(|game_idx| {
                let res = play_one_game(
                    game_idx,
                    prefix,
                    sims_per_move,
                    eval_queue.clone(),
                    tree_threads,
                    iteration,
                );
                span.pb_inc(1);
                res
            })
            .try_reduce(Vec::new, |mut acc, mut samples| {
                acc.append(&mut samples);
                Ok(acc)
            })
    });

    all_samples
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
